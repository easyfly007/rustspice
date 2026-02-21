# Native KLU 并行分解技术文档

本文档以 `native_klu.rs` 中的列级并行 LU 分解为例，详细解释 Rust 多线程编程的核心机制。

## 1. 概述

Native KLU 求解器在**重分解**（refactorization）阶段支持列级并行。当矩阵稀疏模式不变、仅数值改变时（SPICE 中 Newton 迭代的常见情况），多个线程可以同时处理不同列的分解。

**启用条件**（必须同时满足）：
1. 编译时启用 `parallel` feature（`--features parallel`）
2. `.option solver_parallel=N`，N ≥ 2
3. 矩阵块大小 ≥ 64
4. 重分解阶段（相同稀疏模式的第二次及后续分解）

**网表用法**：
```spice
.option solver=nativeklu solver_parallel=4
```

## 2. Rust 所有权与线程安全

### 2.1 为什么多线程在 C/C++ 中很危险？

在 C/C++ 中，任何指针都可以在线程间传递，编译器不会阻止。两个线程同时写同一块内存 = **数据竞争** = 未定义行为（程序崩溃、结果错误、偶发 bug）。

Rust 通过**所有权系统**在编译期杜绝数据竞争：

| 规则 | 含义 |
|------|------|
| 所有权（Ownership） | 每个值只有一个所有者 |
| 借用（Borrowing） | 要么有**一个**可变引用 `&mut T`，要么有**多个**不可变引用 `&T`，不能同时存在 |

违反这两条规则的代码**编译不过**，从根本上杜绝了数据竞争。

### 2.2 `Send` 和 `Sync` —— 线程安全标记

Rust 有两个特殊的 trait 来控制什么类型可以跨线程使用：

| Trait | 含义 | 满足条件的类型示例 |
|-------|------|-------------------|
| `Send` | 可以**移动**到另一个线程 | `Vec<f64>`, `String`, `i32` |
| `Sync` | 可以通过 `&T` 在多个线程间**共享** | `AtomicBool`, `Mutex<T>` |

大部分标准库类型自动实现了这两个 trait。但**裸指针 `*mut f64` 默认既不是 `Send` 也不是 `Sync`**。如果试图在线程闭包里捕获裸指针，编译器直接拒绝：

```
error: `*mut f64` cannot be sent between threads safely
```

## 3. 并行分解实现详解

以下按 `factor_block_parallel()` 的代码结构逐段解释。

> 源码位置：`crates/sim-core/src/native_klu.rs`

### 3.1 共享缓冲区与 SendPtr 包装器

```rust
// 预分配共享输出缓冲区
let mut l_values = vec![0.0f64; sym.l_row_idx.len()];
let mut u_values = vec![0.0f64; sym.u_row_idx.len()];
let mut u_diag = vec![0.0f64; n];

// 获取裸指针
let l_ptr = l_values.as_mut_ptr();
let u_ptr = u_values.as_mut_ptr();
let d_ptr = u_diag.as_mut_ptr();

// 包装器，让裸指针可以跨线程传递
struct SendPtr(*mut f64);
unsafe impl Send for SendPtr {}
unsafe impl Sync for SendPtr {}
```

**问题**：多个线程需要同时写入 `l_values`、`u_values`、`u_diag`。在安全 Rust 中，同一时间只能有一个 `&mut` 指向这些 `Vec`，编译器不允许多线程共享可变引用。

**解决方案**：使用裸指针 `*mut f64` 绕过借用检查，再用 `SendPtr` 包装器让指针可以跨线程传递。`unsafe impl Send/Sync` 是程序员向编译器做出的承诺："我保证这是安全的"。

**安全论据**：
- 每列写入的索引范围互不重叠（列 k 写 `l_col_ptr[k]..l_col_ptr[k+1]`）
- 符号分解的结构保证这些范围不会重叠
- 读取列 j 的数据之前，一定先通过原子操作等待 `ready[j]` 变为 true

### 3.2 `std::thread::scope` —— 作用域线程

```rust
thread::scope(|s| {
    for tid in 0..num_threads {
        let ready_ref = &ready;
        let sym_ref = sym;
        let l_s = &l_send;

        s.spawn(move || {
            // 线程体
        });
    }
});  // <-- 在这里阻塞，直到所有线程结束
```

**作用**：创建真正的 OS 线程，且**保证**在 `scope` 返回之前所有线程都已退出。

**为什么需要"作用域"线程？** 普通的 `std::thread::spawn` 要求所有捕获的数据必须是 `'static`（永久存活），因为编译器不知道线程什么时候结束。但作用域线程不同 —— 编译器知道 scope 结束时所有线程一定已退出，因此允许线程**借用**栈上的局部变量：

```
factor_block_parallel() 的栈帧
├── l_values: Vec<f64>        ← 在 scope 结束前一直存活
├── ready: Vec<AtomicBool>    ← 线程可以安全地引用它
├── sym: &SymbolicFactor      ← 线程可以安全地引用它
│
└── thread::scope {
        thread 0 ─→ 引用 &ready, &sym  ✓ 安全
        thread 1 ─→ 引用 &ready, &sym  ✓ 安全
        thread 2 ─→ 引用 &ready, &sym  ✓ 安全
    }  ← 所有线程在此 join，之后 l_values 等才被释放
```

**对比 C/pthreads**：在 C 中需要手动 `pthread_create` + `pthread_join`，编译器无法检查你是否正确 join 了线程。Rust 把 join 的保证编码进了类型系统。

**为什么不用 Rayon？** Rayon 使用协作式工作窃取调度。如果线程 A 自旋等待线程 B 的结果，但两者在同一个 Rayon worker 上，就会**死锁** —— 自旋的线程无法释放 worker 去处理其他任务。`std::thread::scope` 创建的是真正的 OS 线程，每个线程有独立的内核调度单元，自旋等待不会阻塞其他线程。

### 3.3 `move` 闭包

```rust
s.spawn(move || {
    let l_raw = l_s.0;
    let u_raw = u_s.0;
    let mut work = vec![0.0f64; n];
    // ...
});
```

`move` 关键字将闭包捕获的所有变量的所有权转移到闭包内部。这里实际被 move 的是：

| 变量 | 类型 | 开销 |
|------|------|------|
| `ready_ref` | `&Vec<AtomicBool>`（共享引用） | 复制一个指针，8 字节 |
| `sym_ref` | `&SymbolicFactor`（共享引用） | 复制一个指针，8 字节 |
| `l_s`, `u_s`, `d_s` | `&SendPtr`（指针的引用） | 复制一个指针 |
| `tid` | `usize` | 复制一个整数，8 字节 |

Move 引用非常廉价 —— 只是复制指针值，不复制底层数据。

### 3.4 交错列分配

```rust
// 线程 tid 处理列: tid, tid+T, tid+2T, ...
let mut k = tid;
while k < n {
    // 处理第 k 列
    k += num_threads;
}
```

假设 4 个线程处理 12 列：

```
线程 0: 列 0,  列 4,  列 8
线程 1: 列 1,  列 5,  列 9
线程 2: 列 2,  列 6,  列 10
线程 3: 列 3,  列 7,  列 11
```

**为什么用交错而不是连续分块？** 相邻列之间最可能存在依赖（列 k 常常依赖列 k-1、k-2 等）。交错分配让相邻列落在不同线程上 —— 当线程 t 需要列 k-1 的结果时，线程 t-1 很可能已经完成了它，从而最小化等待时间。

### 3.5 `AtomicBool` —— 无锁同步

```rust
// 创建：每列一个就绪标志
let ready: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

// 等待列 j 完成（读端）
while !ready_ref[j].load(Ordering::Acquire) {
    std::hint::spin_loop();
}

// 通知列 k 完成（写端）
ready_ref[k].store(true, Ordering::Release);
```

**为什么不用普通 `Vec<bool>`？** 普通 `bool` 不是 `Sync`，Rust 不允许两个线程同时读写同一个 `bool`。`AtomicBool` 保证每次读写都是**原子的**（不会出现"读了一半"的状态）。

#### 内存排序（Memory Ordering）

这是多线程编程中最微妙的部分。现代 CPU 会对指令进行重排序以提高性能，这意味着代码中写入的顺序 ≠ 其他线程看到的顺序。

```
线程 1（写者）：                    线程 2（读者）：
  ① 写入 L 值到缓冲区
  ② 写入 U 值到缓冲区
  ③ ready[k].store(true, Release)   ④ while !ready[k].load(Acquire) { 自旋 }
        │                                    │
        └──────── happens-before ────────────┘
                                    ⑤ 读取 L 值 ← 保证能看到步骤①②的写入
```

| 排序 | 保证 |
|------|------|
| `Release`（写端） | 这次 store **之前**的所有内存写入，对看到此 store 的线程可见 |
| `Acquire`（读端） | 看到 `Release` store 之后，能看到该线程在 store 之前的所有写入 |

两者配合建立了 **happens-before** 关系：线程 2 在看到 `ready[k] == true` 之后，保证能看到线程 1 在设置 ready 之前写入的所有 L/U 数据。

**如果不用这些排序会怎样？** CPU 可能重排指令 —— `ready` 标志可能比 L/U 数据先变得可见。线程 2 看到 `ready[k] == true`，但读到的 L 值还是旧的或未初始化的。这在 ARM 等弱排序架构上是真实存在的问题。

#### `std::hint::spin_loop()`

```rust
while !ready_ref[j].load(Ordering::Acquire) {
    std::hint::spin_loop();  // 提示 CPU：我在忙等
}
```

在 x86 上生成 `PAUSE` 指令，作用：
- 降低自旋时的功耗
- 让同一物理核心上的超线程获得更多执行资源
- 避免内存总线饱和

### 3.6 `unsafe` 块 —— 裸指针写入

```rust
// 安全 Rust 写法（编译不过 —— 多线程不能共享 &mut l_values）：
l_values[l_idx] = work[row] / diag;

// 实际使用的 unsafe 写法：
unsafe { *l_raw.add(l_idx) = work[row] / diag; }
```

等同于 C 语言的 `l_raw[l_idx] = work[row] / diag`。需要 `unsafe` 的原因：
1. 多个线程同时写入 `l_values` 缓冲区
2. Rust 的借用检查器在编译期无法证明写入范围不重叠
3. 程序员通过算法不变量（符号分解的列范围）保证安全

## 4. 线程间数据流

```
                    共享数据（只读）               共享数据（原子）
                 ┌─────────────────┐          ┌──────────────────┐
                 │ sym (符号因子)   │          │ ready[0..n]      │
                 │ ap, ai, ax      │          │ (AtomicBool)     │
                 │ row_perm_inv    │          └──────────────────┘
                 └─────────────────┘

  线程 0                线程 1                线程 2
  ┌──────────┐          ┌──────────┐          ┌──────────┐
  │ work[0..n]│          │ work[0..n]│          │ work[0..n]│  ← 私有工作区
  │（私有）    │          │（私有）    │          │（私有）    │
  └──────────┘          └──────────┘          └──────────┘
       │                     │                     │
       ▼                     ▼                     ▼
  ┌────────────────────────────────────────────────────┐
  │ l_values[...]    u_values[...]    u_diag[...]      │  ← 共享缓冲区
  │ (通过裸指针写入，每列范围互不重叠)                    │     (unsafe 写入)
  └────────────────────────────────────────────────────┘
```

每个线程：
- **读取**：`sym`、`ap`、`ai`、`ax`、`row_perm_inv`（只读，无竞争）
- **读写**：自己的 `work[]` 工作区（私有，无竞争）
- **写入**：`l_values`、`u_values`、`u_diag` 中属于自己处理的列的范围（互不重叠）
- **原子操作**：`ready[]` 标志（`Acquire`/`Release` 保证正确性）

## 5. 每列处理的 6 个步骤

```rust
while k < n {
    // 步骤 1: 将 A(:,k) 散射到线程私有的 work[] 中
    for idx in start..end {
        work[perm_row] += ax[idx];
    }

    // 步骤 2: 左看更新 —— 等待依赖列完成，应用 L(:,j) * U(j,k)
    for each dependency column j < k {
        while !ready[j].load(Acquire) { spin_loop(); }  // 等待
        for l_idx in L(:,j) {
            work[row] -= l_values[l_idx] * u_jk;        // 读取已完成列的 L 值
        }
    }

    // 步骤 3: 从 work[] 提取 U 值
    unsafe { *u_raw.add(u_idx) = work[row]; }

    // 步骤 4: 计算 L(:,k) = work[对角线以下] / 对角元
    unsafe { *l_raw.add(l_idx) = work[row] / diag; }

    // 步骤 5: 清零 work[]（仅清除本列触及的元素）

    // 步骤 6: 通知本列完成
    ready[k].store(true, Release);

    k += num_threads;  // 跳到下一个属于本线程的列
}
```

## 6. 选项控制

通过 `.option solver_parallel` 控制并行行为：

| 值 | 行为 |
|----|------|
| `0`（默认） | 关闭并行，使用顺序分解 |
| `1` | 关闭并行（单线程无意义） |
| `≥ 2` | 启用并行，使用指定数量的线程 |

线程数上限为矩阵块大小（线程数不可能超过列数）。

## 7. 安全模型总结

```
┌───────────────────────────────────────────────────────────┐
│                     安全 Rust 层                           │
│  • 所有权系统 → 编译期防止数据竞争                          │
│  • Send/Sync trait → 控制什么类型能跨线程                   │
│  • thread::scope → 保证 join，允许安全借用局部变量           │
│  • AtomicBool + Acquire/Release → 无锁的列间同步            │
│  • 每个线程有自己的 work[] → 无共享可变状态                   │
├───────────────────────────────────────────────────────────┤
│                     unsafe 逃逸口                          │
│  • SendPtr 包装器 → "我保证，不同列写入不重叠"               │
│  • 裸指针解引用 → "我保证，没有别名冲突"                     │
│  • 安全论据来自算法层面的不变量（符号分解的列范围）            │
└───────────────────────────────────────────────────────────┘
```

**核心思想**：Rust 不是禁止一切 unsafe —— 它把 unsafe **隔离**起来。`factor_block_parallel()` 中 95% 的代码是安全的 Rust，`unsafe` 集中在少数几行，有明确的文档说明安全理由。编译器保护你不犯低级错误，只在编译器能力不够的地方由程序员接管。

## 8. 参考

- [The Rustonomicon - Send and Sync](https://doc.rust-lang.org/nomicon/send-and-sync.html)
- [std::thread::scope 文档](https://doc.rust-lang.org/std/thread/fn.scope.html)
- [std::sync::atomic::Ordering](https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html)
- Davis, T.A. "Algorithm 907: KLU" ACM TOMS, 2010
- Gilbert, J.R., Peierls, T. "Sparse partial pivoting" SIAM J. Sci. Stat. Comput., 1988
