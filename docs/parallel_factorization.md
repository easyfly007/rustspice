# Native KLU 并行分解技术文档

本文档以 `native_klu.rs` 中的列级并行 LU 分解为例，详细解释 Rust 多线程编程的核心机制。

## 1. 概述

Native KLU 求解器在**重分解**（refactorization）阶段支持列级并行。当矩阵稀疏模式不变、仅数值改变时（SPICE 中 Newton 迭代的常见情况），多个线程可以同时处理不同列的分解。

实现采用**持久线程池**（persistent thread pool）：工作线程在首次并行分解时创建，之后在所有后续的 `factor()` 调用中复用，直到求解器被销毁。这消除了每次分解时创建/销毁 OS 线程的开销（约 70-100ms），使并行分解在较小矩阵（n > ~10K）上也能获得加速。

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

## 3. 持久线程池实现详解

以下按线程池的创建、工作分发、列处理逐段解释。

> 源码位置：`crates/sim-core/src/native_klu.rs`

### 3.1 线程池生命周期

```
NativeKluSolver::new()          thread_pool = None（尚未创建）
        │
        ▼
第一次 factor() 调用            顺序分解，无线程池
        │
        ▼
第二次 factor() 调用            若 parallel_threads >= 2 且块大小 >= 64：
（同一稀疏模式 = 重分解）         └─ 惰性创建线程池（FactorThreadPool::new()）
        │                        └─ 创建 N 个 OS 线程，进入 worker_loop 等待
        ▼
第 3, 4, 5... 次 factor()      复用同一个线程池，线程不重新创建
        │                        └─ 只传递 WorkDescriptor，唤醒线程，等待完成
        ▼
NativeKluSolver::drop()         FactorThreadPool::drop()
                                 └─ 设置 shutdown = true
                                 └─ 唤醒所有线程
                                 └─ join 所有线程（等待退出）
```

**关键点**：
- **惰性创建**：线程池不在 `NativeKluSolver::new()` 时创建，而是等到第一次需要并行分解时才创建
- **线程数变化**：如果用户在运行中改变了 `solver_parallel` 的值，旧的线程池会被 drop（join 所有线程），然后创建新的线程池
- **顺序分解时的开销**：当走顺序分解路径时（首次分解、小块、parallel_threads < 2），工作线程在 condvar 上休眠，CPU 开销为零

### 3.2 核心数据结构

#### WorkDescriptor —— 每次分解的工作描述

```rust
struct WorkDescriptor {
    n: usize,                    // 矩阵维数
    num_threads: usize,          // 线程数
    // 符号因子的裸指针（只读）
    col_perm_inv: *const usize,
    col_perm: *const usize,
    l_col_ptr: *const usize,     // L 矩阵列指针
    // ... 更多只读指针 ...
    // 输出缓冲区的裸指针（各列写入不重叠）
    l_values: *mut f64,
    u_values: *mut f64,
    u_diag: *mut f64,
    // 同步标志
    ready: *const AtomicBool,
}
unsafe impl Send for WorkDescriptor {}
```

`WorkDescriptor` 包含裸指针指向 `factor_block_parallel()` 栈上的数据。**安全论据**：主线程通过 condvar 阻塞直到所有工作线程完成，因此所有指针指向的数据在工作线程访问期间保证存活。

#### SharedPoolState —— Mutex 保护的共享状态

```rust
struct SharedPoolState {
    work: Option<WorkDescriptor>,  // 当前工作项（None = 空闲）
    generation: u64,               // 每次 dispatch 递增
    done_count: usize,             // 本轮已完成的工作线程数
    num_workers: usize,            // 总工作线程数
    shutdown: bool,                // 终止标志
}
```

`generation` 计数器是核心同步机制：
- 主线程递增 `generation` 并设置 `work = Some(desc)` 来分发工作
- 工作线程比较自己上次看到的 generation 来检测新工作
- `done_count == num_workers` 时主线程被唤醒

#### FactorThreadPool —— 持有线程和共享状态

```rust
struct FactorThreadPool {
    shared: Arc<(Mutex<SharedPoolState>, Condvar, Condvar)>,
    //                                    wake_workers  wake_main
    threads: Vec<Option<JoinHandle<()>>>,
    num_threads: usize,
}
```

使用两个 Condvar 分别用于：
- `wake_workers`：主线程唤醒工作线程开始处理
- `wake_main`：最后一个完成的工作线程唤醒主线程

### 3.3 通信协议

```
主线程                               工作线程（持久存在）
    │                                     │ (在 wake_workers condvar 上休眠)
    ├─ lock mutex                         │
    ├─ work = Some(descriptor)            │
    ├─ generation += 1                    │
    ├─ done_count = 0                     │
    ├─ notify_all(wake_workers) ─────────→├─ 醒来，读取 WorkDescriptor
    ├─ wait(wake_main) ←─ (阻塞)          ├─ 处理交错列
    │                                     ├─ 自旋等待 ready[] 获取依赖
    │                                     ├─ lock mutex, done_count += 1
    │                                     ├─ 若是最后一个: notify(wake_main) ───→├─ 主线程醒来
    │                                     ├─ 回到循环继续休眠                     ├─ work = None
    │                                                                           ├─ 返回 NumericFactor
```

### 3.4 工作线程主循环

每个工作线程在 `worker_loop()` 中无限循环：

```rust
loop {
    // 1. 等待新工作或关闭信号（在 condvar 上休眠，CPU 开销为零）
    let mut state = mutex.lock().unwrap();
    loop {
        if state.shutdown { return; }                     // 收到关闭信号，退出
        if state.generation > last_generation {           // 有新工作
            last_generation = state.generation;
            desc_copy = /* 复制 WorkDescriptor */;
            break;
        }
        state = wake_workers.wait(state).unwrap();        // 继续休眠
    }

    // 2. 调整持久工作区大小（首次分配后复用，不重新分配）
    work_buf.resize(n, 0.0);

    // 3. 处理属于自己的列（交错分配）
    execute_columns(tid, &desc_copy, &mut work_buf);

    // 4. 通知完成
    state.done_count += 1;
    if state.done_count == state.num_workers {
        wake_main.notify_one();                           // 唤醒主线程
    }
}
```

**持久工作区**：每个工作线程拥有一个 `Vec<f64>` 工作缓冲区。它在线程的整个生命周期中复用 —— 首次分配后，后续调用只需 `resize`（通常不会重新分配，因为大小相同或已经足够大）。

### 3.5 为什么使用 Mutex + Condvar 而不是 channel？

对比：

| 方案 | 优点 | 缺点 |
|------|------|------|
| **Mutex + Condvar** | 零分配分发（复制一个 WorkDescriptor 值）；精确控制唤醒时机 | 需要手动管理 generation 计数 |
| `mpsc::channel` | API 更简单 | 每次 send 需要堆分配；无法广播给所有线程 |
| `crossbeam` channel | 高性能 | 引入外部依赖 |

选择 Mutex + Condvar 是因为：无外部依赖、零分配、一次 `notify_all` 即可唤醒所有工作线程。

### 3.6 安全的共享缓冲区写入

**问题**：多个线程需要同时写入 `l_values`、`u_values`、`u_diag`。在安全 Rust 中，同一时间只能有一个 `&mut` 指向这些 `Vec`，编译器不允许多线程共享可变引用。

**解决方案**：在 `factor_block_parallel()` 栈上分配缓冲区，通过 `WorkDescriptor` 的裸指针传递给工作线程。`WorkDescriptor` 通过 `unsafe impl Send` 让裸指针可以跨线程传递。

**安全论据**：
- 每列写入的索引范围互不重叠（列 k 写 `l_col_ptr[k]..l_col_ptr[k+1]`）
- 符号分解的结构保证这些范围不会重叠
- 读取列 j 的数据之前，一定先通过原子操作等待 `ready[j]` 变为 true
- 主线程通过 condvar 阻塞直到所有工作线程完成，保证缓冲区在工作线程访问期间存活

**为什么不用 Rayon？** Rayon 使用协作式工作窃取调度。如果线程 A 自旋等待线程 B 的结果，但两者在同一个 Rayon worker 上，就会**死锁** —— 自旋的线程无法释放 worker 去处理其他任务。持久线程池创建的是真正的 OS 线程，每个线程有独立的内核调度单元，自旋等待不会阻塞其他线程。

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
  FactorThreadPool（持久）
  ┌──────────────────────────────────────────────────────────────┐
  │ shared: Arc<(Mutex<SharedPoolState>, Condvar, Condvar)>      │
  │                                                              │
  │  工作线程 0               工作线程 1              工作线程 2   │
  │  ┌──────────────┐        ┌──────────────┐       ┌──────────┐ │
  │  │ work_buf (持久)│        │ work_buf (持久)│       │ work_buf │ │ ← 持久私有工作区
  │  └──────────────┘        └──────────────┘       └──────────┘ │
  └──────────────────────────────────────────────────────────────┘
              │                     │                     │
    每次 factor() 调用传入 WorkDescriptor（裸指针）：
              │                     │                     │
              ▼                     ▼                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ WorkDescriptor 指向的数据（在 factor_block_parallel 栈上）：  │
  │                                                             │
  │  只读：sym, ap, ai, ax, row_perm_inv                        │
  │  写入：l_values[...], u_values[...], u_diag[...]            │
  │  同步：ready[0..n] (AtomicBool)                             │
  └─────────────────────────────────────────────────────────────┘
```

每个线程：
- **读取**：`sym`、`ap`、`ai`、`ax`、`row_perm_inv`（只读，无竞争）
- **读写**：自己的持久 `work_buf[]` 工作区（私有，无竞争，跨调用复用）
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
│  • Mutex + Condvar → 线程同步和工作分发                     │
│  • Arc → 线程池共享状态的引用计数                            │
│  • Drop trait → 线程池销毁时自动 join 所有线程              │
│  • AtomicBool + Acquire/Release → 无锁的列间同步            │
│  • 每个线程有自己的持久 work_buf → 无共享可变状态             │
├───────────────────────────────────────────────────────────┤
│                     unsafe 逃逸口                          │
│  • WorkDescriptor 的裸指针 → "主线程阻塞直到所有工作完成"    │
│  • unsafe impl Send for WorkDescriptor → "指针跨线程安全"   │
│  • execute_columns 中裸指针解引用 → "不同列写入不重叠"       │
│  • 安全论据来自：算法不变量 + condvar 阻塞保证              │
└───────────────────────────────────────────────────────────┘
```

**核心思想**：Rust 不是禁止一切 unsafe —— 它把 unsafe **隔离**起来。线程池的创建、销毁、同步全部使用安全 Rust（`Mutex`、`Condvar`、`Arc`、`Drop`）。`unsafe` 只集中在 `execute_columns()` 中的裸指针解引用，有明确的文档说明安全理由。编译器保护你不犯低级错误，只在编译器能力不够的地方由程序员接管。

## 8. 参考

- [The Rustonomicon - Send and Sync](https://doc.rust-lang.org/nomicon/send-and-sync.html)
- [std::thread::scope 文档](https://doc.rust-lang.org/std/thread/fn.scope.html)
- [std::sync::atomic::Ordering](https://doc.rust-lang.org/std/sync/atomic/enum.Ordering.html)
- Davis, T.A. "Algorithm 907: KLU" ACM TOMS, 2010
- Gilbert, J.R., Peierls, T. "Sparse partial pivoting" SIAM J. Sci. Stat. Comput., 1988
