# BBD (Bordered Block Diagonal) Solver

BBD 求解器通过图分割将稀疏矩阵分解为多个独立的对角子块和一个较小的 border（分隔节点），利用 Schur 补方法求解。与 BTF 不同，BBD 不依赖强连通分量，而是主动切割矩阵图结构，因此对真实电路矩阵（通常只有一个大 SCC）更有效。

---

## 矩阵结构

BBD 排列后的矩阵形式：

```
┌──────┬──────┬──────┬────────┐
│ B₁₁  │  0   │  0   │  C₁   │
├──────┼──────┼──────┼────────┤
│  0   │ B₂₂  │  0   │  C₂   │
├──────┼──────┼──────┼────────┤
│  0   │  0   │ B₃₃  │  C₃   │
├──────┼──────┼──────┼────────┤
│  R₁  │  R₂  │  R₃  │  B_T  │
└──────┴──────┴──────┴────────┘
```

- **B_kk**: 独立对角子块，可独立做 LU 分解
- **C_k**: 子块行 → border 列的耦合 (block_size × border_size)
- **R_k**: border 行 → 子块列的耦合 (border_size × block_size)
- **B_T**: border 自身的连接 (border_size × border_size)

## 求解算法（Schur 补方法）

### Factor 阶段

1. 对每个对角块 B_kk 做 SparseLU 分解
2. 提取 C_k、R_k、B_T 的数值
3. 计算 Schur 补：`S = B_T - Σ_k R_k · B_kk⁻¹ · C_k`
   - 逐列处理：对 C_k 的每列 c_j，求解 `B_kk · y_j = c_j`，再做 `S[:,j] -= R_k · y_j`
4. 对 Schur 补矩阵 S 做 dense LU 分解

### Solve 阶段

1. 应用行排列到 RHS
2. 对每个子块 k：求解 `B_kk · z_k = b_k`（临时结果）
3. 修正 border RHS：`b'_T = b_T - Σ_k R_k · z_k`
4. 求解 border 系统：`S · x_T = b'_T`
5. 回代：对每个子块 `x_k = z_k - B_kk⁻¹ · (C_k · x_T)`
6. 应用逆列排列

## 分割算法

通过 `Partitioner` trait 抽象，可插拔不同的分割实现。

### GreedyBisectionPartitioner（当前默认）

基于 BFS level-set 的递归二分法：

1. 从度最小的节点开始 BFS，生成层级集
2. 按层级中点切分为两组
3. 割边上的节点提升为 border
4. 递归二分最大的子块，直到达到目标块数

复杂度：O((n + nnz) · log₂(num_blocks))

### 分割生效条件

`SolverSelector` 在以下条件下选择 BBD：
- n ≥ 200（矩阵足够大）
- density < 15%（稀疏矩阵）
- BTF 未检测到有效块结构
- border_ratio < 50%（否则自动降级为 SparseLU）

## 文件结构

| 文件 | 说明 |
|------|------|
| `crates/sim-core/src/bbd.rs` | Partitioner trait、GreedyBisectionPartitioner、BbdDecomposition |
| `crates/sim-core/src/bbd_solver.rs` | BbdSolver (LinearSolver 实现)、DenseSchurSolver |
| `crates/sim-core/src/solver.rs` | SolverType::Bbd、create_bbd_solver()、选择逻辑 |

## 使用方式

```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

// 直接指定 BBD
let mut solver = create_solver(SolverType::Bbd, n);

// 或通过工厂函数控制分块数
use sim_core::solver::create_bbd_solver;
let mut solver = create_bbd_solver(n, 4); // 目标 4 个子块
```

也可以通过 `SolverType::Auto` 自动选择——大矩阵且无 KLU/Faer 时会自动启用。

---

## 实现状态

### 已完成

- [x] `Partitioner` trait 可插拔分割接口
- [x] `GreedyBisectionPartitioner` BFS 递归二分
- [x] `BbdDecomposition` 结构（block_nodes、border_nodes、perm/inv_perm、block_ptr）
- [x] `bbd_decompose()` 分割 + border 提升 + 排列构建
- [x] `should_use_bbd()` 启发式
- [x] `BbdSolver` 实现完整 `LinearSolver` trait（analyze/factor/solve）
- [x] Dense Schur 补求解器
- [x] 不划算时自动降级为 SparseLU
- [x] 集成到 `SolverType` 枚举和 `SolverSelector`
- [x] 18 个单元测试全部通过

### 已知限制

| 限制 | 影响 | 说明 |
|------|------|------|
| 测试矩阵偏小 | 中 | 最大测到 8×8，未在 n≥200 真实电路上验证 |
| Schur 补用 dense | 中 | border 较大时 O(bs³) 开销高 |
| 分割质量有限 | 中 | GreedyBisection 对复杂电路拓扑不如 METIS |
| 无并行化 | 低 | 子块 factor/solve 串行执行 |
| 无 refactorization | 低 | 模式不变时仍重新提取+分解 |
| 非对称矩阵 border 检测 | 低 | 高度非对称矩阵可能漏掉部分 border 节点 |

### TODO（按优先级）

#### P0 — 端到端验证

- [ ] 用 `tests/fixtures/netlists/` 中的实际电路跑 DC 分析，对比 BBD 与其他 solver 结果
- [ ] 构造 n=200+ 合成矩阵（大规模 RC ladder/mesh）做对比测试
- [ ] 补充非对称矩阵测试用例

#### P1 — 实用性优化

- [ ] 大 border 时用稀疏 Schur 补求解器（替代 dense LU）
- [ ] 改进分割质量：KL 边交换精化 / 多级分割
- [ ] 对接 METIS 库（`MetisPartitioner`）

#### P2 — 性能优化

- [ ] 用 rayon 并行化子块 factor 和 solve
- [ ] 缓存 B_kk⁻¹ · C_k 避免 Schur 补重复计算
- [ ] Refactorization：模式不变时只更新数值，跳过 symbolic analysis
- [ ] 块内子矩阵提取避免每次 factor 重新分配内存

#### P3 — 完善集成

- [ ] `SolverSelector` 增加 pre-check：先试分割，检查 border_ratio < 0.3 才选 BBD
- [ ] 运行时统计收集（block 大小分布、border ratio、Schur 补耗时）
- [ ] BBD 求解器性能 benchmark（与 SparseLU、KLU 对比）

## 参考文献

- Saad, Y. "Iterative Methods for Sparse Linear Systems", Chapter 3: Sparse Matrices — BBD and domain decomposition
- Karypis, G., Kumar, V. "A Fast and High Quality Multilevel Scheme for Partitioning Irregular Graphs" — METIS 算法
- Davis, T.A. "Direct Methods for Sparse Linear Systems" — Schur complement methods
