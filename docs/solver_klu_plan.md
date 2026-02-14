# KLU 求解器规划

本文档定义 RustSpice 的线性求解器实现方案，采用 SuiteSparse 的 KLU。

## 1. 调用链流程

```
构建 CSC 矩阵 (Ap, Ai, Ax)
        |
        v
klu_defaults(&mut common)
        |
        v
klu_analyze(n, Ap, Ai, &mut common)  // 仅结构变化时
        |
        v
klu_factor(Ap, Ai, Ax, symbolic, &mut common)
        |
        v
klu_solve(symbolic, numeric, n, 1, b, &mut common)
```

说明:
- `Ap/Ai` 结构不变时可复用 `symbolic`
- 仅数值变化时重复 `klu_factor` + `klu_solve`

## 2. CSC 稀疏矩阵结构

- `n`: 矩阵维度
- `Ap: Vec<i64>`（列指针，长度 n+1）
- `Ai: Vec<i64>`（行索引）
- `Ax: Vec<f64>`（数值）

要求:
- 行索引在每列内递增
- 不允许重复项（后续需要合并）

## 3. SparseBuilder 设计

### 3.1 目标
- 用于构建稀疏矩阵结构和数值
- 支持“结构固定、数值更新”模式

### 3.2 数据结构草案
- `n: usize`
- `col_entries: Vec<Vec<(usize, f64)>>`
- `index_map: Vec<HashMap<usize, usize>>`（可选，用于快速更新）

### 3.3 核心方法
- `insert(col, row, value)`：累计/覆盖
- `finalize() -> (Ap, Ai, Ax)`：生成 CSC
- `clear_values()`：保留结构，清零数值
- `update(col, row, value)`：结构固定时更新

## 4. Rust 层封装

建议在 `crates/sim-core/src/solver/klu.rs` 封装:

### 4.1 Solver 接口草案

```
struct KluSolver {
    n: usize,
    symbolic: *mut klu_symbolic,
    numeric: *mut klu_numeric,
    common: klu_common,
    ap: Vec<i64>,
    ai: Vec<i64>,
}
```

### 4.2 公开方法

- `new(n, ap, ai) -> Self`
- `analyze()`（仅结构变化时）
- `factor(ax: &[f64])`
- `solve(b: &mut [f64])`（原地求解）
- `reset_pattern()`（释放并重建 symbolic）
- `drop` 时释放 `symbolic/numeric`

### 4.3 Trait 设计（可选）

```
trait LinearSolver {
    fn factor(&mut self, ax: &[f64]) -> Result<(), SolverError>;
    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError>;
}
```

### 4.4 错误类型

- `SolverError::AnalyzeFailed`
- `SolverError::FactorFailed`
- `SolverError::SolveFailed`

## 5. 依赖与构建

- 依赖 SuiteSparse (KLU)
- Windows 需要预编译或本地构建 KLU

## 7. 构建集成说明

当前采用 `feature = "klu"` 开关启用 KLU:

```
cargo build -p sim-core --features klu
```

构建脚本 `build.rs` 会读取:

- `KLU_INCLUDE_DIR`
- `KLU_LIB_DIR`
- `SUITESPARSE_DIR`

## 6. SparseBuilder 接口草案

```
struct SparseBuilder {
    n: usize,
    col_entries: Vec<Vec<(usize, f64)>>,
    index_map: Vec<HashMap<usize, usize>>,
}
```

方法建议:

- `new(n) -> Self`
- `insert(col, row, value)`：累计或覆盖
- `finalize() -> (Ap, Ai, Ax)`：导出 CSC
- `clear_values()`：结构不变，仅清零
- `update(col, row, value)`：结构固定时更新
