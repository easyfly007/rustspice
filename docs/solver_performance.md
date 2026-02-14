# Linear Solver Performance Guide

This document provides comprehensive information about the linear solvers available in RustSpice, their performance characteristics, limitations, and best practices.

## Overview

RustSpice provides five linear solver backends for circuit simulation:

| Solver | Feature Flag | Dependencies | Performance | Best For |
|--------|--------------|--------------|-------------|----------|
| **Dense** | (always available) | None | O(n³) | n < 100 nodes |
| **SparseLU** | (always available) | None (native Rust) | O(nnz·fill) | 100-5000 nodes |
| **SparseLU-BTF** | (always available) | None (native Rust) | O(nnz·fill), block-optimized | Block-structured circuits |
| **Faer** | `faer-solver` (default) | Pure Rust | O(nnz·fill) | General use |
| **KLU** | `klu` | SuiteSparse (C) | O(nnz·fill), fastest | Large circuits |

## Quick Start

### Default Setup (Faer - Recommended for Beginners)

```bash
# Build with default features (includes Faer)
cargo build

# Or explicitly enable
cargo build --features faer-solver
```

Faer is enabled by default and requires no external dependencies.

### High-Performance Setup (KLU)

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install libsuitesparse-dev

# macOS
brew install suite-sparse

# Build with KLU
cargo build --features klu
```

For Windows, see the [KLU Installation Guide](#klu-installation-on-windows) below.

---

## Solver Details

### 1. Dense Solver

**Description:** Standard LU factorization with partial pivoting.

**Complexity:**
- Factorization: O(n³)
- Solve: O(n²)
- Memory: O(n²)

**When to Use:**
- Small circuits (< 100 nodes)
- Fallback when sparse solvers unavailable
- Debugging and validation

**Limitations:**
- Memory grows quadratically with circuit size
- Impractical for circuits > 500 nodes
- No pattern reuse optimization

**Example:**
```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

let mut solver = create_solver(SolverType::Dense, n);
solver.prepare(n);
solver.analyze(&ap, &ai)?;
solver.factor(&ap, &ai, &ax)?;
solver.solve(&mut rhs)?;
```

---

### 2. SparseLU Solver (Native Rust)

**Description:** Native Rust sparse LU factorization with AMD ordering, optimized for circuit simulation matrices.

**Complexity:**
- Symbolic analysis: O(nnz + n·avg_degree) for AMD ordering
- Factorization: O(nnz · fill), typically O(n log n) for circuits
- Solve: O(nnz)
- Memory: O(nnz + fill)

**When to Use:**
- Medium-sized circuits (100-5000 nodes)
- When no external dependencies are desired
- As a fallback when Faer/KLU unavailable
- Debugging and validation

**Features:**
- Pure Rust implementation, no external dependencies
- Approximate Minimum Degree (AMD) ordering for fill reduction
- Left-looking factorization algorithm
- Pattern caching for repeated solves
- Pivot tolerance handling for near-singular matrices

**Limitations:**
- Slower than Faer (~1.5x) and KLU (~2x) for large matrices
- Simple AMD (not full approximate minimum degree with mass elimination)
- For block-structured matrices, use SparseLU-BTF instead

**Algorithm Details:**

The SparseLU solver uses a three-phase approach:

1. **AMD Ordering**: Computes a fill-reducing column permutation by simulating Gaussian elimination and always selecting the minimum-degree node:
   ```
   For i = 0 to n-1:
       p = node with minimum degree among uneliminated nodes
       perm[p] = i
       Eliminate p: connect all neighbors to form clique
       Update degrees of affected nodes
   ```

2. **Symbolic Analysis**: Determines L and U sparsity patterns without computing values:
   ```
   For each column k in permuted order:
       L_pattern[k] = rows in A(:,k) below diagonal
       For each j < k where L(k,j) ≠ 0:
           L_pattern[k] = L_pattern[k] ∪ L_pattern[j]  // Fill-in
       U_pattern[k] = rows above diagonal
   ```

3. **Left-Looking Factorization**: Computes L and U values column by column:
   ```
   For each column k:
       work = column k of permuted A
       For each j < k where U(j,k) ≠ 0:
           work -= L(:,j) * U(j,k)
       U(:,k) = work[rows < k]
       L(:,k) = work[rows > k] / U(k,k)
   ```

**Example:**
```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

let mut solver = create_solver(SolverType::SparseLu, n);
solver.prepare(n);
solver.analyze(&ap, &ai)?;  // AMD ordering + symbolic analysis
solver.factor(&ap, &ai, &ax)?;  // Numeric factorization
solver.solve(&mut rhs)?;  // Forward/backward solve
```

---

### 3. SparseLU-BTF Solver (Native Rust with Block Optimization)

**Description:** Native Rust sparse LU factorization with Block Triangular Form (BTF) decomposition, optimized for circuits with block structure.

**Complexity:**
- BTF decomposition: O(nnz + n) for matching and SCC
- Per-block factorization: O(nnz_block · fill_block)
- Solve: O(nnz) with block-wise operations
- Memory: O(nnz + fill) distributed across blocks

**When to Use:**
- Circuits with natural block structure (multi-stage amplifiers, cascaded filters)
- Systems with loosely coupled subcircuits
- Large circuits where BTF can identify independent blocks
- When the matrix has many 1×1 blocks (singletons)

**Features:**
- Pure Rust implementation, no external dependencies
- Automatic BTF detection and application
- Maximum transversal for structural rank detection
- Tarjan's algorithm for strongly connected components
- Falls back to standard SparseLU when BTF is not beneficial
- Block-wise factorization reduces fill-in

**How BTF Works:**

BTF permutes the matrix to upper block triangular form:

```
Original matrix:          After BTF:
[ * * * * * ]            [ B₁₁  U₁₂  U₁₃ ]
[ * * * * * ]      →     [  0   B₂₂  U₂₃ ]
[ * * * * * ]            [  0    0   B₃₃ ]
[ * * * * * ]
[ * * * * * ]
```

Benefits:
- Only diagonal blocks (B₁₁, B₂₂, B₃₃) need LU factorization
- For k equal-sized blocks, factorization is k² times faster
- 1×1 blocks (singletons) require no factorization at all
- Off-diagonal blocks are used only during solve phase

**Performance Gain:**

For a matrix with k blocks of roughly equal size n/k:
- Standard LU: O((n)³) operations
- BTF LU: O(k × (n/k)³) = O(n³/k²) operations

Example: A 1000-node circuit with 10 equal blocks:
- Standard: 1,000,000,000 operations
- BTF: 10,000,000 operations (100× faster)

**Example:**
```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

let mut solver = create_solver(SolverType::SparseLuBtf, n);
solver.prepare(n);
solver.analyze(&ap, &ai)?;  // BTF + AMD ordering per block
solver.factor(&ap, &ai, &ax)?;  // Block-wise factorization
solver.solve(&mut rhs)?;  // Block backward solve

// Check BTF statistics
println!("Solver mode: {}", solver.name());  // "SparseLU-BTF" or "SparseLU"
```

**When BTF Falls Back:**

BTF is not used when:
- Matrix has only 1 block (fully connected)
- The largest block equals the matrix size
- Matrix is too small (< 10 nodes by default)
- BTF would add overhead without benefit

---

### 5. Faer Solver (Default)

**Description:** Pure Rust sparse LU factorization using the faer library.

**Complexity:**
- Symbolic analysis: O(nnz)
- Factorization: O(nnz · fill), typically O(n log n) for circuits
- Solve: O(nnz)
- Memory: O(nnz + fill)

**When to Use:**
- General circuit simulation
- When C dependencies are undesirable
- Cross-platform builds
- Moderate-sized circuits (100 - 10,000 nodes)

**Features:**
- Pure Rust, no external dependencies
- Automatic fill-reducing ordering (AMD)
- Symbolic analysis caching
- Works on all platforms without special setup

**Limitations:**
- Slightly slower than KLU (typically 1.5-2x)
- No BTF (Block Triangular Form) optimization
- No refactorization support (must re-factor when values change)

**Example:**
```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

let mut solver = create_solver(SolverType::Faer, n);
// Or use auto-selection:
let mut solver = create_solver_auto(n);
```

---

### 6. KLU Solver (Best Performance)

**Description:** High-performance sparse LU factorization from SuiteSparse, specifically designed for circuit simulation matrices.

**Complexity:**
- Symbolic analysis: O(nnz)
- Factorization: O(nnz · fill)
- Refactorization: O(nnz · fill) but faster due to reused structure
- Solve: O(nnz)

**When to Use:**
- Large circuits (> 1,000 nodes)
- Performance-critical applications
- Repeated simulations with same topology
- Transient analysis with many time steps

**Features:**
- Block Triangular Form (BTF) decomposition
- Pattern caching with refactorization
- Multiple ordering strategies (AMD, COLAMD)
- Condition number monitoring
- Memory usage statistics

**Limitations:**
- Requires SuiteSparse C library installation
- Platform-dependent setup
- More complex build configuration

**Example:**
```rust
use sim_core::solver::{KluSolver, LinearSolver};

let mut solver = KluSolver::new(n);
solver.set_ordering(0);  // AMD ordering
solver.set_btf(true);    // Enable BTF

solver.prepare(n);
solver.analyze(&ap, &ai)?;
solver.factor(&ap, &ai, &ax)?;

// For repeated solves with same pattern:
// ax values changed, but ap/ai unchanged
solver.factor(&ap, &ai, &new_ax)?;  // Uses refactorization
println!("Refactors: {}", solver.refactor_count);
```

---

## Performance Comparison

### Benchmark Results (Representative)

| Circuit Size | Dense | SparseLU | SparseLU-BTF* | Faer | KLU | Notes |
|-------------|-------|----------|---------------|------|-----|-------|
| 10 nodes | 0.01 ms | 0.02 ms | 0.02 ms | 0.02 ms | 0.02 ms | Dense faster for tiny |
| 100 nodes | 0.5 ms | 0.2 ms | 0.15 ms | 0.3 ms | 0.2 ms | Sparse wins |
| 1,000 nodes | 500 ms | 8 ms | 2-6 ms | 5 ms | 3 ms | BTF faster if blocks exist |
| 5,000 nodes | N/A | 80 ms | 10-60 ms | 40 ms | 25 ms | BTF depends on structure |
| 10,000 nodes | N/A | 200 ms | 30-150 ms | 100 ms | 50 ms | KLU fastest overall |

*BTF performance varies significantly based on circuit structure. The lower bound applies to circuits with many independent blocks.

### Memory Usage (Approximate)

| Circuit Size | Dense | SparseLU | SparseLU-BTF | Faer | KLU |
|-------------|-------|----------|--------------|------|-----|
| 100 nodes | 80 KB | 50 KB | 55 KB | 20 KB | 15 KB |
| 1,000 nodes | 8 MB | 500 KB | 400 KB | 200 KB | 150 KB |
| 5,000 nodes | 200 MB | 3 MB | 2.5 MB | 2 MB | 1.5 MB |
| 10,000 nodes | 800 MB | 8 MB | 6 MB | 5 MB | 4 MB |

---

## SparseBuilder Pattern Caching

The `SparseBuilder` supports pattern freezing for efficient repeated assembly:

```rust
use sim_core::mna::SparseBuilder;

let mut builder = SparseBuilder::new(n);

// First assembly (dynamic mode)
builder.insert(0, 0, 1.0);
builder.insert(0, 1, 2.0);
builder.insert(1, 1, 3.0);
let (ap, ai, ax) = builder.finalize_merged();

// Freeze pattern for repeated solves
builder.freeze_pattern();

// Fast updates (O(1) per entry)
builder.clear_values();
builder.update(0, 0, 1.5);  // O(1) lookup
builder.update(0, 1, 2.5);

// Get values without rebuilding structure
let (ap, ai, ax) = builder.finalize();  // O(n) instead of O(nnz log nnz)
```

**Performance Impact:**

| Operation | Dynamic Mode | Frozen Mode |
|-----------|--------------|-------------|
| insert() | O(1) amortized | O(1) with index_map |
| update() | N/A | O(1) |
| clear_values() | O(nnz) | O(nnz) |
| finalize() | O(nnz log nnz) | O(nnz) |

---

## Known Limitations

### 1. Matrix Structure Requirements

All solvers require:
- Square matrices (n × n)
- CSC (Compressed Sparse Column) format
- Sorted row indices within each column
- No explicit zeros on diagonal (implicit for circuit matrices)

### 2. Numerical Stability

**Ill-conditioned matrices:**
- Very large conductance ratios (e.g., 1e15 Ω vs 1e-3 Ω)
- Floating nodes (not connected to ground)
- Singular or near-singular matrices

**Mitigation:**
- Add Gmin (minimum conductance) to diagonal
- Use source stepping for DC convergence
- Check condition number with KLU's `rcond()`

### 3. Memory Constraints

**Large circuits (> 100,000 nodes):**
- Fill-in can cause memory explosion
- Consider hierarchical simulation
- Use iterative solvers for very large problems (not yet implemented)

### 4. Pattern Changes

When circuit topology changes (e.g., switching):
- All solvers must re-analyze
- KLU's refactorization cannot be used
- Consider subcircuit-based approaches

---

## KLU Installation on Windows

### Option 1: vcpkg (Recommended)

```powershell
# Install vcpkg
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install SuiteSparse
.\vcpkg install suitesparse:x64-windows

# Set environment
$env:SUITESPARSE_DIR = "C:\path\to\vcpkg\installed\x64-windows"

# Build
cargo build --features klu
```

### Option 2: Pre-built Binaries

Download pre-built SuiteSparse from:
- https://github.com/DrTimothyAldenDavis/SuiteSparse/releases

Set environment variables:
```powershell
$env:KLU_LIB_DIR = "C:\suitesparse\lib"
$env:KLU_INCLUDE_DIR = "C:\suitesparse\include"
```

### Option 3: Build from Source

Requires CMake and Visual Studio:
```powershell
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=C:\suitesparse
cmake --build . --config Release
cmake --install .
```

---

## Troubleshooting

### "KLU not available, falling back to Dense"

**Cause:** KLU feature enabled but library not found.

**Solution:**
1. Install SuiteSparse
2. Set `SUITESPARSE_DIR` or `KLU_LIB_DIR` environment variable
3. Rebuild

### "Symbolic analysis failed"

**Cause:** Invalid matrix structure.

**Check:**
- Column pointer length is n+1
- Row indices are within bounds
- No negative indices

### "Numerical factorization failed"

**Cause:** Singular or nearly singular matrix.

**Solutions:**
- Add Gmin to diagonal
- Check for floating nodes
- Verify circuit connectivity

### "Solve step failed"

**Cause:** Factorization not performed or corrupted.

**Check:**
- Call `factor()` before `solve()`
- Ensure matrix values are finite (no NaN/Inf)

---

## Best Practices

### 1. Solver Selection

```rust
// Automatic selection (recommended)
let solver = create_solver_auto(n);

// Or explicit selection based on circuit size and structure
let solver = if n < 100 {
    create_solver(SolverType::Dense, n)
} else if cfg!(feature = "klu") {
    create_solver(SolverType::Klu, n)
} else if cfg!(feature = "faer-solver") {
    create_solver(SolverType::Faer, n)
} else {
    // For block-structured circuits, use SparseLU-BTF
    // For general circuits, use SparseLU
    create_solver(SolverType::SparseLuBtf, n)
};
```

### 2. Pattern Reuse

For transient analysis with many time steps:

```rust
// Analyze once
solver.analyze(&ap, &ai)?;

// Factor and solve for each time step
for step in 0..num_steps {
    // Matrix values change, pattern stays same
    update_matrix_values(&mut ax);
    solver.factor(&ap, &ai, &ax)?;  // KLU uses refactorization
    solver.solve(&mut rhs)?;
}
```

### 3. SparseBuilder for Newton Iteration

```rust
let mut builder = SparseBuilder::new(n);

// First Newton iteration: build pattern
stamp_all_devices(&mut builder);
let (ap, ai, ax) = builder.finalize_merged();
builder.freeze_pattern();

// Subsequent iterations: fast updates
for iter in 1..max_newton {
    builder.clear_values();
    stamp_all_devices(&mut builder);  // Uses O(1) updates
    let ax = builder.get_values().unwrap();
    solver.factor(&ap, &ai, ax)?;
    solver.solve(&mut rhs)?;
}
```

### 4. Condition Monitoring (KLU)

```rust
#[cfg(feature = "klu")]
{
    let mut solver = KluSolver::new(n);
    solver.factor(&ap, &ai, &ax)?;

    let rcond = solver.rcond();
    if rcond < 1e-12 {
        eprintln!("Warning: Matrix is ill-conditioned (rcond = {:.2e})", rcond);
    }
}
```

---

## Automatic Solver Selection

RustSpice includes intelligent automatic solver selection based on matrix properties.

### Selection Criteria

The `SolverSelector` analyzes the following matrix properties:

| Property | How It's Used |
|----------|---------------|
| **Size (n)** | Small matrices (n ≤ 50) use Dense; larger use sparse |
| **Density (nnz/n²)** | Dense matrices (>30% fill) may use Dense up to n=200 |
| **Block Structure** | Detected via BTF; enables SparseLU-BTF for speedup |
| **Average Degree** | High degree matrices benefit from advanced solvers |

### Decision Tree

```
                    n ≤ 50?
                   /      \
                 Yes       No
                  |         |
               Dense    density > 30% && n ≤ 200?
                        /                    \
                      Yes                     No
                       |                       |
                    Dense              KLU available?
                                      /            \
                                    Yes             No
                                     |               |
                                   KLU        Faer available?
                                             /            \
                                           Yes             No
                                            |               |
                                          Faer      block structure?
                                                   /              \
                                                 Yes               No
                                                  |                 |
                                          SparseLU-BTF         SparseLU
```

### Usage

```rust
use sim_core::solver::{create_solver_for_matrix, SolverSelector};

// Automatic selection (recommended)
let solver = create_solver_for_matrix(n, &ap, &ai);

// Or get selection details
let selector = SolverSelector::select(n, &ap, &ai);
println!("Selected: {:?}", selector.selected);
println!("Reason: {}", selector.reason);
let solver = selector.create_solver();
```

### Quick vs Full Analysis

- **Full analysis** (`SolverSelector::select`): Includes BTF decomposition to detect block structure
- **Quick analysis** (`SolverSelector::select_quick`): Faster but may miss block structure opportunities

---

## Future Improvements

- [x] Native Rust sparse LU solver (SparseLU) - no external dependencies
- [x] BTF (Block Triangular Form) decomposition for SparseLU
- [x] Full AMD algorithm with quotient graph for SparseLU (see [AMD Algorithm](amd_algorithm.md))
- [x] Automatic solver selection based on matrix properties
- [ ] Iterative solvers (GMRES, BiCGSTAB) for very large circuits
- [ ] GPU-accelerated solvers (cuSPARSE)
- [ ] Parallel direct solvers (PARDISO, SuperLU_MT)
- [ ] Automatic solver selection based on matrix properties
- [ ] Matrix reordering visualization tools

---

## References

1. Davis, T.A. "Direct Methods for Sparse Linear Systems", SIAM, 2006
2. Davis, T.A., Palamadai Natarajan, E. "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems", ACM TOMS, 2010
3. Duff, I.S., Erisman, A.M., Reid, J.K. "Direct Methods for Sparse Matrices", Oxford, 2017
4. Faer documentation: https://docs.rs/faer
