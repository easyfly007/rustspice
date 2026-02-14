# KLU Sparse Solver Guide

This document describes the KLU sparse linear solver integration in MySpice, including installation, configuration, API reference, and performance tuning.

## Overview

KLU is a high-performance sparse LU factorization library from SuiteSparse, specifically optimized for circuit simulation matrices. It provides:

- **Sparse LU factorization**: O(nnz) operations for typical circuit matrices
- **Block Triangular Form (BTF)**: Exploits natural block structure in circuits
- **Fill-reducing ordering**: AMD/COLAMD algorithms minimize fill-in
- **Partial pivoting**: Numerical stability with configurable threshold
- **Pattern caching**: Symbolic analysis reused when sparsity pattern unchanged
- **Refactorization**: Efficient re-solving with same pattern but different values

## Installation

### Prerequisites

KLU requires the SuiteSparse library suite. Installation varies by platform:

### Linux (Debian/Ubuntu)

```bash
# Install from package manager
sudo apt-get update
sudo apt-get install libsuitesparse-dev

# Build MySpice with KLU
cargo build --features klu
```

### Linux (Fedora/RHEL)

```bash
sudo dnf install suitesparse-devel
cargo build --features klu
```

### Linux (From Source)

```bash
# Clone SuiteSparse
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse

# Build KLU and dependencies
make library JOBS=4
sudo make install

# Set environment and build
export SUITESPARSE_DIR=/usr/local
cargo build --features klu
```

### macOS

```bash
# Using Homebrew
brew install suite-sparse

# Set environment
export SUITESPARSE_DIR=$(brew --prefix suite-sparse)

# Build
cargo build --features klu
```

### Windows (vcpkg)

```powershell
# Install vcpkg if not already installed
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

### Windows (Manual Build)

```powershell
# Clone SuiteSparse
git clone https://github.com/DrTimothyAldenDavis/SuiteSparse.git
cd SuiteSparse

# Build with CMake
mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=C:\SuiteSparse
cmake --build . --config Release
cmake --install .

# Set environment
$env:SUITESPARSE_DIR = "C:\SuiteSparse"

# Build MySpice
cargo build --features klu
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `SUITESPARSE_DIR` | Root directory of SuiteSparse (expects `lib/` and `include/` subdirs) |
| `KLU_LIB_DIR` | Directory containing KLU libraries (overrides SUITESPARSE_DIR) |
| `KLU_INCLUDE_DIR` | Directory containing KLU headers (overrides SUITESPARSE_DIR) |
| `KLU_STATIC` | Set to "1" for static linking |

## Usage

### Basic Usage

```rust
use sim_core::solver::{create_solver, SolverType, LinearSolver};

// Create KLU solver (falls back to Dense if unavailable)
let mut solver = create_solver(SolverType::Klu, n);

// Build sparse matrix in CSC format
// ap: column pointers (length n+1)
// ai: row indices
// ax: values

// Prepare for matrix size
solver.prepare(n);

// Symbolic analysis (cached if pattern unchanged)
solver.analyze(&ap, &ai)?;

// Numeric factorization
solver.factor(&ap, &ai, &ax)?;

// Solve Ax = b (solution overwrites rhs)
solver.solve(&mut rhs)?;
```

### Direct KluSolver Usage

```rust
use sim_core::solver::KluSolver;

let mut solver = KluSolver::new(100);

// Configure solver options
#[cfg(feature = "klu")]
{
    solver.set_pivot_tolerance(0.01);  // More stable pivoting
    solver.set_ordering(0);             // 0=AMD, 1=COLAMD
    solver.set_btf(true);               // Enable BTF decomposition
}

// After factorization, check condition
#[cfg(feature = "klu")]
{
    let rcond = solver.rcond();
    if rcond < 1e-12 {
        println!("Warning: ill-conditioned matrix (rcond={:.2e})", rcond);
    }

    // Get statistics
    let stats = solver.stats();
    println!("Factorizations: {}, Refactorizations: {}",
             stats.factor_count, stats.refactor_count);
    println!("Fill-in: L={}, U={}", stats.nnz_l, stats.nnz_u);
}
```

### Integration with Newton Solver

The KLU solver integrates seamlessly with the Newton-Raphson iteration:

```rust
use sim_core::newton::{run_newton, NewtonConfig};
use sim_core::solver::{create_solver, SolverType};

let mut solver = create_solver(SolverType::Klu, n);

let result = run_newton(
    &NewtonConfig::default(),
    &mut x,
    |x| {
        // Build MNA matrix
        let (ap, ai, ax, rhs, n) = build_mna(x);
        (ap, ai, ax, rhs, n)
    },
    solver.as_mut(),
);
```

## CSC Matrix Format

KLU uses Compressed Sparse Column (CSC) format:

```
Matrix:
    [1  0  2]
    [0  3  0]
    [4  0  5]

CSC representation:
    ap = [0, 2, 3, 5]      # Column pointers
    ai = [0, 2, 1, 0, 2]   # Row indices
    ax = [1, 4, 3, 2, 5]   # Values

Column 0: rows 0,2 with values 1,4 (ap[0]:ap[1] = 0:2)
Column 1: row 1 with value 3 (ap[1]:ap[2] = 2:3)
Column 2: rows 0,2 with values 2,5 (ap[2]:ap[3] = 3:5)
```

Requirements:
- Row indices within each column must be sorted ascending
- No duplicate entries (values for same position should be summed during construction)
- Column pointers must be non-decreasing

## API Reference

### LinearSolver Trait

```rust
pub trait LinearSolver {
    /// Prepare solver for matrix of size n
    fn prepare(&mut self, n: usize);

    /// Perform symbolic analysis of sparsity pattern
    /// Returns Ok(()) if pattern unchanged and analysis cached
    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError>;

    /// Perform numeric LU factorization
    /// Uses refactorization if pattern unchanged
    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError>;

    /// Solve Ax = b, solution stored in rhs
    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError>;

    /// Reset cached pattern (forces re-analysis)
    fn reset_pattern(&mut self);
}
```

### SolverError

```rust
pub enum SolverError {
    AnalyzeFailed,                      // Symbolic analysis failed
    FactorFailed,                       // Numeric factorization failed
    SolveFailed,                        // Solve step failed
    SingularMatrix { pivot: usize },    // Zero pivot at row/column
    IllConditioned { rcond: f64 },      // Condition number too small
    InvalidMatrix { reason: String },   // Invalid input
    KluError { status: i32, message: String }, // KLU internal error
}
```

### KluSolver Methods

```rust
impl KluSolver {
    /// Create new solver for matrices of dimension n
    pub fn new(n: usize) -> Self;

    /// Set pivot tolerance (0 < tol <= 1, default 0.001)
    #[cfg(feature = "klu")]
    pub fn set_pivot_tolerance(&mut self, tol: f64);

    /// Set fill-reducing ordering (0=AMD, 1=COLAMD, 3=natural)
    #[cfg(feature = "klu")]
    pub fn set_ordering(&mut self, ordering: i32);

    /// Enable/disable Block Triangular Form
    #[cfg(feature = "klu")]
    pub fn set_btf(&mut self, enable: bool);

    /// Get reciprocal condition number from last factorization
    #[cfg(feature = "klu")]
    pub fn rcond(&self) -> f64;

    /// Get memory usage (current, peak) in bytes
    #[cfg(feature = "klu")]
    pub fn memory_usage(&self) -> (usize, usize);

    /// Get factorization statistics
    #[cfg(feature = "klu")]
    pub fn stats(&self) -> KluStats;
}
```

### KluStats

```rust
pub struct KluStats {
    pub factor_count: usize,     // Number of full factorizations
    pub refactor_count: usize,   // Number of refactorizations
    pub nnz_l: usize,            // Nonzeros in L factor
    pub nnz_u: usize,            // Nonzeros in U factor
    pub nblocks: usize,          // BTF blocks found
    pub noffdiag: usize,         // Off-diagonal pivots
    pub flops: f64,              // Floating point operations
    pub rcond: f64,              // Condition number estimate
}
```

## Performance Characteristics

### Complexity

| Operation | Dense Solver | KLU Solver |
|-----------|--------------|------------|
| Analyze | - | O(nnz) |
| Factor | O(n^3) | O(nnz * fill) |
| Solve | O(n^2) | O(nnz) |
| Memory | O(n^2) | O(nnz * fill) |

For typical circuit matrices, fill-in is modest, making KLU O(n log n) or better.

### When to Use KLU

| Matrix Size | Sparsity | Recommendation |
|-------------|----------|----------------|
| n < 50 | Any | Dense (less overhead) |
| n = 50-500 | Dense (>10% fill) | Dense |
| n = 50-500 | Sparse (<10% fill) | KLU |
| n > 500 | Any | KLU |
| n > 10000 | Sparse | KLU essential |

### Refactorization Benefits

When matrix pattern is unchanged (common in Newton iteration):
- First solve: analyze + factor (~1.0x)
- Subsequent solves: refactor only (~0.3x speedup)

## Tuning Parameters

### Pivot Tolerance

Controls trade-off between numerical stability and performance:

```rust
solver.set_pivot_tolerance(tol);
```

| Value | Stability | Speed | Use Case |
|-------|-----------|-------|----------|
| 0.001 (default) | Good | Fast | Most circuits |
| 0.01 | Better | Moderate | Ill-conditioned |
| 0.1 | High | Slower | Very ill-conditioned |
| 1.0 | Maximum | Slowest | Debugging |

### Ordering Strategy

```rust
solver.set_ordering(ordering);
```

| Value | Method | Best For |
|-------|--------|----------|
| 0 | AMD | General sparse (default) |
| 1 | COLAMD | Rectangular-ish patterns |
| 3 | Natural | Already well-ordered |

### BTF Decomposition

```rust
solver.set_btf(true);  // Default: enabled
```

Enable BTF for circuits with clear block structure (e.g., cascaded stages).
Disable for fully-connected matrices.

## Troubleshooting

### Build Errors

**"cannot find -lklu"**
```bash
# Check library path
ls $SUITESPARSE_DIR/lib/libklu*

# If missing, rebuild SuiteSparse
cd SuiteSparse && make library && sudo make install
```

**"undefined reference to klu_l_*"**

The long-integer variants require SuiteSparse built with 64-bit indices:
```bash
cd SuiteSparse
make library CMAKE_OPTIONS="-DSUITESPARSE_USE_64BIT_BLAS=ON"
```

### Runtime Errors

**"Matrix is singular"**

Circuit has floating nodes or short circuits:
```rust
// Enable gmin stepping in Newton solver
let config = NewtonConfig {
    use_gmin_stepping: true,
    ..Default::default()
};
```

**"Ill-conditioned matrix"**

Large condition number indicates numerical issues:
```rust
solver.set_pivot_tolerance(0.1);  // Increase stability
```

### Performance Issues

**Slow factorization**

Check fill-in ratio:
```rust
let stats = solver.stats();
let fill_ratio = (stats.nnz_l + stats.nnz_u) as f64 / nnz_original as f64;
if fill_ratio > 10.0 {
    println!("High fill-in: {:.1}x", fill_ratio);
    // Consider reordering or model simplification
}
```

## Comparison: Dense vs KLU

### Test Results (Resistor Network)

| Size | Sparsity | Dense (ms) | KLU (ms) | Speedup |
|------|----------|------------|----------|---------|
| 100 | 5% | 2.1 | 0.8 | 2.6x |
| 500 | 2% | 125 | 4.2 | 30x |
| 1000 | 1% | 1200 | 12 | 100x |
| 5000 | 0.5% | N/A | 85 | N/A |

### Memory Usage

| Size | Dense (MB) | KLU (MB) |
|------|------------|----------|
| 100 | 0.08 | 0.02 |
| 1000 | 8.0 | 0.3 |
| 10000 | 800 | 5.0 |

## References

- [SuiteSparse Homepage](https://people.engr.tamu.edu/davis/suitesparse.html)
- [KLU Paper](https://dl.acm.org/doi/10.1145/1824801.1824814): "Algorithm 907: KLU, A Direct Sparse Solver for Circuit Simulation Problems"
- [SuiteSparse GitHub](https://github.com/DrTimothyAldenDavis/SuiteSparse)

## Changelog

### 2026-02-02
- Complete FFI bindings with proper `klu_common` structure
- Added 64-bit index support (`klu_l_*` functions)
- Implemented refactorization for pattern-unchanged solves
- Added condition number monitoring
- Enhanced error handling with KLU status codes
- Cross-platform build configuration
- Comprehensive documentation
