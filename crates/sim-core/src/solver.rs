//! Linear Solver Module
//!
//! This module provides linear equation solvers for circuit simulation.
//! Multiple solver backends are available:
//!
//! | Solver | Feature | Performance | Dependencies |
//! |--------|---------|-------------|--------------|
//! | Dense  | (always) | O(n³) | None |
//! | SparseLU | (always) | O(nnz·fill) | None (native Rust) |
//! | SparseLU-BTF | (always) | O(nnz·fill), block-optimized | None (native Rust) |
//! | Faer   | `faer-solver` (default) | O(nnz·fill) | Pure Rust |
//! | KLU    | `klu` | O(nnz·fill), fastest | SuiteSparse (C) |
//!
//! # Faer Sparse Solver (Recommended for Easy Setup)
//!
//! Faer is a pure Rust linear algebra library with sparse support.
//! It's enabled by default and requires no external dependencies:
//!
//! ```bash
//! # Build with default features (includes faer)
//! cargo build
//!
//! # Or explicitly enable
//! cargo build --features faer-solver
//! ```
//!
//! # KLU Sparse Solver (Best Performance)
//!
//! KLU is a high-performance sparse LU factorization library from SuiteSparse,
//! optimized for circuit simulation matrices. To enable KLU:
//!
//! ```bash
//! # Set environment variables pointing to SuiteSparse installation
//! export SUITESPARSE_DIR=/path/to/suitesparse
//! # Or set individual paths
//! export KLU_LIB_DIR=/path/to/lib
//! export KLU_INCLUDE_DIR=/path/to/include
//!
//! # Build with KLU feature
//! cargo build --features klu
//! ```
//!
//! # Solver Selection
//!
//! Use `create_solver_auto()` for automatic selection based on available features:
//! - Prefers KLU if available (best performance)
//! - Falls back to Faer if available (pure Rust)
//! - Falls back to Dense as last resort
//!
//! # Usage
//!
//! ```ignore
//! use sim_core::solver::{create_solver_auto, LinearSolver};
//!
//! // Create best available solver
//! let mut solver = create_solver_auto(100);
//!
//! // Prepare for matrix of size n
//! solver.prepare(n);
//!
//! // Analyze sparsity pattern (cached if unchanged)
//! solver.analyze(&ap, &ai)?;
//!
//! // Factor the matrix
//! solver.factor(&ap, &ai, &ax)?;
//!
//! // Solve Ax = b (result stored in rhs)
//! solver.solve(&mut rhs)?;
//! ```

use std::fmt;

/// Error types for linear solver operations
#[derive(Debug, Clone)]
pub enum SolverError {
    /// Symbolic analysis failed (invalid sparsity pattern)
    AnalyzeFailed,
    /// Numerical factorization failed (singular or ill-conditioned matrix)
    FactorFailed,
    /// Solve step failed
    SolveFailed,
    /// Matrix is singular (zero pivot encountered)
    SingularMatrix { pivot: usize },
    /// Matrix is ill-conditioned (reciprocal condition number too small)
    IllConditioned { rcond: f64 },
    /// Invalid matrix dimensions or structure
    InvalidMatrix { reason: String },
    /// KLU internal error with status code
    #[cfg(feature = "klu")]
    KluError { status: i32, message: String },
}

impl fmt::Display for SolverError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverError::AnalyzeFailed => write!(f, "Symbolic analysis failed"),
            SolverError::FactorFailed => write!(f, "Numerical factorization failed"),
            SolverError::SolveFailed => write!(f, "Solve step failed"),
            SolverError::SingularMatrix { pivot } => {
                write!(f, "Singular matrix: zero pivot at row/column {}", pivot)
            }
            SolverError::IllConditioned { rcond } => {
                write!(f, "Ill-conditioned matrix: rcond = {:.2e}", rcond)
            }
            SolverError::InvalidMatrix { reason } => {
                write!(f, "Invalid matrix: {}", reason)
            }
            #[cfg(feature = "klu")]
            SolverError::KluError { status, message } => {
                write!(f, "KLU error (status {}): {}", status, message)
            }
        }
    }
}

impl std::error::Error for SolverError {}

/// Solver type selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SolverType {
    /// Dense LU solver - O(n³), suitable for n < 100
    #[default]
    Dense,
    /// Native Rust sparse LU solver - O(nnz·fill), no dependencies
    SparseLu,
    /// Native Rust sparse LU solver with BTF decomposition - best for block-structured matrices
    SparseLuBtf,
    /// Faer sparse solver - Pure Rust, O(nnz·fill)
    Faer,
    /// KLU sparse solver - SuiteSparse C library, fastest
    Klu,
    /// Automatic selection based on available features
    Auto,
}

pub trait LinearSolver: Send {
    /// Prepare the solver for a matrix of size n
    fn prepare(&mut self, n: usize);

    /// Analyze the sparsity pattern (can be cached)
    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError>;

    /// Factorize the matrix
    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError>;

    /// Solve Ax = b, result overwrites rhs
    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError>;

    /// Reset cached pattern (call when matrix structure changes)
    fn reset_pattern(&mut self);

    /// Get the solver name for diagnostics
    fn name(&self) -> &'static str {
        "Unknown"
    }
}

/// Create solver based on explicit type selection
pub fn create_solver(solver_type: SolverType, n: usize) -> Box<dyn LinearSolver> {
    match solver_type {
        SolverType::Dense => Box::new(DenseSolver::new(n)),
        SolverType::SparseLu => Box::new(crate::sparse_lu::SparseLuSolver::new(n)),
        SolverType::SparseLuBtf => Box::new(crate::sparse_lu_btf::SparseLuBtfSolver::new(n)),
        SolverType::Faer => {
            #[cfg(feature = "faer-solver")]
            {
                Box::new(FaerSolver::new(n))
            }
            #[cfg(not(feature = "faer-solver"))]
            {
                eprintln!("Warning: Faer not available, falling back to SparseLU solver");
                Box::new(crate::sparse_lu::SparseLuSolver::new(n))
            }
        }
        SolverType::Klu => {
            #[cfg(feature = "klu")]
            {
                Box::new(KluSolver::new(n))
            }
            #[cfg(not(feature = "klu"))]
            {
                eprintln!("Warning: KLU not available, falling back to SparseLU solver");
                Box::new(crate::sparse_lu::SparseLuSolver::new(n))
            }
        }
        SolverType::Auto => create_solver_auto(n),
    }
}

/// Create the best available solver automatically (size-based heuristic)
///
/// This is a simple size-based selection. For smarter selection based on
/// matrix properties, use `SolverSelector::select()` after analyzing the matrix.
///
/// # Selection Rules (by size)
///
/// | Size | KLU available | Faer available | Neither |
/// |------|---------------|----------------|---------|
/// | n ≤ 50 | Dense | Dense | Dense |
/// | 50 < n ≤ 500 | KLU | Faer | SparseLU |
/// | n > 500 | KLU | Faer | SparseLU-BTF |
///
pub fn create_solver_auto(n: usize) -> Box<dyn LinearSolver> {
    // Small matrices: Dense is fast and has low overhead
    if n <= 50 {
        return Box::new(DenseSolver::new(n));
    }

    // Medium to large matrices: use best available sparse solver
    #[cfg(feature = "klu")]
    {
        return Box::new(KluSolver::new(n));
    }

    #[cfg(all(feature = "faer-solver", not(feature = "klu")))]
    {
        return Box::new(FaerSolver::new(n));
    }

    // No external solvers: use native SparseLU
    #[cfg(not(any(feature = "klu", feature = "faer-solver")))]
    {
        if n > 500 {
            // Large matrices benefit from BTF
            Box::new(crate::sparse_lu_btf::SparseLuBtfSolver::new(n))
        } else {
            Box::new(crate::sparse_lu::SparseLuSolver::new(n))
        }
    }
}

/// Matrix properties used for solver selection
#[derive(Debug, Clone)]
pub struct MatrixProperties {
    /// Matrix dimension
    pub n: usize,
    /// Number of nonzeros
    pub nnz: usize,
    /// Sparsity ratio: nnz / n² (0.0 = empty, 1.0 = dense)
    pub density: f64,
    /// Average entries per row/column
    pub avg_degree: f64,
    /// Whether the matrix has block structure (from BTF analysis)
    pub has_block_structure: bool,
    /// Number of BTF blocks (1 = no useful block structure)
    pub num_blocks: usize,
    /// Largest block size (as fraction of n)
    pub max_block_ratio: f64,
}

impl MatrixProperties {
    /// Analyze matrix properties from CSC format
    ///
    /// # Arguments
    /// * `n` - Matrix dimension
    /// * `ap` - Column pointers (length n+1)
    /// * `ai` - Row indices
    ///
    /// # Example
    /// ```ignore
    /// let props = MatrixProperties::analyze(n, &ap, &ai);
    /// println!("Density: {:.2}%", props.density * 100.0);
    /// ```
    pub fn analyze(n: usize, ap: &[i64], ai: &[i64]) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                nnz: 0,
                density: 0.0,
                avg_degree: 0.0,
                has_block_structure: false,
                num_blocks: 0,
                max_block_ratio: 0.0,
            };
        }

        let nnz = ap[n] as usize;
        let density = nnz as f64 / (n * n) as f64;
        let avg_degree = nnz as f64 / n as f64;

        // Analyze block structure using BTF
        let btf = crate::btf::btf_decompose(n, ap, ai);
        let has_block_structure = btf.num_blocks > 1 && btf.max_block_size < n;
        let max_block_ratio = btf.max_block_size as f64 / n as f64;

        Self {
            n,
            nnz,
            density,
            avg_degree,
            has_block_structure,
            num_blocks: btf.num_blocks,
            max_block_ratio,
        }
    }

    /// Quick analysis without BTF (faster, less accurate)
    pub fn analyze_quick(n: usize, ap: &[i64], _ai: &[i64]) -> Self {
        if n == 0 {
            return Self {
                n: 0,
                nnz: 0,
                density: 0.0,
                avg_degree: 0.0,
                has_block_structure: false,
                num_blocks: 1,
                max_block_ratio: 1.0,
            };
        }

        let nnz = ap[n] as usize;
        let density = nnz as f64 / (n * n) as f64;
        let avg_degree = nnz as f64 / n as f64;

        Self {
            n,
            nnz,
            density,
            avg_degree,
            has_block_structure: false,  // Unknown without BTF
            num_blocks: 1,
            max_block_ratio: 1.0,
        }
    }
}

/// Intelligent solver selector based on matrix properties
///
/// This selector analyzes matrix characteristics and chooses the most
/// appropriate solver based on:
///
/// # Selection Criteria
///
/// ## 1. Matrix Size
/// - **n ≤ 50**: Dense solver (low overhead, O(n³) is acceptable)
/// - **50 < n ≤ 200**: Sparse solvers start to win
/// - **n > 200**: Sparse solvers essential
///
/// ## 2. Sparsity (density = nnz/n²)
/// - **density > 0.3 (30%)**: Matrix is "dense", use Dense solver up to n=200
/// - **density < 0.1 (10%)**: Typical sparse matrix, use sparse solvers
/// - **density < 0.01 (1%)**: Very sparse, BTF may help
///
/// ## 3. Block Structure
/// - **Multiple BTF blocks**: SparseLU-BTF provides speedup proportional to k²
///   where k is the number of blocks
/// - **Single block**: Regular sparse solver
/// - **Many 1×1 blocks**: Nearly triangular, BTF very beneficial
///
/// ## 4. Available Solvers
/// - **KLU**: Best performance, use for large matrices if available
/// - **Faer**: Good pure-Rust alternative
/// - **SparseLU-BTF**: Best native option for block-structured matrices
/// - **SparseLU**: Best native option for general sparse matrices
/// - **Dense**: Fallback for small/dense matrices
///
/// # Decision Tree
///
/// ```text
///                         n ≤ 50?
///                        /      \
///                      Yes       No
///                       |         |
///                    Dense    density > 0.3 && n ≤ 200?
///                             /                    \
///                           Yes                     No
///                            |                       |
///                         Dense              KLU available?
///                                           /            \
///                                         Yes             No
///                                          |               |
///                                        KLU        Faer available?
///                                                  /            \
///                                                Yes             No
///                                                 |               |
///                                               Faer      has_block_structure?
///                                                        /              \
///                                                      Yes               No
///                                                       |                 |
///                                               SparseLU-BTF         SparseLU
/// ```
#[derive(Debug, Clone)]
pub struct SolverSelector {
    /// Matrix properties
    pub properties: MatrixProperties,
    /// Selected solver type
    pub selected: SolverType,
    /// Reason for selection
    pub reason: String,
}

impl SolverSelector {
    /// Analyze matrix and select the best solver
    ///
    /// Performs full BTF analysis to detect block structure.
    pub fn select(n: usize, ap: &[i64], ai: &[i64]) -> Self {
        let properties = MatrixProperties::analyze(n, ap, ai);
        Self::select_from_properties(properties)
    }

    /// Quick selection without BTF analysis
    ///
    /// Faster but may miss opportunities to use SparseLU-BTF.
    pub fn select_quick(n: usize, ap: &[i64], ai: &[i64]) -> Self {
        let properties = MatrixProperties::analyze_quick(n, ap, ai);
        Self::select_from_properties(properties)
    }

    /// Select solver based on pre-computed properties
    pub fn select_from_properties(properties: MatrixProperties) -> Self {
        let n = properties.n;
        let density = properties.density;

        // Rule 1: Very small matrices -> Dense
        if n <= 50 {
            return Self {
                properties,
                selected: SolverType::Dense,
                reason: format!("Small matrix (n={}) - Dense solver has lowest overhead", n),
            };
        }

        // Rule 2: Dense matrices up to moderate size -> Dense
        if density > 0.3 && n <= 200 {
            return Self {
                properties,
                selected: SolverType::Dense,
                reason: format!(
                    "Dense matrix ({:.1}% fill, n={}) - Dense solver efficient",
                    density * 100.0, n
                ),
            };
        }

        // Rule 3: KLU if available (best performance for large sparse)
        #[cfg(feature = "klu")]
        {
            return Self {
                properties,
                selected: SolverType::Klu,
                reason: format!(
                    "Large sparse matrix (n={}, {:.1}% fill) - KLU optimal",
                    n, density * 100.0
                ),
            };
        }

        // Rule 4: Faer if available
        #[cfg(all(feature = "faer-solver", not(feature = "klu")))]
        {
            return Self {
                properties,
                selected: SolverType::Faer,
                reason: format!(
                    "Sparse matrix (n={}, {:.1}% fill) - Faer selected",
                    n, density * 100.0
                ),
            };
        }

        // Rule 5: Native solvers - choose based on structure
        #[cfg(not(any(feature = "klu", feature = "faer-solver")))]
        {
            if properties.has_block_structure && properties.num_blocks > 1 {
                let speedup_est = (properties.num_blocks as f64).powi(2) /
                    (1.0 + properties.max_block_ratio.powi(3) * (properties.num_blocks as f64));
                Self {
                    selected: SolverType::SparseLuBtf,
                    reason: format!(
                        "Block structure detected ({} blocks, max {:.0}% of n) - \
                         SparseLU-BTF ~{:.1}x faster",
                        properties.num_blocks,
                        properties.max_block_ratio * 100.0,
                        speedup_est.max(1.0)
                    ),
                    properties,
                }
            } else if n > 500 {
                // Large matrix without clear block structure - try BTF anyway
                Self {
                    properties,
                    selected: SolverType::SparseLuBtf,
                    reason: format!(
                        "Large matrix (n={}) - SparseLU-BTF may find hidden structure",
                        n
                    ),
                }
            } else {
                Self {
                    properties,
                    selected: SolverType::SparseLu,
                    reason: format!(
                        "Medium sparse matrix (n={}, {:.1}% fill) - SparseLU selected",
                        n, density * 100.0
                    ),
                }
            }
        }
    }

    /// Create the selected solver
    pub fn create_solver(&self) -> Box<dyn LinearSolver> {
        create_solver(self.selected, self.properties.n)
    }
}

/// Create solver with automatic selection based on matrix properties
///
/// This is the recommended way to create a solver when you have the matrix
/// structure available. It performs BTF analysis to detect block structure.
///
/// # Arguments
/// * `n` - Matrix dimension
/// * `ap` - Column pointers (CSC format)
/// * `ai` - Row indices (CSC format)
///
/// # Returns
/// A boxed solver optimized for the given matrix structure
///
/// # Example
/// ```ignore
/// use sim_core::solver::create_solver_for_matrix;
///
/// let solver = create_solver_for_matrix(n, &ap, &ai);
/// solver.analyze(&ap, &ai)?;
/// solver.factor(&ap, &ai, &ax)?;
/// solver.solve(&mut rhs)?;
/// ```
pub fn create_solver_for_matrix(n: usize, ap: &[i64], ai: &[i64]) -> Box<dyn LinearSolver> {
    let selector = SolverSelector::select(n, ap, ai);
    selector.create_solver()
}

/// Create solver with quick automatic selection (no BTF analysis)
///
/// Faster than `create_solver_for_matrix` but may not detect block structure.
pub fn create_solver_for_matrix_quick(n: usize, ap: &[i64], ai: &[i64]) -> Box<dyn LinearSolver> {
    let selector = SolverSelector::select_quick(n, ap, ai);
    selector.create_solver()
}

/// Get a description of the solver that would be selected by create_solver_auto
pub fn describe_solver_selection() -> &'static str {
    #[cfg(feature = "klu")]
    {
        return "KLU (SuiteSparse) - Optimal for circuit simulation";
    }

    #[cfg(all(feature = "faer-solver", not(feature = "klu")))]
    {
        return "Faer (Pure Rust) - Good performance, no C dependencies";
    }

    #[cfg(not(any(feature = "klu", feature = "faer-solver")))]
    {
        "SparseLU/SparseLU-BTF (Native Rust) - No external dependencies"
    }
}

#[derive(Debug)]
pub struct DenseSolver {
    pub n: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
}

impl DenseSolver {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            lu: vec![0.0; n * n],
            pivots: (0..n).collect(),
        }
    }

    fn ensure_capacity(&mut self, n: usize) {
        if self.n != n {
            self.n = n;
            self.lu.resize(n * n, 0.0);
            self.pivots = (0..n).collect();
        }
    }

    fn build_dense(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        let n = self.n;
        if ap.len() != n + 1 {
            return Err(SolverError::AnalyzeFailed);
        }
        self.lu.fill(0.0);
        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for idx in start..end {
                let row = ai[idx] as usize;
                if row < n {
                    self.lu[row * n + col] += ax[idx];
                }
            }
        }
        Ok(())
    }

    fn factorize(&mut self) -> Result<(), SolverError> {
        let n = self.n;
        for i in 0..n {
            self.pivots[i] = i;
        }
        for k in 0..n {
            let mut pivot = k;
            let mut max_val = self.lu[k * n + k].abs();
            for i in (k + 1)..n {
                let val = self.lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    pivot = i;
                }
            }
            if max_val == 0.0 {
                return Err(SolverError::FactorFailed);
            }
            if pivot != k {
                for j in 0..n {
                    self.lu.swap(k * n + j, pivot * n + j);
                }
                self.pivots.swap(k, pivot);
            }
            let pivot_val = self.lu[k * n + k];
            for i in (k + 1)..n {
                let factor = self.lu[i * n + k] / pivot_val;
                self.lu[i * n + k] = factor;
                for j in (k + 1)..n {
                    self.lu[i * n + j] -= factor * self.lu[k * n + j];
                }
            }
        }
        Ok(())
    }
}

impl LinearSolver for DenseSolver {
    fn prepare(&mut self, n: usize) {
        self.ensure_capacity(n);
    }

    fn analyze(&mut self, _ap: &[i64], _ai: &[i64]) -> Result<(), SolverError> {
        Ok(())
    }

    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        self.build_dense(ap, ai, ax)?;
        self.factorize()
    }

    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        let n = self.n;
        if rhs.len() != n {
            return Err(SolverError::SolveFailed);
        }
        let mut b = vec![0.0; n];
        for i in 0..n {
            b[i] = rhs[self.pivots[i]];
        }
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.lu[i * n + j] * b[j];
            }
            b[i] = sum;
        }
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= self.lu[i * n + j] * rhs[j];
            }
            let diag = self.lu[i * n + i];
            if diag == 0.0 {
                return Err(SolverError::SolveFailed);
            }
            rhs[i] = sum / diag;
        }
        Ok(())
    }

    fn reset_pattern(&mut self) {}

    fn name(&self) -> &'static str {
        "Dense"
    }
}

// ============================================================================
// Faer Sparse Solver
// ============================================================================

/// Faer Sparse Solver
///
/// Pure Rust sparse LU factorization solver using the faer library.
/// This is the recommended solver for easy setup as it requires no external
/// C dependencies.
///
/// # Performance Characteristics
///
/// - Symbolic analysis: O(nnz)
/// - Numeric factorization: O(nnz * fill)
/// - Solve: O(nnz) per right-hand side
///
/// # Comparison with KLU
///
/// | Aspect | Faer | KLU |
/// |--------|------|-----|
/// | Dependencies | Pure Rust | SuiteSparse (C) |
/// | Performance | Good | Best |
/// | Setup | Trivial | Requires install |
/// | Portability | All platforms | Platform-dependent |
#[cfg(feature = "faer-solver")]
pub struct FaerSolver {
    /// Matrix dimension
    pub n: usize,
    /// Cached symbolic analysis
    symbolic: Option<faer::sparse::linalg::solvers::SymbolicLu<usize>>,
    /// Cached numeric factorization
    lu: Option<faer::sparse::linalg::solvers::Lu<usize, f64>>,
    /// Cached column pointers
    last_ap: Vec<i64>,
    /// Cached row indices
    last_ai: Vec<i64>,
    /// Statistics
    pub factor_count: usize,
}

#[cfg(feature = "faer-solver")]
impl FaerSolver {
    pub fn new(n: usize) -> Self {
        Self {
            n,
            symbolic: None,
            lu: None,
            last_ap: Vec::new(),
            last_ai: Vec::new(),
            factor_count: 0,
        }
    }

    /// Check if pattern matches cached pattern
    fn pattern_matches(&self, ap: &[i64], ai: &[i64]) -> bool {
        self.last_ap == ap && self.last_ai == ai
    }

    /// Convert CSC arrays to triplet format for faer
    fn csc_to_triplets(
        n: usize,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
    ) -> Vec<(usize, usize, f64)> {
        let mut triplets = Vec::with_capacity(ax.len());
        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for idx in start..end {
                let row = ai[idx] as usize;
                triplets.push((row, col, ax[idx]));
            }
        }
        triplets
    }
}

#[cfg(feature = "faer-solver")]
impl LinearSolver for FaerSolver {
    fn prepare(&mut self, n: usize) {
        if n != self.n {
            self.reset_pattern();
            self.n = n;
        }
    }

    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        use faer::sparse::SparseColMat;
        use faer::sparse::linalg::solvers::SymbolicLu;

        // Check if pattern is unchanged
        if self.symbolic.is_some() && self.pattern_matches(ap, ai) {
            return Ok(());
        }

        // Validate input
        if ap.len() != self.n + 1 {
            return Err(SolverError::InvalidMatrix {
                reason: format!(
                    "Column pointer length {} != expected {}",
                    ap.len(),
                    self.n + 1
                ),
            });
        }

        // Create dummy values for symbolic analysis
        let nnz = ai.len();
        let dummy_values = vec![1.0f64; nnz];

        // Convert to triplets for faer
        let triplets = Self::csc_to_triplets(self.n, ap, ai, &dummy_values);

        // Create sparse matrix from triplets
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(
            self.n,
            self.n,
            &triplets,
        )
        .map_err(|e| SolverError::InvalidMatrix {
            reason: format!("Failed to create sparse matrix: {:?}", e),
        })?;

        // Perform symbolic analysis using high-level API
        let symbolic = SymbolicLu::try_new(mat.symbolic())
            .map_err(|_| SolverError::AnalyzeFailed)?;

        self.symbolic = Some(symbolic);
        self.last_ap = ap.to_vec();
        self.last_ai = ai.to_vec();

        Ok(())
    }

    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        use faer::sparse::SparseColMat;
        use faer::sparse::linalg::solvers::Lu;

        // Ensure symbolic analysis is done
        if self.symbolic.is_none() || !self.pattern_matches(ap, ai) {
            self.analyze(ap, ai)?;
        }

        let symbolic = self.symbolic.clone().ok_or(SolverError::FactorFailed)?;

        // Convert to triplets for faer
        let triplets = Self::csc_to_triplets(self.n, ap, ai, ax);

        // Create sparse matrix from triplets
        let mat = SparseColMat::<usize, f64>::try_new_from_triplets(
            self.n,
            self.n,
            &triplets,
        )
        .map_err(|e| SolverError::InvalidMatrix {
            reason: format!("Failed to create sparse matrix: {:?}", e),
        })?;

        // Perform numeric factorization using high-level API
        let lu = Lu::try_new_with_symbolic(symbolic, mat.as_ref())
            .map_err(|_| SolverError::FactorFailed)?;

        self.lu = Some(lu);
        self.factor_count += 1;

        Ok(())
    }

    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        use faer::prelude::SpSolver;
        use faer::Mat;

        let lu = self.lu.as_ref().ok_or(SolverError::SolveFailed)?;

        if rhs.len() != self.n {
            return Err(SolverError::InvalidMatrix {
                reason: format!("RHS length {} != matrix dimension {}", rhs.len(), self.n),
            });
        }

        // Create a column matrix from rhs
        let b = Mat::from_fn(self.n, 1, |i, _| rhs[i]);

        // Solve the system
        let x = lu.solve(&b);

        // Copy result back to rhs
        for i in 0..self.n {
            rhs[i] = x[(i, 0)];
        }

        Ok(())
    }

    fn reset_pattern(&mut self) {
        self.symbolic = None;
        self.lu = None;
        self.last_ap.clear();
        self.last_ai.clear();
    }

    fn name(&self) -> &'static str {
        "Faer"
    }
}

#[cfg(feature = "faer-solver")]
unsafe impl Send for FaerSolver {}

/// KLU Sparse Solver
///
/// High-performance sparse LU factorization solver using SuiteSparse KLU.
/// Optimized for circuit simulation matrices with the following features:
///
/// - **Pattern caching**: Symbolic analysis is cached when sparsity pattern unchanged
/// - **Refactorization**: Uses `klu_refactor` for efficient re-solving with same pattern
/// - **Block triangular form**: Exploits BTF structure for better performance
/// - **Condition monitoring**: Tracks reciprocal condition number
///
/// # Performance Characteristics
///
/// - Symbolic analysis: O(nnz) - done once per pattern
/// - Numeric factorization: O(nnz * fill) - typically O(n log n) for circuits
/// - Solve: O(nnz) per right-hand side
///
/// # Memory Usage
///
/// Approximately 2-3x the size of the original matrix, depending on fill-in.
pub struct KluSolver {
    /// Matrix dimension
    pub n: usize,
    /// Whether KLU is available and enabled
    pub enabled: bool,
    /// Cached column pointers for pattern comparison
    last_ap: Vec<i64>,
    /// Cached row indices for pattern comparison
    last_ai: Vec<i64>,
    /// Whether numeric factorization exists and can be refactored
    #[cfg(feature = "klu")]
    has_numeric: bool,
    /// Symbolic analysis result (pattern-dependent)
    #[cfg(feature = "klu")]
    symbolic: *mut klu_sys::klu_symbolic,
    /// Numeric factorization result (value-dependent)
    #[cfg(feature = "klu")]
    numeric: *mut klu_sys::klu_numeric,
    /// KLU control parameters and statistics
    #[cfg(feature = "klu")]
    common: klu_sys::klu_common,
    /// Last computed reciprocal condition number
    #[cfg(feature = "klu")]
    pub last_rcond: f64,
    /// Number of factorizations performed
    #[cfg(feature = "klu")]
    pub factor_count: usize,
    /// Number of refactorizations performed (more efficient)
    #[cfg(feature = "klu")]
    pub refactor_count: usize,
}

// KluSolver cannot be sent between threads due to raw pointers
// but it can be used within a single thread safely
#[cfg(feature = "klu")]
unsafe impl Send for KluSolver {}

impl KluSolver {
    /// Create a new KLU solver for matrices of dimension n
    ///
    /// # Arguments
    /// * `n` - Expected matrix dimension (can be resized later)
    ///
    /// # Example
    /// ```ignore
    /// let solver = KluSolver::new(100);
    /// assert!(solver.enabled); // true if KLU feature enabled
    /// ```
    pub fn new(n: usize) -> Self {
        #[cfg(feature = "klu")]
        {
            let mut common = klu_sys::klu_common::default();
            unsafe {
                // Use long-integer variant for 64-bit indices
                klu_sys::klu_l_defaults(&mut common);
            }
            Self {
                n,
                enabled: true,
                last_ap: Vec::new(),
                last_ai: Vec::new(),
                has_numeric: false,
                symbolic: std::ptr::null_mut(),
                numeric: std::ptr::null_mut(),
                common,
                last_rcond: 1.0,
                factor_count: 0,
                refactor_count: 0,
            }
        }
        #[cfg(not(feature = "klu"))]
        Self {
            n,
            enabled: false,
            last_ap: Vec::new(),
            last_ai: Vec::new(),
        }
    }

    /// Set pivot tolerance for partial pivoting
    ///
    /// Larger values (closer to 1.0) give more stable factorization
    /// but may be slower. Default is 0.001.
    ///
    /// # Arguments
    /// * `tol` - Pivot tolerance in range (0, 1]
    #[cfg(feature = "klu")]
    pub fn set_pivot_tolerance(&mut self, tol: f64) {
        self.common.tol = tol.clamp(1e-15, 1.0);
    }

    /// Set ordering strategy
    ///
    /// # Arguments
    /// * `ordering` - 0=AMD (default), 1=COLAMD, 3=natural
    #[cfg(feature = "klu")]
    pub fn set_ordering(&mut self, ordering: i32) {
        self.common.ordering = ordering;
    }

    /// Enable or disable block triangular form decomposition
    ///
    /// BTF can significantly improve performance for circuits with
    /// natural block structure.
    #[cfg(feature = "klu")]
    pub fn set_btf(&mut self, enable: bool) {
        self.common.btf = if enable { 1 } else { 0 };
    }

    /// Get the reciprocal condition number from last factorization
    ///
    /// Returns a value in (0, 1]. Small values (< 1e-12) indicate
    /// an ill-conditioned matrix that may produce inaccurate results.
    #[cfg(feature = "klu")]
    pub fn rcond(&self) -> f64 {
        self.last_rcond
    }

    /// Get memory usage statistics
    #[cfg(feature = "klu")]
    pub fn memory_usage(&self) -> (usize, usize) {
        (self.common.memusage, self.common.mempeak)
    }

    /// Get factorization statistics
    #[cfg(feature = "klu")]
    pub fn stats(&self) -> KluStats {
        KluStats {
            factor_count: self.factor_count,
            refactor_count: self.refactor_count,
            nnz_l: self.common.lnz as usize,
            nnz_u: self.common.unz as usize,
            nblocks: self.common.nblocks as usize,
            noffdiag: self.common.noffdiag as usize,
            flops: self.common.flops,
            rcond: self.last_rcond,
        }
    }

    /// Check if pattern matches cached pattern
    #[cfg(feature = "klu")]
    fn pattern_matches(&self, ap: &[i64], ai: &[i64]) -> bool {
        self.last_ap == ap && self.last_ai == ai
    }

    /// Convert KLU status to SolverError
    #[cfg(feature = "klu")]
    fn check_status(&self) -> Result<(), SolverError> {
        match self.common.status {
            klu_sys::KLU_OK => Ok(()),
            klu_sys::KLU_SINGULAR => Err(SolverError::SingularMatrix {
                pivot: self.common.noffdiag as usize,
            }),
            status => Err(SolverError::KluError {
                status,
                message: klu_sys::status_message(status).to_string(),
            }),
        }
    }
}

/// Statistics from KLU factorization
#[cfg(feature = "klu")]
#[derive(Debug, Clone)]
pub struct KluStats {
    /// Number of full factorizations
    pub factor_count: usize,
    /// Number of refactorizations (pattern reuse)
    pub refactor_count: usize,
    /// Number of nonzeros in L
    pub nnz_l: usize,
    /// Number of nonzeros in U
    pub nnz_u: usize,
    /// Number of BTF blocks
    pub nblocks: usize,
    /// Number of off-diagonal pivots
    pub noffdiag: usize,
    /// Floating point operations for factorization
    pub flops: f64,
    /// Reciprocal condition number
    pub rcond: f64,
}

impl LinearSolver for KluSolver {
    fn prepare(&mut self, n: usize) {
        if n != self.n {
            self.reset_pattern();
        }
        self.n = n;
    }

    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        if !self.enabled {
            return Err(SolverError::AnalyzeFailed);
        }

        // Validate input
        if ap.len() != self.n + 1 {
            return Err(SolverError::InvalidMatrix {
                reason: format!(
                    "Column pointer length {} != expected {}",
                    ap.len(),
                    self.n + 1
                ),
            });
        }

        #[cfg(feature = "klu")]
        {
            // Check if pattern is unchanged - can skip analysis
            if !self.symbolic.is_null() && self.pattern_matches(ap, ai) {
                return Ok(());
            }

            // Pattern changed - need new symbolic analysis
            unsafe {
                // Free old symbolic if exists
                if !self.symbolic.is_null() {
                    klu_sys::klu_l_free_symbolic(&mut self.symbolic, &mut self.common);
                    self.symbolic = std::ptr::null_mut();
                }
                // Also invalidate numeric since pattern changed
                if !self.numeric.is_null() {
                    klu_sys::klu_l_free_numeric(&mut self.numeric, &mut self.common);
                    self.numeric = std::ptr::null_mut();
                }
                self.has_numeric = false;

                // Perform symbolic analysis using long-integer variant
                self.symbolic = klu_sys::klu_l_analyze(
                    self.n as i64,
                    ap.as_ptr(),
                    ai.as_ptr(),
                    &mut self.common,
                );

                if self.symbolic.is_null() {
                    self.check_status()?;
                    return Err(SolverError::AnalyzeFailed);
                }
            }

            // Cache the pattern
            self.last_ap = ap.to_vec();
            self.last_ai = ai.to_vec();
        }

        #[cfg(not(feature = "klu"))]
        {
            let _ = (ap, ai);
            Err(SolverError::AnalyzeFailed)
        }
        #[cfg(feature = "klu")]
        Ok(())
    }

    #[allow(unused_variables)]
    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        if !self.enabled {
            return Err(SolverError::FactorFailed);
        }

        #[cfg(feature = "klu")]
        {
            if self.symbolic.is_null() {
                return Err(SolverError::InvalidMatrix {
                    reason: "Symbolic analysis not performed".to_string(),
                });
            }

            unsafe {
                // Check if we can refactor (same pattern, existing numeric)
                let pattern_same = self.pattern_matches(ap, ai);

                if pattern_same && self.has_numeric && !self.numeric.is_null() {
                    // Refactor: more efficient when pattern unchanged
                    let ok = klu_sys::klu_l_refactor(
                        ap.as_ptr(),
                        ai.as_ptr(),
                        ax.as_ptr(),
                        self.symbolic,
                        self.numeric,
                        &mut self.common,
                    );

                    if ok == 0 {
                        self.check_status()?;
                        return Err(SolverError::FactorFailed);
                    }
                    self.refactor_count += 1;
                } else {
                    // Full factorization needed
                    if !self.numeric.is_null() {
                        klu_sys::klu_l_free_numeric(&mut self.numeric, &mut self.common);
                        self.numeric = std::ptr::null_mut();
                    }

                    self.numeric = klu_sys::klu_l_factor(
                        ap.as_ptr(),
                        ai.as_ptr(),
                        ax.as_ptr(),
                        self.symbolic,
                        &mut self.common,
                    );

                    if self.numeric.is_null() {
                        self.check_status()?;
                        return Err(SolverError::FactorFailed);
                    }
                    self.has_numeric = true;
                    self.factor_count += 1;
                }

                // Compute condition number estimate
                let rcond_ok = klu_sys::klu_l_rcond(
                    self.symbolic,
                    self.numeric,
                    &mut self.common,
                );
                if rcond_ok != 0 {
                    self.last_rcond = self.common.rcond;
                    // Warn if ill-conditioned (but don't fail)
                    if self.last_rcond < 1e-14 {
                        // Matrix is very ill-conditioned
                        // Could return warning but continue
                    }
                }
            }
        }

        #[cfg(not(feature = "klu"))]
        {
            let _ = (ap, ai, ax);
            Err(SolverError::FactorFailed)
        }
        #[cfg(feature = "klu")]
        Ok(())
    }

    #[allow(unused_variables)]
    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        if !self.enabled {
            return Err(SolverError::SolveFailed);
        }

        #[cfg(feature = "klu")]
        {
            if self.symbolic.is_null() || self.numeric.is_null() {
                return Err(SolverError::InvalidMatrix {
                    reason: "Factorization not performed".to_string(),
                });
            }

            if rhs.len() != self.n {
                return Err(SolverError::InvalidMatrix {
                    reason: format!("RHS length {} != matrix dimension {}", rhs.len(), self.n),
                });
            }

            unsafe {
                let ok = klu_sys::klu_l_solve(
                    self.symbolic,
                    self.numeric,
                    self.n as i64,
                    1, // single right-hand side
                    rhs.as_mut_ptr(),
                    &mut self.common,
                );

                if ok == 0 {
                    self.check_status()?;
                    return Err(SolverError::SolveFailed);
                }
            }
            Ok(())
        }

        #[cfg(not(feature = "klu"))]
        Err(SolverError::SolveFailed)
    }

    fn reset_pattern(&mut self) {
        #[cfg(feature = "klu")]
        {
            unsafe {
                if !self.numeric.is_null() {
                    klu_sys::klu_l_free_numeric(&mut self.numeric, &mut self.common);
                    self.numeric = std::ptr::null_mut();
                }
                if !self.symbolic.is_null() {
                    klu_sys::klu_l_free_symbolic(&mut self.symbolic, &mut self.common);
                    self.symbolic = std::ptr::null_mut();
                }
            }
            self.has_numeric = false;
        }
        self.last_ap.clear();
        self.last_ai.clear();
    }

    fn name(&self) -> &'static str {
        "KLU"
    }
}

pub fn debug_dump_solver() {
    println!("solver: klu solver stub");
}

impl Drop for KluSolver {
    fn drop(&mut self) {
        self.reset_pattern();
    }
}

// ============================================================================
// KLU FFI Bindings
// ============================================================================
//
// KLU is part of SuiteSparse by Tim Davis.
// Reference: https://github.com/DrTimothyAldenDavis/SuiteSparse
//
// KLU is designed for sparse LU factorization of circuit simulation matrices.
// It uses a left-looking algorithm with partial pivoting.

#[cfg(feature = "klu")]
#[allow(non_camel_case_types, non_snake_case, dead_code)]
pub mod klu_sys {
    use std::os::raw::{c_double, c_int};

    // ========================================================================
    // KLU Status Codes
    // ========================================================================

    /// KLU completed successfully
    pub const KLU_OK: c_int = 0;
    /// Matrix is singular (zero pivot)
    pub const KLU_SINGULAR: c_int = 1;
    /// Out of memory
    pub const KLU_OUT_OF_MEMORY: c_int = -2;
    /// Invalid input
    pub const KLU_INVALID: c_int = -3;
    /// Pivot is too small (ill-conditioned)
    pub const KLU_TOO_LARGE: c_int = -4;

    // ========================================================================
    // Type Definitions
    // ========================================================================

    /// Index type for KLU (matches SuiteSparse_long on 64-bit systems)
    /// Note: Standard KLU uses int (i32), but klu_l_* variants use int64_t
    #[cfg(target_pointer_width = "64")]
    pub type KluInt = i64;
    #[cfg(target_pointer_width = "32")]
    pub type KluInt = i32;

    // ========================================================================
    // Opaque Structures
    // ========================================================================

    /// Opaque structure containing symbolic analysis results
    /// Created by klu_analyze, freed by klu_free_symbolic
    #[repr(C)]
    pub struct klu_symbolic {
        _private: [u8; 0],
    }

    /// Opaque structure containing numeric factorization results
    /// Created by klu_factor, freed by klu_free_numeric
    #[repr(C)]
    pub struct klu_numeric {
        _private: [u8; 0],
    }

    // ========================================================================
    // KLU Common Structure
    // ========================================================================

    /// KLU common control and statistics structure
    ///
    /// This structure contains control parameters for KLU operations and
    /// statistics from the most recent operation.
    #[repr(C)]
    #[derive(Debug, Clone)]
    pub struct klu_common {
        // ------------------------------------------------------------------
        // Control parameters (set by klu_defaults, user may modify)
        // ------------------------------------------------------------------
        /// Pivot tolerance for partial pivoting (default: 0.001)
        /// Larger values give more stable factorization but may be slower
        pub tol: c_double,

        /// Memory growth factor (default: 1.2)
        pub memgrow: c_double,

        /// Memory realloc factor (default: 1.2)
        pub initmem_amd: c_double,

        /// Initial memory allocation for matrix L (default: 10.0)
        pub initmem: c_double,

        /// Maximum work (for BTF, default: 0 = no limit)
        pub maxwork: c_double,

        /// Block triangular form control:
        /// 0 = do not use BTF
        /// 1 = use BTF if advantageous (default)
        pub btf: c_int,

        /// Ordering method:
        /// 0 = AMD (default)
        /// 1 = COLAMD
        /// 2 = user-provided P and Q
        /// 3 = natural ordering (no permutation)
        pub ordering: c_int,

        /// Scaling method:
        /// 0 = no scaling
        /// 1 = sum scaling
        /// 2 = max scaling (default)
        pub scale: c_int,

        // ------------------------------------------------------------------
        // Statistics (output from KLU operations)
        // ------------------------------------------------------------------
        /// Status code from most recent operation
        pub status: c_int,

        /// Number of off-diagonal pivots
        pub noffdiag: c_int,

        /// Number of block triangular blocks
        pub nblocks: c_int,

        /// Estimated reciprocal condition number (from klu_rcond)
        pub rcond: c_double,

        /// Reciprocal pivot growth (from klu_rgrowth)
        pub rgrowth: c_double,

        /// Work done during factorization
        pub work: c_double,

        /// Memory usage in bytes
        pub memusage: usize,

        /// Peak memory usage in bytes
        pub mempeak: usize,

        /// Flop count for factorization
        pub flops: c_double,

        /// Number of entries in L
        pub lnz: c_double,

        /// Number of entries in U
        pub unz: c_double,

        /// Number of entries in L+U (excluding diagonal)
        pub nrealloc: c_double,

        // ------------------------------------------------------------------
        // For internal use
        // ------------------------------------------------------------------
        /// User data pointer
        pub user_data: *mut std::ffi::c_void,

        /// User ordering function
        pub user_order: *mut std::ffi::c_void,
    }

    impl Default for klu_common {
        fn default() -> Self {
            Self {
                tol: 0.001,
                memgrow: 1.2,
                initmem_amd: 1.2,
                initmem: 10.0,
                maxwork: 0.0,
                btf: 1,
                ordering: 0,
                scale: 2,
                status: 0,
                noffdiag: 0,
                nblocks: 0,
                rcond: 0.0,
                rgrowth: 0.0,
                work: 0.0,
                memusage: 0,
                mempeak: 0,
                flops: 0.0,
                lnz: 0.0,
                unz: 0.0,
                nrealloc: 0.0,
                user_data: std::ptr::null_mut(),
                user_order: std::ptr::null_mut(),
            }
        }
    }

    // ========================================================================
    // Standard KLU Functions (32-bit indices)
    // ========================================================================

    #[link(name = "klu")]
    extern "C" {
        /// Initialize klu_common with default values
        ///
        /// Must be called before any other KLU function.
        /// Returns 1 on success, 0 on failure.
        pub fn klu_defaults(common: *mut klu_common) -> c_int;

        /// Perform symbolic analysis of matrix structure
        ///
        /// Analyzes the sparsity pattern and computes fill-reducing ordering.
        /// The result can be reused for matrices with the same pattern.
        ///
        /// # Arguments
        /// * `n` - Matrix dimension
        /// * `Ap` - Column pointers (size n+1)
        /// * `Ai` - Row indices
        /// * `common` - Control/statistics structure
        ///
        /// # Returns
        /// Pointer to symbolic analysis result, or NULL on failure
        pub fn klu_analyze(
            n: c_int,
            Ap: *const c_int,
            Ai: *const c_int,
            common: *mut klu_common,
        ) -> *mut klu_symbolic;

        /// Compute numeric LU factorization
        ///
        /// Uses the symbolic analysis to compute L and U factors.
        ///
        /// # Arguments
        /// * `Ap` - Column pointers (size n+1)
        /// * `Ai` - Row indices
        /// * `Ax` - Matrix values
        /// * `symbolic` - Result from klu_analyze
        /// * `common` - Control/statistics structure
        ///
        /// # Returns
        /// Pointer to numeric factorization, or NULL on failure
        pub fn klu_factor(
            Ap: *const c_int,
            Ai: *const c_int,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            common: *mut klu_common,
        ) -> *mut klu_numeric;

        /// Refactorize matrix with same pattern but different values
        ///
        /// More efficient than klu_factor when pattern is unchanged.
        /// Uses existing symbolic analysis and numeric structure.
        ///
        /// # Returns
        /// 1 on success, 0 on failure
        pub fn klu_refactor(
            Ap: *const c_int,
            Ai: *const c_int,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        /// Solve Ax = b using factorization
        ///
        /// The solution overwrites the right-hand side vector b.
        ///
        /// # Arguments
        /// * `symbolic` - From klu_analyze
        /// * `numeric` - From klu_factor
        /// * `ldim` - Leading dimension of B (usually n)
        /// * `nrhs` - Number of right-hand sides
        /// * `B` - Right-hand side(s), overwritten with solution
        /// * `common` - Control/statistics structure
        ///
        /// # Returns
        /// 1 on success, 0 on failure
        pub fn klu_solve(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            ldim: c_int,
            nrhs: c_int,
            B: *mut c_double,
            common: *mut klu_common,
        ) -> c_int;

        /// Solve A'x = b (transpose solve)
        pub fn klu_tsolve(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            ldim: c_int,
            nrhs: c_int,
            B: *mut c_double,
            common: *mut klu_common,
        ) -> c_int;

        /// Estimate reciprocal condition number
        ///
        /// Computes an estimate of 1/cond(A) using the factorization.
        /// A small value indicates an ill-conditioned matrix.
        ///
        /// # Returns
        /// 1 on success, 0 on failure. Result stored in common.rcond.
        pub fn klu_rcond(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        /// Compute reciprocal pivot growth
        ///
        /// # Returns
        /// 1 on success, 0 on failure. Result stored in common.rgrowth.
        pub fn klu_rgrowth(
            Ap: *const c_int,
            Ai: *const c_int,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        /// Free symbolic analysis result
        pub fn klu_free_symbolic(
            symbolic: *mut *mut klu_symbolic,
            common: *mut klu_common,
        ) -> c_int;

        /// Free numeric factorization result
        pub fn klu_free_numeric(
            numeric: *mut *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;
    }

    // ========================================================================
    // Long Integer KLU Functions (64-bit indices)
    // ========================================================================
    //
    // These variants use 64-bit integers for large matrices.
    // Function names have _l suffix (e.g., klu_l_analyze).

    #[link(name = "klu")]
    extern "C" {
        pub fn klu_l_defaults(common: *mut klu_common) -> c_int;

        pub fn klu_l_analyze(
            n: i64,
            Ap: *const i64,
            Ai: *const i64,
            common: *mut klu_common,
        ) -> *mut klu_symbolic;

        pub fn klu_l_factor(
            Ap: *const i64,
            Ai: *const i64,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            common: *mut klu_common,
        ) -> *mut klu_numeric;

        pub fn klu_l_refactor(
            Ap: *const i64,
            Ai: *const i64,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_solve(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            ldim: i64,
            nrhs: i64,
            B: *mut c_double,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_tsolve(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            ldim: i64,
            nrhs: i64,
            B: *mut c_double,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_rcond(
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_rgrowth(
            Ap: *const i64,
            Ai: *const i64,
            Ax: *const c_double,
            symbolic: *mut klu_symbolic,
            numeric: *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_free_symbolic(
            symbolic: *mut *mut klu_symbolic,
            common: *mut klu_common,
        ) -> c_int;

        pub fn klu_l_free_numeric(
            numeric: *mut *mut klu_numeric,
            common: *mut klu_common,
        ) -> c_int;
    }

    // ========================================================================
    // Helper Functions
    // ========================================================================

    /// Convert KLU status code to error message
    pub fn status_message(status: c_int) -> &'static str {
        match status {
            KLU_OK => "OK",
            KLU_SINGULAR => "Matrix is singular",
            KLU_OUT_OF_MEMORY => "Out of memory",
            KLU_INVALID => "Invalid input",
            KLU_TOO_LARGE => "Problem too large",
            _ => "Unknown error",
        }
    }
}
