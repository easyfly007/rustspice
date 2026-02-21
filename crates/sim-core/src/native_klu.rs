//! Native Rust KLU Solver
//!
//! Pure Rust implementation of the KLU algorithm for sparse LU factorization,
//! optimized for circuit simulation matrices. Eliminates the need for the
//! SuiteSparse C library dependency.
//!
//! # Algorithm
//!
//! KLU uses a three-phase approach:
//!
//! 1. **Analyze**: BTF decomposition + AMD ordering per block + symbolic factorization
//! 2. **Factor**: Gilbert-Peierls left-looking LU with threshold partial pivoting
//! 3. **Solve**: Block back-substitution with triangular solves
//!
//! # Features
//!
//! - BTF (Block Triangular Form) decomposition via `btf_decompose()`
//! - AMD fill-reducing ordering via `amd_order()`
//! - Gilbert-Peierls DFS-based symbolic + numeric factorization
//! - Threshold partial pivoting (configurable tolerance)
//! - Refactorization: reuses symbolic pattern when matrix values change
//! - Column-level parallel factorization (feature: `parallel`,
//!   active during refactorization when block size >= 64)
//! - Condition number estimation via `rcond()`
//!
//! # References
//!
//! - Davis, T.A., Palamadai Natarajan, E. "Algorithm 907: KLU" ACM TOMS, 2010.
//! - Gilbert, J.R., Peierls, T. "Sparse partial pivoting in time proportional to
//!   arithmetic operations" SIAM J. Sci. Stat. Comput., 1988.

use crate::amd::amd_order;
use crate::btf::{btf_decompose, BtfDecomposition};
use crate::solver::{LinearSolver, SolverError};

#[cfg(feature = "parallel")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "parallel")]
use std::thread;

// ============================================================================
// Configuration
// ============================================================================

/// Configuration parameters for the Native KLU solver.
#[derive(Debug, Clone)]
pub struct KluConfig {
    /// Pivot tolerance for threshold partial pivoting.
    /// During column factorization, a row is eligible as pivot if
    /// `|a_ik| >= tol * max|a_*k|`. Default: 0.001.
    pub pivot_tol: f64,
    /// Ordering strategy: 0 = AMD (default), 1 = COLAMD (falls back to AMD with warning), 3 = natural.
    pub ordering: i32,
    /// Whether to use BTF decomposition. Default: true.
    pub use_btf: bool,
    /// Minimum block size below which BTF is skipped (single-block fallback).
    pub btf_min_blocks: usize,
}

impl Default for KluConfig {
    fn default() -> Self {
        Self {
            pivot_tol: 0.001,
            ordering: 0,
            use_btf: true,
            btf_min_blocks: 2,
        }
    }
}

// ============================================================================
// Solver state
// ============================================================================

/// Solver lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SolverState {
    Empty,
    Analyzed,
    Factored,
}

// ============================================================================
// Per-block symbolic + numeric factors
// ============================================================================

/// Symbolic factorization for one diagonal block.
/// Stores the non-zero pattern of L and U discovered by Gilbert-Peierls DFS.
#[derive(Debug, Clone)]
struct SymbolicFactor {
    /// Block dimension.
    n: usize,
    /// Column permutation from AMD: new_col = col_perm[old_col].
    col_perm: Vec<usize>,
    /// Inverse column permutation: old_col = col_perm_inv[new_col].
    col_perm_inv: Vec<usize>,
    /// For each column k (in permuted order), the list of row indices in L
    /// (below diagonal, in permuted row space). Stored flat with pointers.
    l_col_ptr: Vec<usize>,
    l_row_idx: Vec<usize>,
    /// For each column k, the list of row indices in U (above diagonal + diagonal).
    /// Diagonal is stored last in each column segment.
    u_col_ptr: Vec<usize>,
    u_row_idx: Vec<usize>,
}

/// Numeric factorization for one diagonal block.
#[derive(Debug, Clone)]
struct NumericFactor {
    /// L values corresponding to SymbolicFactor::l_row_idx.
    l_values: Vec<f64>,
    /// U values corresponding to SymbolicFactor::u_row_idx.
    u_values: Vec<f64>,
    /// Row permutation from partial pivoting: row_perm[k] = original row chosen as pivot for step k.
    row_perm: Vec<usize>,
    /// Inverse row permutation.
    row_perm_inv: Vec<usize>,
    /// Diagonal of U (redundant with u_values but handy for rcond).
    u_diag: Vec<f64>,
}

/// Off-diagonal block between two BTF blocks, stored in CSC.
#[derive(Debug, Clone)]
struct OffDiagBlock {
    block_row: usize,
    block_col: usize,
    row_start: usize,
    col_start: usize,
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

// ============================================================================
// Statistics
// ============================================================================

/// Statistics from Native KLU factorization.
#[derive(Debug, Clone, Default)]
pub struct NativeKluStats {
    pub factor_count: usize,
    pub refactor_count: usize,
    pub nnz_l: usize,
    pub nnz_u: usize,
    pub nblocks: usize,
    pub noffdiag: usize,
    pub rcond: f64,
    /// Whether the last factorization used the parallel path.
    pub parallel_factor: bool,
}

// ============================================================================
// Main solver struct
// ============================================================================

/// Native Rust KLU Solver.
///
/// Implements the KLU algorithm entirely in Rust, with BTF decomposition,
/// AMD ordering, Gilbert-Peierls LU with threshold partial pivoting,
/// refactorization, and condition number estimation.
pub struct NativeKluSolver {
    n: usize,
    config: KluConfig,

    // BTF decomposition (None when BTF disabled or single block)
    btf: Option<BtfDecomposition>,

    // Per-block factors
    block_symbolic: Vec<SymbolicFactor>,
    block_numeric: Vec<NumericFactor>,

    // Off-diagonal blocks in BTF
    off_diag: Vec<OffDiagBlock>,

    // Pattern cache for refactorization detection
    last_ap: Vec<i64>,
    last_ai: Vec<i64>,

    state: SolverState,
    stats: NativeKluStats,

    // Workspace (reused)
    work: Vec<f64>,

    // Parallel factorization thread count (0 = off, >0 = thread count)
    parallel_threads: usize,
}

unsafe impl Send for NativeKluSolver {}

impl NativeKluSolver {
    /// Create a new Native KLU solver for matrices of dimension `n`.
    pub fn new(n: usize) -> Self {
        Self {
            n,
            config: KluConfig::default(),
            btf: None,
            block_symbolic: Vec::new(),
            block_numeric: Vec::new(),
            off_diag: Vec::new(),
            last_ap: Vec::new(),
            last_ai: Vec::new(),
            state: SolverState::Empty,
            stats: NativeKluStats::default(),
            work: vec![0.0; n],
            parallel_threads: 0,
        }
    }

    /// Create with custom configuration.
    pub fn with_config(n: usize, config: KluConfig) -> Self {
        let mut s = Self::new(n);
        s.config = config;
        s
    }

    /// Set pivot tolerance (0, 1].
    pub fn set_pivot_tolerance(&mut self, tol: f64) {
        self.config.pivot_tol = tol.clamp(1e-15, 1.0);
    }

    /// Set ordering strategy: 0=AMD, 1=COLAMD (fallback AMD), 3=natural.
    pub fn set_ordering(&mut self, ordering: i32) {
        self.config.ordering = ordering;
    }

    /// Enable or disable BTF.
    pub fn set_btf(&mut self, enable: bool) {
        self.config.use_btf = enable;
    }

    /// Get reciprocal condition number estimate from last factorization.
    pub fn rcond(&self) -> f64 {
        self.stats.rcond
    }

    /// Get statistics.
    pub fn stats(&self) -> &NativeKluStats {
        &self.stats
    }

    // ========================================================================
    // Pattern matching
    // ========================================================================

    fn pattern_matches(&self, ap: &[i64], ai: &[i64]) -> bool {
        self.last_ap == ap && self.last_ai == ai
    }

    // ========================================================================
    // Phase 1: Analyze
    // ========================================================================

    fn do_analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        let n = self.n;

        // Reset (preserve factor/refactor counts across re-analysis)
        let saved_factor_count = self.stats.factor_count;
        let saved_refactor_count = self.stats.refactor_count;
        self.block_symbolic.clear();
        self.block_numeric.clear();
        self.off_diag.clear();
        self.stats = NativeKluStats::default();
        self.stats.factor_count = saved_factor_count;
        self.stats.refactor_count = saved_refactor_count;

        // BTF decomposition
        let use_btf = self.config.use_btf && n >= 4;
        let btf = if use_btf {
            let b = btf_decompose(n, ap, ai);
            if b.num_blocks >= self.config.btf_min_blocks && b.max_block_size < n {
                Some(b)
            } else {
                None
            }
        } else {
            None
        };

        if let Some(ref btf) = btf {
            self.stats.nblocks = btf.num_blocks;
            // Analyze each diagonal block
            for k in 0..btf.num_blocks {
                let (blk_start, blk_end) = btf.block_range(k);
                let blk_size = blk_end - blk_start;
                let (blk_ap, blk_ai) = self.extract_block_pattern(btf, ap, ai, blk_start, blk_end);
                let sym = self.symbolic_for_block(blk_size, &blk_ap, &blk_ai)?;
                self.block_symbolic.push(sym);
            }
            // Extract off-diagonal blocks
            self.extract_off_diag_patterns(btf, ap, ai);
        } else {
            self.stats.nblocks = 1;
            // Single block: whole matrix
            let sym = self.symbolic_for_block(n, ap, ai)?;
            self.block_symbolic.push(sym);
        }

        self.btf = btf;
        self.state = SolverState::Analyzed;
        self.last_ap = ap.to_vec();
        self.last_ai = ai.to_vec();

        // Accumulate nnz stats
        for sym in &self.block_symbolic {
            self.stats.nnz_l += sym.l_row_idx.len();
            self.stats.nnz_u += sym.u_row_idx.len();
        }

        Ok(())
    }

    /// Extract CSC pattern for a diagonal block.
    fn extract_block_pattern(
        &self,
        btf: &BtfDecomposition,
        ap: &[i64],
        ai: &[i64],
        blk_start: usize,
        blk_end: usize,
    ) -> (Vec<i64>, Vec<i64>) {
        let n = self.n;
        let blk_size = blk_end - blk_start;
        let mut blk_ap = vec![0i64; blk_size + 1];
        let mut blk_ai = Vec::new();

        for local_col in 0..blk_size {
            let global_col_new = blk_start + local_col;
            let global_col_old = btf.col_perm[global_col_new];
            let col_start = ap[global_col_old] as usize;
            let col_end = ap[global_col_old + 1] as usize;

            for idx in col_start..col_end {
                let global_row_old = ai[idx] as usize;
                if global_row_old >= n {
                    continue;
                }
                let global_row_new = btf.row_perm_inv[global_row_old];
                if global_row_new >= blk_start && global_row_new < blk_end {
                    blk_ai.push((global_row_new - blk_start) as i64);
                }
            }
            blk_ap[local_col + 1] = blk_ai.len() as i64;
        }

        (blk_ap, blk_ai)
    }

    /// Extract off-diagonal block patterns for BTF.
    fn extract_off_diag_patterns(
        &mut self,
        btf: &BtfDecomposition,
        ap: &[i64],
        ai: &[i64],
    ) {
        let n = self.n;
        self.off_diag.clear();
        let mut noffdiag = 0usize;

        for blk_row in 0..btf.num_blocks {
            let (row_start, row_end) = btf.block_range(blk_row);
            let nrows = row_end - row_start;

            for blk_col in (blk_row + 1)..btf.num_blocks {
                let (col_start, col_end) = btf.block_range(blk_col);
                let ncols = col_end - col_start;

                let mut col_ptr = vec![0usize; ncols + 1];
                let mut row_idx = Vec::new();

                for local_col in 0..ncols {
                    let global_col_new = col_start + local_col;
                    let global_col_old = btf.col_perm[global_col_new];
                    let c_start = ap[global_col_old] as usize;
                    let c_end = ap[global_col_old + 1] as usize;

                    for idx in c_start..c_end {
                        let global_row_old = ai[idx] as usize;
                        if global_row_old >= n {
                            continue;
                        }
                        let global_row_new = btf.row_perm_inv[global_row_old];
                        if global_row_new >= row_start && global_row_new < row_end {
                            row_idx.push(global_row_new - row_start);
                        }
                    }
                    col_ptr[local_col + 1] = row_idx.len();
                }

                if !row_idx.is_empty() {
                    noffdiag += row_idx.len();
                    self.off_diag.push(OffDiagBlock {
                        block_row: blk_row,
                        block_col: blk_col,
                        row_start,
                        col_start,
                        nrows,
                        ncols,
                        col_ptr,
                        row_idx,
                        values: Vec::new(),
                    });
                }
            }
        }
        self.stats.noffdiag = noffdiag;
    }

    // ========================================================================
    // Symbolic factorization for a single block (Gilbert-Peierls DFS)
    // ========================================================================

    fn symbolic_for_block(
        &self,
        n: usize,
        ap: &[i64],
        ai: &[i64],
    ) -> Result<SymbolicFactor, SolverError> {
        if n == 0 {
            return Ok(SymbolicFactor {
                n: 0,
                col_perm: Vec::new(),
                col_perm_inv: Vec::new(),
                l_col_ptr: vec![0],
                l_row_idx: Vec::new(),
                u_col_ptr: vec![0],
                u_row_idx: Vec::new(),
            });
        }

        // Compute ordering
        let (col_perm, col_perm_inv) = if self.config.ordering == 3 {
            // Natural ordering
            let p: Vec<usize> = (0..n).collect();
            let q: Vec<usize> = (0..n).collect();
            (p, q)
        } else {
            if self.config.ordering == 1 {
                eprintln!("Warning: COLAMD not implemented, falling back to AMD");
            }
            let amd_result = amd_order(n, ap, ai);
            (amd_result.perm, amd_result.inv_perm)
        };

        // Build adjacency for the permuted matrix (CSC) for DFS
        // perm_ap/perm_ai: CSC of P*A*P^T
        let mut perm_ap = vec![0i64; n + 1];
        let mut perm_ai = Vec::new();

        for new_col in 0..n {
            let old_col = col_perm_inv[new_col];
            let start = ap[old_col] as usize;
            let end = ap[old_col + 1] as usize;
            for idx in start..end {
                let old_row = ai[idx] as usize;
                if old_row < n {
                    let new_row = col_perm[old_row];
                    perm_ai.push(new_row as i64);
                }
            }
            perm_ap[new_col + 1] = perm_ai.len() as i64;
        }

        // Gilbert-Peierls symbolic: DFS per column to find L/U patterns
        let mut l_col_ptr = vec![0usize; n + 1];
        let mut l_row_idx = Vec::new();
        let mut u_col_ptr = vec![0usize; n + 1];
        let mut u_row_idx = Vec::new();

        // Workspace for DFS
        let mut mark = vec![0usize; n];
        let mut current_mark = 0usize;
        let mut stack = Vec::with_capacity(n);
        let mut pattern = Vec::with_capacity(n);

        // For each column k (in permuted order)
        for k in 0..n {
            current_mark += 1;
            pattern.clear();

            // Seed DFS with non-zero rows in column k of permuted A
            let col_start = perm_ap[k] as usize;
            let col_end = perm_ap[k + 1] as usize;

            for idx in col_start..col_end {
                let row = perm_ai[idx] as usize;
                if row != k && mark[row] != current_mark {
                    mark[row] = current_mark;
                    if row < k {
                        // Row in U region: DFS through L columns
                        stack.push((row, true)); // (node, first_visit)
                    } else {
                        // Row in L region: direct entry
                        pattern.push(row);
                    }
                }
            }

            // DFS through upper part to discover fill-in
            while let Some((node, first_visit)) = stack.pop() {
                if first_visit {
                    // Push node back as "visited", then push its L-column children
                    stack.push((node, false));
                    // node < k: look at L column of node for fill
                    // We use the already-computed l_row_idx for columns < k
                    let l_start = l_col_ptr[node];
                    let l_end = l_col_ptr[node + 1];
                    for l_idx in l_start..l_end {
                        let child = l_row_idx[l_idx];
                        if mark[child] != current_mark {
                            mark[child] = current_mark;
                            if child < k {
                                stack.push((child, true));
                            } else {
                                pattern.push(child);
                            }
                        }
                    }
                }
                // On second visit: node is done, add to U pattern
                if !first_visit {
                    pattern.push(node);
                }
            }

            // Now split pattern into L part (> k) and U part (< k)
            let mut u_rows = Vec::new();
            let mut l_rows = Vec::new();
            for &row in &pattern {
                if row < k {
                    u_rows.push(row);
                } else if row > k {
                    l_rows.push(row);
                }
            }

            // Sort for deterministic order
            u_rows.sort_unstable();
            l_rows.sort_unstable();

            // Store U column (above-diagonal rows + diagonal at end)
            let u_start = u_row_idx.len();
            u_row_idx.extend_from_slice(&u_rows);
            u_row_idx.push(k); // diagonal
            u_col_ptr[k] = u_start;
            u_col_ptr[k + 1] = u_row_idx.len();

            // Store L column (below-diagonal rows)
            let l_start = l_row_idx.len();
            l_row_idx.extend_from_slice(&l_rows);
            l_col_ptr[k] = l_start;
            l_col_ptr[k + 1] = l_row_idx.len();
        }

        Ok(SymbolicFactor {
            n,
            col_perm,
            col_perm_inv,
            l_col_ptr,
            l_row_idx,
            u_col_ptr,
            u_row_idx,
        })
    }

    // ========================================================================
    // Phase 2: Numeric factorization (full factor or refactorization)
    // ========================================================================

    fn do_factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        let pattern_same = self.state == SolverState::Factored && self.pattern_matches(ap, ai);
        #[allow(unused_mut)]
        let mut used_parallel = false;

        if let Some(ref btf) = self.btf {
            let btf = btf.clone();
            // Factor each diagonal block
            let mut new_numerics: Vec<NumericFactor> = Vec::with_capacity(btf.num_blocks);

            for k in 0..btf.num_blocks {
                let (blk_start, blk_end) = btf.block_range(k);
                let (blk_ap, blk_ai, blk_ax) =
                    self.extract_block_values(&btf, ap, ai, ax, blk_start, blk_end);

                let sym = &self.block_symbolic[k];
                let old_numeric = if pattern_same && k < self.block_numeric.len() {
                    Some(&self.block_numeric[k])
                } else {
                    None
                };

                let num = {
                    #[cfg(feature = "parallel")]
                    {
                        if self.parallel_threads >= 2 && pattern_same && old_numeric.is_some() && sym.n >= 64 {
                            used_parallel = true;
                            self.factor_block_parallel(sym, &blk_ap, &blk_ai, &blk_ax, old_numeric.unwrap())?
                        } else {
                            self.factor_block(sym, &blk_ap, &blk_ai, &blk_ax, old_numeric, pattern_same)?
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    {
                        self.factor_block(sym, &blk_ap, &blk_ai, &blk_ax, old_numeric, pattern_same)?
                    }
                };
                new_numerics.push(num);
            }

            self.block_numeric = new_numerics;

            // Extract off-diagonal values (inlined to avoid borrow conflict)
            let n = self.n;
            for off_blk in &mut self.off_diag {
                let row_start = off_blk.row_start;
                let row_end = row_start + off_blk.nrows;
                let col_start = off_blk.col_start;

                off_blk.values.clear();

                for local_col in 0..off_blk.ncols {
                    let global_col_new = col_start + local_col;
                    let global_col_old = btf.col_perm[global_col_new];
                    let c_start = ap[global_col_old] as usize;
                    let c_end = ap[global_col_old + 1] as usize;

                    for idx in c_start..c_end {
                        let global_row_old = ai[idx] as usize;
                        if global_row_old >= n {
                            continue;
                        }
                        let global_row_new = btf.row_perm_inv[global_row_old];
                        if global_row_new >= row_start && global_row_new < row_end {
                            off_blk.values.push(ax[idx]);
                        }
                    }
                }
            }
        } else {
            // Single block
            let sym = &self.block_symbolic[0];
            let old_numeric = if pattern_same && !self.block_numeric.is_empty() {
                Some(&self.block_numeric[0])
            } else {
                None
            };

            let num = {
                #[cfg(feature = "parallel")]
                {
                    if self.parallel_threads >= 2 && pattern_same && old_numeric.is_some() && sym.n >= 64 {
                        used_parallel = true;
                        self.factor_block_parallel(sym, ap, ai, ax, old_numeric.unwrap())?
                    } else {
                        self.factor_block(sym, ap, ai, ax, old_numeric, pattern_same)?
                    }
                }
                #[cfg(not(feature = "parallel"))]
                {
                    self.factor_block(sym, ap, ai, ax, old_numeric, pattern_same)?
                }
            };

            self.block_numeric.clear();
            self.block_numeric.push(num);
        }

        // Update stats
        self.stats.parallel_factor = used_parallel;
        if pattern_same && self.state == SolverState::Factored {
            self.stats.refactor_count += 1;
        } else {
            self.stats.factor_count += 1;
        }

        // Compute rcond
        self.compute_rcond();

        self.state = SolverState::Factored;
        Ok(())
    }

    /// Extract CSC values for a diagonal block.
    fn extract_block_values(
        &self,
        btf: &BtfDecomposition,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
        blk_start: usize,
        blk_end: usize,
    ) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
        let n = self.n;
        let blk_size = blk_end - blk_start;
        let mut blk_ap = vec![0i64; blk_size + 1];
        let mut blk_ai = Vec::new();
        let mut blk_ax = Vec::new();

        for local_col in 0..blk_size {
            let global_col_new = blk_start + local_col;
            let global_col_old = btf.col_perm[global_col_new];
            let col_start = ap[global_col_old] as usize;
            let col_end = ap[global_col_old + 1] as usize;

            for idx in col_start..col_end {
                let global_row_old = ai[idx] as usize;
                if global_row_old >= n {
                    continue;
                }
                let global_row_new = btf.row_perm_inv[global_row_old];
                if global_row_new >= blk_start && global_row_new < blk_end {
                    blk_ai.push((global_row_new - blk_start) as i64);
                    blk_ax.push(ax[idx]);
                }
            }
            blk_ap[local_col + 1] = blk_ai.len() as i64;
        }

        (blk_ap, blk_ai, blk_ax)
    }


    // ========================================================================
    // Gilbert-Peierls numeric LU with threshold partial pivoting
    // ========================================================================

    fn factor_block(
        &self,
        sym: &SymbolicFactor,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
        old_numeric: Option<&NumericFactor>,
        refactor: bool,
    ) -> Result<NumericFactor, SolverError> {
        let n = sym.n;
        if n == 0 {
            return Ok(NumericFactor {
                l_values: Vec::new(),
                u_values: Vec::new(),
                row_perm: Vec::new(),
                row_perm_inv: Vec::new(),
                u_diag: Vec::new(),
            });
        }

        // Handle 1x1 block (singleton) fast path
        if n == 1 {
            // Find the single diagonal value
            let old_col = sym.col_perm_inv[0];
            let start = ap[old_col] as usize;
            let end = ap[old_col + 1] as usize;
            let mut diag = 0.0;
            for idx in start..end {
                let row = ai[idx] as usize;
                if row < n && sym.col_perm[row] == 0 {
                    diag += ax[idx];
                }
            }
            if diag.abs() < 1e-300 {
                diag = 1e-14;
            }
            return Ok(NumericFactor {
                l_values: Vec::new(),
                u_values: vec![diag], // diagonal stored
                row_perm: vec![0],
                row_perm_inv: vec![0],
                u_diag: vec![diag],
            });
        }

        let mut l_values = vec![0.0; sym.l_row_idx.len()];
        let mut u_values = vec![0.0; sym.u_row_idx.len()];
        let mut u_diag = vec![0.0; n];

        // Row permutation: if refactoring, reuse old pivots; else identity to start
        let mut row_perm: Vec<usize>;
        let mut row_perm_inv: Vec<usize>;
        let use_old_pivots = refactor && old_numeric.is_some();

        if use_old_pivots {
            let old = old_numeric.unwrap();
            row_perm = old.row_perm.clone();
            row_perm_inv = old.row_perm_inv.clone();
        } else {
            row_perm = (0..n).collect();
            row_perm_inv = (0..n).collect();
        }

        // Workspace
        let mut work = vec![0.0; n];

        // For each column k (in column-permuted order)
        for k in 0..n {
            let orig_col = sym.col_perm_inv[k];

            // Step 1: Scatter column k of A (in current row permutation) into work
            let start = ap[orig_col] as usize;
            let end = ap[orig_col + 1] as usize;

            for idx in start..end {
                let orig_row = ai[idx] as usize;
                if orig_row < n {
                    let perm_row = row_perm_inv[sym.col_perm[orig_row]];
                    work[perm_row] += ax[idx];
                }
            }

            // Step 2: Left-looking update — for each j < k where U(j,k) ≠ 0
            let u_start = sym.u_col_ptr[k];
            let u_end = sym.u_col_ptr[k + 1] - 1; // exclude diagonal

            for u_idx in u_start..u_end {
                let j = sym.u_row_idx[u_idx];
                let u_jk = work[j];

                if u_jk != 0.0 {
                    // Subtract L(:,j) * U(j,k)
                    let l_start = sym.l_col_ptr[j];
                    let l_end = sym.l_col_ptr[j + 1];
                    for l_idx in l_start..l_end {
                        let row = sym.l_row_idx[l_idx];
                        work[row] -= l_values[l_idx] * u_jk;
                    }
                }
            }

            // Step 3: Threshold partial pivoting (skip if refactoring)
            if !use_old_pivots {
                // Find max absolute value in L region (below diagonal)
                let l_start_k = sym.l_col_ptr[k];
                let l_end_k = sym.l_col_ptr[k + 1];

                let diag_abs = work[k].abs();
                let mut max_abs = diag_abs;
                let mut max_row = k;

                for l_idx in l_start_k..l_end_k {
                    let row = sym.l_row_idx[l_idx];
                    let val = work[row].abs();
                    if val > max_abs {
                        max_abs = val;
                        max_row = row;
                    }
                }

                // Swap if pivot is better than diagonal by threshold
                if max_row != k && diag_abs < self.config.pivot_tol * max_abs {
                    // Swap rows k and max_row in work
                    work.swap(k, max_row);

                    // Update row permutation
                    let orig_k = row_perm[k];
                    let orig_max = row_perm[max_row];
                    row_perm[k] = orig_max;
                    row_perm[max_row] = orig_k;
                    row_perm_inv[orig_k] = max_row;
                    row_perm_inv[orig_max] = k;

                    // Also swap any already-computed L values in previous columns
                    // that reference these rows — this is the key row-swap propagation
                    // Actually, since we're using the symbolic pattern which is fixed,
                    // the row swap only affects work[] for this column going forward.
                    // The symbolic pattern uses permuted indices, so swapping in work
                    // is sufficient.
                }
            }

            // Step 4: Extract U entries for this column
            for u_idx in u_start..u_end {
                let row = sym.u_row_idx[u_idx];
                u_values[u_idx] = work[row];
            }

            // Diagonal
            let mut diag = work[k];
            if diag.abs() < 1e-14 {
                diag = if diag >= 0.0 { 1e-14 } else { -1e-14 };
            }
            u_values[sym.u_col_ptr[k + 1] - 1] = diag;
            u_diag[k] = diag;

            // Step 5: Compute L(:,k) = work(below diagonal) / diag
            let l_start_k = sym.l_col_ptr[k];
            let l_end_k = sym.l_col_ptr[k + 1];
            for l_idx in l_start_k..l_end_k {
                let row = sym.l_row_idx[l_idx];
                l_values[l_idx] = work[row] / diag;
            }

            // Step 6: Clear work (only touched entries)
            for idx in start..end {
                let orig_row = ai[idx] as usize;
                if orig_row < n {
                    let perm_row = row_perm_inv[sym.col_perm[orig_row]];
                    work[perm_row] = 0.0;
                }
            }
            for l_idx in l_start_k..l_end_k {
                work[sym.l_row_idx[l_idx]] = 0.0;
            }
            for u_idx in u_start..u_end {
                work[sym.u_row_idx[u_idx]] = 0.0;
            }
            work[k] = 0.0;
        }

        Ok(NumericFactor {
            l_values,
            u_values,
            row_perm,
            row_perm_inv,
            u_diag,
        })
    }

    // ========================================================================
    // Parallel numeric refactorization (column-level parallelism)
    // ========================================================================

    /// Parallel refactorization using interleaved column assignment with
    /// spin-wait synchronization. Only called during refactorization when
    /// pivot order is fixed (use_old_pivots=true), block size >= 64, and
    /// the `parallel` feature is enabled.
    ///
    /// Safety: Uses raw pointers for shared output buffers (`l_values`,
    /// `u_values`, `u_diag`). This is safe because:
    /// - Each column writes to a disjoint range determined by `l_col_ptr`/`u_col_ptr`
    /// - Reads from column j only happen after `ready[j]` is set (Acquire ordering)
    /// - Writes are followed by `ready[k].store(true, Release)` (happens-before)
    #[cfg(feature = "parallel")]
    fn factor_block_parallel(
        &self,
        sym: &SymbolicFactor,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
        old_numeric: &NumericFactor,
    ) -> Result<NumericFactor, SolverError> {
        let n = sym.n;

        // Pre-allocate shared output buffers
        let mut l_values = vec![0.0f64; sym.l_row_idx.len()];
        let mut u_values = vec![0.0f64; sym.u_row_idx.len()];
        let mut u_diag = vec![0.0f64; n];

        // Ready flags for column synchronization
        let ready: Vec<AtomicBool> = (0..n).map(|_| AtomicBool::new(false)).collect();

        // Row permutation from previous factorization (read-only)
        let row_perm_inv = &old_numeric.row_perm_inv;

        // Raw pointers for disjoint writes from multiple threads
        let l_ptr = l_values.as_mut_ptr();
        let u_ptr = u_values.as_mut_ptr();
        let d_ptr = u_diag.as_mut_ptr();

        // Wrapper to send raw pointers across thread boundaries.
        // Safety: each column writes to a disjoint range, and reads
        // from column j only occur after ready[j] is set (Acquire/Release).
        struct SendPtr(*mut f64);
        unsafe impl Send for SendPtr {}
        unsafe impl Sync for SendPtr {}

        let l_send = SendPtr(l_ptr);
        let u_send = SendPtr(u_ptr);
        let d_send = SendPtr(d_ptr);

        // Use real OS threads (not rayon tasks) because spin-waiting
        // would block rayon's cooperative work-stealing scheduler.
        // Thread count is controlled by solver_parallel option.
        let num_threads = self.parallel_threads.min(n);

        thread::scope(|s| {
            for tid in 0..num_threads {
                let ready_ref = &ready;
                let sym_ref = sym;
                let l_s = &l_send;
                let u_s = &u_send;
                let d_s = &d_send;

                s.spawn(move || {
                    let l_raw = l_s.0;
                    let u_raw = u_s.0;
                    let d_raw = d_s.0;

                    // Per-thread workspace
                    let mut work = vec![0.0f64; n];

                    // Interleaved column assignment: thread tid handles
                    // columns tid, tid+T, tid+2T, ...
                    let mut k = tid;
                    while k < n {
                        let orig_col = sym_ref.col_perm_inv[k];

                        // Step 1: Scatter column k of A into work using row_perm_inv (read-only)
                        let start = ap[orig_col] as usize;
                        let end = ap[orig_col + 1] as usize;
                        for idx in start..end {
                            let orig_row = ai[idx] as usize;
                            if orig_row < n {
                                let perm_row = row_perm_inv[sym_ref.col_perm[orig_row]];
                                work[perm_row] += ax[idx];
                            }
                        }

                        // Step 2: Left-looking update — for each j < k where U(j,k) != 0
                        let u_start = sym_ref.u_col_ptr[k];
                        let u_end = sym_ref.u_col_ptr[k + 1] - 1; // exclude diagonal

                        for u_idx in u_start..u_end {
                            let j = sym_ref.u_row_idx[u_idx];

                            // Spin-wait until column j is ready
                            while !ready_ref[j].load(Ordering::Acquire) {
                                std::hint::spin_loop();
                            }

                            let u_jk = work[j];
                            if u_jk != 0.0 {
                                let l_start_j = sym_ref.l_col_ptr[j];
                                let l_end_j = sym_ref.l_col_ptr[j + 1];
                                for l_idx in l_start_j..l_end_j {
                                    let row = sym_ref.l_row_idx[l_idx];
                                    // Read L value from shared buffer (column j is ready)
                                    let l_val = unsafe { *l_raw.add(l_idx) };
                                    work[row] -= l_val * u_jk;
                                }
                            }
                        }

                        // Step 3: Extract U entries for this column
                        for u_idx in u_start..u_end {
                            let row = sym_ref.u_row_idx[u_idx];
                            unsafe { *u_raw.add(u_idx) = work[row]; }
                        }

                        // Diagonal
                        let mut diag = work[k];
                        if diag.abs() < 1e-14 {
                            diag = if diag >= 0.0 { 1e-14 } else { -1e-14 };
                        }
                        let diag_idx = sym_ref.u_col_ptr[k + 1] - 1;
                        unsafe {
                            *u_raw.add(diag_idx) = diag;
                            *d_raw.add(k) = diag;
                        }

                        // Step 4: Compute L(:,k) = work[below diag] / diag
                        let l_start_k = sym_ref.l_col_ptr[k];
                        let l_end_k = sym_ref.l_col_ptr[k + 1];
                        for l_idx in l_start_k..l_end_k {
                            let row = sym_ref.l_row_idx[l_idx];
                            unsafe { *l_raw.add(l_idx) = work[row] / diag; }
                        }

                        // Step 5: Clear work (only touched entries)
                        for idx in start..end {
                            let orig_row = ai[idx] as usize;
                            if orig_row < n {
                                let perm_row = row_perm_inv[sym_ref.col_perm[orig_row]];
                                work[perm_row] = 0.0;
                            }
                        }
                        for l_idx in l_start_k..l_end_k {
                            work[sym_ref.l_row_idx[l_idx]] = 0.0;
                        }
                        for u_idx in u_start..u_end {
                            work[sym_ref.u_row_idx[u_idx]] = 0.0;
                        }
                        work[k] = 0.0;

                        // Step 6: Signal this column is done
                        ready_ref[k].store(true, Ordering::Release);

                        k += num_threads;
                    }
                });
            }
        });

        Ok(NumericFactor {
            l_values,
            u_values,
            row_perm: old_numeric.row_perm.clone(),
            row_perm_inv: old_numeric.row_perm_inv.clone(),
            u_diag,
        })
    }

    // ========================================================================
    // Condition number estimation
    // ========================================================================

    fn compute_rcond(&mut self) {
        let mut min_diag = f64::MAX;
        let mut max_diag = 0.0f64;

        for num in &self.block_numeric {
            for &d in &num.u_diag {
                let a = d.abs();
                if a < min_diag {
                    min_diag = a;
                }
                if a > max_diag {
                    max_diag = a;
                }
            }
        }

        self.stats.rcond = if max_diag > 0.0 {
            min_diag / max_diag
        } else {
            0.0
        };
    }

    // ========================================================================
    // Phase 3: Solve
    // ========================================================================

    fn do_solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        if let Some(ref btf) = self.btf {
            self.solve_btf(btf.clone(), rhs)
        } else {
            self.solve_single_block(rhs)
        }
    }

    /// Solve when the matrix is a single block (no BTF).
    fn solve_single_block(&self, rhs: &mut [f64]) -> Result<(), SolverError> {
        let sym = &self.block_symbolic[0];
        let num = &self.block_numeric[0];
        let n = sym.n;

        let mut temp = vec![0.0; n];

        // Apply permutations: temp[row_perm_inv[col_perm[i]]] = rhs[i]
        // Combined: P_row * P_col * rhs
        for i in 0..n {
            let col_new = sym.col_perm[i];
            let row_new = num.row_perm_inv[col_new];
            temp[row_new] = rhs[i];
        }

        // Forward solve: L * z = temp (L is unit lower triangular, column-oriented)
        for k in 0..n {
            let z_k = temp[k];
            let l_start = sym.l_col_ptr[k];
            let l_end = sym.l_col_ptr[k + 1];
            for l_idx in l_start..l_end {
                let row = sym.l_row_idx[l_idx];
                temp[row] -= num.l_values[l_idx] * z_k;
            }
        }

        // Backward solve: U * x = temp
        for k in (0..n).rev() {
            let diag = num.u_diag[k];
            temp[k] /= diag;
            let x_k = temp[k];

            let u_start = sym.u_col_ptr[k];
            let u_end = sym.u_col_ptr[k + 1] - 1;
            for u_idx in u_start..u_end {
                let row = sym.u_row_idx[u_idx];
                temp[row] -= num.u_values[u_idx] * x_k;
            }
        }

        // Apply inverse column permutation
        for i in 0..n {
            rhs[i] = temp[sym.col_perm[i]];
        }

        Ok(())
    }

    /// Solve with BTF block structure.
    fn solve_btf(&mut self, btf: BtfDecomposition, rhs: &mut [f64]) -> Result<(), SolverError> {
        let n = self.n;
        self.work.resize(n, 0.0);

        // Apply BTF row permutation
        for old_row in 0..n {
            let new_pos = btf.row_perm_inv[old_row];
            self.work[new_pos] = rhs[old_row];
        }

        // Solve blocks from last to first (backward for upper triangular BTF)
        for k in (0..btf.num_blocks).rev() {
            let (blk_start, blk_end) = btf.block_range(k);
            let blk_size = blk_end - blk_start;

            // Subtract contributions from later blocks via off-diagonal entries
            for off_blk in &self.off_diag {
                if off_blk.block_row == k {
                    for local_col in 0..off_blk.ncols {
                        let global_col = off_blk.col_start + local_col;
                        let x_j = self.work[global_col];
                        let cs = off_blk.col_ptr[local_col];
                        let ce = off_blk.col_ptr[local_col + 1];
                        for idx in cs..ce {
                            let local_row = off_blk.row_idx[idx];
                            let global_row = off_blk.row_start + local_row;
                            self.work[global_row] -= off_blk.values[idx] * x_j;
                        }
                    }
                }
            }

            // Extract block RHS
            let mut blk_rhs: Vec<f64> = self.work[blk_start..blk_end].to_vec();

            // Solve this block
            let sym = &self.block_symbolic[k];
            let num = &self.block_numeric[k];

            if blk_size == 1 {
                // Singleton
                let diag = num.u_diag[0];
                blk_rhs[0] /= diag;
            } else {
                self.solve_block_internal(sym, num, &mut blk_rhs)?;
            }

            // Store back
            for (i, &val) in blk_rhs.iter().enumerate() {
                self.work[blk_start + i] = val;
            }
        }

        // Apply inverse BTF column permutation
        for old_col in 0..n {
            let new_pos = btf.col_perm_inv[old_col];
            rhs[old_col] = self.work[new_pos];
        }

        Ok(())
    }

    /// Solve a single diagonal block (L\U triangular solves with permutations).
    fn solve_block_internal(
        &self,
        sym: &SymbolicFactor,
        num: &NumericFactor,
        rhs: &mut [f64],
    ) -> Result<(), SolverError> {
        let n = sym.n;
        let mut temp = vec![0.0; n];

        // Apply permutations
        for i in 0..n {
            let col_new = sym.col_perm[i];
            let row_new = num.row_perm_inv[col_new];
            temp[row_new] = rhs[i];
        }

        // Forward solve L
        for k in 0..n {
            let z_k = temp[k];
            let l_start = sym.l_col_ptr[k];
            let l_end = sym.l_col_ptr[k + 1];
            for l_idx in l_start..l_end {
                let row = sym.l_row_idx[l_idx];
                temp[row] -= num.l_values[l_idx] * z_k;
            }
        }

        // Backward solve U
        for k in (0..n).rev() {
            let diag = num.u_diag[k];
            temp[k] /= diag;
            let x_k = temp[k];
            let u_start = sym.u_col_ptr[k];
            let u_end = sym.u_col_ptr[k + 1] - 1;
            for u_idx in u_start..u_end {
                let row = sym.u_row_idx[u_idx];
                temp[row] -= num.u_values[u_idx] * x_k;
            }
        }

        // Inverse column permutation
        for i in 0..n {
            rhs[i] = temp[sym.col_perm[i]];
        }

        Ok(())
    }
}

// ============================================================================
// LinearSolver trait implementation
// ============================================================================

impl LinearSolver for NativeKluSolver {
    fn prepare(&mut self, n: usize) {
        if n != self.n {
            self.reset_pattern();
            self.n = n;
            self.work.resize(n, 0.0);
        }
    }

    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        if self.state != SolverState::Empty && self.pattern_matches(ap, ai) {
            return Ok(());
        }

        if ap.len() != self.n + 1 {
            return Err(SolverError::InvalidMatrix {
                reason: format!(
                    "Column pointer length {} != expected {}",
                    ap.len(),
                    self.n + 1
                ),
            });
        }

        self.do_analyze(ap, ai)
    }

    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        // Ensure analysis done (handles pattern change)
        if self.state == SolverState::Empty || !self.pattern_matches(ap, ai) {
            self.do_analyze(ap, ai)?;
        }

        self.do_factor(ap, ai, ax)
    }

    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        if self.state != SolverState::Factored {
            return Err(SolverError::SolveFailed);
        }

        if rhs.len() != self.n {
            return Err(SolverError::InvalidMatrix {
                reason: format!("RHS length {} != matrix dimension {}", rhs.len(), self.n),
            });
        }

        self.do_solve(rhs)
    }

    fn reset_pattern(&mut self) {
        self.state = SolverState::Empty;
        self.btf = None;
        self.block_symbolic.clear();
        self.block_numeric.clear();
        self.off_diag.clear();
        self.last_ap.clear();
        self.last_ai.clear();
        self.stats = NativeKluStats::default();
    }

    fn name(&self) -> &'static str {
        "NativeKLU"
    }

    fn set_parallel_threads(&mut self, threads: usize) {
        self.parallel_threads = threads;
    }
}
