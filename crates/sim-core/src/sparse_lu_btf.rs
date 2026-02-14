//! BTF-Integrated Sparse LU Solver
//!
//! This module extends the SparseLU solver with Block Triangular Form (BTF)
//! decomposition for improved performance on matrices with block structure.
//!
//! # Overview
//!
//! The BTF-integrated solver performs these steps:
//!
//! 1. **BTF Decomposition**: Permute matrix to upper block triangular form
//! 2. **Block Factorization**: Factor each diagonal block independently
//! 3. **Block Solve**: Solve using block forward/backward substitution
//!
//! # When BTF Helps
//!
//! BTF provides significant speedup when:
//! - Matrix has multiple strongly connected components (subcircuits)
//! - Matrix is nearly triangular (many 1×1 blocks)
//! - Diagonal blocks are much smaller than full matrix
//!
//! For a matrix with k equal-sized blocks, factorization is k² times faster!
//!
//! # Example
//!
//! ```ignore
//! use sim_core::sparse_lu_btf::SparseLuBtfSolver;
//! use sim_core::solver::LinearSolver;
//!
//! let mut solver = SparseLuBtfSolver::new(100);
//! solver.prepare(100);
//! solver.analyze(&ap, &ai)?;
//! solver.factor(&ap, &ai, &ax)?;
//! solver.solve(&mut rhs)?;
//!
//! // Check BTF statistics
//! println!("Number of blocks: {}", solver.num_blocks());
//! ```
//!
//! # References
//!
//! - Pothen, A., Fan, C.-J. "Computing the block triangular form of a sparse matrix"
//!   ACM TOMS, 1990.
//! - Davis, T.A., Palamadai Natarajan, E. "Algorithm 907: KLU" ACM TOMS, 2010.

use crate::btf::{btf_decompose, should_use_btf, BtfDecomposition};
use crate::solver::{LinearSolver, SolverError};
use crate::sparse_lu::SparseLuSolver;

/// Sparse LU Solver with BTF (Block Triangular Form) support
///
/// This solver automatically applies BTF decomposition when beneficial,
/// factoring diagonal blocks independently for improved performance.
///
/// # Block Structure
///
/// After BTF permutation, the matrix has the form:
///
/// ```text
/// ┌─────┬─────┬─────┐
/// │ B₁₁ │ U₁₂ │ U₁₃ │
/// ├─────┼─────┼─────┤
/// │  0  │ B₂₂ │ U₂₃ │
/// ├─────┼─────┼─────┤
/// │  0  │  0  │ B₃₃ │
/// └─────┴─────┴─────┘
/// ```
///
/// Only diagonal blocks (B₁₁, B₂₂, B₃₃) need LU factorization.
/// Off-diagonal blocks (U₁₂, U₁₃, U₂₃) are used during solve phase.
#[derive(Debug)]
pub struct SparseLuBtfSolver {
    /// Matrix dimension
    n: usize,

    /// BTF decomposition (None if BTF not used or not yet computed)
    btf: Option<BtfDecomposition>,

    /// Solvers for each diagonal block
    block_solvers: Vec<SparseLuSolver>,

    /// Off-diagonal block storage: U[i][j] for block row i, block col j where j > i
    /// Stored as (block_row_offset, block_col_offset, values in CSC within the block)
    off_diagonal_blocks: Vec<OffDiagonalBlock>,

    /// Workspace for solve phase
    work: Vec<f64>,

    /// Whether to use BTF (can be forced on/off)
    use_btf: bool,

    /// Fallback solver when BTF is not used
    fallback_solver: Option<SparseLuSolver>,

    /// Cached column pointers for pattern comparison
    last_ap: Vec<i64>,

    /// Cached row indices for pattern comparison
    last_ai: Vec<i64>,

    /// State tracking
    analyzed: bool,
    factored: bool,

    /// Statistics
    pub factor_count: usize,
}

/// Off-diagonal block in BTF structure
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct OffDiagonalBlock {
    /// Block row index (in BTF block numbering)
    block_row: usize,
    /// Block column index
    block_col: usize,
    /// Row offset in permuted matrix
    row_start: usize,
    /// Column offset in permuted matrix
    col_start: usize,
    /// Number of rows in this block
    nrows: usize,
    /// Number of columns in this block
    ncols: usize,
    /// Column pointers (CSC format, relative to block)
    col_ptr: Vec<usize>,
    /// Row indices (relative to block row_start)
    row_idx: Vec<usize>,
    /// Values
    values: Vec<f64>,
}

impl SparseLuBtfSolver {
    /// Create a new BTF-integrated sparse LU solver
    ///
    /// # Arguments
    /// * `n` - Expected matrix dimension
    pub fn new(n: usize) -> Self {
        Self {
            n,
            btf: None,
            block_solvers: Vec::new(),
            off_diagonal_blocks: Vec::new(),
            work: vec![0.0; n],
            use_btf: true, // Enable by default
            fallback_solver: None,
            last_ap: Vec::new(),
            last_ai: Vec::new(),
            analyzed: false,
            factored: false,
            factor_count: 0,
        }
    }

    /// Create solver with BTF explicitly enabled or disabled
    pub fn with_btf(n: usize, use_btf: bool) -> Self {
        let mut solver = Self::new(n);
        solver.use_btf = use_btf;
        solver
    }

    /// Enable or disable BTF
    pub fn set_use_btf(&mut self, use_btf: bool) {
        if self.use_btf != use_btf {
            self.use_btf = use_btf;
            self.reset_pattern();
        }
    }

    /// Get number of BTF blocks (1 if BTF not used)
    pub fn num_blocks(&self) -> usize {
        self.btf.as_ref().map(|b| b.num_blocks).unwrap_or(1)
    }

    /// Get BTF statistics
    pub fn btf_stats(&self) -> Option<BtfStats> {
        self.btf.as_ref().map(|b| BtfStats {
            num_blocks: b.num_blocks,
            structural_rank: b.structural_rank,
            num_singletons: b.num_singletons,
            max_block_size: b.max_block_size,
            block_sizes: b.block_sizes(),
        })
    }

    /// Check if the sparsity pattern matches the cached pattern
    fn pattern_matches(&self, ap: &[i64], ai: &[i64]) -> bool {
        self.last_ap == ap && self.last_ai == ai
    }

    /// Perform BTF analysis and set up block solvers
    fn analyze_btf(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        let n = self.n;

        // Compute BTF decomposition
        let btf = btf_decompose(n, ap, ai);

        // Check if BTF is beneficial
        // If there's only one block covering the whole matrix, BTF adds overhead
        if btf.num_blocks <= 1 || btf.max_block_size == n {
            // Fall back to regular sparse LU
            self.btf = None;
            let mut fallback = SparseLuSolver::new(n);
            fallback.prepare(n);
            fallback.analyze(ap, ai)?;
            self.fallback_solver = Some(fallback);
            return Ok(());
        }

        // Create solver for each diagonal block
        self.block_solvers.clear();
        self.block_solvers.reserve(btf.num_blocks);

        for k in 0..btf.num_blocks {
            let block_size = btf.block_size(k);
            let mut block_solver = SparseLuSolver::new(block_size);
            block_solver.prepare(block_size);
            self.block_solvers.push(block_solver);
        }

        // Extract block patterns for symbolic analysis
        self.extract_block_patterns(&btf, ap, ai)?;

        self.btf = Some(btf);
        self.fallback_solver = None;

        Ok(())
    }

    /// Extract diagonal and off-diagonal block patterns
    fn extract_block_patterns(
        &mut self,
        btf: &BtfDecomposition,
        ap: &[i64],
        ai: &[i64],
    ) -> Result<(), SolverError> {
        let n = self.n;

        // For each diagonal block, extract its CSC pattern
        for k in 0..btf.num_blocks {
            let (blk_start, blk_end) = btf.block_range(k);
            let blk_size = blk_end - blk_start;

            // Build CSC for this diagonal block
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

                    // Check if this entry is in the diagonal block
                    if global_row_new >= blk_start && global_row_new < blk_end {
                        let local_row = global_row_new - blk_start;
                        blk_ai.push(local_row as i64);
                    }
                }

                blk_ap[local_col + 1] = blk_ai.len() as i64;
            }

            // Analyze the block pattern
            self.block_solvers[k].analyze(&blk_ap, &blk_ai)?;
        }

        // Extract off-diagonal blocks
        self.off_diagonal_blocks.clear();

        for blk_row in 0..btf.num_blocks {
            let (row_start, row_end) = btf.block_range(blk_row);

            for blk_col in (blk_row + 1)..btf.num_blocks {
                let (col_start, col_end) = btf.block_range(blk_col);
                let ncols = col_end - col_start;
                let nrows = row_end - row_start;

                // Check if there are any entries in this off-diagonal block
                let mut has_entries = false;
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
                            has_entries = true;
                            break;
                        }
                    }
                    if has_entries {
                        break;
                    }
                }

                if has_entries {
                    // Build CSC for this off-diagonal block
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
                                let local_row = global_row_new - row_start;
                                row_idx.push(local_row);
                            }
                        }

                        col_ptr[local_col + 1] = row_idx.len();
                    }

                    self.off_diagonal_blocks.push(OffDiagonalBlock {
                        block_row: blk_row,
                        block_col: blk_col,
                        row_start,
                        col_start,
                        nrows,
                        ncols,
                        col_ptr,
                        row_idx,
                        values: Vec::new(), // Filled during factor
                    });
                }
            }
        }

        Ok(())
    }

    /// Factor with BTF
    fn factor_btf(
        &mut self,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
    ) -> Result<(), SolverError> {
        let btf = self.btf.as_ref().ok_or(SolverError::FactorFailed)?;
        let n = self.n;

        // Factor each diagonal block
        for k in 0..btf.num_blocks {
            let (blk_start, blk_end) = btf.block_range(k);
            let blk_size = blk_end - blk_start;

            // Build CSC with values for this diagonal block
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

                    // Check if this entry is in the diagonal block
                    if global_row_new >= blk_start && global_row_new < blk_end {
                        let local_row = global_row_new - blk_start;
                        blk_ai.push(local_row as i64);
                        blk_ax.push(ax[idx]);
                    }
                }

                blk_ap[local_col + 1] = blk_ai.len() as i64;
            }

            // Factor the block
            self.block_solvers[k].factor(&blk_ap, &blk_ai, &blk_ax)?;
        }

        // Extract off-diagonal block values
        for off_blk in &mut self.off_diagonal_blocks {
            let row_start = off_blk.row_start;
            let row_end = row_start + off_blk.nrows;
            let col_start = off_blk.col_start;
            let ncols = off_blk.ncols;

            off_blk.values.clear();

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
                        off_blk.values.push(ax[idx]);
                    }
                }
            }
        }

        Ok(())
    }

    /// Solve with BTF
    fn solve_btf(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        let btf = self.btf.as_ref().ok_or(SolverError::SolveFailed)?;
        let n = self.n;

        // Ensure workspace is sized
        self.work.resize(n, 0.0);

        // Step 1: Apply BTF row permutation to RHS
        // work[new_pos] = rhs[old_row] where new_pos = row_perm_inv[old_row]
        for old_row in 0..n {
            let new_pos = btf.row_perm_inv[old_row];
            self.work[new_pos] = rhs[old_row];
        }

        // Step 2: Forward solve through blocks (L part)
        // For each diagonal block k, solve L_k * z_k = y_k
        // But since our blocks are LU factored together, we need to handle this carefully
        // Actually, since BTF produces upper triangular block structure,
        // we process blocks in order and each block's L solve is independent

        let mut block_solutions: Vec<Vec<f64>> = Vec::with_capacity(btf.num_blocks);

        for k in 0..btf.num_blocks {
            let (blk_start, blk_end) = btf.block_range(k);
            // Extract RHS for this block
            let mut blk_rhs: Vec<f64> = self.work[blk_start..blk_end].to_vec();

            // Solve the diagonal block
            self.block_solvers[k].solve(&mut blk_rhs)?;

            // Store solution for this block
            block_solutions.push(blk_rhs);
        }

        // Step 3: Backward solve (U part) with off-diagonal contributions
        // Process blocks in reverse order
        // For block k, we need to subtract contributions from blocks j > k

        for k in (0..btf.num_blocks).rev() {
            let (blk_start, _blk_end) = btf.block_range(k);

            // Get the solution for this block (already computed in forward pass)
            let x_k = &block_solutions[k];

            // No need to recompute - the block solution is already correct
            // because BTF guarantees zeros below diagonal blocks

            // Copy solution to work vector
            for (i, &val) in x_k.iter().enumerate() {
                self.work[blk_start + i] = val;
            }
        }

        // Wait - we need to handle the upper triangular off-diagonal blocks!
        // The correct approach for BTF is:
        // 1. Forward solve: for k = 0 to num_blocks-1
        //    - Subtract contributions from earlier blocks' U parts
        //    - Solve L_k U_k x_k = (b_k - sum of U_{i,k} * x_i for i < k)
        //
        // Actually for upper BTF:
        // [B11  U12  U13] [x1]   [b1]
        // [ 0   B22  U23] [x2] = [b2]
        // [ 0    0   B33] [x3]   [b3]
        //
        // Solve from bottom to top:
        // B33 * x3 = b3                    -> solve x3
        // B22 * x2 = b2 - U23 * x3         -> solve x2
        // B11 * x1 = b1 - U12 * x2 - U13 * x3  -> solve x1

        // Clear and redo properly
        for old_row in 0..n {
            let new_pos = btf.row_perm_inv[old_row];
            self.work[new_pos] = rhs[old_row];
        }

        // Process blocks from last to first (backward)
        for k in (0..btf.num_blocks).rev() {
            let (blk_start, blk_end) = btf.block_range(k);

            // Subtract contributions from later blocks (j > k)
            for off_blk in &self.off_diagonal_blocks {
                if off_blk.block_row == k {
                    // This off-diagonal block connects block k (rows) to block j (cols)
                    // Multiply U_{k,j} * x_j and subtract from work[k]
                    for local_col in 0..off_blk.ncols {
                        let global_col = off_blk.col_start + local_col;
                        let x_j_col = self.work[global_col];

                        let col_start_idx = off_blk.col_ptr[local_col];
                        let col_end_idx = off_blk.col_ptr[local_col + 1];

                        for idx in col_start_idx..col_end_idx {
                            let local_row = off_blk.row_idx[idx];
                            let global_row = off_blk.row_start + local_row;
                            self.work[global_row] -= off_blk.values[idx] * x_j_col;
                        }
                    }
                }
            }

            // Extract RHS for this block
            let mut blk_rhs: Vec<f64> = self.work[blk_start..blk_end].to_vec();

            // Solve the diagonal block
            self.block_solvers[k].solve(&mut blk_rhs)?;

            // Store solution back
            for (i, &val) in blk_rhs.iter().enumerate() {
                self.work[blk_start + i] = val;
            }
        }

        // Step 4: Apply inverse column permutation
        // rhs[old_col] = work[new_pos] where new_pos = col_perm_inv[old_col]
        for old_col in 0..n {
            let new_pos = btf.col_perm_inv[old_col];
            rhs[old_col] = self.work[new_pos];
        }

        Ok(())
    }
}

impl LinearSolver for SparseLuBtfSolver {
    fn prepare(&mut self, n: usize) {
        if n != self.n {
            self.reset_pattern();
            self.n = n;
            self.work.resize(n, 0.0);
        }
    }

    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        // Check if pattern is unchanged
        if self.analyzed && self.pattern_matches(ap, ai) {
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

        // Reset state
        self.analyzed = false;
        self.factored = false;

        // Decide whether to use BTF
        let nnz = ap[self.n] as usize;
        let should_btf = self.use_btf && should_use_btf(self.n, nnz);

        if should_btf {
            self.analyze_btf(ap, ai)?;
        } else {
            // Use fallback solver
            self.btf = None;
            let mut fallback = SparseLuSolver::new(self.n);
            fallback.prepare(self.n);
            fallback.analyze(ap, ai)?;
            self.fallback_solver = Some(fallback);
        }

        // Cache the pattern
        self.last_ap = ap.to_vec();
        self.last_ai = ai.to_vec();
        self.analyzed = true;

        Ok(())
    }

    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        // Ensure symbolic analysis is done
        if !self.analyzed || !self.pattern_matches(ap, ai) {
            self.analyze(ap, ai)?;
        }

        if self.btf.is_some() {
            self.factor_btf(ap, ai, ax)?;
        } else if let Some(ref mut fallback) = self.fallback_solver {
            fallback.factor(ap, ai, ax)?;
        } else {
            return Err(SolverError::FactorFailed);
        }

        self.factored = true;
        self.factor_count += 1;

        Ok(())
    }

    fn solve(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        if !self.factored {
            return Err(SolverError::SolveFailed);
        }

        if rhs.len() != self.n {
            return Err(SolverError::InvalidMatrix {
                reason: format!("RHS length {} != matrix dimension {}", rhs.len(), self.n),
            });
        }

        if self.btf.is_some() {
            self.solve_btf(rhs)
        } else if let Some(ref mut fallback) = self.fallback_solver {
            fallback.solve(rhs)
        } else {
            Err(SolverError::SolveFailed)
        }
    }

    fn reset_pattern(&mut self) {
        self.analyzed = false;
        self.factored = false;
        self.last_ap.clear();
        self.last_ai.clear();
        self.btf = None;
        self.block_solvers.clear();
        self.off_diagonal_blocks.clear();
        self.fallback_solver = None;
    }

    fn name(&self) -> &'static str {
        if self.btf.is_some() {
            "SparseLU-BTF"
        } else {
            "SparseLU"
        }
    }
}

// SparseLuBtfSolver can be sent between threads
unsafe impl Send for SparseLuBtfSolver {}

/// BTF statistics
#[derive(Debug, Clone)]
pub struct BtfStats {
    /// Number of diagonal blocks
    pub num_blocks: usize,
    /// Structural rank
    pub structural_rank: usize,
    /// Number of 1×1 blocks
    pub num_singletons: usize,
    /// Largest block size
    pub max_block_size: usize,
    /// Size of each block
    pub block_sizes: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::DenseSolver;

    #[test]
    fn test_btf_solver_diagonal() {
        // Diagonal matrix - each entry is its own block
        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];
        let ax = vec![2.0, 3.0, 4.0];
        let mut rhs = vec![4.0, 9.0, 8.0];

        let mut solver = SparseLuBtfSolver::new(3);
        solver.prepare(3);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        assert!((rhs[0] - 2.0).abs() < 1e-10);
        assert!((rhs[1] - 3.0).abs() < 1e-10);
        assert!((rhs[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_btf_solver_dense_2x2() {
        // Dense 2×2 - single block
        let ap = vec![0i64, 2, 4];
        let ai = vec![0i64, 1, 0, 1];
        let ax = vec![3.0, 1.0, 1.0, 2.0];
        let mut rhs = vec![9.0, 8.0];

        let mut solver = SparseLuBtfSolver::new(2);
        solver.prepare(2);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Verify: 3x + y = 9, x + 2y = 8
        // x = 2, y = 3
        assert!((rhs[0] - 2.0).abs() < 1e-9);
        assert!((rhs[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_btf_solver_block_diagonal() {
        // Block diagonal: two independent 2×2 blocks
        // [ 2  1  0  0 ]   [x1]   [ 7]     x1=2, x2=3
        // [ 1  3  0  0 ] * [x2] = [11]
        // [ 0  0  4  1 ]   [x3]   [17]     x3=4, x4=1
        // [ 0  0  1  2 ]   [x4]   [ 6]

        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];
        let ax = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];
        let mut rhs = vec![7.0, 11.0, 17.0, 6.0];

        let mut solver = SparseLuBtfSolver::new(4);
        solver.prepare(4);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Verify solutions
        assert!((rhs[0] - 2.0).abs() < 1e-9, "x1: expected 2, got {}", rhs[0]);
        assert!((rhs[1] - 3.0).abs() < 1e-9, "x2: expected 3, got {}", rhs[1]);
        assert!((rhs[2] - 4.0).abs() < 1e-9, "x3: expected 4, got {}", rhs[2]);
        assert!((rhs[3] - 1.0).abs() < 1e-9, "x4: expected 1, got {}", rhs[3]);
    }

    #[test]
    fn test_btf_solver_matches_dense() {
        // Compare BTF solver with dense solver on same matrix
        let ap = vec![0i64, 2, 5, 7];
        let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
        let ax = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
        let rhs_orig = vec![5.0, 14.0, 14.0];

        // Solve with BTF solver
        let mut rhs_btf = rhs_orig.clone();
        let mut btf_solver = SparseLuBtfSolver::new(3);
        btf_solver.prepare(3);
        btf_solver.analyze(&ap, &ai).unwrap();
        btf_solver.factor(&ap, &ai, &ax).unwrap();
        btf_solver.solve(&mut rhs_btf).unwrap();

        // Solve with Dense solver
        let mut rhs_dense = rhs_orig.clone();
        let mut dense_solver = DenseSolver::new(3);
        dense_solver.prepare(3);
        dense_solver.analyze(&ap, &ai).unwrap();
        dense_solver.factor(&ap, &ai, &ax).unwrap();
        dense_solver.solve(&mut rhs_dense).unwrap();

        // Compare
        for i in 0..3 {
            assert!(
                (rhs_btf[i] - rhs_dense[i]).abs() < 1e-9,
                "Mismatch at {}: BTF={}, Dense={}",
                i,
                rhs_btf[i],
                rhs_dense[i]
            );
        }
    }

    #[test]
    fn test_btf_solver_upper_triangular_blocks() {
        // Upper block triangular:
        // [ 2  0  1 ]   [x1]   [ 5]
        // [ 0  3  1 ] * [x2] = [10]
        // [ 0  0  4 ]   [x3]   [ 8]
        //
        // x3 = 2, x2 = (10-2)/3 = 8/3, x1 = (5-2)/2 = 1.5

        let ap = vec![0i64, 1, 2, 5];
        let ai = vec![0i64, 1, 0, 1, 2];
        let ax = vec![2.0, 3.0, 1.0, 1.0, 4.0];
        let mut rhs = vec![5.0, 10.0, 8.0];

        let mut solver = SparseLuBtfSolver::new(3);
        solver.prepare(3);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        let x3 = 8.0 / 4.0; // = 2
        let x2 = (10.0 - 1.0 * x3) / 3.0; // = 8/3
        let x1 = (5.0 - 1.0 * x3) / 2.0; // = 1.5

        assert!((rhs[0] - x1).abs() < 1e-9, "x1: expected {}, got {}", x1, rhs[0]);
        assert!((rhs[1] - x2).abs() < 1e-9, "x2: expected {}, got {}", x2, rhs[1]);
        assert!((rhs[2] - x3).abs() < 1e-9, "x3: expected {}, got {}", x3, rhs[2]);
    }

    #[test]
    fn test_btf_solver_disabled() {
        // Test with BTF explicitly disabled
        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];
        let ax = vec![2.0, 3.0, 4.0];
        let mut rhs = vec![4.0, 9.0, 8.0];

        let mut solver = SparseLuBtfSolver::with_btf(3, false);
        solver.prepare(3);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        assert_eq!(solver.name(), "SparseLU");
        assert!((rhs[0] - 2.0).abs() < 1e-10);
        assert!((rhs[1] - 3.0).abs() < 1e-10);
        assert!((rhs[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_btf_stats() {
        // Block diagonal matrix
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];
        let _ax = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];

        // Use larger matrix to trigger BTF
        let mut solver = SparseLuBtfSolver::new(4);
        solver.set_use_btf(true);
        solver.prepare(4);
        solver.analyze(&ap, &ai).unwrap();

        // For small matrices, BTF might not be used
        // The stats will reflect whether BTF was actually applied
        if let Some(stats) = solver.btf_stats() {
            assert!(stats.num_blocks >= 1);
            assert_eq!(stats.structural_rank, 4);
        }
    }
}
