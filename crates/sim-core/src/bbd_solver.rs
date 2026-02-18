//! BBD (Bordered Block Diagonal) Solver
//!
//! This module implements a linear solver based on BBD decomposition with
//! Schur complement reduction. It is designed for large, sparse circuit
//! matrices where BTF is ineffective (single strongly connected component).
//!
//! # Algorithm
//!
//! Given a matrix in BBD form:
//!
//! ```text
//! ┌──────┬──────┬────────┐
//! │ B₁₁  │  0   │  C₁   │   ┐
//! ├──────┼──────┼────────┤   ├─ sub-blocks
//! │  0   │ B₂₂  │  C₂   │   ┘
//! ├──────┼──────┼────────┤
//! │  R₁  │  R₂  │  B_T  │   ← border (top block)
//! └──────┴──────┴────────┘
//! ```
//!
//! The Schur complement method:
//!
//! 1. **Factor** each diagonal block B_kk independently
//! 2. **Compute** Schur complement: S = B_T - Σ R_k · B_kk⁻¹ · C_k
//! 3. **Solve** border system: S · x_T = b_T - Σ R_k · B_kk⁻¹ · b_k
//! 4. **Back-substitute** for each block: x_k = B_kk⁻¹ · (b_k - C_k · x_T)
//!
//! # Usage
//!
//! ```ignore
//! use sim_core::bbd_solver::BbdSolver;
//! use sim_core::bbd::GreedyBisectionPartitioner;
//! use sim_core::solver::LinearSolver;
//!
//! let partitioner = Box::new(GreedyBisectionPartitioner::new());
//! let mut solver = BbdSolver::new(n, partitioner, 4);
//! solver.prepare(n);
//! solver.analyze(&ap, &ai)?;
//! solver.factor(&ap, &ai, &ax)?;
//! solver.solve(&mut rhs)?;
//! ```

use crate::bbd::{bbd_decompose, BbdDecomposition, Partitioner};
use crate::solver::{LinearSolver, SolverError};
use crate::sparse_lu::SparseLuSolver;

/// Sparse block in CSC format (for C_k and R_k coupling blocks)
#[derive(Debug, Clone)]
struct SparseBlock {
    #[allow(dead_code)]
    nrows: usize,
    ncols: usize,
    col_ptr: Vec<usize>,
    row_idx: Vec<usize>,
    values: Vec<f64>,
}

impl SparseBlock {
    fn new(nrows: usize, ncols: usize) -> Self {
        Self {
            nrows,
            ncols,
            col_ptr: vec![0; ncols + 1],
            row_idx: Vec::new(),
            values: Vec::new(),
        }
    }

    /// Multiply: y += A * x (sparse matrix-vector product)
    fn matvec_add(&self, x: &[f64], y: &mut [f64]) {
        for col in 0..self.ncols {
            let xc = x[col];
            if xc == 0.0 {
                continue;
            }
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for idx in start..end {
                y[self.row_idx[idx]] += self.values[idx] * xc;
            }
        }
    }
}

/// Dense Schur complement solver (for the border system S · x = b)
#[derive(Debug)]
struct DenseSchurSolver {
    n: usize,
    lu: Vec<f64>,
    pivots: Vec<usize>,
    factored: bool,
}

impl DenseSchurSolver {
    fn new(n: usize) -> Self {
        Self {
            n,
            lu: vec![0.0; n * n],
            pivots: (0..n).collect(),
            factored: false,
        }
    }

    fn resize(&mut self, n: usize) {
        self.n = n;
        self.lu.resize(n * n, 0.0);
        self.pivots = (0..n).collect();
        self.factored = false;
    }

    /// Set the dense matrix from accumulated values, then factorize.
    fn factor(&mut self, matrix: &[f64]) -> Result<(), SolverError> {
        let n = self.n;
        if n == 0 {
            self.factored = true;
            return Ok(());
        }

        self.lu.copy_from_slice(&matrix[..n * n]);

        // LU factorization with partial pivoting
        for k in 0..n {
            self.pivots[k] = k;
        }
        for k in 0..n {
            // Find pivot
            let mut pivot = k;
            let mut max_val = self.lu[k * n + k].abs();
            for i in (k + 1)..n {
                let val = self.lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    pivot = i;
                }
            }
            if max_val < 1e-30 {
                return Err(SolverError::SingularMatrix { pivot: k });
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

        self.factored = true;
        Ok(())
    }

    /// Solve LU · x = b, result overwrites rhs.
    fn solve(&self, rhs: &mut [f64]) -> Result<(), SolverError> {
        if !self.factored {
            return Err(SolverError::SolveFailed);
        }
        let n = self.n;
        if n == 0 {
            return Ok(());
        }

        // Apply row permutation
        let mut b = vec![0.0; n];
        for i in 0..n {
            b[i] = rhs[self.pivots[i]];
        }

        // Forward substitution (L · y = b)
        for i in 0..n {
            let mut sum = b[i];
            for j in 0..i {
                sum -= self.lu[i * n + j] * b[j];
            }
            b[i] = sum;
        }

        // Backward substitution (U · x = y)
        for i in (0..n).rev() {
            let mut sum = b[i];
            for j in (i + 1)..n {
                sum -= self.lu[i * n + j] * rhs[j];
            }
            let diag = self.lu[i * n + i];
            if diag.abs() < 1e-30 {
                return Err(SolverError::SolveFailed);
            }
            rhs[i] = sum / diag;
        }

        Ok(())
    }
}

/// BBD (Bordered Block Diagonal) linear solver.
///
/// Uses graph partitioning to decompose the matrix into independent sub-blocks
/// and a border, then applies Schur complement reduction.
pub struct BbdSolver {
    n: usize,
    decomp: Option<BbdDecomposition>,
    partitioner: Box<dyn Partitioner>,
    num_target_blocks: usize,

    /// Solvers for each diagonal block B_kk
    block_solvers: Vec<SparseLuSolver>,

    /// C_k blocks: coupling from block k columns to border rows
    /// C_k has dimensions (border_size × block_k_size)
    c_blocks: Vec<SparseBlock>,

    /// R_k blocks: coupling from border columns to block k rows
    /// R_k has dimensions (block_k_size × border_size)
    r_blocks: Vec<SparseBlock>,

    /// Dense Schur complement solver
    schur_solver: DenseSchurSolver,

    /// B_T values (border×border, dense, row-major)
    bt_dense: Vec<f64>,

    /// Schur complement matrix (dense, row-major)
    schur_matrix: Vec<f64>,

    /// Workspace
    work: Vec<f64>,

    /// Pattern cache
    last_ap: Vec<i64>,
    last_ai: Vec<i64>,
    analyzed: bool,
    factored: bool,
    pub factor_count: usize,

    /// Fallback solver when BBD is not beneficial
    fallback_solver: Option<SparseLuSolver>,
}

impl std::fmt::Debug for BbdSolver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BbdSolver")
            .field("n", &self.n)
            .field("num_target_blocks", &self.num_target_blocks)
            .field("analyzed", &self.analyzed)
            .field("factored", &self.factored)
            .field("factor_count", &self.factor_count)
            .field("has_decomp", &self.decomp.is_some())
            .field("has_fallback", &self.fallback_solver.is_some())
            .finish()
    }
}

impl BbdSolver {
    /// Create a new BBD solver.
    ///
    /// # Arguments
    /// * `n` - Expected matrix dimension
    /// * `partitioner` - Graph partitioning algorithm
    /// * `num_blocks` - Target number of diagonal blocks
    pub fn new(n: usize, partitioner: Box<dyn Partitioner>, num_blocks: usize) -> Self {
        Self {
            n,
            decomp: None,
            partitioner,
            num_target_blocks: num_blocks.max(2),
            block_solvers: Vec::new(),
            c_blocks: Vec::new(),
            r_blocks: Vec::new(),
            schur_solver: DenseSchurSolver::new(0),
            bt_dense: Vec::new(),
            schur_matrix: Vec::new(),
            work: vec![0.0; n],
            last_ap: Vec::new(),
            last_ai: Vec::new(),
            analyzed: false,
            factored: false,
            factor_count: 0,
            fallback_solver: None,
        }
    }

    /// Check if pattern matches cached pattern
    fn pattern_matches(&self, ap: &[i64], ai: &[i64]) -> bool {
        self.last_ap == ap && self.last_ai == ai
    }

    /// Analyze: compute BBD decomposition and set up block structures
    fn analyze_bbd(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        let n = self.n;

        // Compute BBD decomposition
        let decomp = bbd_decompose(n, ap, ai, &*self.partitioner, self.num_target_blocks);

        // Check if BBD is beneficial
        if decomp.num_blocks <= 1 || decomp.border_size == 0 {
            // Trivial decomposition — fall back to SparseLU
            return self.setup_fallback(ap, ai);
        }

        // Check border ratio heuristic
        let border_ratio = decomp.border_size as f64 / n as f64;
        if border_ratio > 0.5 {
            // Border too large — Schur complement won't help
            return self.setup_fallback(ap, ai);
        }

        // Set up block solvers
        self.block_solvers.clear();
        self.block_solvers.reserve(decomp.num_blocks);

        for k in 0..decomp.num_blocks {
            let block_size = decomp.block_ptr[k + 1] - decomp.block_ptr[k];
            let mut solver = SparseLuSolver::new(block_size);
            solver.prepare(block_size);
            self.block_solvers.push(solver);
        }

        // Extract block patterns for symbolic analysis
        self.extract_block_patterns(&decomp, ap, ai)?;

        // Set up Schur complement solver
        let bs = decomp.border_size;
        self.schur_solver.resize(bs);
        self.bt_dense.resize(bs * bs, 0.0);
        self.schur_matrix.resize(bs * bs, 0.0);

        self.decomp = Some(decomp);
        self.fallback_solver = None;

        Ok(())
    }

    fn setup_fallback(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        self.decomp = None;
        let mut fallback = SparseLuSolver::new(self.n);
        fallback.prepare(self.n);
        fallback.analyze(ap, ai)?;
        self.fallback_solver = Some(fallback);
        Ok(())
    }

    /// Extract CSC patterns for diagonal blocks, C_k, R_k
    fn extract_block_patterns(
        &mut self,
        decomp: &BbdDecomposition,
        ap: &[i64],
        ai: &[i64],
    ) -> Result<(), SolverError> {
        let n = self.n;
        let bs = decomp.border_size;

        // For each diagonal block, extract CSC pattern and analyze
        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];
            let blk_size = blk_end - blk_start;

            let mut blk_ap = vec![0i64; blk_size + 1];
            let mut blk_ai = Vec::new();

            for local_col in 0..blk_size {
                let new_col = blk_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    // Entry in diagonal block
                    if new_row >= blk_start && new_row < blk_end {
                        blk_ai.push((new_row - blk_start) as i64);
                    }
                }

                blk_ap[local_col + 1] = blk_ai.len() as i64;
            }

            self.block_solvers[k].analyze(&blk_ap, &blk_ai)?;
        }

        // Extract C_k patterns (border_rows × block_k_cols)
        // C_k(i,j) = A_permuted(border_start + i, blk_start + j)
        self.c_blocks.clear();
        self.c_blocks.reserve(decomp.num_blocks);

        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];
            let blk_size = blk_end - blk_start;

            let mut c_block = SparseBlock::new(bs, blk_size);

            for local_col in 0..blk_size {
                let new_col = blk_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    // Entry in border rows, block columns → C_k
                    if new_row >= decomp.border_start {
                        let border_row = new_row - decomp.border_start;
                        c_block.row_idx.push(border_row);
                    }
                }

                c_block.col_ptr[local_col + 1] = c_block.row_idx.len();
            }

            self.c_blocks.push(c_block);
        }

        // Extract R_k patterns (block_k_rows × border_cols)
        // R_k(i,j) = A_permuted(blk_start + i, border_start + j)
        self.r_blocks.clear();
        self.r_blocks.reserve(decomp.num_blocks);

        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];
            let blk_size = blk_end - blk_start;

            let mut r_block = SparseBlock::new(blk_size, bs);

            for local_col in 0..bs {
                let new_col = decomp.border_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    // Entry in block rows, border columns → R_k
                    if new_row >= blk_start && new_row < blk_end {
                        let block_row = new_row - blk_start;
                        r_block.row_idx.push(block_row);
                    }
                }

                r_block.col_ptr[local_col + 1] = r_block.row_idx.len();
            }

            self.r_blocks.push(r_block);
        }

        Ok(())
    }

    /// Factor: compute LU of each block and the Schur complement
    fn factor_bbd(
        &mut self,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
    ) -> Result<(), SolverError> {
        let decomp = self.decomp.as_ref().ok_or(SolverError::FactorFailed)?;
        let n = self.n;
        let bs = decomp.border_size;

        // Factor each diagonal block
        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];
            let blk_size = blk_end - blk_start;

            let mut blk_ap = vec![0i64; blk_size + 1];
            let mut blk_ai = Vec::new();
            let mut blk_ax = Vec::new();

            for local_col in 0..blk_size {
                let new_col = blk_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    if new_row >= blk_start && new_row < blk_end {
                        blk_ai.push((new_row - blk_start) as i64);
                        blk_ax.push(ax[idx]);
                    }
                }

                blk_ap[local_col + 1] = blk_ai.len() as i64;
            }

            self.block_solvers[k].factor(&blk_ap, &blk_ai, &blk_ax)?;
        }

        // Extract C_k values
        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];
            let blk_size = blk_end - blk_start;

            self.c_blocks[k].values.clear();

            for local_col in 0..blk_size {
                let new_col = blk_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    if new_row >= decomp.border_start {
                        self.c_blocks[k].values.push(ax[idx]);
                    }
                }
            }
        }

        // Extract R_k values
        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];

            self.r_blocks[k].values.clear();

            for local_col in 0..bs {
                let new_col = decomp.border_start + local_col;
                let old_col = decomp.perm[new_col];

                let col_start = ap[old_col] as usize;
                let col_end = ap[old_col + 1] as usize;

                for idx in col_start..col_end {
                    let old_row = ai[idx] as usize;
                    if old_row >= n {
                        continue;
                    }
                    let new_row = decomp.inv_perm[old_row];

                    if new_row >= blk_start && new_row < blk_end {
                        self.r_blocks[k].values.push(ax[idx]);
                    }
                }
            }
        }

        // Extract B_T (border×border) as dense matrix
        self.bt_dense.fill(0.0);
        for local_col in 0..bs {
            let new_col = decomp.border_start + local_col;
            let old_col = decomp.perm[new_col];

            let col_start = ap[old_col] as usize;
            let col_end = ap[old_col + 1] as usize;

            for idx in col_start..col_end {
                let old_row = ai[idx] as usize;
                if old_row >= n {
                    continue;
                }
                let new_row = decomp.inv_perm[old_row];

                if new_row >= decomp.border_start {
                    let local_row = new_row - decomp.border_start;
                    self.bt_dense[local_row * bs + local_col] += ax[idx];
                }
            }
        }

        // Compute Schur complement: S = B_T - Σ c_blocks[k] · B_kk⁻¹ · r_blocks[k]
        //
        // In our naming convention:
        //   c_blocks[k]: (border_size × block_size) — entries at (border_row, block_col)
        //   r_blocks[k]: (block_size × border_size) — entries at (block_row, border_col)
        //
        // For each column j of r_blocks[k] (j = 0..border_size-1):
        //   r_j = r_blocks[k][:,j] ∈ R^{block_size}
        //   Solve B_kk · y_j = r_j  → y_j ∈ R^{block_size}
        //   S[:,j] -= c_blocks[k] · y_j  (result ∈ R^{border_size})
        self.schur_matrix.copy_from_slice(&self.bt_dense);

        for k in 0..decomp.num_blocks {
            let blk_size = decomp.block_ptr[k + 1] - decomp.block_ptr[k];

            for j in 0..bs {
                // Extract column j of r_blocks[k]
                let r_col_start = self.r_blocks[k].col_ptr[j];
                let r_col_end = self.r_blocks[k].col_ptr[j + 1];

                if r_col_start == r_col_end {
                    continue; // No entries in this column
                }

                let mut r_j = vec![0.0; blk_size];
                for idx in r_col_start..r_col_end {
                    r_j[self.r_blocks[k].row_idx[idx]] = self.r_blocks[k].values[idx];
                }

                // Solve B_kk · y_j = r_j
                self.block_solvers[k].solve(&mut r_j)?;

                // S[:,j] -= c_blocks[k] · y_j
                // c_blocks[k] is (border_size × block_size) in CSC with ncols=block_size
                for col in 0..blk_size {
                    let y_val = r_j[col];
                    if y_val == 0.0 {
                        continue;
                    }
                    let c_start = self.c_blocks[k].col_ptr[col];
                    let c_end = self.c_blocks[k].col_ptr[col + 1];
                    for idx in c_start..c_end {
                        let border_row = self.c_blocks[k].row_idx[idx];
                        let c_val = self.c_blocks[k].values[idx];
                        self.schur_matrix[border_row * bs + j] -= c_val * y_val;
                    }
                }
            }
        }

        // Factor the Schur complement
        self.schur_solver.factor(&self.schur_matrix)?;

        Ok(())
    }

    /// Solve using BBD + Schur complement
    fn solve_bbd(&mut self, rhs: &mut [f64]) -> Result<(), SolverError> {
        let decomp = self.decomp.as_ref().ok_or(SolverError::SolveFailed)?;
        let n = self.n;
        let bs = decomp.border_size;

        self.work.resize(n, 0.0);

        // Step 1: Apply row permutation to RHS
        // work[new] = rhs[perm[new]]... no.
        // perm[new] = old, so work[new] = rhs[old] = rhs[perm[new]]
        for new in 0..n {
            self.work[new] = rhs[decomp.perm[new]];
        }

        // Step 2: For each block k, solve B_kk · z_k = b_k (temporary)
        let mut z_blocks: Vec<Vec<f64>> = Vec::with_capacity(decomp.num_blocks);

        for k in 0..decomp.num_blocks {
            let blk_start = decomp.block_ptr[k];
            let blk_end = decomp.block_ptr[k + 1];

            let mut z_k: Vec<f64> = self.work[blk_start..blk_end].to_vec();
            self.block_solvers[k].solve(&mut z_k)?;
            z_blocks.push(z_k);
        }

        // Step 3: Compute modified border RHS: b'_T = b_T - Σ c_blocks[k] · z_k
        // c_blocks[k] is (border_size × block_size)
        let mut b_border: Vec<f64> = self.work[decomp.border_start..].to_vec();

        for k in 0..decomp.num_blocks {
            // b_border -= c_blocks[k] · z_k
            self.c_blocks[k].matvec_add_neg(&z_blocks[k], &mut b_border);
        }

        // Step 4: Solve S · x_T = b'_T
        self.schur_solver.solve(&mut b_border)?;

        // Step 5: Back-substitute for each block:
        //   x_k = z_k - B_kk⁻¹ · r_blocks[k] · x_T
        // r_blocks[k] is (block_size × border_size)
        for k in 0..decomp.num_blocks {
            let blk_size = decomp.block_ptr[k + 1] - decomp.block_ptr[k];

            // Compute w = r_blocks[k] · x_T
            let mut w = vec![0.0; blk_size];
            self.r_blocks[k].matvec_add(&b_border, &mut w);

            // Solve B_kk · v = w
            self.block_solvers[k].solve(&mut w)?;

            // x_k = z_k - v
            let blk_start = decomp.block_ptr[k];
            for i in 0..blk_size {
                self.work[blk_start + i] = z_blocks[k][i] - w[i];
            }
        }

        // Copy border solution
        for i in 0..bs {
            self.work[decomp.border_start + i] = b_border[i];
        }

        // Step 6: Apply inverse column permutation
        // rhs[old] = work[inv_perm[old]]
        for old in 0..n {
            rhs[old] = self.work[decomp.inv_perm[old]];
        }

        Ok(())
    }
}

impl SparseBlock {
    /// Multiply: y -= A * x (negate and add)
    fn matvec_add_neg(&self, x: &[f64], y: &mut [f64]) {
        for col in 0..self.ncols {
            let xc = x[col];
            if xc == 0.0 {
                continue;
            }
            let start = self.col_ptr[col];
            let end = self.col_ptr[col + 1];
            for idx in start..end {
                y[self.row_idx[idx]] -= self.values[idx] * xc;
            }
        }
    }
}

impl LinearSolver for BbdSolver {
    fn prepare(&mut self, n: usize) {
        if n != self.n {
            self.reset_pattern();
            self.n = n;
            self.work.resize(n, 0.0);
        }
    }

    fn analyze(&mut self, ap: &[i64], ai: &[i64]) -> Result<(), SolverError> {
        if self.analyzed && self.pattern_matches(ap, ai) {
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

        self.analyzed = false;
        self.factored = false;

        self.analyze_bbd(ap, ai)?;

        self.last_ap = ap.to_vec();
        self.last_ai = ai.to_vec();
        self.analyzed = true;

        Ok(())
    }

    fn factor(&mut self, ap: &[i64], ai: &[i64], ax: &[f64]) -> Result<(), SolverError> {
        if !self.analyzed || !self.pattern_matches(ap, ai) {
            self.analyze(ap, ai)?;
        }

        if self.decomp.is_some() {
            self.factor_bbd(ap, ai, ax)?;
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

        if self.decomp.is_some() {
            self.solve_bbd(rhs)
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
        self.decomp = None;
        self.block_solvers.clear();
        self.c_blocks.clear();
        self.r_blocks.clear();
        self.fallback_solver = None;
    }

    fn name(&self) -> &'static str {
        if self.decomp.is_some() {
            "BBD"
        } else {
            "SparseLU (BBD fallback)"
        }
    }
}

unsafe impl Send for BbdSolver {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbd::GreedyBisectionPartitioner;
    use crate::solver::DenseSolver;

    fn create_bbd_solver(n: usize) -> BbdSolver {
        BbdSolver::new(n, Box::new(GreedyBisectionPartitioner::new()), 2)
    }

    /// Compute ||Ax - b|| / ||b|| for verification
    fn residual_norm(
        n: usize,
        ap: &[i64],
        ai: &[i64],
        ax: &[f64],
        x: &[f64],
        b: &[f64],
    ) -> f64 {
        let mut r = b.to_vec();
        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for idx in start..end {
                let row = ai[idx] as usize;
                if row < n {
                    r[row] -= ax[idx] * x[col];
                }
            }
        }
        let r_norm: f64 = r.iter().map(|&x| x * x).sum::<f64>().sqrt();
        let b_norm: f64 = b.iter().map(|&x| x * x).sum::<f64>().sqrt();
        if b_norm == 0.0 {
            r_norm
        } else {
            r_norm / b_norm
        }
    }

    #[test]
    fn test_bbd_solver_diagonal() {
        // 3×3 diagonal — falls back to SparseLU
        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];
        let ax = vec![2.0, 3.0, 4.0];
        let mut rhs = vec![4.0, 9.0, 8.0];

        let mut solver = create_bbd_solver(3);
        solver.prepare(3);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        assert!((rhs[0] - 2.0).abs() < 1e-10);
        assert!((rhs[1] - 3.0).abs() < 1e-10);
        assert!((rhs[2] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_bbd_solver_dense_2x2() {
        let ap = vec![0i64, 2, 4];
        let ai = vec![0i64, 1, 0, 1];
        let ax = vec![3.0, 1.0, 1.0, 2.0];
        let mut rhs = vec![9.0, 8.0];

        let mut solver = create_bbd_solver(2);
        solver.prepare(2);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        assert!((rhs[0] - 2.0).abs() < 1e-9);
        assert!((rhs[1] - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_bbd_solver_block_diagonal() {
        // Two independent 2×2 blocks
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];
        let ax = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];
        let b = vec![7.0, 11.0, 17.0, 6.0];
        let mut rhs = b.clone();

        let mut solver = create_bbd_solver(4);
        solver.prepare(4);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        let rel_err = residual_norm(4, &ap, &ai, &ax, &rhs, &b);
        assert!(
            rel_err < 1e-10,
            "Residual too large: {:.2e}",
            rel_err
        );
    }

    #[test]
    fn test_bbd_solver_matches_dense() {
        // Compare BBD vs Dense on a general matrix
        let ap = vec![0i64, 2, 5, 7];
        let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
        let ax = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
        let b = vec![5.0, 14.0, 14.0];

        // Dense solve
        let mut rhs_dense = b.clone();
        let mut dense_solver = DenseSolver::new(3);
        dense_solver.prepare(3);
        dense_solver.analyze(&ap, &ai).unwrap();
        dense_solver.factor(&ap, &ai, &ax).unwrap();
        dense_solver.solve(&mut rhs_dense).unwrap();

        // BBD solve
        let mut rhs_bbd = b.clone();
        let mut bbd_solver = create_bbd_solver(3);
        bbd_solver.prepare(3);
        bbd_solver.analyze(&ap, &ai).unwrap();
        bbd_solver.factor(&ap, &ai, &ax).unwrap();
        bbd_solver.solve(&mut rhs_bbd).unwrap();

        for i in 0..3 {
            assert!(
                (rhs_bbd[i] - rhs_dense[i]).abs() < 1e-9,
                "Mismatch at {}: BBD={}, Dense={}",
                i,
                rhs_bbd[i],
                rhs_dense[i]
            );
        }
    }

    #[test]
    fn test_bbd_solver_bordered_block_diagonal() {
        // Construct a proper BBD matrix:
        // Block 1 (nodes 0,1): [4 1; 1 4]
        // Block 2 (nodes 2,3): [4 1; 1 4]
        // Border (node 4): connected to nodes 1 and 3
        //
        // Full 5×5 matrix:
        //   [4 1 0 0 0]   [x0]   [b0]
        //   [1 4 0 0 1] * [x1] = [b1]
        //   [0 0 4 1 0]   [x2]   [b2]
        //   [0 0 1 4 1]   [x3]   [b3]
        //   [0 1 0 1 3]   [x4]   [b4]
        let n = 5;
        let ap = vec![0i64, 2, 5, 7, 10, 13];
        let ai = vec![
            0i64, 1, // col 0: (0,0)=4, (1,0)=1
            0, 1, 4, // col 1: (0,1)=1, (1,1)=4, (4,1)=1
            2, 3, // col 2: (2,2)=4, (3,2)=1
            2, 3, 4, // col 3: (2,3)=1, (3,3)=4, (4,3)=1
            1, 3, 4, // col 4: (1,4)=1, (3,4)=1, (4,4)=3
        ];
        let ax = vec![
            4.0, 1.0, // col 0
            1.0, 4.0, 1.0, // col 1
            4.0, 1.0, // col 2
            1.0, 4.0, 1.0, // col 3
            1.0, 1.0, 3.0, // col 4
        ];

        let b = vec![5.0, 10.0, 5.0, 10.0, 8.0];

        // Dense solve for reference
        let mut rhs_dense = b.clone();
        let mut dense_solver = DenseSolver::new(n);
        dense_solver.prepare(n);
        dense_solver.factor(&ap, &ai, &ax).unwrap();
        dense_solver.solve(&mut rhs_dense).unwrap();

        // BBD solve
        let mut rhs_bbd = b.clone();
        let mut bbd_solver = create_bbd_solver(n);
        bbd_solver.prepare(n);
        bbd_solver.analyze(&ap, &ai).unwrap();
        bbd_solver.factor(&ap, &ai, &ax).unwrap();
        bbd_solver.solve(&mut rhs_bbd).unwrap();

        // Verify solutions match
        for i in 0..n {
            assert!(
                (rhs_bbd[i] - rhs_dense[i]).abs() < 1e-8,
                "Mismatch at {}: BBD={:.10}, Dense={:.10}",
                i,
                rhs_bbd[i],
                rhs_dense[i]
            );
        }

        // Verify residual
        let rel_err = residual_norm(n, &ap, &ai, &ax, &rhs_bbd, &b);
        assert!(rel_err < 1e-10, "Residual too large: {:.2e}", rel_err);
    }

    #[test]
    fn test_bbd_solver_larger_bbd_matrix() {
        // 8×8 matrix: two 3-node blocks + 2 border nodes
        // Block 1: nodes 0,1,2
        // Block 2: nodes 3,4,5
        // Border: nodes 6,7
        //
        // Each block is tridiagonal [2,-1,0; -1,2,-1; 0,-1,2]
        // Border nodes connect to block endpoints
        let n = 8;

        // Build CSC manually
        // Row indices for each column:
        // col 0: rows 0,1           vals: 2,-1
        // col 1: rows 0,1,2         vals: -1,2,-1
        // col 2: rows 1,2,6         vals: -1,2,1  (connects to border node 6)
        // col 3: rows 3,4           vals: 2,-1
        // col 4: rows 3,4,5         vals: -1,2,-1
        // col 5: rows 4,5,7         vals: -1,2,1  (connects to border node 7)
        // col 6: rows 2,6,7         vals: 1,3,0.5 (border, connects to block 1)
        // col 7: rows 5,6,7         vals: 1,0.5,3 (border, connects to block 2)
        let ap = vec![0i64, 2, 5, 8, 10, 13, 16, 19, 22];
        let ai = vec![
            0i64, 1,    // col 0
            0, 1, 2,    // col 1
            1, 2, 6,    // col 2
            3, 4,       // col 3
            3, 4, 5,    // col 4
            4, 5, 7,    // col 5
            2, 6, 7,    // col 6
            5, 6, 7,    // col 7
        ];
        let ax = vec![
            2.0, -1.0,           // col 0
            -1.0, 2.0, -1.0,     // col 1
            -1.0, 2.0, 1.0,      // col 2
            2.0, -1.0,           // col 3
            -1.0, 2.0, -1.0,     // col 4
            -1.0, 2.0, 1.0,      // col 5
            1.0, 3.0, 0.5,       // col 6
            1.0, 0.5, 3.0,       // col 7
        ];

        let b = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 4.0];

        // Dense reference
        let mut rhs_dense = b.clone();
        let mut dense_solver = DenseSolver::new(n);
        dense_solver.prepare(n);
        dense_solver.factor(&ap, &ai, &ax).unwrap();
        dense_solver.solve(&mut rhs_dense).unwrap();

        // BBD solve
        let mut rhs_bbd = b.clone();
        let mut bbd_solver = create_bbd_solver(n);
        bbd_solver.prepare(n);
        bbd_solver.analyze(&ap, &ai).unwrap();
        bbd_solver.factor(&ap, &ai, &ax).unwrap();
        bbd_solver.solve(&mut rhs_bbd).unwrap();

        // Verify
        for i in 0..n {
            assert!(
                (rhs_bbd[i] - rhs_dense[i]).abs() < 1e-8,
                "Mismatch at {}: BBD={:.10}, Dense={:.10}",
                i,
                rhs_bbd[i],
                rhs_dense[i]
            );
        }
    }

    #[test]
    fn test_bbd_solver_repeated_factor_solve() {
        // Verify that factoring with different values (same pattern) works
        let ap = vec![0i64, 2, 4, 6, 8];
        let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

        let mut solver = create_bbd_solver(4);
        solver.prepare(4);
        solver.analyze(&ap, &ai).unwrap();

        // First solve
        let ax1 = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];
        let b1 = vec![7.0, 11.0, 17.0, 6.0];
        let mut rhs1 = b1.clone();
        solver.factor(&ap, &ai, &ax1).unwrap();
        solver.solve(&mut rhs1).unwrap();

        let err1 = residual_norm(4, &ap, &ai, &ax1, &rhs1, &b1);
        assert!(err1 < 1e-10);

        // Second solve with different values
        let ax2 = vec![5.0, 2.0, 2.0, 6.0, 3.0, 1.0, 1.0, 4.0];
        let b2 = vec![1.0, 2.0, 3.0, 4.0];
        let mut rhs2 = b2.clone();
        solver.factor(&ap, &ai, &ax2).unwrap();
        solver.solve(&mut rhs2).unwrap();

        let err2 = residual_norm(4, &ap, &ai, &ax2, &rhs2, &b2);
        assert!(err2 < 1e-10);
    }

    #[test]
    fn test_bbd_solver_name() {
        let mut solver = create_bbd_solver(3);
        // Before analysis, name is implementation-dependent
        solver.prepare(3);

        let ap = vec![0i64, 1, 2, 3];
        let ai = vec![0i64, 1, 2];
        let ax = vec![1.0, 2.0, 3.0];
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        // Small diagonal matrix — should fall back
        let name = solver.name();
        assert!(
            name == "BBD" || name == "SparseLU (BBD fallback)",
            "Unexpected name: {}",
            name
        );
    }
}
