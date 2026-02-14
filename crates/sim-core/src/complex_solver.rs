//! Complex linear solver for AC analysis.
//!
//! This module provides solvers for complex linear systems Y·V = I
//! where Y is the complex admittance matrix.

use num_complex::Complex64;

/// Trait for complex linear solvers.
pub trait ComplexLinearSolver {
    /// Prepare the solver for a given matrix size.
    fn prepare(&mut self, size: usize);

    /// Factor and solve the system Ax = b.
    /// The matrix A is in CSC format (column pointers, row indices, values).
    /// Returns true if solve succeeded.
    fn solve(
        &mut self,
        ap: &[i64],
        ai: &[i64],
        ax: &[Complex64],
        b: &[Complex64],
        x: &mut [Complex64],
    ) -> bool;
}

/// Dense LU solver for complex matrices.
/// Suitable for small to medium-sized circuits.
pub struct ComplexDenseSolver {
    size: usize,
    a: Vec<Complex64>,
    pivot: Vec<usize>,
}

impl ComplexDenseSolver {
    pub fn new() -> Self {
        Self {
            size: 0,
            a: Vec::new(),
            pivot: Vec::new(),
        }
    }
}

impl Default for ComplexDenseSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl ComplexLinearSolver for ComplexDenseSolver {
    fn prepare(&mut self, size: usize) {
        self.size = size;
        self.a = vec![Complex64::new(0.0, 0.0); size * size];
        self.pivot = vec![0; size];
    }

    fn solve(
        &mut self,
        ap: &[i64],
        ai: &[i64],
        ax: &[Complex64],
        b: &[Complex64],
        x: &mut [Complex64],
    ) -> bool {
        let n = self.size;
        if n == 0 {
            return true;
        }

        // Clear and fill dense matrix from CSC format
        // Note: CSC format may have duplicate entries that need to be summed
        self.a.fill(Complex64::new(0.0, 0.0));
        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for k in start..end {
                let row = ai[k] as usize;
                self.a[row * n + col] += ax[k];  // Add instead of overwrite
            }
        }

        // Copy b to x
        x.copy_from_slice(b);

        // LU factorization with partial pivoting
        for k in 0..n {
            // Find pivot
            let mut max_val = self.a[k * n + k].norm();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = self.a[i * n + k].norm();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }
            self.pivot[k] = max_row;

            // Swap rows if needed
            if max_row != k {
                for j in 0..n {
                    self.a.swap(k * n + j, max_row * n + j);
                }
                x.swap(k, max_row);
            }

            // Check for singular matrix
            let pivot = self.a[k * n + k];
            if pivot.norm() < 1e-30 {
                return false;
            }

            // Elimination
            for i in (k + 1)..n {
                let factor = self.a[i * n + k] / pivot;
                self.a[i * n + k] = factor;
                for j in (k + 1)..n {
                    let a_kj = self.a[k * n + j];
                    self.a[i * n + j] -= factor * a_kj;
                }
                x[i] -= factor * x[k];
            }
        }

        // Back substitution
        for i in (0..n).rev() {
            for j in (i + 1)..n {
                x[i] -= self.a[i * n + j] * x[j];
            }
            let pivot = self.a[i * n + i];
            if pivot.norm() < 1e-30 {
                return false;
            }
            x[i] /= pivot;
        }

        true
    }
}

/// Create a complex linear solver.
pub fn create_complex_solver() -> Box<dyn ComplexLinearSolver> {
    Box::new(ComplexDenseSolver::new())
}
