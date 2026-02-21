//! Tests for the Native KLU Solver

use sim_core::native_klu::{KluConfig, NativeKluSolver};
use sim_core::solver::{DenseSolver, LinearSolver, SolverType, create_solver};

/// Helper: verify Ax = rhs_orig by computing residual
fn check_residual(ap: &[i64], ai: &[i64], ax: &[f64], x: &[f64], b: &[f64], tol: f64) {
    let n = b.len();
    for row in 0..n {
        let mut ax_row = 0.0;
        for col in 0..n {
            let start = ap[col] as usize;
            let end = ap[col + 1] as usize;
            for idx in start..end {
                if ai[idx] as usize == row {
                    ax_row += ax[idx] * x[col];
                }
            }
        }
        let residual = (ax_row - b[row]).abs();
        assert!(
            residual < tol,
            "Residual too large at row {}: |Ax-b| = {:.2e} (Ax={}, b={})",
            row, residual, ax_row, b[row]
        );
    }
}

// ============================================================================
// Basic correctness tests
// ============================================================================

#[test]
fn test_native_klu_2x2() {
    // [ 2  0 ]   [ 4 ]
    // [ 1  3 ] * [ 7 ] => x=2, y=5/3
    let ap = vec![0i64, 2, 3];
    let ai = vec![0i64, 1, 1];
    let ax = vec![2.0, 1.0, 3.0];
    let b = vec![4.0, 7.0];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-10);
}

#[test]
fn test_native_klu_3x3_dense() {
    // [ 4  1  0 ]
    // [ 1  5  2 ]
    // [ 0  2  6 ]
    let ap = vec![0i64, 2, 5, 7];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
    let b = vec![5.0, 14.0, 14.0];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-10);
}

#[test]
fn test_native_klu_diagonal() {
    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];
    let ax = vec![2.0, 3.0, 4.0];
    let b = vec![4.0, 9.0, 8.0];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(3);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
    assert!((rhs[2] - 2.0).abs() < 1e-10);
}

#[test]
fn test_native_klu_tridiagonal() {
    // Tridiagonal 5x5
    // [ 4 -1  0  0  0]
    // [-1  4 -1  0  0]
    // [ 0 -1  4 -1  0]
    // [ 0  0 -1  4 -1]
    // [ 0  0  0 -1  4]
    let n = 5;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    for col in 0..n {
        if col > 0 {
            ai.push((col - 1) as i64);
            ax.push(-1.0);
        }
        ai.push(col as i64);
        ax.push(4.0);
        if col < n - 1 {
            ai.push((col + 1) as i64);
            ax.push(-1.0);
        }
        ap.push(ai.len() as i64);
    }

    let b = vec![1.0; n];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(n);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-10);
}

// ============================================================================
// Comparison with DenseSolver
// ============================================================================

#[test]
fn test_native_klu_matches_dense() {
    let ap = vec![0i64, 2, 5, 7];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
    let b = vec![5.0, 14.0, 14.0];

    // Solve with NativeKlu
    let mut rhs_klu = b.clone();
    let mut klu_solver = NativeKluSolver::new(3);
    klu_solver.factor(&ap, &ai, &ax).unwrap();
    klu_solver.solve(&mut rhs_klu).unwrap();

    // Solve with Dense
    let mut rhs_dense = b.clone();
    let mut dense_solver = DenseSolver::new(3);
    dense_solver.factor(&ap, &ai, &ax).unwrap();
    dense_solver.solve(&mut rhs_dense).unwrap();

    for i in 0..3 {
        assert!(
            (rhs_klu[i] - rhs_dense[i]).abs() < 1e-9,
            "Mismatch at {}: NativeKlu={}, Dense={}",
            i, rhs_klu[i], rhs_dense[i]
        );
    }
}

// ============================================================================
// BTF block tests
// ============================================================================

#[test]
fn test_native_klu_block_diagonal() {
    // Two independent 2x2 blocks:
    // [ 2  1  0  0 ]   [ 7]     x1=2, x2=3
    // [ 1  3  0  0 ] * [11]
    // [ 0  0  4  1 ]   [17]     x3=4, x4=1
    // [ 0  0  1  2 ]   [ 6]
    let ap = vec![0i64, 2, 4, 6, 8];
    let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];
    let ax = vec![2.0, 1.0, 1.0, 3.0, 4.0, 1.0, 1.0, 2.0];
    let b = vec![7.0, 11.0, 17.0, 6.0];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(4);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-9);
}

#[test]
fn test_native_klu_upper_block_triangular() {
    // [ 2  0  1 ]
    // [ 0  3  1 ]
    // [ 0  0  4 ]
    let ap = vec![0i64, 1, 2, 5];
    let ai = vec![0i64, 1, 0, 1, 2];
    let ax = vec![2.0, 3.0, 1.0, 1.0, 4.0];
    let b = vec![5.0, 10.0, 8.0];
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(3);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-9);
}

// ============================================================================
// Refactorization tests
// ============================================================================

#[test]
fn test_native_klu_refactorization() {
    let ap = vec![0i64, 2, 5, 7];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax1 = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
    let ax2 = vec![8.0, 2.0, 2.0, 10.0, 4.0, 4.0, 12.0]; // doubled values

    let mut solver = NativeKluSolver::new(3);

    // First factor
    solver.factor(&ap, &ai, &ax1).unwrap();
    assert_eq!(solver.stats().factor_count, 1);
    assert_eq!(solver.stats().refactor_count, 0);

    let b1 = vec![5.0, 14.0, 14.0];
    let mut rhs1 = b1.clone();
    solver.solve(&mut rhs1).unwrap();
    check_residual(&ap, &ai, &ax1, &rhs1, &b1, 1e-10);

    // Refactor with same pattern
    solver.factor(&ap, &ai, &ax2).unwrap();
    assert_eq!(solver.stats().factor_count, 1);
    assert_eq!(solver.stats().refactor_count, 1);

    let b2 = vec![10.0, 28.0, 28.0];
    let mut rhs2 = b2.clone();
    solver.solve(&mut rhs2).unwrap();
    check_residual(&ap, &ai, &ax2, &rhs2, &b2, 1e-10);

    // Results should be similar (same solution since values are proportional)
    for i in 0..3 {
        assert!(
            (rhs1[i] - rhs2[i]).abs() < 1e-9,
            "Refactor solution mismatch at {}: {} vs {}",
            i, rhs1[i], rhs2[i]
        );
    }
}

// ============================================================================
// Condition number tests
// ============================================================================

#[test]
fn test_native_klu_rcond() {
    // Well-conditioned diagonal
    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];
    let ax = vec![1.0, 1.0, 1.0];

    let mut solver = NativeKluSolver::new(3);
    solver.factor(&ap, &ai, &ax).unwrap();
    assert!((solver.rcond() - 1.0).abs() < 1e-10, "Identity should have rcond=1.0");

    // Ill-conditioned: one very small diagonal
    let ax_ill = vec![1.0, 1e-15, 1.0];
    solver.factor(&ap, &ai, &ax_ill).unwrap();
    assert!(solver.rcond() < 1e-10, "Ill-conditioned matrix should have small rcond");
}

// ============================================================================
// Configuration tests
// ============================================================================

#[test]
fn test_native_klu_btf_disabled() {
    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];
    let ax = vec![2.0, 3.0, 4.0];
    let b = vec![4.0, 9.0, 8.0];
    let mut rhs = b.clone();

    let config = KluConfig {
        use_btf: false,
        ..Default::default()
    };
    let mut solver = NativeKluSolver::with_config(3, config);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
    assert!((rhs[2] - 2.0).abs() < 1e-10);
    assert_eq!(solver.stats().nblocks, 1);
}

#[test]
fn test_native_klu_natural_ordering() {
    let ap = vec![0i64, 2, 5, 7];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
    let b = vec![5.0, 14.0, 14.0];
    let mut rhs = b.clone();

    let config = KluConfig {
        ordering: 3, // natural
        use_btf: false,
        ..Default::default()
    };
    let mut solver = NativeKluSolver::with_config(3, config);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-10);
}

// ============================================================================
// Solver factory test
// ============================================================================

#[test]
fn test_native_klu_via_factory() {
    let mut solver = create_solver(SolverType::NativeKlu, 3);
    assert_eq!(solver.name(), "NativeKLU");

    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];
    let ax = vec![2.0, 3.0, 4.0];
    let b = vec![4.0, 9.0, 8.0];
    let mut rhs = b.clone();

    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
    assert!((rhs[2] - 2.0).abs() < 1e-10);
}

// ============================================================================
// Larger matrix test
// ============================================================================

#[test]
fn test_native_klu_rc_ladder_20() {
    // Build a 20-node RC ladder circuit matrix
    // This produces a tridiagonal-like structure
    let n = 20;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    for col in 0..n {
        if col > 0 {
            ai.push((col - 1) as i64);
            ax.push(-1.0);
        }
        ai.push(col as i64);
        // Diagonal: 2 for interior, 1 for endpoints
        ax.push(if col == 0 || col == n - 1 { 2.0 } else { 3.0 });
        if col < n - 1 {
            ai.push((col + 1) as i64);
            ax.push(-1.0);
        }
        ap.push(ai.len() as i64);
    }

    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(n);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-9);
}

#[test]
fn test_native_klu_rc_ladder_100() {
    let n = 100;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    for col in 0..n {
        if col > 0 {
            ai.push((col - 1) as i64);
            ax.push(-1.0);
        }
        ai.push(col as i64);
        ax.push(if col == 0 || col == n - 1 { 2.0 } else { 3.0 });
        if col < n - 1 {
            ai.push((col + 1) as i64);
            ax.push(-1.0);
        }
        ap.push(ai.len() as i64);
    }

    let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
    let mut rhs = b.clone();

    let mut solver = NativeKluSolver::new(n);
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    check_residual(&ap, &ai, &ax, &rhs, &b, 1e-8);
}

// ============================================================================
// Pattern change test
// ============================================================================

#[test]
fn test_native_klu_pattern_change() {
    // First solve with one pattern
    let ap1 = vec![0i64, 1, 2, 3];
    let ai1 = vec![0i64, 1, 2];
    let ax1 = vec![2.0, 3.0, 4.0];
    let mut rhs1 = vec![4.0, 9.0, 8.0];

    let mut solver = NativeKluSolver::new(3);
    solver.factor(&ap1, &ai1, &ax1).unwrap();
    solver.solve(&mut rhs1).unwrap();

    // Now solve with a different pattern
    let ap2 = vec![0i64, 2, 5, 7];
    let ai2 = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax2 = vec![4.0, 1.0, 1.0, 5.0, 2.0, 2.0, 6.0];
    let b2 = vec![5.0, 14.0, 14.0];
    let mut rhs2 = b2.clone();

    solver.factor(&ap2, &ai2, &ax2).unwrap();
    solver.solve(&mut rhs2).unwrap();

    check_residual(&ap2, &ai2, &ax2, &rhs2, &b2, 1e-10);
    // factor_count should be 2 (both were full factors, pattern changed)
    assert_eq!(solver.stats().factor_count, 2);
}
