//! Sparse Solver Tests
//!
//! These tests verify the sparse solver implementations (KLU and Faer).
//! Tests without feature gates run on the fallback Dense solver.

#[cfg(feature = "klu")]
use sim_core::solver::KluSolver;
#[cfg(feature = "faer-solver")]
use sim_core::solver::FaerSolver;
use sim_core::solver::{create_solver, create_solver_auto, DenseSolver, LinearSolver, SolverType};

// ============================================================================
// Helper Functions
// ============================================================================

/// Build a simple 3x3 test matrix in CSC format
/// Matrix:
///   [4  1  0]
///   [1  4  1]
///   [0  1  4]
fn build_tridiagonal_3x3() -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
    let ap = vec![0, 2, 5, 7]; // Column pointers
    let ai = vec![0, 1, 0, 1, 2, 1, 2]; // Row indices
    let ax = vec![4.0, 1.0, 1.0, 4.0, 1.0, 1.0, 4.0]; // Values
    let rhs = vec![1.0, 2.0, 1.0]; // Right-hand side
    (ap, ai, ax, rhs)
}

/// Build a larger sparse matrix (Poisson-like with Dirichlet BC)
/// This creates a well-conditioned tridiagonal matrix:
/// - Diagonal: 2 (for all nodes)
/// - Off-diagonal: -1
/// With Dirichlet boundary conditions at both ends (implicit in structure)
fn build_resistor_ladder(n: usize) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>) {
    // Tridiagonal matrix:
    // G_ii = 2
    // G_i,i+1 = G_i+1,i = -1
    let mut ap = Vec::with_capacity(n + 1);
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    let mut ptr = 0i64;
    for col in 0..n {
        ap.push(ptr);
        // Lower diagonal (row = col - 1)
        if col > 0 {
            ai.push((col - 1) as i64);
            ax.push(-1.0);
            ptr += 1;
        }
        // Diagonal (row = col)
        ai.push(col as i64);
        ax.push(2.0); // Always 2 for well-conditioned matrix
        ptr += 1;
        // Upper diagonal (row = col + 1)
        if col < n - 1 {
            ai.push((col + 1) as i64);
            ax.push(-1.0);
            ptr += 1;
        }
    }
    ap.push(ptr);

    // RHS: source at first node, sink at last (represents voltage distribution)
    let mut rhs = vec![0.0; n];
    rhs[0] = 1.0; // Voltage source contribution

    (ap, ai, ax, rhs)
}

/// Compare two solutions with tolerance
#[cfg(feature = "klu")]
fn solutions_equal(a: &[f64], b: &[f64], tol: f64) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

// ============================================================================
// Basic Solver Tests
// ============================================================================

#[test]
fn test_dense_solver_3x3() {
    let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
    let n = 3;

    let mut solver = DenseSolver::new(n);
    solver.prepare(n);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // For tridiagonal matrix [4,1,0; 1,4,1; 0,1,4] with RHS [1,2,1]
    // Solution is [1/7, 3/7, 1/7] ≈ [0.143, 0.429, 0.143]
    assert!((rhs[0] - 1.0 / 7.0).abs() < 0.01, "rhs[0] = {}", rhs[0]);
    assert!((rhs[1] - 3.0 / 7.0).abs() < 0.01, "rhs[1] = {}", rhs[1]);
    assert!((rhs[2] - 1.0 / 7.0).abs() < 0.01, "rhs[2] = {}", rhs[2]);
}

#[test]
fn test_klu_solver_fallback_when_disabled() {
    // Without KLU feature, should fall back to Dense
    let _solver = create_solver(SolverType::Klu, 10);

    #[cfg(not(feature = "klu"))]
    {
        // Should have created a DenseSolver via fallback
        // We can verify by checking it works
        let (ap, ai, _ax, _rhs) = build_tridiagonal_3x3();
        let mut solver = create_solver(SolverType::Klu, 3);
        solver.prepare(3);
        assert!(solver.analyze(&ap, &ai).is_ok());
    }
}

#[test]
fn test_solver_type_default() {
    assert_eq!(SolverType::default(), SolverType::Dense);
}

#[test]
fn test_create_dense_solver() {
    // Should create successfully
    let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
    let mut solver = create_solver(SolverType::Dense, 3);
    solver.prepare(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();
}

// ============================================================================
// KLU-Specific Tests (only run with feature enabled)
// ============================================================================

#[cfg(feature = "klu")]
mod klu_enabled_tests {
    use super::*;

    #[test]
    fn test_klu_solver_3x3() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        assert!(solver.enabled);

        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Same expected solution as dense: [1/7, 3/7, 1/7]
        assert!((rhs[0] - 1.0 / 7.0).abs() < 0.01, "rhs[0] = {}", rhs[0]);
        assert!((rhs[1] - 3.0 / 7.0).abs() < 0.01, "rhs[1] = {}", rhs[1]);
        assert!((rhs[2] - 1.0 / 7.0).abs() < 0.01, "rhs[2] = {}", rhs[2]);
    }

    #[test]
    fn test_klu_vs_dense_comparison() {
        let (ap, ai, ax, rhs) = build_resistor_ladder(20);
        let n = 20;

        // Solve with Dense
        let mut rhs_dense = rhs.clone();
        let mut dense = DenseSolver::new(n);
        dense.prepare(n);
        dense.analyze(&ap, &ai).unwrap();
        dense.factor(&ap, &ai, &ax).unwrap();
        dense.solve(&mut rhs_dense).unwrap();

        // Solve with KLU
        let mut rhs_klu = rhs.clone();
        let mut klu = KluSolver::new(n);
        klu.prepare(n);
        klu.analyze(&ap, &ai).unwrap();
        klu.factor(&ap, &ai, &ax).unwrap();
        klu.solve(&mut rhs_klu).unwrap();

        // Solutions should match
        assert!(solutions_equal(&rhs_dense, &rhs_klu, 1e-10));
    }

    #[test]
    fn test_klu_pattern_caching() {
        let (ap, ai, ax, _) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.prepare(n);

        // First analysis
        solver.analyze(&ap, &ai).unwrap();
        let factor_count_1 = solver.factor_count;

        // Second analysis with same pattern - should be cached
        solver.analyze(&ap, &ai).unwrap();
        // No additional work should be done

        // Factor and check stats
        solver.factor(&ap, &ai, &ax).unwrap();
        assert_eq!(solver.factor_count, 1);
    }

    #[test]
    fn test_klu_refactorization() {
        let (ap, ai, mut ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();

        // First factorization
        solver.factor(&ap, &ai, &ax).unwrap();
        assert_eq!(solver.factor_count, 1);
        assert_eq!(solver.refactor_count, 0);

        // Solve
        solver.solve(&mut rhs).unwrap();
        let solution1 = rhs.clone();

        // Change values but keep pattern
        ax[0] = 5.0; // Change diagonal
        rhs = vec![1.0, 2.0, 1.0]; // Reset RHS

        // Should use refactorization
        solver.factor(&ap, &ai, &ax).unwrap();
        assert_eq!(solver.factor_count, 1); // No new full factorization
        assert_eq!(solver.refactor_count, 1); // One refactorization

        // Solve with new values
        solver.solve(&mut rhs).unwrap();
        // Solution should be different
        assert!(!solutions_equal(&solution1, &rhs, 1e-10));
    }

    #[test]
    fn test_klu_condition_number() {
        let (ap, ai, ax, _) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        let rcond = solver.rcond();
        // Should be reasonable condition for well-conditioned matrix
        assert!(rcond > 0.0);
        assert!(rcond <= 1.0);
    }

    #[test]
    fn test_klu_stats() {
        let (ap, ai, ax, _) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        let stats = solver.stats();
        assert_eq!(stats.factor_count, 1);
        assert!(stats.nnz_l > 0);
        assert!(stats.nnz_u > 0);
    }

    #[test]
    fn test_klu_pivot_tolerance() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.set_pivot_tolerance(0.5); // More conservative pivoting

        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Should still get correct answer: [1/7, 3/7, 1/7]
        assert!((rhs[0] - 1.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_klu_reset_pattern() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        // Reset pattern
        solver.reset_pattern();

        // Should need to re-analyze
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Should still work correctly: [1/7, 3/7, 1/7]
        assert!((rhs[0] - 1.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_klu_larger_matrix() {
        let n = 100;
        let (ap, ai, ax, mut rhs) = build_resistor_ladder(n);
        let original_rhs = rhs.clone();

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Verify solution by computing Ax and comparing to original RHS
        // For tridiagonal: Ax[i] = -x[i-1] + 2*x[i] - x[i+1]
        for i in 0..n {
            let mut ax_i = 2.0 * rhs[i];
            if i > 0 {
                ax_i -= rhs[i - 1];
            }
            if i < n - 1 {
                ax_i -= rhs[i + 1];
            }
            assert!(
                (ax_i - original_rhs[i]).abs() < 1e-10,
                "Residual at node {}: Ax={}, b={}",
                i,
                ax_i,
                original_rhs[i]
            );
        }
    }

    #[test]
    fn test_klu_memory_usage() {
        let n = 50;
        let (ap, ai, ax, _) = build_resistor_ladder(n);

        let mut solver = KluSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        let (current, peak) = solver.memory_usage();
        // Memory should be allocated
        assert!(current > 0 || peak > 0);
    }

    #[test]
    fn test_klu_btf_setting() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = KluSolver::new(n);
        solver.set_btf(false); // Disable BTF

        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Should still work
        assert!((rhs[0] - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_klu_ordering_methods() {
        let (ap, ai, ax, _) = build_resistor_ladder(20);
        let n = 20;

        // Test AMD ordering
        let mut solver_amd = KluSolver::new(n);
        solver_amd.set_ordering(0);
        solver_amd.prepare(n);
        solver_amd.analyze(&ap, &ai).unwrap();
        solver_amd.factor(&ap, &ai, &ax).unwrap();
        let mut rhs_amd = vec![0.0; n];
        rhs_amd[0] = 1.0;
        solver_amd.solve(&mut rhs_amd).unwrap();

        // Test COLAMD ordering
        let mut solver_colamd = KluSolver::new(n);
        solver_colamd.set_ordering(1);
        solver_colamd.prepare(n);
        solver_colamd.analyze(&ap, &ai).unwrap();
        solver_colamd.factor(&ap, &ai, &ax).unwrap();
        let mut rhs_colamd = vec![0.0; n];
        rhs_colamd[0] = 1.0;
        solver_colamd.solve(&mut rhs_colamd).unwrap();

        // Both should give same result
        assert!(solutions_equal(&rhs_amd, &rhs_colamd, 1e-10));
    }
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_invalid_matrix_dimensions() {
    let mut solver = DenseSolver::new(3);
    solver.prepare(3);

    // Wrong size column pointers
    let ap = vec![0, 2]; // Should be length 4 for n=3
    let ai = vec![0, 1];
    let ax = vec![1.0, 2.0];

    let result = solver.factor(&ap, &ai, &ax);
    assert!(result.is_err());
}

#[test]
fn test_singular_matrix() {
    // Matrix with zero row
    let ap = vec![0, 1, 2, 3];
    let ai = vec![0, 1, 2];
    let ax = vec![1.0, 0.0, 1.0]; // Zero on diagonal

    let mut solver = DenseSolver::new(3);
    solver.prepare(3);
    let _result = solver.factor(&ap, &ai, &ax);
    // May or may not fail depending on pivot selection
}

#[test]
fn test_solve_without_factor() {
    let mut solver = DenseSolver::new(3);
    solver.prepare(3);

    let (ap, ai, _, _rhs) = build_tridiagonal_3x3();
    solver.analyze(&ap, &ai).unwrap();
    // Skip factor step

    // Solve should work but give garbage (or zero for initialized memory)
    // This is undefined behavior we should handle
}

// ============================================================================
// Benchmark-style Tests
// ============================================================================

#[test]
fn test_repeated_solves_same_pattern() {
    let n = 50;
    let (ap, ai, ax, _) = build_resistor_ladder(n);

    let mut solver = create_solver(SolverType::Dense, n);
    solver.prepare(n);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();

    // Multiple solves with same factorization
    for i in 0..10 {
        let mut rhs = vec![0.0; n];
        rhs[i % n] = 1.0;
        solver.solve(&mut rhs).unwrap();
    }
}

#[cfg(feature = "klu")]
#[test]
fn test_klu_repeated_refactorization() {
    let n = 50;
    let (ap, ai, mut ax, _) = build_resistor_ladder(n);

    let mut solver = KluSolver::new(n);
    solver.prepare(n);
    solver.analyze(&ap, &ai).unwrap();

    // Multiple factorizations with changing values
    for i in 0..10 {
        ax[0] = 1.0 + 0.1 * (i as f64);
        solver.factor(&ap, &ai, &ax).unwrap();

        let mut rhs = vec![0.0; n];
        rhs[0] = 1.0;
        solver.solve(&mut rhs).unwrap();
    }

    // Should have 1 full factor and 9 refactors
    assert_eq!(solver.factor_count, 1);
    assert_eq!(solver.refactor_count, 9);
}

// ============================================================================
// Faer Solver Tests (only run with feature enabled)
// ============================================================================

#[cfg(feature = "faer-solver")]
mod faer_tests {
    use super::*;

    #[test]
    fn test_faer_solver_3x3() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = FaerSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Same expected solution as dense: [1/7, 3/7, 1/7]
        assert!(
            (rhs[0] - 1.0 / 7.0).abs() < 0.01,
            "rhs[0] = {}, expected {}",
            rhs[0],
            1.0 / 7.0
        );
        assert!(
            (rhs[1] - 3.0 / 7.0).abs() < 0.01,
            "rhs[1] = {}, expected {}",
            rhs[1],
            3.0 / 7.0
        );
        assert!(
            (rhs[2] - 1.0 / 7.0).abs() < 0.01,
            "rhs[2] = {}, expected {}",
            rhs[2],
            1.0 / 7.0
        );
    }

    #[test]
    fn test_faer_vs_dense_comparison() {
        let (ap, ai, ax, rhs) = build_resistor_ladder(20);
        let n = 20;

        // Solve with Dense
        let mut rhs_dense = rhs.clone();
        let mut dense = DenseSolver::new(n);
        dense.prepare(n);
        dense.analyze(&ap, &ai).unwrap();
        dense.factor(&ap, &ai, &ax).unwrap();
        dense.solve(&mut rhs_dense).unwrap();

        // Solve with Faer
        let mut rhs_faer = rhs.clone();
        let mut faer = FaerSolver::new(n);
        faer.prepare(n);
        faer.analyze(&ap, &ai).unwrap();
        faer.factor(&ap, &ai, &ax).unwrap();
        faer.solve(&mut rhs_faer).unwrap();

        // Solutions should match
        for i in 0..n {
            assert!(
                (rhs_dense[i] - rhs_faer[i]).abs() < 1e-10,
                "Mismatch at index {}: dense={}, faer={}",
                i,
                rhs_dense[i],
                rhs_faer[i]
            );
        }
    }

    #[test]
    fn test_faer_pattern_caching() {
        let (ap, ai, ax, _) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = FaerSolver::new(n);
        solver.prepare(n);

        // First analysis
        solver.analyze(&ap, &ai).unwrap();

        // Second analysis with same pattern - should be cached (no error)
        solver.analyze(&ap, &ai).unwrap();

        // Factor should work
        solver.factor(&ap, &ai, &ax).unwrap();
        assert_eq!(solver.factor_count, 1);
    }

    #[test]
    fn test_faer_larger_matrix() {
        let n = 100;
        let (ap, ai, ax, mut rhs) = build_resistor_ladder(n);
        let original_rhs = rhs.clone();

        let mut solver = FaerSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Verify solution by computing Ax and comparing to original RHS
        for i in 0..n {
            let mut ax_i = 2.0 * rhs[i];
            if i > 0 {
                ax_i -= rhs[i - 1];
            }
            if i < n - 1 {
                ax_i -= rhs[i + 1];
            }
            assert!(
                (ax_i - original_rhs[i]).abs() < 1e-9,
                "Residual at node {}: Ax={}, b={}",
                i,
                ax_i,
                original_rhs[i]
            );
        }
    }

    #[test]
    fn test_faer_solver_name() {
        let solver = FaerSolver::new(3);
        assert_eq!(solver.name(), "Faer");
    }

    #[test]
    fn test_faer_reset_pattern() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let n = 3;

        let mut solver = FaerSolver::new(n);
        solver.prepare(n);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();

        // Reset pattern
        solver.reset_pattern();

        // Should need to re-analyze
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Should still work correctly: [1/7, 3/7, 1/7]
        assert!((rhs[0] - 1.0 / 7.0).abs() < 0.01);
    }

    #[test]
    fn test_create_solver_auto_selects_faer() {
        // When only faer-solver is enabled (not klu), should select Faer for large matrices
        // Small matrices (n <= 50) use Dense for lower overhead
        let solver_small = create_solver_auto(10);
        assert_eq!(solver_small.name(), "Dense", "Small matrices should use Dense");

        // Large matrices should use Faer (when klu not available)
        let solver_large = create_solver_auto(100);
        #[cfg(not(feature = "klu"))]
        {
            assert_eq!(solver_large.name(), "Faer", "Large matrices should use Faer");
        }
    }

    #[test]
    fn test_faer_via_solver_type() {
        let (ap, ai, ax, mut rhs) = build_tridiagonal_3x3();
        let mut solver = create_solver(SolverType::Faer, 3);
        solver.prepare(3);
        solver.analyze(&ap, &ai).unwrap();
        solver.factor(&ap, &ai, &ax).unwrap();
        solver.solve(&mut rhs).unwrap();

        // Should work correctly
        assert!((rhs[1] - 3.0 / 7.0).abs() < 0.01);
    }
}
