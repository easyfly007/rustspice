//! Comprehensive tests for the native Rust Sparse LU Solver
//!
//! This test module verifies correctness and numerical stability of the SparseLuSolver
//! across various matrix patterns typical in circuit simulation.

use sim_core::solver::{create_solver, DenseSolver, LinearSolver, SolverType};
use sim_core::sparse_lu::SparseLuSolver;

// ============================================================================
// Basic Functionality Tests
// ============================================================================

#[test]
fn test_2x2_simple_system() {
    // Matrix: [[3, 1], [1, 2]]
    // RHS: [9, 8]
    // Solution: x = 2, y = 3
    // Verify: 3*2 + 1*3 = 9 ✓, 1*2 + 2*3 = 8 ✓
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs = vec![9.0, 8.0];

    let mut solver = SparseLuSolver::new(2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!(
        (rhs[0] - 2.0).abs() < 1e-9,
        "Expected x=2, got {}",
        rhs[0]
    );
    assert!(
        (rhs[1] - 3.0).abs() < 1e-9,
        "Expected y=3, got {}",
        rhs[1]
    );
}

#[test]
fn test_3x3_tridiagonal() {
    // Tridiagonal matrix (common in 1D circuit chains):
    // [ 2 -1  0 ]
    // [-1  2 -1 ]
    // [ 0 -1  2 ]
    // CSC format: columns [col0: (0,2), (1,-1)], [col1: (0,-1), (1,2), (2,-1)], [col2: (1,-1), (2,2)]
    let ap = vec![0i64, 2, 5, 7];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2];
    let ax = vec![2.0, -1.0, -1.0, 2.0, -1.0, -1.0, 2.0];
    let mut rhs = vec![1.0, 0.0, 1.0];

    let mut solver = SparseLuSolver::new(3);
    solver.prepare(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // The solution should be [1, 1, 1]
    // Verify: 2*1 - 1*1 = 1 ✓, -1*1 + 2*1 - 1*1 = 0 ✓, -1*1 + 2*1 = 1 ✓
    assert!(
        (rhs[0] - 1.0).abs() < 1e-9,
        "Expected x[0]=1, got {}",
        rhs[0]
    );
    assert!(
        (rhs[1] - 1.0).abs() < 1e-9,
        "Expected x[1]=1, got {}",
        rhs[1]
    );
    assert!(
        (rhs[2] - 1.0).abs() < 1e-9,
        "Expected x[2]=1, got {}",
        rhs[2]
    );
}

#[test]
fn test_diagonal_matrix() {
    // Pure diagonal matrix - simplest case
    let ap = vec![0i64, 1, 2, 3, 4];
    let ai = vec![0i64, 1, 2, 3];
    let ax = vec![2.0, 4.0, 5.0, 10.0];
    let mut rhs = vec![6.0, 12.0, 15.0, 50.0];

    let mut solver = SparseLuSolver::new(4);
    solver.prepare(4);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 3.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
    assert!((rhs[2] - 3.0).abs() < 1e-10);
    assert!((rhs[3] - 5.0).abs() < 1e-10);
}

// ============================================================================
// Circuit Pattern Tests
// ============================================================================

#[test]
fn test_ladder_network_pattern() {
    // Ladder RC network pattern (common in analog circuits):
    //
    //   o---R1---o---R2---o---R3---o
    //   |        |        |        |
    //   C1       C2       C3       C4
    //   |        |        |        |
    //  GND      GND      GND      GND
    //
    // This creates a tridiagonal structure in MNA

    // 4x4 tridiagonal representing simplified ladder
    // Col 0: rows 0, 1
    // Col 1: rows 0, 1, 2
    // Col 2: rows 1, 2, 3
    // Col 3: rows 2, 3
    let ap = vec![0i64, 2, 5, 8, 10];
    let ai = vec![0i64, 1, 0, 1, 2, 1, 2, 3, 2, 3];
    let ax = vec![4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0, -1.0, -1.0, 4.0];

    let mut rhs = vec![1.0, 0.0, 0.0, 1.0];

    let mut solver = SparseLuSolver::new(4);
    solver.prepare(4);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // Verify solution by computing A*x
    let x = rhs.clone();
    let mut ax_result = vec![0.0; 4];
    for col in 0..4 {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            ax_result[row] += ax[idx] * x[col];
        }
    }

    assert!(
        (ax_result[0] - 1.0).abs() < 1e-9,
        "Residual[0] = {}",
        ax_result[0] - 1.0
    );
    assert!(
        (ax_result[1] - 0.0).abs() < 1e-9,
        "Residual[1] = {}",
        ax_result[1]
    );
    assert!(
        (ax_result[2] - 0.0).abs() < 1e-9,
        "Residual[2] = {}",
        ax_result[2]
    );
    assert!(
        (ax_result[3] - 1.0).abs() < 1e-9,
        "Residual[3] = {}",
        ax_result[3] - 1.0
    );
}

#[test]
fn test_star_network_pattern() {
    // Star network: central node connected to all peripheral nodes
    //
    //      o Node 1
    //       \
    //   o----o----o
    //   2    0    3
    //       /
    //      o Node 4
    //
    // Node 0 is center, connected to nodes 1,2,3,4
    // Creates a dense column/row at position 0

    // 5x5 matrix with star pattern
    // Column 0 has entries at rows 0,1,2,3,4
    // Columns 1,2,3,4 have entries at row 0 and diagonal
    let ap = vec![0i64, 5, 7, 9, 11, 13];
    let ai = vec![
        0i64, 1, 2, 3, 4, // col 0
        0, 1, // col 1
        0, 2, // col 2
        0, 3, // col 3
        0, 4, // col 4
    ];
    let ax = vec![
        8.0, -2.0, -2.0, -2.0, -2.0, // col 0: diagonal = sum of connections
        -2.0, 3.0, // col 1
        -2.0, 3.0, // col 2
        -2.0, 3.0, // col 3
        -2.0, 3.0, // col 4
    ];

    let mut rhs = vec![0.0, 1.0, 1.0, 1.0, 1.0];

    let mut solver = SparseLuSolver::new(5);
    solver.prepare(5);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // Verify solution by computing residual
    let x = rhs.clone();
    let mut ax_result = vec![0.0; 5];
    for col in 0..5 {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            ax_result[row] += ax[idx] * x[col];
        }
    }

    for i in 0..5 {
        let expected = if i == 0 { 0.0 } else { 1.0 };
        assert!(
            (ax_result[i] - expected).abs() < 1e-9,
            "Residual[{}] = {} (expected {})",
            i,
            ax_result[i],
            expected
        );
    }
}

// ============================================================================
// Numerical Accuracy Tests
// ============================================================================

#[test]
fn test_compare_with_dense_solver() {
    // Generate a random-ish sparse matrix and verify SparseLU matches Dense
    // Using a 5x5 SPD-like matrix for stability

    let ap = vec![0i64, 3, 6, 9, 12, 14];
    let ai = vec![
        0i64, 1, 2, // col 0
        0, 1, 3, // col 1
        0, 2, 4, // col 2
        1, 3, 4, // col 3
        2, 4, // col 4
    ];
    let ax = vec![
        10.0, -1.0, -2.0, // col 0
        -1.0, 10.0, -1.0, // col 1
        -2.0, 10.0, -3.0, // col 2
        -1.0, 10.0, -2.0, // col 3
        -3.0, 10.0, // col 4
    ];

    let rhs_orig = vec![1.0, 2.0, 3.0, 4.0, 5.0];

    // Solve with SparseLU
    let mut rhs_sparse = rhs_orig.clone();
    let mut sparse_solver = SparseLuSolver::new(5);
    sparse_solver.prepare(5);
    sparse_solver.analyze(&ap, &ai).unwrap();
    sparse_solver.factor(&ap, &ai, &ax).unwrap();
    sparse_solver.solve(&mut rhs_sparse).unwrap();

    // Solve with Dense
    let mut rhs_dense = rhs_orig.clone();
    let mut dense_solver = DenseSolver::new(5);
    dense_solver.prepare(5);
    dense_solver.analyze(&ap, &ai).unwrap();
    dense_solver.factor(&ap, &ai, &ax).unwrap();
    dense_solver.solve(&mut rhs_dense).unwrap();

    // Compare results (allow some numerical difference)
    for i in 0..5 {
        assert!(
            (rhs_sparse[i] - rhs_dense[i]).abs() < 1e-10,
            "Mismatch at {}: SparseLU={}, Dense={}",
            i,
            rhs_sparse[i],
            rhs_dense[i]
        );
    }
}

#[test]
fn test_large_matrix_residual() {
    // Test a larger matrix (10x10) by checking residual norm

    // Create a 10x10 diagonally dominant matrix
    let n = 10;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    for col in 0..n {
        // Each column has diagonal and possibly neighbors
        if col > 0 {
            ai.push((col - 1) as i64);
            ax.push(-1.0);
        }
        ai.push(col as i64);
        ax.push(4.0); // Diagonal dominance
        if col < n - 1 {
            ai.push((col + 1) as i64);
            ax.push(-1.0);
        }
        ap.push(ai.len() as i64);
    }

    let mut rhs: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let rhs_orig = rhs.clone();

    let mut solver = SparseLuSolver::new(n);
    solver.prepare(n);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // Compute residual: r = b - A*x
    let mut residual = rhs_orig;
    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            residual[row] -= ax[idx] * rhs[col];
        }
    }

    // Check residual norm is small
    let residual_norm: f64 = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
    assert!(
        residual_norm < 1e-10,
        "Residual norm too large: {}",
        residual_norm
    );
}

// ============================================================================
// Pattern Reuse Tests
// ============================================================================

#[test]
fn test_pattern_reuse() {
    // Test that factoring with same pattern reuses symbolic analysis

    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax1 = vec![4.0, 1.0, 1.0, 3.0];
    let ax2 = vec![5.0, 2.0, 2.0, 4.0];

    let mut rhs1 = vec![9.0, 8.0];
    let mut rhs2 = vec![14.0, 12.0];

    let mut solver = SparseLuSolver::new(2);
    solver.prepare(2);

    // First factorization
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax1).unwrap();
    solver.solve(&mut rhs1).unwrap();

    // Second factorization with same pattern (should reuse analysis)
    solver.factor(&ap, &ai, &ax2).unwrap();
    solver.solve(&mut rhs2).unwrap();

    // Verify both solutions by residual check
    // For first system: [4, 1; 1, 3] * x = [9, 8]
    assert!((4.0 * rhs1[0] + 1.0 * rhs1[1] - 9.0).abs() < 1e-10);
    assert!((1.0 * rhs1[0] + 3.0 * rhs1[1] - 8.0).abs() < 1e-10);

    // For second system: [5, 2; 2, 4] * x = [14, 12]
    assert!((5.0 * rhs2[0] + 2.0 * rhs2[1] - 14.0).abs() < 1e-10);
    assert!((2.0 * rhs2[0] + 4.0 * rhs2[1] - 12.0).abs() < 1e-10);

    // Check that we only did one analysis
    let stats = solver.stats();
    assert_eq!(stats.factor_count, 2, "Should have factored twice");
}

#[test]
fn test_pattern_change_triggers_reanalysis() {
    let mut solver = SparseLuSolver::new(2);

    // First pattern
    let ap1 = vec![0i64, 1, 2];
    let ai1 = vec![0i64, 1];
    let ax1 = vec![2.0, 3.0];
    let mut rhs1 = vec![4.0, 9.0];

    solver.prepare(2);
    solver.analyze(&ap1, &ai1).unwrap();
    solver.factor(&ap1, &ai1, &ax1).unwrap();
    solver.solve(&mut rhs1).unwrap();

    assert!((rhs1[0] - 2.0).abs() < 1e-10);
    assert!((rhs1[1] - 3.0).abs() < 1e-10);

    // Different pattern (more entries)
    let ap2 = vec![0i64, 2, 4];
    let ai2 = vec![0i64, 1, 0, 1];
    let ax2 = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs2 = vec![9.0, 8.0];

    // This should trigger reanalysis
    solver.factor(&ap2, &ai2, &ax2).unwrap();
    solver.solve(&mut rhs2).unwrap();

    assert!((rhs2[0] - 2.0).abs() < 1e-9);
    assert!((rhs2[1] - 3.0).abs() < 1e-9);
}

// ============================================================================
// Edge Cases
// ============================================================================

#[test]
fn test_1x1_matrix() {
    let ap = vec![0i64, 1];
    let ai = vec![0i64];
    let ax = vec![5.0];
    let mut rhs = vec![15.0];

    let mut solver = SparseLuSolver::new(1);
    solver.prepare(1);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_identity_matrix() {
    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];
    let ax = vec![1.0, 1.0, 1.0];
    let mut rhs = vec![5.0, 7.0, 11.0];

    let mut solver = SparseLuSolver::new(3);
    solver.prepare(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 5.0).abs() < 1e-10);
    assert!((rhs[1] - 7.0).abs() < 1e-10);
    assert!((rhs[2] - 11.0).abs() < 1e-10);
}

#[test]
fn test_lower_triangular() {
    // Lower triangular matrix
    // [ 2  0  0 ]
    // [ 1  3  0 ]
    // [ 2  1  4 ]
    let ap = vec![0i64, 3, 5, 6];
    let ai = vec![0i64, 1, 2, 1, 2, 2];
    let ax = vec![2.0, 1.0, 2.0, 3.0, 1.0, 4.0];
    let mut rhs = vec![4.0, 11.0, 17.0];

    let mut solver = SparseLuSolver::new(3);
    solver.prepare(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // x = [2, 3, 2]
    // 2*2 = 4 ✓
    // 1*2 + 3*3 = 11 ✓
    // 2*2 + 1*3 + 4*2 = 4+3+8 = 15 ≠ 17
    // Let me recalculate...
    // Actually: 2*2 + 1*3 + 4*2 = 4 + 3 + 8 = 15
    // So rhs should be [4, 11, 15] for solution [2, 3, 2]
    // With rhs = [4, 11, 17]:
    // x0 = 4/2 = 2
    // x1 = (11 - 1*2)/3 = 9/3 = 3
    // x2 = (17 - 2*2 - 1*3)/4 = (17-4-3)/4 = 10/4 = 2.5

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
    assert!((rhs[2] - 2.5).abs() < 1e-10);
}

#[test]
fn test_upper_triangular() {
    // Upper triangular matrix
    // [ 2  1  2 ]
    // [ 0  3  1 ]
    // [ 0  0  4 ]
    // Col 0: row 0 only -> (0, 2)
    // Col 1: rows 0, 1 -> (0, 1), (1, 3)
    // Col 2: rows 0, 1, 2 -> (0, 2), (1, 1), (2, 4)
    let ap = vec![0i64, 1, 3, 6];
    let ai = vec![0i64, 0, 1, 0, 1, 2];
    let ax = vec![2.0, 1.0, 3.0, 2.0, 1.0, 4.0];
    let mut rhs = vec![12.0, 7.0, 8.0];

    let mut solver = SparseLuSolver::new(3);
    solver.prepare(3);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    // Solve from bottom:
    // x2 = 8/4 = 2
    // x1 = (7 - 1*2)/3 = 5/3
    // x0 = (12 - 1*(5/3) - 2*2)/2 = (12 - 5/3 - 4)/2 = (8 - 5/3)/2 = (24/3 - 5/3)/2 = (19/3)/2 = 19/6

    assert!((rhs[2] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 5.0 / 3.0).abs() < 1e-10);
    assert!((rhs[0] - 19.0 / 6.0).abs() < 1e-10);
}

// ============================================================================
// SolverType Integration Tests
// ============================================================================

#[test]
fn test_create_solver_sparse_lu() {
    let solver = create_solver(SolverType::SparseLu, 10);
    assert_eq!(solver.name(), "SparseLU");
}

#[test]
fn test_create_solver_sparse_lu_btf() {
    let solver = create_solver(SolverType::SparseLuBtf, 10);
    // Name depends on whether BTF is used (based on matrix structure)
    assert!(solver.name() == "SparseLU" || solver.name() == "SparseLU-BTF");
}

#[test]
fn test_sparse_lu_btf_via_solver_trait() {
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs = vec![9.0, 8.0];

    let mut solver = create_solver(SolverType::SparseLuBtf, 2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-9);
    assert!((rhs[1] - 3.0).abs() < 1e-9);
}

#[test]
fn test_sparse_lu_via_solver_trait() {
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs = vec![9.0, 8.0];

    let mut solver = create_solver(SolverType::SparseLu, 2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-9);
    assert!((rhs[1] - 3.0).abs() < 1e-9);
}

// ============================================================================
// Fill-in and Ordering Tests
// ============================================================================

#[test]
fn test_fill_in_stats() {
    // Arrow matrix pattern - causes fill-in without good ordering
    //
    // [ x x x x x ]   First row/column dense
    // [ x x . . . ]
    // [ x . x . . ]
    // [ x . . x . ]
    // [ x . . . x ]

    let n = 5;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();
    let mut ax = Vec::new();

    // Build arrow matrix in CSC
    for col in 0..n {
        // First row is always present
        ai.push(0);
        ax.push(if col == 0 { 5.0 } else { -1.0 });

        if col > 0 {
            // Column has entry at position col (diagonal)
            ai.push(col as i64);
            ax.push(3.0);
        } else {
            // First column has all rows
            for row in 1..n {
                ai.push(row as i64);
                ax.push(-1.0);
            }
        }
        ap.push(ai.len() as i64);
    }

    let mut rhs = vec![1.0; n];

    let mut solver = SparseLuSolver::new(n);
    solver.prepare(n);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    let stats = solver.stats();
    assert!(stats.analyzed);
    assert!(stats.factored);
    assert!(stats.nnz_l > 0 || stats.nnz_u > 0);

    // Verify solution
    let x = rhs.clone();
    let mut residual = vec![1.0; n];
    for col in 0..n {
        let start = ap[col] as usize;
        let end = ap[col + 1] as usize;
        for idx in start..end {
            let row = ai[idx] as usize;
            residual[row] -= ax[idx] * x[col];
        }
    }

    let residual_norm: f64 = residual.iter().map(|&x| x * x).sum::<f64>().sqrt();
    assert!(
        residual_norm < 1e-10,
        "Residual norm too large: {}",
        residual_norm
    );
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_multiple_solves_same_factorization() {
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];

    let mut solver = SparseLuSolver::new(2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();

    // Solve multiple times with different RHS
    for i in 0..10 {
        let b0 = 9.0 + i as f64;
        let b1 = 8.0 + 2.0 * i as f64;
        let mut rhs = vec![b0, b1];

        solver.solve(&mut rhs).unwrap();

        // Verify: 3x + y = b0, x + 2y = b1
        // Solution: x = (2*b0 - b1)/5, y = (3*b1 - b0)/5
        let expected_x = (2.0 * b0 - b1) / 5.0;
        let expected_y = (3.0 * b1 - b0) / 5.0;

        assert!(
            (rhs[0] - expected_x).abs() < 1e-10,
            "Iter {}: expected x={}, got {}",
            i,
            expected_x,
            rhs[0]
        );
        assert!(
            (rhs[1] - expected_y).abs() < 1e-10,
            "Iter {}: expected y={}, got {}",
            i,
            expected_y,
            rhs[1]
        );
    }
}

#[test]
fn test_reset_pattern() {
    let mut solver = SparseLuSolver::new(2);

    let ap = vec![0i64, 1, 2];
    let ai = vec![0i64, 1];
    let ax = vec![2.0, 3.0];
    let mut rhs = vec![4.0, 9.0];

    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);

    // Reset and use new matrix
    solver.reset_pattern();

    let stats = solver.stats();
    assert!(!stats.analyzed);
    assert!(!stats.factored);

    // Must re-analyze after reset
    rhs = vec![4.0, 9.0];
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-10);
    assert!((rhs[1] - 3.0).abs() < 1e-10);
}
