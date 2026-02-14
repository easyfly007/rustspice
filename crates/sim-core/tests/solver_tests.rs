use sim_core::solver::{
    create_solver_auto, create_solver_for_matrix, MatrixProperties, SolverSelector, SolverType,
};

// ============================================================================
// Automatic Solver Selection Tests
// ============================================================================

#[test]
fn test_solver_selector_small_matrix() {
    // Small matrices should use Dense
    let ap = vec![0i64, 2, 4, 6];
    let ai = vec![0i64, 1, 0, 1, 1, 2];

    let selector = SolverSelector::select(3, &ap, &ai);
    assert_eq!(selector.selected, SolverType::Dense);
    assert!(selector.reason.contains("Small"));
}

#[test]
fn test_solver_selector_dense_matrix() {
    // Dense matrix (high fill ratio) up to moderate size should use Dense
    // Create a 100x100 matrix with 80% density
    let n = 100;
    let mut ap = vec![0i64];
    let mut ai = Vec::new();

    for col in 0..n {
        for row in 0..n {
            if (row + col) % 5 != 0 {
                // ~80% density
                ai.push(row as i64);
            }
        }
        ap.push(ai.len() as i64);
    }

    let props = MatrixProperties::analyze_quick(n, &ap, &ai);
    assert!(props.density > 0.3, "Matrix should be dense");

    let selector = SolverSelector::select_quick(n, &ap, &ai);
    assert_eq!(selector.selected, SolverType::Dense);
}

#[test]
fn test_matrix_properties_diagonal() {
    // Diagonal matrix
    let ap = vec![0i64, 1, 2, 3, 4];
    let ai = vec![0i64, 1, 2, 3];

    let props = MatrixProperties::analyze(4, &ap, &ai);

    assert_eq!(props.n, 4);
    assert_eq!(props.nnz, 4);
    assert!((props.density - 0.25).abs() < 0.01); // 4/16 = 0.25
    assert!((props.avg_degree - 1.0).abs() < 0.01);
}

#[test]
fn test_matrix_properties_block_structure() {
    // Block diagonal matrix: two 2x2 blocks
    let ap = vec![0i64, 2, 4, 6, 8];
    let ai = vec![0i64, 1, 0, 1, 2, 3, 2, 3];

    let props = MatrixProperties::analyze(4, &ap, &ai);

    assert_eq!(props.n, 4);
    // Should detect 2 blocks
    assert!(
        props.num_blocks >= 2,
        "Should detect block structure, got {} blocks",
        props.num_blocks
    );
}

#[test]
fn test_create_solver_auto_small() {
    // Small matrix should get Dense
    let solver = create_solver_auto(10);
    assert_eq!(solver.name(), "Dense");
}

#[test]
fn test_create_solver_auto_medium() {
    // Medium matrix should get a sparse solver
    let solver = create_solver_auto(100);
    // Could be Faer, SparseLU, or KLU depending on features
    assert!(
        solver.name() != "Dense" || cfg!(not(any(feature = "klu", feature = "faer-solver"))),
        "Medium matrix should use sparse solver if available"
    );
}

#[test]
fn test_create_solver_for_matrix() {
    // Test the matrix-aware solver selection
    let ap = vec![0i64, 2, 4];
    let ai = vec![0i64, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];

    let mut solver = create_solver_for_matrix(2, &ap, &ai);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();

    let mut rhs = vec![9.0, 8.0];
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-9);
    assert!((rhs[1] - 3.0).abs() < 1e-9);
}

#[test]
fn test_solver_selector_provides_reason() {
    let ap = vec![0i64, 1, 2, 3];
    let ai = vec![0i64, 1, 2];

    let selector = SolverSelector::select(3, &ap, &ai);
    assert!(!selector.reason.is_empty(), "Selector should provide a reason");
}

// ============================================================================
// Original Tests
// ============================================================================

#[test]
fn solver_module_placeholder() {
    assert!(true);
}

#[test]
fn dense_solver_solves_simple_system() {
    use sim_core::solver::{DenseSolver, LinearSolver};

    let ap = vec![0, 2, 4];
    let ai = vec![0, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs = vec![9.0, 8.0];

    let mut solver = DenseSolver::new(2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-9);
    assert!((rhs[1] - 3.0).abs() < 1e-9);
}

#[cfg(feature = "klu")]
#[test]
fn klu_solver_solves_simple_system() {
    use sim_core::solver::{KluSolver, LinearSolver};

    let ap = vec![0, 2, 4];
    let ai = vec![0, 1, 0, 1];
    let ax = vec![3.0, 1.0, 1.0, 2.0];
    let mut rhs = vec![9.0, 8.0];

    let mut solver = KluSolver::new(2);
    solver.prepare(2);
    solver.analyze(&ap, &ai).unwrap();
    solver.factor(&ap, &ai, &ax).unwrap();
    solver.solve(&mut rhs).unwrap();

    assert!((rhs[0] - 2.0).abs() < 1e-9);
    assert!((rhs[1] - 3.0).abs() < 1e-9);
}
