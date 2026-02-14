use sim_core::newton::{
    apply_damping, check_convergence, norm2, GminSchedule, NewtonConfig, SourceSchedule,
};
use sim_core::analysis::estimate_error;
use sim_core::solver::{LinearSolver, SolverError};
use sim_core::analysis::estimate_error_weighted;

#[test]
fn convergence_check_passes_with_small_delta() {
    let config = NewtonConfig::default();
    let x = vec![1.0, 2.0];
    let dx = vec![1e-12, 1e-12];
    assert!(check_convergence(&dx, &x, &config));
}

#[test]
fn damping_moves_towards_new_solution() {
    let mut x = vec![0.0, 0.0];
    let x_new = vec![1.0, 2.0];
    apply_damping(&mut x, &x_new, 0.5);
    assert_eq!(x, vec![0.5, 1.0]);
}

#[test]
fn gmin_schedule_steps() {
    let mut sched = GminSchedule::new(2, 1e-6, 1e-12);
    let v1 = sched.value();
    assert!(sched.advance());
    let v2 = sched.value();
    assert!(v2 <= v1);
}

#[test]
fn source_schedule_scales() {
    let mut sched = SourceSchedule::new(4);
    assert_eq!(sched.scale(), 0.0);
    sched.advance();
    assert_eq!(sched.scale(), 0.25);
}

#[test]
fn norm2_basic() {
    let vec = vec![3.0, 4.0];
    assert_eq!(norm2(&vec), 5.0);
}

#[test]
fn error_estimate_accepts_small_delta() {
    let prev = vec![0.0, 1.0];
    let next = vec![0.0, 1.0000001];
    let est = estimate_error(&prev, &next, 1e-3);
    assert!(est.accept);
}

#[test]
fn convergence_fails_for_large_delta() {
    let config = NewtonConfig::default();
    let x = vec![1.0, 2.0];
    let dx = vec![1.0, 1.0];
    assert!(!check_convergence(&dx, &x, &config));
}

#[test]
fn run_newton_converges_on_identity_system() {
    let config = NewtonConfig::default();
    let mut x = vec![0.0, 0.0];
    let mut solver = DummySolver;
    let result = sim_core::newton::run_newton(
        &config,
        &mut x,
        |_x| {
            let ap = vec![0, 1, 2];
            let ai = vec![0, 1];
            let ax = vec![1.0, 1.0];
            let rhs = vec![1.0, 2.0];
            (ap, ai, ax, rhs, 2)
        },
        &mut solver,
    );
    assert!(result.converged);
    assert_eq!(x, vec![1.0, 2.0]);
}

struct DummySolver;

impl LinearSolver for DummySolver {
    fn prepare(&mut self, _n: usize) {}

    fn analyze(&mut self, _ap: &[i64], _ai: &[i64]) -> Result<(), SolverError> {
        Ok(())
    }

    fn factor(&mut self, _ap: &[i64], _ai: &[i64], _ax: &[f64]) -> Result<(), SolverError> {
        Ok(())
    }

    fn solve(&mut self, _rhs: &mut [f64]) -> Result<(), SolverError> {
        Ok(())
    }

    fn reset_pattern(&mut self) {}
}

#[test]
fn weighted_error_rejects_large_delta() {
    let prev = vec![1.0, 1.0];
    let next = vec![10.0, 10.0];
    let est = estimate_error_weighted(&prev, &next, 1e-6, 1e-3);
    assert!(!est.accept);
}
