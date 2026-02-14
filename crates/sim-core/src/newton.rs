#[derive(Debug, Clone)]
pub struct NewtonConfig {
    pub max_iters: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub gmin: f64,
    pub damping: f64,
    pub damping_min: f64,
    pub gmin_steps: usize,
    pub source_steps: usize,
}

impl Default for NewtonConfig {
    fn default() -> Self {
        Self {
            max_iters: 50,
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            gmin: 1e-12,
            damping: 1.0,
            damping_min: 0.1,
            gmin_steps: 0,
            source_steps: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewtonState {
    pub iter: usize,
    pub converged: bool,
    pub last_norm: f64,
    pub last_dx_norm: f64,
    pub damping: f64,
}

impl NewtonState {
    pub fn new() -> Self {
        Self {
            iter: 0,
            converged: false,
            last_norm: 0.0,
            last_dx_norm: 0.0,
            damping: 1.0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NewtonResult {
    pub converged: bool,
    pub iterations: usize,
    pub final_norm: f64,
    pub reason: NewtonExitReason,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NewtonExitReason {
    Converged,
    MaxIters,
    SolverFailure,
}

pub fn run_newton<FBuild, S>(
    config: &NewtonConfig,
    x: &mut Vec<f64>,
    mut build: FBuild,
    solver: &mut S,
) -> NewtonResult
where
    FBuild: FnMut(&[f64]) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>, usize),
    S: crate::solver::LinearSolver + ?Sized,
{
    let mut state = NewtonState::new();
    state.damping = config.damping;
    let mut prev_dx_norm = f64::MAX;
    let mut reason = NewtonExitReason::MaxIters;
    let mut message = None;

    for iter in 0..config.max_iters {
        state.iter = iter + 1;
        let (ap, ai, ax, mut rhs, n) = build(x);
        solver.prepare(n);
        if solver.analyze(&ap, &ai).is_err()
            || solver.factor(&ap, &ai, &ax).is_err()
            || solver.solve(&mut rhs).is_err()
        {
            reason = NewtonExitReason::SolverFailure;
            message = Some("linear solver failed".to_string());
            break;
        }
        let x_new = rhs;
        let dx: Vec<f64> = x_new.iter().zip(x.iter()).map(|(a, b)| a - b).collect();
        state.last_dx_norm = norm2(&dx);
        state.last_norm = norm2(&x_new);
        if check_convergence(&dx, &x_new, config) {
            *x = x_new;
            state.converged = true;
            reason = NewtonExitReason::Converged;
            break;
        }
        if state.last_dx_norm > prev_dx_norm && state.damping > config.damping_min {
            state.damping = (state.damping * 0.5).max(config.damping_min);
        }
        prev_dx_norm = state.last_dx_norm;
        apply_damping(x, &x_new, state.damping);
    }

    NewtonResult {
        converged: state.converged,
        iterations: state.iter,
        final_norm: state.last_norm,
        reason,
        message,
    }
}

pub fn debug_dump_newton(result: &NewtonResult) {
    println!(
        "newton: converged={} iters={} norm={} reason={:?} msg={:?}",
        result.converged,
        result.iterations,
        result.final_norm,
        result.reason,
        result.message
    );
}

pub fn debug_dump_newton_with_tag(tag: &str, result: &NewtonResult) {
    println!(
        "newton[{}]: converged={} iters={} norm={} reason={:?} msg={:?}",
        tag,
        result.converged,
        result.iterations,
        result.final_norm,
        result.reason,
        result.message
    );
}

pub fn run_newton_with_stepping<FBuild, S>(
    config: &NewtonConfig,
    x: &mut Vec<f64>,
    mut build: FBuild,
    solver: &mut S,
) -> NewtonResult
where
    FBuild: FnMut(&[f64], f64, f64) -> (Vec<i64>, Vec<i64>, Vec<f64>, Vec<f64>, usize),
    S: crate::solver::LinearSolver + ?Sized,
{
    let gmin_start = (config.gmin * 1e3).max(1e-6);
    let mut gmin_sched = GminSchedule::new(config.gmin_steps, gmin_start, config.gmin);
    let mut last_result = NewtonResult {
        converged: false,
        iterations: 0,
        final_norm: 0.0,
        reason: NewtonExitReason::MaxIters,
        message: None,
    };

    for _ in 0..=config.gmin_steps {
        let gmin = gmin_sched.value();
        let mut source_sched = SourceSchedule::new(config.source_steps);

        for _ in 0..=config.source_steps {
            let source_scale = source_sched.scale();
            let result = run_newton(
                config,
                x,
                |x| build(x, gmin, source_scale),
                solver,
            );
            if result.converged {
                return result;
            }
            last_result = result;
            source_sched.advance();
        }

        gmin_sched.advance();
    }

    last_result
}

pub fn check_convergence(dx: &[f64], x: &[f64], config: &NewtonConfig) -> bool {
    dx.iter().zip(x.iter()).all(|(dx_i, x_i)| {
        dx_i.abs() <= config.abs_tol + config.rel_tol * x_i.abs()
    })
}

pub fn norm2(vec: &[f64]) -> f64 {
    vec.iter().map(|v| v * v).sum::<f64>().sqrt()
}

pub fn apply_damping(x: &mut [f64], x_new: &[f64], damping: f64) {
    for (xi, xni) in x.iter_mut().zip(x_new.iter()) {
        *xi = *xi + damping * (*xni - *xi);
    }
}

#[derive(Debug, Clone)]
pub struct GminSchedule {
    pub steps: usize,
    pub current: usize,
    pub start: f64,
    pub end: f64,
}

impl GminSchedule {
    pub fn new(steps: usize, start: f64, end: f64) -> Self {
        Self {
            steps,
            current: 0,
            start,
            end,
        }
    }

    pub fn value(&self) -> f64 {
        if self.steps == 0 {
            return self.end;
        }
        let t = self.current as f64 / self.steps as f64;
        self.start + (self.end - self.start) * t
    }

    pub fn advance(&mut self) -> bool {
        if self.current < self.steps {
            self.current += 1;
            true
        } else {
            false
        }
    }
}

#[derive(Debug, Clone)]
pub struct SourceSchedule {
    pub steps: usize,
    pub current: usize,
}

impl SourceSchedule {
    pub fn new(steps: usize) -> Self {
        Self { steps, current: 0 }
    }

    pub fn scale(&self) -> f64 {
        if self.steps == 0 {
            return 1.0;
        }
        self.current as f64 / self.steps as f64
    }

    pub fn advance(&mut self) -> bool {
        if self.current < self.steps {
            self.current += 1;
            true
        } else {
            false
        }
    }
}
