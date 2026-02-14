use crate::circuit::AnalysisCmd;

#[derive(Debug, Clone)]
pub struct ConvergenceConfig {
    pub max_iters: usize,
    pub abs_tol: f64,
    pub rel_tol: f64,
    pub gmin: f64,
    pub damping: f64,
}

#[derive(Debug, Clone)]
pub struct ConvergenceState {
    pub iter: usize,
    pub last_norm: f64,
    pub converged: bool,
}

#[derive(Debug, Clone)]
pub struct TimeStepConfig {
    pub tstep: f64,
    pub tstop: f64,
    pub tstart: f64,
    pub tmax: f64,
    pub min_dt: f64,
    pub max_dt: f64,
    pub abs_tol: f64,
    pub rel_tol: f64,
}

#[derive(Debug, Clone)]
pub struct TimeStepState {
    pub time: f64,
    pub step: usize,
    pub dt: f64,
    pub last_dt: f64,
    pub accepted: bool,
}

#[derive(Debug, Clone)]
pub struct AnalysisPlan {
    pub cmd: AnalysisCmd,
}

pub fn debug_dump_analysis(plan: &AnalysisPlan) {
    println!("analysis: {:?}", plan.cmd);
}

#[derive(Debug, Clone)]
pub struct NewtonPlan {
    pub config: crate::newton::NewtonConfig,
}

#[derive(Debug, Clone)]
pub struct ErrorEstimate {
    pub error_norm: f64,
    pub accept: bool,
}

pub fn estimate_error(prev: &[f64], next: &[f64], tol: f64) -> ErrorEstimate {
    let mut max_err = 0.0;
    for (p, n) in prev.iter().zip(next.iter()) {
        let err = (n - p).abs();
        if err > max_err {
            max_err = err;
        }
    }
    ErrorEstimate {
        error_norm: max_err,
        accept: max_err <= tol,
    }
}

pub fn estimate_error_weighted(
    prev: &[f64],
    next: &[f64],
    abs_tol: f64,
    rel_tol: f64,
) -> ErrorEstimate {
    let mut max_ratio = 0.0;
    for (p, n) in prev.iter().zip(next.iter()) {
        let denom = abs_tol + rel_tol * p.abs().max(n.abs());
        if denom == 0.0 {
            continue;
        }
        let ratio = (n - p).abs() / denom;
        if ratio > max_ratio {
            max_ratio = ratio;
        }
    }
    ErrorEstimate {
        error_norm: max_ratio,
        accept: max_ratio <= 1.0,
    }
}

// ============================================================================
// Phase 1: Local Truncation Error (LTE) Estimation
// ============================================================================

/// Local Truncation Error estimate using Milne's Device.
///
/// Milne's Device compares solutions from two methods of different orders:
/// - Backward Euler (1st order): LTE ∝ dt²
/// - Trapezoidal (2nd order): LTE ∝ dt³
///
/// The difference between them estimates the lower-order LTE:
/// LTE ≈ |x_trap - x_be| / 3
#[derive(Debug, Clone)]
pub struct LteEstimate {
    /// Maximum normalized error ratio across all nodes
    pub error_norm: f64,
    /// Index of the node with maximum error
    pub max_error_node: usize,
    /// Whether this step should be accepted (error_norm <= 1.0)
    pub accept: bool,
    /// Suggested step size adjustment factor
    pub suggested_factor: f64,
    /// Raw LTE value (before normalization)
    pub raw_lte: f64,
}

/// Estimate Local Truncation Error using Milne's Device.
///
/// This function compares solutions from Backward Euler and Trapezoidal
/// methods to estimate the local truncation error.
///
/// # Arguments
/// * `x_be` - Solution from Backward Euler method
/// * `x_trap` - Solution from Trapezoidal method
/// * `abs_tol` - Absolute tolerance (typically 1e-9 V)
/// * `rel_tol` - Relative tolerance (typically 1e-6)
///
/// # Returns
/// * `LteEstimate` containing error metrics and step size suggestion
///
/// # Algorithm
/// For each node i:
///   LTE_i ≈ |x_trap[i] - x_be[i]| / 3
///   tol_i = abs_tol + rel_tol * max(|x_be[i]|, |x_trap[i]|)
///   ratio_i = LTE_i / tol_i
///
/// The factor 3 comes from the difference in truncation error orders:
/// BE: error ≈ (dt²/2) * x''
/// Trap: error ≈ (dt³/12) * x'''
/// For small dt, the difference is dominated by the BE error term.
pub fn estimate_lte_milne(
    x_be: &[f64],
    x_trap: &[f64],
    abs_tol: f64,
    rel_tol: f64,
) -> LteEstimate {
    let mut max_ratio = 0.0;
    let mut max_node = 0;
    let mut max_raw_lte = 0.0;

    for (i, (be, tr)) in x_be.iter().zip(x_trap.iter()).enumerate() {
        // Skip if both values are essentially zero
        if be.abs() < 1e-30 && tr.abs() < 1e-30 {
            continue;
        }

        // LTE ≈ |x_trap - x_be| / 3 (Milne's Device)
        let lte = (tr - be).abs() / 3.0;

        // Weighted tolerance: allows larger errors for larger signals
        let tol = abs_tol + rel_tol * be.abs().max(tr.abs());

        // Normalized error ratio
        let ratio = if tol > 0.0 { lte / tol } else { 0.0 };

        if ratio > max_ratio {
            max_ratio = ratio;
            max_node = i;
            max_raw_lte = lte;
        }
    }

    // Suggested step size factor based on error
    // For a method of order p, optimal factor is (1/error)^(1/(p+1))
    // Using p=1 (BE order), so factor = (1/error)^0.5
    let suggested_factor = if max_ratio > 1e-10 {
        let raw_factor = (1.0 / max_ratio).powf(0.5);
        // Apply safety factor of 0.9
        let safe_factor = raw_factor * 0.9;
        // Clamp to reasonable range [0.1, 2.0]
        safe_factor.clamp(0.1, 2.0)
    } else {
        // Error is very small, allow step to double
        2.0
    };

    LteEstimate {
        error_norm: max_ratio,
        max_error_node: max_node,
        accept: max_ratio <= 1.0,
        suggested_factor,
        raw_lte: max_raw_lte,
    }
}

/// Estimate LTE using solution difference (simpler method).
///
/// This is a fallback when only one integration method is available.
/// It estimates error from the change in solution between time steps.
///
/// Less accurate than Milne's Device but doesn't require dual solve.
///
/// # Arguments
/// * `x_prev` - Solution at previous time step
/// * `x_curr` - Solution at current time step
/// * `dt` - Current time step size
/// * `dt_prev` - Previous time step size
/// * `abs_tol` - Absolute tolerance
/// * `rel_tol` - Relative tolerance
pub fn estimate_lte_difference(
    x_prev: &[f64],
    x_curr: &[f64],
    dt: f64,
    dt_prev: f64,
    abs_tol: f64,
    rel_tol: f64,
) -> LteEstimate {
    let mut max_ratio = 0.0;
    let mut max_node = 0;
    let mut max_raw_lte = 0.0;

    // Ratio of time steps for scaling
    let dt_ratio = dt / dt_prev;

    for (i, (prev, curr)) in x_prev.iter().zip(x_curr.iter()).enumerate() {
        if prev.abs() < 1e-30 && curr.abs() < 1e-30 {
            continue;
        }

        // Estimate second derivative from solution change
        // LTE for BE ≈ (dt²/2) * x'' ≈ (dt/2) * (dx/dt)
        // Approximate dx/dt from finite difference
        let dx = curr - prev;
        let lte = (dx.abs() * dt_ratio) / 2.0;

        let tol = abs_tol + rel_tol * prev.abs().max(curr.abs());
        let ratio = if tol > 0.0 { lte / tol } else { 0.0 };

        if ratio > max_ratio {
            max_ratio = ratio;
            max_node = i;
            max_raw_lte = lte;
        }
    }

    let suggested_factor = if max_ratio > 1e-10 {
        ((1.0 / max_ratio).powf(0.5) * 0.9).clamp(0.1, 2.0)
    } else {
        2.0
    };

    LteEstimate {
        error_norm: max_ratio,
        max_error_node: max_node,
        accept: max_ratio <= 1.0,
        suggested_factor,
        raw_lte: max_raw_lte,
    }
}

/// Configuration for adaptive time stepping with LTE control.
#[derive(Debug, Clone)]
pub struct AdaptiveConfig {
    /// Absolute tolerance for voltage nodes (default: 1e-9 V)
    pub abs_tol: f64,
    /// Relative tolerance (default: 1e-6)
    pub rel_tol: f64,
    /// Minimum allowed time step (default: 1e-15 s)
    pub dt_min: f64,
    /// Maximum allowed time step (default: tstop/10)
    pub dt_max: f64,
    /// Initial time step (default: auto)
    pub dt_init: f64,
    /// Maximum step growth factor (default: 2.0)
    pub growth_limit: f64,
    /// Minimum step shrink factor (default: 0.1)
    pub shrink_limit: f64,
    /// Safety factor applied to suggested step (default: 0.9)
    pub safety_factor: f64,
    /// Error threshold for rejection (default: 1.0)
    pub reject_threshold: f64,
    /// Use Milne's Device for LTE (requires dual solve)
    pub use_milne: bool,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            abs_tol: 1e-9,
            rel_tol: 1e-6,
            dt_min: 1e-15,
            dt_max: 1e-3,
            dt_init: 1e-9,
            growth_limit: 2.0,
            shrink_limit: 0.1,
            safety_factor: 0.9,
            reject_threshold: 1.0,
            use_milne: true,
        }
    }
}

impl AdaptiveConfig {
    /// Create config from TimeStepConfig with adaptive defaults.
    pub fn from_timestep_config(tsc: &TimeStepConfig) -> Self {
        Self {
            abs_tol: tsc.abs_tol,
            rel_tol: tsc.rel_tol,
            dt_min: tsc.min_dt,
            dt_max: tsc.max_dt,
            dt_init: tsc.tstep.min(tsc.max_dt),
            ..Default::default()
        }
    }

    /// Compute suggested new time step from LTE estimate.
    pub fn compute_new_dt(&self, current_dt: f64, lte: &LteEstimate) -> f64 {
        let factor = lte.suggested_factor;
        let new_dt = current_dt * factor;

        // Apply limits
        new_dt.clamp(self.dt_min, self.dt_max)
    }

    /// Compute emergency reduced time step after repeated failures.
    pub fn emergency_dt(&self, current_dt: f64) -> f64 {
        (current_dt * 0.25).max(self.dt_min)
    }
}

// ============================================================================
// Phase 2: PI Controller for Step Size Adjustment
// ============================================================================

/// Configuration for the PI (Proportional-Integral) step size controller.
///
/// The PI controller uses error history to smoothly adjust time steps,
/// avoiding the oscillations that occur with simple fixed-factor adjustments.
///
/// # Algorithm
/// ```text
/// dt_new = dt * (e_tol/e_n)^k_p * (e_tol/e_{n-1})^k_i
/// ```
///
/// Where:
/// - `e_n` = current error ratio (LTE / tolerance)
/// - `e_{n-1}` = previous error ratio
/// - `e_tol` = target error (1.0)
/// - `k_p` = proportional gain (0.7/p for method order p)
/// - `k_i` = integral gain (0.4/p for method order p)
///
/// # References
/// - Gustafsson, K. "Control-theoretic techniques for stepsize selection
///   in implicit Runge-Kutta methods", ACM TOMS, 1994
/// - Söderlind, G. "Automatic control and adaptive time-stepping",
///   Numerical Algorithms, 2002
#[derive(Debug, Clone)]
pub struct PiControllerConfig {
    /// Proportional gain (default: 0.7/p where p is method order)
    pub k_p: f64,
    /// Integral gain (default: 0.4/p where p is method order)
    pub k_i: f64,
    /// Maximum step growth factor per step (default: 2.0)
    pub growth_limit: f64,
    /// Minimum step shrink factor per step (default: 0.1)
    pub shrink_limit: f64,
    /// Safety factor applied to computed step (default: 0.9)
    pub safety_factor: f64,
    /// Error threshold above which step is rejected (default: 1.0)
    pub reject_threshold: f64,
    /// Order of the integration method (1=BE, 2=Trap)
    pub method_order: usize,
}

impl Default for PiControllerConfig {
    fn default() -> Self {
        Self::for_order(2) // Default to 2nd order (Trapezoidal)
    }
}

impl PiControllerConfig {
    /// Create configuration for a specific method order.
    ///
    /// # Arguments
    /// * `order` - Order of the integration method (1 for BE, 2 for Trap/BDF2)
    ///
    /// # Optimal Gains
    /// For order p:
    /// - k_p = 0.7 / p
    /// - k_i = 0.4 / p
    pub fn for_order(order: usize) -> Self {
        let p = order.max(1) as f64;
        Self {
            k_p: 0.7 / p,
            k_i: 0.4 / p,
            growth_limit: 2.0,
            shrink_limit: 0.1,
            safety_factor: 0.9,
            reject_threshold: 1.0,
            method_order: order,
        }
    }

    /// Create configuration for Backward Euler (1st order).
    pub fn backward_euler() -> Self {
        Self::for_order(1)
    }

    /// Create configuration for Trapezoidal rule (2nd order).
    pub fn trapezoidal() -> Self {
        Self::for_order(2)
    }
}

/// PI Controller for adaptive time step selection.
///
/// This controller maintains error history and uses control theory
/// principles to compute optimal step sizes that minimize computational
/// work while maintaining accuracy.
///
/// # Advantages over simple step control:
/// - Smoother step size evolution (no oscillations)
/// - Better tracking of error target
/// - Faster convergence to optimal step size
/// - Handles varying solution dynamics gracefully
#[derive(Debug, Clone)]
pub struct StepController {
    /// Controller configuration
    config: PiControllerConfig,
    /// Error from previous accepted step (e_{n-1})
    prev_error: Option<f64>,
    /// Error from two steps ago (e_{n-2}), for PID extension
    prev_prev_error: Option<f64>,
    /// Time step from previous accepted step
    prev_dt: Option<f64>,
    /// Number of consecutive rejected steps
    consecutive_rejects: usize,
    /// Total number of accepted steps
    accepted_count: usize,
    /// Total number of rejected steps
    rejected_count: usize,
    /// Minimum dt used during simulation
    min_dt_used: f64,
    /// Maximum dt used during simulation
    max_dt_used: f64,
}

impl StepController {
    /// Create a new step controller with given configuration.
    pub fn new(config: PiControllerConfig) -> Self {
        Self {
            config,
            prev_error: None,
            prev_prev_error: None,
            prev_dt: None,
            consecutive_rejects: 0,
            accepted_count: 0,
            rejected_count: 0,
            min_dt_used: f64::MAX,
            max_dt_used: 0.0,
        }
    }

    /// Create a controller for Backward Euler method.
    pub fn backward_euler() -> Self {
        Self::new(PiControllerConfig::backward_euler())
    }

    /// Create a controller for Trapezoidal method.
    pub fn trapezoidal() -> Self {
        Self::new(PiControllerConfig::trapezoidal())
    }

    /// Create a controller for a specific method order.
    pub fn for_order(order: usize) -> Self {
        Self::new(PiControllerConfig::for_order(order))
    }

    /// Compute the suggested new time step based on current error.
    ///
    /// # Arguments
    /// * `current_error` - Current normalized error ratio (LTE/tolerance)
    /// * `current_dt` - Current time step size
    ///
    /// # Returns
    /// Suggested new time step (not yet clamped to min/max)
    ///
    /// # Algorithm
    /// Uses PI controller formula:
    /// ```text
    /// factor = (1/e_n)^k_p * (1/e_{n-1})^k_i * safety
    /// dt_new = dt * clamp(factor, shrink_limit, growth_limit)
    /// ```
    pub fn suggest_dt(&self, current_error: f64, current_dt: f64) -> f64 {
        // Avoid division by zero or log of zero
        let e_n = current_error.max(1e-10);
        let e_tol = 1.0; // Target error ratio

        // Proportional term: (e_tol / e_n)^k_p
        let p_factor = (e_tol / e_n).powf(self.config.k_p);

        // Integral term: (e_tol / e_{n-1})^k_i
        let i_factor = if let Some(e_prev) = self.prev_error {
            let e_prev = e_prev.max(1e-10);
            (e_tol / e_prev).powf(self.config.k_i)
        } else {
            // No history yet, use only proportional term
            1.0
        };

        // Combined factor with safety margin
        let raw_factor = p_factor * i_factor * self.config.safety_factor;

        // Clamp to prevent extreme changes
        let factor = raw_factor.clamp(self.config.shrink_limit, self.config.growth_limit);

        current_dt * factor
    }

    /// Record the result of a time step attempt.
    ///
    /// # Arguments
    /// * `error` - The error ratio for this step
    /// * `dt` - The time step that was attempted
    /// * `accepted` - Whether the step was accepted
    ///
    /// This updates the error history for future PI calculations.
    pub fn record_step(&mut self, error: f64, dt: f64, accepted: bool) {
        if accepted {
            // Shift error history
            self.prev_prev_error = self.prev_error;
            self.prev_error = Some(error);
            self.prev_dt = Some(dt);

            // Update statistics
            self.consecutive_rejects = 0;
            self.accepted_count += 1;
            self.min_dt_used = self.min_dt_used.min(dt);
            self.max_dt_used = self.max_dt_used.max(dt);
        } else {
            self.consecutive_rejects += 1;
            self.rejected_count += 1;
        }
    }

    /// Check if the current error should cause step rejection.
    ///
    /// # Arguments
    /// * `error` - Current normalized error ratio
    ///
    /// # Returns
    /// `true` if the step should be rejected and retried with smaller dt
    pub fn should_reject(&self, error: f64) -> bool {
        error > self.config.reject_threshold
    }

    /// Compute an emergency time step after repeated failures.
    ///
    /// When multiple consecutive steps are rejected, this provides
    /// an aggressively reduced step size to help recovery.
    ///
    /// # Arguments
    /// * `current_dt` - Current time step
    /// * `dt_min` - Minimum allowed time step
    ///
    /// # Returns
    /// Emergency reduced time step
    pub fn emergency_dt(&self, current_dt: f64, dt_min: f64) -> f64 {
        match self.consecutive_rejects {
            0..=2 => current_dt * 0.5,      // Moderate reduction
            3..=5 => current_dt * 0.25,     // Aggressive reduction
            6..=10 => current_dt * 0.1,     // Very aggressive
            _ => dt_min,                     // Fall back to minimum
        }
    }

    /// Check if we're in a failure recovery situation.
    pub fn is_struggling(&self) -> bool {
        self.consecutive_rejects >= 3
    }

    /// Get the number of consecutive rejected steps.
    pub fn consecutive_rejects(&self) -> usize {
        self.consecutive_rejects
    }

    /// Get statistics about the controller's performance.
    pub fn statistics(&self) -> StepControllerStats {
        StepControllerStats {
            accepted_count: self.accepted_count,
            rejected_count: self.rejected_count,
            rejection_rate: if self.accepted_count + self.rejected_count > 0 {
                self.rejected_count as f64 / (self.accepted_count + self.rejected_count) as f64
            } else {
                0.0
            },
            min_dt_used: if self.min_dt_used == f64::MAX { 0.0 } else { self.min_dt_used },
            max_dt_used: self.max_dt_used,
            dt_ratio: if self.min_dt_used > 0.0 && self.min_dt_used != f64::MAX {
                self.max_dt_used / self.min_dt_used
            } else {
                1.0
            },
        }
    }

    /// Reset the controller state (e.g., at simulation start).
    pub fn reset(&mut self) {
        self.prev_error = None;
        self.prev_prev_error = None;
        self.prev_dt = None;
        self.consecutive_rejects = 0;
        self.accepted_count = 0;
        self.rejected_count = 0;
        self.min_dt_used = f64::MAX;
        self.max_dt_used = 0.0;
    }

    /// Get the previous error (if available).
    pub fn prev_error(&self) -> Option<f64> {
        self.prev_error
    }

    /// Get the previous time step (if available).
    pub fn prev_dt(&self) -> Option<f64> {
        self.prev_dt
    }
}

/// Statistics from the step controller.
#[derive(Debug, Clone)]
pub struct StepControllerStats {
    /// Number of accepted time steps
    pub accepted_count: usize,
    /// Number of rejected time steps
    pub rejected_count: usize,
    /// Rejection rate (rejected / total)
    pub rejection_rate: f64,
    /// Minimum time step used
    pub min_dt_used: f64,
    /// Maximum time step used
    pub max_dt_used: f64,
    /// Ratio of max to min dt (measure of step variation)
    pub dt_ratio: f64,
}

impl std::fmt::Display for StepControllerStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Steps: {} accepted, {} rejected ({:.1}% rejection rate), dt range: {:.2e} to {:.2e} (ratio: {:.1}x)",
            self.accepted_count,
            self.rejected_count,
            self.rejection_rate * 100.0,
            self.min_dt_used,
            self.max_dt_used,
            self.dt_ratio
        )
    }
}

/// Combined adaptive controller that uses both LTE estimation and PI control.
///
/// This is the main interface for adaptive time stepping, combining:
/// - LTE estimation (Phase 1)
/// - PI step control (Phase 2)
/// - Emergency handling for difficult regions
#[derive(Debug, Clone)]
pub struct AdaptiveStepController {
    /// Adaptive configuration (tolerances, limits)
    pub config: AdaptiveConfig,
    /// PI controller for step size
    pub pi_controller: StepController,
    /// Minimum time step limit
    pub dt_min: f64,
    /// Maximum time step limit
    pub dt_max: f64,
}

impl AdaptiveStepController {
    /// Create a new adaptive controller with default settings.
    pub fn new(dt_min: f64, dt_max: f64) -> Self {
        Self {
            config: AdaptiveConfig::default(),
            pi_controller: StepController::trapezoidal(),
            dt_min,
            dt_max,
        }
    }

    /// Create from a TimeStepConfig.
    pub fn from_config(tsc: &TimeStepConfig) -> Self {
        Self {
            config: AdaptiveConfig::from_timestep_config(tsc),
            pi_controller: StepController::trapezoidal(),
            dt_min: tsc.min_dt,
            dt_max: tsc.max_dt,
        }
    }

    /// Set the integration method order (affects PI gains).
    pub fn set_method_order(&mut self, order: usize) {
        self.pi_controller = StepController::for_order(order);
    }

    /// Process an LTE estimate and decide on step acceptance and next dt.
    ///
    /// # Arguments
    /// * `lte` - LTE estimate from current step
    /// * `current_dt` - Current time step size
    ///
    /// # Returns
    /// `(accept, next_dt)` - Whether to accept and suggested next step
    pub fn process_lte(&mut self, lte: &LteEstimate, current_dt: f64) -> (bool, f64) {
        let accept = lte.accept && !self.pi_controller.should_reject(lte.error_norm);

        let next_dt = if accept {
            // Use PI controller for smooth step adjustment
            let suggested = self.pi_controller.suggest_dt(lte.error_norm, current_dt);
            self.pi_controller.record_step(lte.error_norm, current_dt, true);
            suggested.clamp(self.dt_min, self.dt_max)
        } else {
            // Step rejected - use emergency reduction if struggling
            self.pi_controller.record_step(lte.error_norm, current_dt, false);

            if self.pi_controller.is_struggling() {
                self.pi_controller.emergency_dt(current_dt, self.dt_min)
            } else {
                // Use LTE suggested factor for first few rejections
                let suggested = current_dt * lte.suggested_factor;
                suggested.clamp(self.dt_min, self.dt_max)
            }
        };

        (accept, next_dt)
    }

    /// Get the current statistics.
    pub fn statistics(&self) -> StepControllerStats {
        self.pi_controller.statistics()
    }

    /// Reset the controller for a new simulation.
    pub fn reset(&mut self) {
        self.pi_controller.reset();
    }
}

#[cfg(test)]
mod pi_controller_tests {
    use super::*;

    #[test]
    fn test_pi_config_for_order() {
        let be_config = PiControllerConfig::for_order(1);
        assert!((be_config.k_p - 0.7).abs() < 1e-10);
        assert!((be_config.k_i - 0.4).abs() < 1e-10);

        let trap_config = PiControllerConfig::for_order(2);
        assert!((trap_config.k_p - 0.35).abs() < 1e-10);
        assert!((trap_config.k_i - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_pi_controller_small_error_increases_dt() {
        let controller = StepController::trapezoidal();
        let current_dt = 1e-9;
        let small_error = 0.1; // Error is 10% of tolerance

        let suggested_dt = controller.suggest_dt(small_error, current_dt);

        // Small error should suggest larger step
        assert!(suggested_dt > current_dt);
        // But limited by growth_limit
        assert!(suggested_dt <= current_dt * 2.0);
    }

    #[test]
    fn test_pi_controller_large_error_decreases_dt() {
        let controller = StepController::trapezoidal();
        let current_dt = 1e-9;
        let large_error = 5.0; // Error is 5x tolerance

        let suggested_dt = controller.suggest_dt(large_error, current_dt);

        // Large error should suggest smaller step
        assert!(suggested_dt < current_dt);
        // But limited by shrink_limit
        assert!(suggested_dt >= current_dt * 0.1);
    }

    #[test]
    fn test_pi_controller_error_at_target() {
        let controller = StepController::trapezoidal();
        let current_dt = 1e-9;
        let target_error = 1.0; // Exactly at tolerance

        let suggested_dt = controller.suggest_dt(target_error, current_dt);

        // At target, step should stay roughly the same (with safety factor)
        assert!(suggested_dt < current_dt); // Safety factor reduces it slightly
        assert!(suggested_dt > current_dt * 0.8);
    }

    #[test]
    fn test_pi_controller_uses_history() {
        let mut controller = StepController::trapezoidal();
        let current_dt = 1e-9;

        // First step - no history
        let dt1 = controller.suggest_dt(0.5, current_dt);
        controller.record_step(0.5, current_dt, true);

        // Second step - has history
        let dt2 = controller.suggest_dt(0.5, current_dt);

        // With history, the integral term contributes
        // Both should suggest increase, but amounts may differ
        assert!(dt1 > current_dt);
        assert!(dt2 > current_dt);
    }

    #[test]
    fn test_pi_controller_record_step_updates_stats() {
        let mut controller = StepController::trapezoidal();

        controller.record_step(0.5, 1e-9, true);
        controller.record_step(0.3, 1.5e-9, true);
        controller.record_step(2.0, 2e-9, false);

        let stats = controller.statistics();
        assert_eq!(stats.accepted_count, 2);
        assert_eq!(stats.rejected_count, 1);
        assert!((stats.rejection_rate - 1.0/3.0).abs() < 0.01);
    }

    #[test]
    fn test_pi_controller_consecutive_rejects() {
        let mut controller = StepController::trapezoidal();

        assert_eq!(controller.consecutive_rejects(), 0);
        assert!(!controller.is_struggling());

        controller.record_step(2.0, 1e-9, false);
        controller.record_step(2.0, 5e-10, false);
        controller.record_step(2.0, 2.5e-10, false);

        assert_eq!(controller.consecutive_rejects(), 3);
        assert!(controller.is_struggling());

        // Accepted step resets counter
        controller.record_step(0.5, 1e-10, true);
        assert_eq!(controller.consecutive_rejects(), 0);
        assert!(!controller.is_struggling());
    }

    #[test]
    fn test_pi_controller_emergency_dt() {
        let controller = StepController::trapezoidal();
        let dt = 1e-9;
        let dt_min = 1e-15;

        // Mild emergency
        let mut ctrl = controller.clone();
        ctrl.consecutive_rejects = 2;
        assert!((ctrl.emergency_dt(dt, dt_min) - dt * 0.5).abs() < 1e-20);

        // Moderate emergency
        ctrl.consecutive_rejects = 4;
        assert!((ctrl.emergency_dt(dt, dt_min) - dt * 0.25).abs() < 1e-20);

        // Severe emergency
        ctrl.consecutive_rejects = 8;
        assert!((ctrl.emergency_dt(dt, dt_min) - dt * 0.1).abs() < 1e-20);

        // Extreme emergency - fall back to minimum
        ctrl.consecutive_rejects = 15;
        assert!((ctrl.emergency_dt(dt, dt_min) - dt_min).abs() < 1e-20);
    }

    #[test]
    fn test_pi_controller_reset() {
        let mut controller = StepController::trapezoidal();

        controller.record_step(0.5, 1e-9, true);
        controller.record_step(2.0, 1e-9, false);

        assert!(controller.prev_error.is_some());
        assert_eq!(controller.accepted_count, 1);
        assert_eq!(controller.rejected_count, 1);

        controller.reset();

        assert!(controller.prev_error.is_none());
        assert_eq!(controller.accepted_count, 0);
        assert_eq!(controller.rejected_count, 0);
        assert_eq!(controller.consecutive_rejects, 0);
    }

    #[test]
    fn test_adaptive_controller_accept_and_grow() {
        let mut controller = AdaptiveStepController::new(1e-15, 1e-6);
        let current_dt = 1e-9;

        let lte = LteEstimate {
            error_norm: 0.1,
            max_error_node: 0,
            accept: true,
            suggested_factor: 1.5,
            raw_lte: 1e-10,
        };

        let (accept, next_dt) = controller.process_lte(&lte, current_dt);

        assert!(accept);
        assert!(next_dt > current_dt);
        assert!(next_dt <= controller.dt_max);
    }

    #[test]
    fn test_adaptive_controller_reject_and_shrink() {
        let mut controller = AdaptiveStepController::new(1e-15, 1e-6);
        let current_dt = 1e-9;

        let lte = LteEstimate {
            error_norm: 5.0,
            max_error_node: 0,
            accept: false,
            suggested_factor: 0.3,
            raw_lte: 1e-6,
        };

        let (accept, next_dt) = controller.process_lte(&lte, current_dt);

        assert!(!accept);
        assert!(next_dt < current_dt);
        assert!(next_dt >= controller.dt_min);
    }

    #[test]
    fn test_adaptive_controller_respects_limits() {
        let mut controller = AdaptiveStepController::new(1e-12, 1e-9);

        // Try to grow beyond max
        let lte_small = LteEstimate {
            error_norm: 0.001,
            max_error_node: 0,
            accept: true,
            suggested_factor: 2.0,
            raw_lte: 1e-12,
        };
        let (_, next_dt) = controller.process_lte(&lte_small, 1e-9);
        assert!(next_dt <= controller.dt_max);

        // Try to shrink below min
        controller.reset();
        let lte_large = LteEstimate {
            error_norm: 1000.0,
            max_error_node: 0,
            accept: false,
            suggested_factor: 0.01,
            raw_lte: 1e-3,
        };
        let (_, next_dt) = controller.process_lte(&lte_large, 1e-12);
        assert!(next_dt >= controller.dt_min);
    }

    #[test]
    fn test_step_controller_stats_display() {
        let stats = StepControllerStats {
            accepted_count: 100,
            rejected_count: 10,
            rejection_rate: 0.0909,
            min_dt_used: 1e-12,
            max_dt_used: 1e-9,
            dt_ratio: 1000.0,
        };

        let display = format!("{}", stats);
        assert!(display.contains("100 accepted"));
        assert!(display.contains("10 rejected"));
        assert!(display.contains("9.1%") || display.contains("9.0%"));
    }
}

#[cfg(test)]
mod lte_tests {
    use super::*;

    #[test]
    fn test_lte_milne_identical_solutions() {
        // When BE and Trap give same result, LTE should be zero
        let x_be = vec![1.0, 2.0, 3.0];
        let x_trap = vec![1.0, 2.0, 3.0];

        let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

        assert!(lte.error_norm < 1e-10);
        assert!(lte.accept);
        assert!((lte.suggested_factor - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_lte_milne_small_difference() {
        // Small difference should be accepted
        let x_be = vec![1.0, 2.0, 3.0];
        let x_trap = vec![1.0 + 1e-10, 2.0 + 1e-10, 3.0 + 1e-10];

        let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

        assert!(lte.accept);
        assert!(lte.error_norm < 1.0);
    }

    #[test]
    fn test_lte_milne_large_difference() {
        // Large difference should be rejected
        let x_be = vec![1.0, 2.0, 3.0];
        let x_trap = vec![1.1, 2.1, 3.1]; // 0.1V difference

        let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

        assert!(!lte.accept);
        assert!(lte.error_norm > 1.0);
        assert!(lte.suggested_factor < 1.0); // Should suggest smaller step
    }

    #[test]
    fn test_lte_milne_relative_tolerance() {
        // Relative tolerance allows larger absolute errors for larger signals
        let x_be = vec![1000.0];
        let x_trap = vec![1000.001]; // 1mV difference on 1000V signal

        let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

        // With rel_tol=1e-6, tolerance is ~1e-3 for 1000V signal
        // LTE = 0.001/3 ≈ 3.3e-4, which is less than tolerance
        assert!(lte.accept);
    }

    #[test]
    fn test_lte_milne_finds_max_error_node() {
        let x_be = vec![1.0, 2.0, 3.0];
        let x_trap = vec![1.0, 2.0 + 1e-6, 3.0]; // Node 1 has error

        let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

        assert_eq!(lte.max_error_node, 1);
    }

    #[test]
    fn test_lte_difference_basic() {
        let x_prev = vec![0.0, 0.0];
        let x_curr = vec![1.0, 2.0];
        let dt = 1e-9;
        let dt_prev = 1e-9;

        let lte = estimate_lte_difference(&x_prev, &x_curr, dt, dt_prev, 1e-9, 1e-6);

        // Large change should indicate large error
        assert!(!lte.accept);
    }

    #[test]
    fn test_adaptive_config_defaults() {
        let config = AdaptiveConfig::default();

        assert_eq!(config.abs_tol, 1e-9);
        assert_eq!(config.rel_tol, 1e-6);
        assert_eq!(config.growth_limit, 2.0);
        assert_eq!(config.shrink_limit, 0.1);
    }

    #[test]
    fn test_adaptive_config_compute_new_dt() {
        let config = AdaptiveConfig {
            dt_min: 1e-12,
            dt_max: 1e-6,
            ..Default::default()
        };

        // When error is small, step should grow
        let lte_small = LteEstimate {
            error_norm: 0.1,
            max_error_node: 0,
            accept: true,
            suggested_factor: 2.0,
            raw_lte: 1e-10,
        };
        let new_dt = config.compute_new_dt(1e-9, &lte_small);
        assert!(new_dt > 1e-9);
        assert!(new_dt <= config.dt_max);

        // When error is large, step should shrink
        let lte_large = LteEstimate {
            error_norm: 10.0,
            max_error_node: 0,
            accept: false,
            suggested_factor: 0.3,
            raw_lte: 1e-6,
        };
        let new_dt = config.compute_new_dt(1e-9, &lte_large);
        assert!(new_dt < 1e-9);
        assert!(new_dt >= config.dt_min);
    }
}
