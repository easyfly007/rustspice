//! Waveform Specification and Breakpoint Handling for Transient Analysis
//!
//! This module provides:
//! 1. Waveform specifications (DC, PULSE, PWL, SIN, EXP)
//! 2. Waveform evaluation at arbitrary time points
//! 3. Breakpoint extraction and management for adaptive time stepping
//!
//! # Breakpoint Handling
//!
//! Breakpoints are discontinuities in waveforms that require special handling:
//! - The time step should be limited to hit breakpoints exactly
//! - After a breakpoint, smaller time steps help capture the transient response
//!
//! # Example
//!
//! ```ignore
//! // Extract breakpoints from circuit
//! let mut bp_mgr = BreakpointManager::new();
//! bp_mgr.extract_from_sources(&sources, tstop);
//!
//! // In time stepping loop:
//! let dt = bp_mgr.limit_dt(t, proposed_dt, 1e-15);
//! ```

use std::collections::BTreeSet;

// ============================================================================
// Waveform Specifications
// ============================================================================

/// PULSE waveform parameters
///
/// ```text
///       v2 ─────┬─────┐
///              /│     │\
///             / │     │ \
///            /  │     │  \
///       v1 ─┘   │     │   └─────
///           │   │     │   │
///           td  tr    pw  tf
///           └───────per───────┘
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PulseParams {
    /// Initial value (low level)
    pub v1: f64,
    /// Pulsed value (high level)
    pub v2: f64,
    /// Delay time before first pulse
    pub td: f64,
    /// Rise time
    pub tr: f64,
    /// Fall time
    pub tf: f64,
    /// Pulse width (time at high level)
    pub pw: f64,
    /// Period (0 = single pulse, non-periodic)
    pub per: f64,
}

impl Default for PulseParams {
    fn default() -> Self {
        Self {
            v1: 0.0,
            v2: 1.0,
            td: 0.0,
            tr: 1e-9,
            tf: 1e-9,
            pw: 1e-6,
            per: 0.0, // Non-periodic by default
        }
    }
}

impl PulseParams {
    /// Create a new PULSE specification
    pub fn new(v1: f64, v2: f64, td: f64, tr: f64, tf: f64, pw: f64, per: f64) -> Self {
        Self { v1, v2, td, tr, tf, pw, per }
    }

    /// Evaluate PULSE waveform at time t
    pub fn evaluate(&self, t: f64) -> f64 {
        if t < self.td {
            return self.v1;
        }

        // Time within the current period
        let t_rel = if self.per > 0.0 {
            (t - self.td) % self.per
        } else {
            t - self.td
        };

        // Rise phase
        if t_rel < self.tr {
            return self.v1 + (self.v2 - self.v1) * t_rel / self.tr;
        }

        // High phase
        if t_rel < self.tr + self.pw {
            return self.v2;
        }

        // Fall phase
        if t_rel < self.tr + self.pw + self.tf {
            let t_fall = t_rel - self.tr - self.pw;
            return self.v2 - (self.v2 - self.v1) * t_fall / self.tf;
        }

        // Low phase (rest of period)
        self.v1
    }
}

/// PWL (Piece-Wise Linear) waveform parameters
///
/// ```text
///       │    *───*
///       │   /     \
///       │  /       *───*
///       │ /
///       *─┘
///       └──t1──t2──t3──t4──
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct PwlParams {
    /// Time-value pairs: [(t0, v0), (t1, v1), ...]
    pub points: Vec<(f64, f64)>,
}

impl Default for PwlParams {
    fn default() -> Self {
        Self { points: vec![(0.0, 0.0)] }
    }
}

impl PwlParams {
    /// Create a new PWL specification from time-value pairs
    pub fn new(points: Vec<(f64, f64)>) -> Self {
        let mut params = Self { points };
        // Sort by time to ensure proper ordering
        params.points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        params
    }

    /// Evaluate PWL waveform at time t
    pub fn evaluate(&self, t: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }

        // Before first point
        if t <= self.points[0].0 {
            return self.points[0].1;
        }

        // After last point
        if t >= self.points.last().unwrap().0 {
            return self.points.last().unwrap().1;
        }

        // Find interval containing t
        for i in 0..self.points.len() - 1 {
            let (t0, v0) = self.points[i];
            let (t1, v1) = self.points[i + 1];

            if t >= t0 && t < t1 {
                // Linear interpolation
                let ratio = (t - t0) / (t1 - t0);
                return v0 + (v1 - v0) * ratio;
            }
        }

        // Fallback (shouldn't reach here)
        self.points.last().unwrap().1
    }
}

/// SIN waveform parameters
///
/// ```text
/// v(t) = vo + va * sin(2π * freq * (t - td)) * exp(-theta * (t - td))
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct SinParams {
    /// DC offset
    pub vo: f64,
    /// Amplitude
    pub va: f64,
    /// Frequency in Hz
    pub freq: f64,
    /// Delay time
    pub td: f64,
    /// Damping factor (0 = no damping)
    pub theta: f64,
}

impl Default for SinParams {
    fn default() -> Self {
        Self {
            vo: 0.0,
            va: 1.0,
            freq: 1e6,
            td: 0.0,
            theta: 0.0,
        }
    }
}

impl SinParams {
    /// Evaluate SIN waveform at time t
    pub fn evaluate(&self, t: f64) -> f64 {
        if t < self.td {
            return self.vo;
        }

        let t_rel = t - self.td;
        let phase = 2.0 * std::f64::consts::PI * self.freq * t_rel;
        let damping = if self.theta > 0.0 {
            (-self.theta * t_rel).exp()
        } else {
            1.0
        };

        self.vo + self.va * phase.sin() * damping
    }
}

/// EXP (Exponential) waveform parameters
///
/// ```text
/// v(t) = v1                                    for t < td1
///      = v1 + (v2-v1)*(1 - exp(-(t-td1)/tau1)) for td1 <= t < td2
///      = v1 + (v2-v1)*(1 - exp(-(t-td1)/tau1))
///          + (v1-v2)*(1 - exp(-(t-td2)/tau2)) for t >= td2
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ExpParams {
    /// Initial value
    pub v1: f64,
    /// Target value for first transition
    pub v2: f64,
    /// First transition delay
    pub td1: f64,
    /// First transition time constant
    pub tau1: f64,
    /// Second transition delay
    pub td2: f64,
    /// Second transition time constant
    pub tau2: f64,
}

impl Default for ExpParams {
    fn default() -> Self {
        Self {
            v1: 0.0,
            v2: 1.0,
            td1: 0.0,
            tau1: 1e-9,
            td2: 1e-6,
            tau2: 1e-9,
        }
    }
}

impl ExpParams {
    /// Evaluate EXP waveform at time t
    pub fn evaluate(&self, t: f64) -> f64 {
        if t < self.td1 {
            return self.v1;
        }

        let mut v = self.v1;

        // First exponential transition
        let exp1 = 1.0 - (-(t - self.td1) / self.tau1).exp();
        v += (self.v2 - self.v1) * exp1;

        // Second exponential transition (return to v1)
        if t >= self.td2 {
            let exp2 = 1.0 - (-(t - self.td2) / self.tau2).exp();
            v += (self.v1 - self.v2) * exp2;
        }

        v
    }
}

/// Waveform specification for voltage/current sources
#[derive(Debug, Clone, PartialEq)]
pub enum WaveformSpec {
    /// DC (constant) value
    Dc(f64),
    /// PULSE waveform
    Pulse(PulseParams),
    /// PWL (Piece-Wise Linear) waveform
    Pwl(PwlParams),
    /// SIN waveform
    Sin(SinParams),
    /// EXP waveform
    Exp(ExpParams),
}

impl Default for WaveformSpec {
    fn default() -> Self {
        WaveformSpec::Dc(0.0)
    }
}

impl WaveformSpec {
    /// Evaluate waveform at time t
    pub fn evaluate(&self, t: f64) -> f64 {
        match self {
            WaveformSpec::Dc(v) => *v,
            WaveformSpec::Pulse(p) => p.evaluate(t),
            WaveformSpec::Pwl(p) => p.evaluate(t),
            WaveformSpec::Sin(p) => p.evaluate(t),
            WaveformSpec::Exp(p) => p.evaluate(t),
        }
    }

    /// Check if waveform has discontinuities that need breakpoint handling
    pub fn has_breakpoints(&self) -> bool {
        matches!(self, WaveformSpec::Pulse(_) | WaveformSpec::Pwl(_) | WaveformSpec::Exp(_))
    }
}

// ============================================================================
// Breakpoint Handling
// ============================================================================

/// Type of breakpoint for debugging and analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BreakpointType {
    /// PWL corner point
    PwlCorner,
    /// PULSE delay start
    PulseDelay,
    /// PULSE rise edge start
    PulseRise,
    /// PULSE high level start (after rise)
    PulseHigh,
    /// PULSE fall edge start
    PulseFall,
    /// PULSE low level start (after fall)
    PulseLow,
    /// EXP first transition
    ExpTransition1,
    /// EXP second transition
    ExpTransition2,
    /// User-defined breakpoint
    UserDefined,
}

/// Information about a breakpoint
#[derive(Debug, Clone)]
pub struct Breakpoint {
    /// Time of the breakpoint
    pub time: f64,
    /// Name of the source causing this breakpoint
    pub source_name: String,
    /// Type of breakpoint
    pub bp_type: BreakpointType,
}

/// Source with waveform specification (for breakpoint extraction)
#[derive(Debug, Clone)]
pub struct TransientSource {
    /// Source instance name
    pub name: String,
    /// Waveform specification
    pub waveform: WaveformSpec,
}

/// Breakpoint manager for adaptive time stepping
///
/// Manages breakpoints extracted from circuit sources and provides
/// methods for limiting time steps to hit breakpoints exactly.
#[derive(Debug, Clone)]
pub struct BreakpointManager {
    /// Set of breakpoint times (sorted, unique)
    breakpoint_times: BTreeSet<OrderedF64>,
    /// Detailed breakpoint information
    breakpoints: Vec<Breakpoint>,
    /// Number of settling steps after a breakpoint
    settling_steps: usize,
    /// Current settling counter
    current_settling: usize,
    /// Factor to reduce dt during settling
    settling_factor: f64,
}

/// Wrapper for f64 that implements Ord for use in BTreeSet
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrderedF64(f64);

impl Eq for OrderedF64 {}

impl PartialOrd for OrderedF64 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrderedF64 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(std::cmp::Ordering::Equal)
    }
}

impl Default for BreakpointManager {
    fn default() -> Self {
        Self::new()
    }
}

impl BreakpointManager {
    /// Create a new breakpoint manager
    pub fn new() -> Self {
        Self {
            breakpoint_times: BTreeSet::new(),
            breakpoints: Vec::new(),
            settling_steps: 5,
            current_settling: 0,
            settling_factor: 0.1,
        }
    }

    /// Configure settling behavior after breakpoints
    ///
    /// # Arguments
    /// * `steps` - Number of small time steps after each breakpoint
    /// * `factor` - Factor to reduce time step during settling (0.1 = 10% of normal)
    pub fn configure_settling(&mut self, steps: usize, factor: f64) {
        self.settling_steps = steps;
        self.settling_factor = factor.clamp(0.01, 1.0);
    }

    /// Add a breakpoint at the specified time
    pub fn add_breakpoint(&mut self, time: f64, source_name: &str, bp_type: BreakpointType) {
        // Skip negative times or times very close to existing breakpoints
        if time < 0.0 {
            return;
        }

        self.breakpoint_times.insert(OrderedF64(time));
        self.breakpoints.push(Breakpoint {
            time,
            source_name: source_name.to_string(),
            bp_type,
        });
    }

    /// Extract breakpoints from transient sources
    ///
    /// # Arguments
    /// * `sources` - List of transient sources with waveform specifications
    /// * `tstop` - Stop time for the transient analysis
    pub fn extract_from_sources(&mut self, sources: &[TransientSource], tstop: f64) {
        for source in sources {
            match &source.waveform {
                WaveformSpec::Pulse(p) => self.extract_pulse_breakpoints(&source.name, p, tstop),
                WaveformSpec::Pwl(p) => self.extract_pwl_breakpoints(&source.name, p, tstop),
                WaveformSpec::Exp(p) => self.extract_exp_breakpoints(&source.name, p, tstop),
                WaveformSpec::Sin(p) => self.extract_sin_breakpoints(&source.name, p),
                WaveformSpec::Dc(_) => {} // No breakpoints for DC
            }
        }
    }

    /// Extract breakpoints from PULSE waveform
    fn extract_pulse_breakpoints(&mut self, name: &str, pulse: &PulseParams, tstop: f64) {
        let mut t = pulse.td;
        let period = pulse.per;
        let mut period_count = 0;
        let max_periods = 10000; // Safety limit

        // Add delay point
        if pulse.td > 0.0 {
            self.add_breakpoint(pulse.td, name, BreakpointType::PulseDelay);
        }

        while t <= tstop && period_count < max_periods {
            // Rise edge start
            self.add_breakpoint(t, name, BreakpointType::PulseRise);

            // Rise edge end (high level start)
            let t_high = t + pulse.tr;
            if t_high > tstop {
                break;
            }
            self.add_breakpoint(t_high, name, BreakpointType::PulseHigh);

            // Fall edge start
            let t_fall = t_high + pulse.pw;
            if t_fall > tstop {
                break;
            }
            self.add_breakpoint(t_fall, name, BreakpointType::PulseFall);

            // Fall edge end (low level start)
            let t_low = t_fall + pulse.tf;
            if t_low > tstop {
                break;
            }
            self.add_breakpoint(t_low, name, BreakpointType::PulseLow);

            // Move to next period
            if period > 0.0 {
                t = pulse.td + period * (period_count + 1) as f64;
                period_count += 1;
            } else {
                break; // Non-periodic
            }
        }
    }

    /// Extract breakpoints from PWL waveform
    fn extract_pwl_breakpoints(&mut self, name: &str, pwl: &PwlParams, _tstop: f64) {
        for (t, _v) in &pwl.points {
            self.add_breakpoint(*t, name, BreakpointType::PwlCorner);
        }
    }

    /// Extract breakpoints from EXP waveform
    fn extract_exp_breakpoints(&mut self, name: &str, exp: &ExpParams, _tstop: f64) {
        self.add_breakpoint(exp.td1, name, BreakpointType::ExpTransition1);
        self.add_breakpoint(exp.td2, name, BreakpointType::ExpTransition2);
    }

    /// Extract breakpoints from SIN waveform (only delay if present)
    fn extract_sin_breakpoints(&mut self, name: &str, sin: &SinParams) {
        if sin.td > 0.0 {
            self.add_breakpoint(sin.td, name, BreakpointType::UserDefined);
        }
    }

    /// Get the next breakpoint after time t
    ///
    /// Returns `None` if no more breakpoints exist
    pub fn next_breakpoint(&self, t: f64) -> Option<f64> {
        self.breakpoint_times
            .range((std::ops::Bound::Excluded(OrderedF64(t)), std::ops::Bound::Unbounded))
            .next()
            .map(|of| of.0)
    }

    /// Get the previous breakpoint at or before time t
    pub fn prev_breakpoint(&self, t: f64) -> Option<f64> {
        self.breakpoint_times
            .range((std::ops::Bound::Unbounded, std::ops::Bound::Included(OrderedF64(t))))
            .next_back()
            .map(|of| of.0)
    }

    /// Limit proposed time step to hit the next breakpoint exactly
    ///
    /// # Arguments
    /// * `t` - Current simulation time
    /// * `proposed_dt` - Proposed time step from error control
    /// * `min_margin` - Minimum time to leave before a breakpoint
    ///
    /// # Returns
    /// The (possibly reduced) time step that will either:
    /// - Hit the next breakpoint exactly, or
    /// - Leave enough margin to hit it in the next step
    pub fn limit_dt(&self, t: f64, proposed_dt: f64, min_margin: f64) -> f64 {
        if let Some(next_bp) = self.next_breakpoint(t) {
            let time_to_bp = next_bp - t;

            // If we would overshoot the breakpoint
            if proposed_dt >= time_to_bp {
                // Step exactly to the breakpoint
                return time_to_bp.max(min_margin);
            }

            // If we would get very close but not reach it
            // (within 10% of the step), just go to the breakpoint
            if proposed_dt > 0.9 * time_to_bp {
                return time_to_bp;
            }

            // If the remaining distance after this step would be tiny,
            // stretch this step to hit the breakpoint
            let remaining = time_to_bp - proposed_dt;
            if remaining < 0.1 * proposed_dt && remaining < min_margin * 10.0 {
                return time_to_bp;
            }
        }

        proposed_dt
    }

    /// Check if the time interval crossed a breakpoint
    ///
    /// Returns `true` if there's a breakpoint in the interval (t_prev, t]
    pub fn crossed_breakpoint(&self, t_prev: f64, t: f64) -> bool {
        self.breakpoint_times
            .range((
                std::ops::Bound::Excluded(OrderedF64(t_prev)),
                std::ops::Bound::Included(OrderedF64(t)),
            ))
            .next()
            .is_some()
    }

    /// Update settling state after a time step
    ///
    /// Call this after each accepted time step to track settling behavior.
    ///
    /// # Arguments
    /// * `t_prev` - Previous time
    /// * `t` - Current time
    ///
    /// # Returns
    /// `true` if currently in settling phase (use smaller time steps)
    pub fn update_settling(&mut self, t_prev: f64, t: f64) -> bool {
        if self.crossed_breakpoint(t_prev, t) {
            // Just crossed a breakpoint, start settling
            self.current_settling = self.settling_steps;
        }

        if self.current_settling > 0 {
            self.current_settling -= 1;
            true
        } else {
            false
        }
    }

    /// Get the recommended time step during settling
    ///
    /// # Arguments
    /// * `normal_dt` - Normal (non-settling) time step
    /// * `dt_min` - Minimum allowed time step
    pub fn settling_dt(&self, normal_dt: f64, dt_min: f64) -> f64 {
        (normal_dt * self.settling_factor).max(dt_min)
    }

    /// Check if currently in settling phase
    pub fn is_settling(&self) -> bool {
        self.current_settling > 0
    }

    /// Get the number of breakpoints
    pub fn breakpoint_count(&self) -> usize {
        self.breakpoint_times.len()
    }

    /// Get all breakpoint times as a sorted vector
    pub fn all_breakpoint_times(&self) -> Vec<f64> {
        self.breakpoint_times.iter().map(|of| of.0).collect()
    }

    /// Get detailed breakpoint information
    pub fn breakpoint_info(&self) -> &[Breakpoint] {
        &self.breakpoints
    }

    /// Clear all breakpoints
    pub fn clear(&mut self) {
        self.breakpoint_times.clear();
        self.breakpoints.clear();
        self.current_settling = 0;
    }

    /// Reset settling state without clearing breakpoints
    pub fn reset_settling(&mut self) {
        self.current_settling = 0;
    }
}

// ============================================================================
// Waveform Parsing Helpers
// ============================================================================

/// Parse a PULSE specification string
///
/// Format: `PULSE(v1 v2 td tr tf pw per)`
pub fn parse_pulse(spec: &str) -> Option<PulseParams> {
    // Remove PULSE( prefix and ) suffix
    let inner = spec
        .trim()
        .strip_prefix("PULSE(")
        .or_else(|| spec.trim().strip_prefix("pulse("))
        .and_then(|s| s.strip_suffix(')'))?;

    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let parse_val = |s: &str| -> Option<f64> {
        parse_number_with_suffix(s)
    };

    Some(PulseParams {
        v1: parse_val(parts.get(0)?).unwrap_or(0.0),
        v2: parse_val(parts.get(1)?).unwrap_or(1.0),
        td: parts.get(2).and_then(|s| parse_val(s)).unwrap_or(0.0),
        tr: parts.get(3).and_then(|s| parse_val(s)).unwrap_or(1e-9),
        tf: parts.get(4).and_then(|s| parse_val(s)).unwrap_or(1e-9),
        pw: parts.get(5).and_then(|s| parse_val(s)).unwrap_or(1e-6),
        per: parts.get(6).and_then(|s| parse_val(s)).unwrap_or(0.0),
    })
}

/// Parse a PWL specification string
///
/// Format: `PWL(t1 v1 t2 v2 ...)`
pub fn parse_pwl(spec: &str) -> Option<PwlParams> {
    let inner = spec
        .trim()
        .strip_prefix("PWL(")
        .or_else(|| spec.trim().strip_prefix("pwl("))
        .and_then(|s| s.strip_suffix(')'))?;

    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 || parts.len() % 2 != 0 {
        return None;
    }

    let mut points = Vec::new();
    for chunk in parts.chunks(2) {
        let t = parse_number_with_suffix(chunk[0])?;
        let v = parse_number_with_suffix(chunk[1])?;
        points.push((t, v));
    }

    Some(PwlParams::new(points))
}

/// Parse a SIN specification string
///
/// Format: `SIN(vo va freq td theta)`
/// - vo: DC offset
/// - va: Amplitude
/// - freq: Frequency in Hz
/// - td: Delay time (optional, default 0)
/// - theta: Damping factor (optional, default 0)
pub fn parse_sin(spec: &str) -> Option<SinParams> {
    let inner = spec
        .trim()
        .strip_prefix("SIN(")
        .or_else(|| spec.trim().strip_prefix("sin("))
        .and_then(|s| s.strip_suffix(')'))?;

    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    Some(SinParams {
        vo: parse_number_with_suffix(parts.get(0)?).unwrap_or(0.0),
        va: parse_number_with_suffix(parts.get(1)?).unwrap_or(1.0),
        freq: parts.get(2).and_then(|s| parse_number_with_suffix(s)).unwrap_or(1e6),
        td: parts.get(3).and_then(|s| parse_number_with_suffix(s)).unwrap_or(0.0),
        theta: parts.get(4).and_then(|s| parse_number_with_suffix(s)).unwrap_or(0.0),
    })
}

/// Parse an EXP specification string
///
/// Format: `EXP(v1 v2 td1 tau1 td2 tau2)`
/// - v1: Initial value
/// - v2: Target value
/// - td1: Rise delay time
/// - tau1: Rise time constant
/// - td2: Fall delay time
/// - tau2: Fall time constant
pub fn parse_exp(spec: &str) -> Option<ExpParams> {
    let inner = spec
        .trim()
        .strip_prefix("EXP(")
        .or_else(|| spec.trim().strip_prefix("exp("))
        .and_then(|s| s.strip_suffix(')'))?;

    let parts: Vec<&str> = inner.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    Some(ExpParams {
        v1: parse_number_with_suffix(parts.get(0)?).unwrap_or(0.0),
        v2: parse_number_with_suffix(parts.get(1)?).unwrap_or(1.0),
        td1: parts.get(2).and_then(|s| parse_number_with_suffix(s)).unwrap_or(0.0),
        tau1: parts.get(3).and_then(|s| parse_number_with_suffix(s)).unwrap_or(1e-9),
        td2: parts.get(4).and_then(|s| parse_number_with_suffix(s)).unwrap_or(1e-6),
        tau2: parts.get(5).and_then(|s| parse_number_with_suffix(s)).unwrap_or(1e-9),
    })
}

/// Parse a source value string and return the appropriate WaveformSpec
///
/// Supports:
/// - DC value: "5", "3.3", "1k", "1m", etc.
/// - PULSE: "PULSE(v1 v2 td tr tf pw per)"
/// - PWL: "PWL(t1 v1 t2 v2 ...)"
/// - SIN: "SIN(vo va freq td theta)"
/// - EXP: "EXP(v1 v2 td1 tau1 td2 tau2)"
pub fn parse_source_value(spec: &str) -> Option<WaveformSpec> {
    let upper = spec.trim().to_uppercase();

    // Try PULSE
    if upper.starts_with("PULSE") {
        return parse_pulse(spec).map(WaveformSpec::Pulse);
    }

    // Try PWL
    if upper.starts_with("PWL") {
        return parse_pwl(spec).map(WaveformSpec::Pwl);
    }

    // Try SIN
    if upper.starts_with("SIN") {
        return parse_sin(spec).map(WaveformSpec::Sin);
    }

    // Try EXP
    if upper.starts_with("EXP") {
        return parse_exp(spec).map(WaveformSpec::Exp);
    }

    // Try DC value
    parse_number_with_suffix(spec).map(WaveformSpec::Dc)
}

/// Evaluate a source value string at a given time
///
/// This is the main entry point for time-varying source evaluation.
/// Returns the source value at time `t`.
pub fn evaluate_source_at_time(spec: &str, t: f64) -> Option<f64> {
    parse_source_value(spec).map(|waveform| waveform.evaluate(t))
}

/// Parse a number with engineering suffix (k, m, u, n, p, f, meg, g, t)
pub fn parse_number_with_suffix(token: &str) -> Option<f64> {
    let lower = token.to_ascii_lowercase();
    let trimmed = lower.trim();

    let (num_str, multiplier) = if trimmed.ends_with("meg") {
        (&trimmed[..trimmed.len() - 3], 1e6)
    } else {
        let (value_part, suffix) = trimmed.split_at(trimmed.len().saturating_sub(1));
        match suffix {
            "f" => (value_part, 1e-15),
            "p" => (value_part, 1e-12),
            "n" => (value_part, 1e-9),
            "u" => (value_part, 1e-6),
            "m" => (value_part, 1e-3),
            "k" => (value_part, 1e3),
            "g" => (value_part, 1e9),
            "t" => (value_part, 1e12),
            _ => (trimmed, 1.0),
        }
    };

    num_str.parse::<f64>().ok().map(|n| n * multiplier)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ========================================================================
    // Waveform Evaluation Tests
    // ========================================================================

    #[test]
    fn test_pulse_before_delay() {
        let pulse = PulseParams::new(0.0, 5.0, 1e-6, 1e-9, 1e-9, 1e-6, 0.0);
        assert!((pulse.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((pulse.evaluate(0.5e-6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_rise() {
        let pulse = PulseParams::new(0.0, 5.0, 0.0, 1e-6, 1e-6, 5e-6, 0.0);
        // At t=0, start of rise
        assert!((pulse.evaluate(0.0) - 0.0).abs() < 1e-10);
        // At t=0.5us, halfway through rise
        assert!((pulse.evaluate(0.5e-6) - 2.5).abs() < 1e-10);
        // At t=1us, end of rise
        assert!((pulse.evaluate(1e-6) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_high() {
        let pulse = PulseParams::new(0.0, 5.0, 0.0, 1e-6, 1e-6, 5e-6, 0.0);
        // During high phase
        assert!((pulse.evaluate(2e-6) - 5.0).abs() < 1e-10);
        assert!((pulse.evaluate(5e-6) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_fall() {
        let pulse = PulseParams::new(0.0, 5.0, 0.0, 1e-6, 1e-6, 5e-6, 0.0);
        // At t=6us, start of fall (tr + pw = 1us + 5us)
        assert!((pulse.evaluate(6e-6) - 5.0).abs() < 1e-10);
        // At t=6.5us, halfway through fall
        assert!((pulse.evaluate(6.5e-6) - 2.5).abs() < 1e-10);
        // At t=7us, end of fall
        assert!((pulse.evaluate(7e-6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pulse_periodic() {
        let pulse = PulseParams::new(0.0, 5.0, 0.0, 0.0, 0.0, 5e-6, 10e-6);
        // First period high
        assert!((pulse.evaluate(2e-6) - 5.0).abs() < 1e-10);
        // First period low
        assert!((pulse.evaluate(7e-6) - 0.0).abs() < 1e-10);
        // Second period high
        assert!((pulse.evaluate(12e-6) - 5.0).abs() < 1e-10);
        // Second period low
        assert!((pulse.evaluate(17e-6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_pwl_interpolation() {
        let pwl = PwlParams::new(vec![(0.0, 0.0), (1e-6, 5.0), (2e-6, 3.0)]);
        // At corner points
        assert!((pwl.evaluate(0.0) - 0.0).abs() < 1e-10);
        assert!((pwl.evaluate(1e-6) - 5.0).abs() < 1e-10);
        assert!((pwl.evaluate(2e-6) - 3.0).abs() < 1e-10);
        // Interpolated
        assert!((pwl.evaluate(0.5e-6) - 2.5).abs() < 1e-10);
        assert!((pwl.evaluate(1.5e-6) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_pwl_before_and_after() {
        let pwl = PwlParams::new(vec![(1e-6, 2.0), (2e-6, 4.0)]);
        // Before first point
        assert!((pwl.evaluate(0.0) - 2.0).abs() < 1e-10);
        // After last point
        assert!((pwl.evaluate(3e-6) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_evaluation() {
        let sin = SinParams {
            vo: 1.0,
            va: 2.0,
            freq: 1e6,
            td: 0.0,
            theta: 0.0,
        };
        // At t=0
        assert!((sin.evaluate(0.0) - 1.0).abs() < 1e-10);
        // At t=0.25/freq (quarter period)
        let t_quarter = 0.25e-6;
        assert!((sin.evaluate(t_quarter) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_sin_with_delay() {
        let sin = SinParams {
            vo: 0.0,
            va: 1.0,
            freq: 1e6,
            td: 1e-6,
            theta: 0.0,
        };
        // Before delay
        assert!((sin.evaluate(0.5e-6) - 0.0).abs() < 1e-10);
        // At delay (starts at 0)
        assert!((sin.evaluate(1e-6) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_exp_evaluation() {
        let exp = ExpParams {
            v1: 0.0,
            v2: 5.0,
            td1: 0.0,
            tau1: 1e-6,
            td2: 10e-6,
            tau2: 1e-6,
        };
        // At t=0
        assert!((exp.evaluate(0.0) - 0.0).abs() < 1e-10);
        // At t=tau1 (about 63.2% of transition)
        let v_at_tau = exp.evaluate(1e-6);
        assert!(v_at_tau > 3.0 && v_at_tau < 3.5);
        // At t=5*tau1 (about 99.3% complete)
        assert!((exp.evaluate(5e-6) - 5.0).abs() < 0.1);
    }

    // ========================================================================
    // Breakpoint Extraction Tests
    // ========================================================================

    #[test]
    fn test_breakpoint_manager_new() {
        let bpm = BreakpointManager::new();
        assert_eq!(bpm.breakpoint_count(), 0);
    }

    #[test]
    fn test_add_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PulseRise);
        bpm.add_breakpoint(2e-6, "V1", BreakpointType::PulseFall);
        assert_eq!(bpm.breakpoint_count(), 2);
    }

    #[test]
    fn test_extract_pulse_breakpoints() {
        let mut bpm = BreakpointManager::new();
        let pulse = PulseParams::new(0.0, 5.0, 1e-6, 100e-9, 100e-9, 5e-6, 0.0);
        bpm.extract_pulse_breakpoints("V1", &pulse, 20e-6);

        // Helper to check if a time is approximately in the list
        let times = bpm.all_breakpoint_times();
        let contains_approx = |target: f64| -> bool {
            times.iter().any(|&t| (t - target).abs() < 1e-15)
        };

        // Should have: delay, rise, high, fall, low
        assert!(contains_approx(1e-6));      // delay/rise
        assert!(contains_approx(1.1e-6));    // high (1e-6 + 100n)
        assert!(contains_approx(6.1e-6));    // fall (1e-6 + 100n + 5e-6)
        assert!(contains_approx(6.2e-6));    // low (1e-6 + 100n + 5e-6 + 100n)
    }

    #[test]
    fn test_extract_pulse_periodic_breakpoints() {
        let mut bpm = BreakpointManager::new();
        let pulse = PulseParams::new(0.0, 5.0, 0.0, 0.0, 0.0, 5e-6, 10e-6);
        bpm.extract_pulse_breakpoints("V1", &pulse, 25e-6);

        // Should have breakpoints at 0, 5u, 10u, 15u, 20u, 25u (within tstop)
        let times = bpm.all_breakpoint_times();
        assert!(times.len() >= 5);
    }

    #[test]
    fn test_extract_pwl_breakpoints() {
        let mut bpm = BreakpointManager::new();
        let pwl = PwlParams::new(vec![
            (0.0, 0.0),
            (1e-6, 5.0),
            (2e-6, 3.0),
            (5e-6, 0.0),
        ]);
        bpm.extract_pwl_breakpoints("V1", &pwl, 10e-6);

        let times = bpm.all_breakpoint_times();
        assert!(times.contains(&0.0));
        assert!(times.contains(&1e-6));
        assert!(times.contains(&2e-6));
        assert!(times.contains(&5e-6));
    }

    #[test]
    fn test_extract_exp_breakpoints() {
        let mut bpm = BreakpointManager::new();
        let exp = ExpParams {
            v1: 0.0,
            v2: 5.0,
            td1: 1e-6,
            tau1: 100e-9,
            td2: 5e-6,
            tau2: 100e-9,
        };
        bpm.extract_exp_breakpoints("V1", &exp, 10e-6);

        let times = bpm.all_breakpoint_times();
        assert!(times.contains(&1e-6));
        assert!(times.contains(&5e-6));
    }

    // ========================================================================
    // Breakpoint Limiting Tests
    // ========================================================================

    #[test]
    fn test_next_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PwlCorner);
        bpm.add_breakpoint(3e-6, "V1", BreakpointType::PwlCorner);
        bpm.add_breakpoint(5e-6, "V1", BreakpointType::PwlCorner);

        assert_eq!(bpm.next_breakpoint(0.0), Some(1e-6));
        assert_eq!(bpm.next_breakpoint(1e-6), Some(3e-6));
        assert_eq!(bpm.next_breakpoint(2e-6), Some(3e-6));
        assert_eq!(bpm.next_breakpoint(5e-6), None);
    }

    #[test]
    fn test_limit_dt_no_breakpoints() {
        let bpm = BreakpointManager::new();
        let dt = bpm.limit_dt(0.0, 1e-6, 1e-15);
        assert!((dt - 1e-6).abs() < 1e-15);
    }

    #[test]
    fn test_limit_dt_hits_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PwlCorner);

        // Large step that would overshoot
        let dt = bpm.limit_dt(0.0, 5e-6, 1e-15);
        assert!((dt - 1e-6).abs() < 1e-15);

        // Step exactly to breakpoint
        let dt = bpm.limit_dt(0.5e-6, 1e-6, 1e-15);
        assert!((dt - 0.5e-6).abs() < 1e-15);
    }

    #[test]
    fn test_limit_dt_close_to_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PwlCorner);

        // Step that gets close but doesn't reach (within 90%)
        // At t=0.05us, proposing 0.9us step (would end at 0.95us, 95% of way)
        let dt = bpm.limit_dt(0.05e-6, 0.9e-6, 1e-15);
        // Should stretch to hit the breakpoint
        assert!((dt - 0.95e-6).abs() < 1e-15);
    }

    // ========================================================================
    // Settling Tests
    // ========================================================================

    #[test]
    fn test_crossed_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PwlCorner);

        assert!(!bpm.crossed_breakpoint(0.0, 0.5e-6));
        assert!(bpm.crossed_breakpoint(0.5e-6, 1e-6));
        assert!(bpm.crossed_breakpoint(0.5e-6, 1.5e-6));
        assert!(!bpm.crossed_breakpoint(1e-6, 1.5e-6));
    }

    #[test]
    fn test_settling_after_breakpoint() {
        let mut bpm = BreakpointManager::new();
        bpm.configure_settling(3, 0.1);
        bpm.add_breakpoint(1e-6, "V1", BreakpointType::PwlCorner);

        // Before breakpoint
        assert!(!bpm.update_settling(0.0, 0.5e-6));
        assert!(!bpm.is_settling());

        // Cross breakpoint
        assert!(bpm.update_settling(0.5e-6, 1e-6));
        assert!(bpm.is_settling());

        // Still settling
        assert!(bpm.update_settling(1e-6, 1.1e-6));
        assert!(bpm.update_settling(1.1e-6, 1.2e-6));

        // Settling done (3 steps)
        assert!(!bpm.update_settling(1.2e-6, 1.3e-6));
        assert!(!bpm.is_settling());
    }

    #[test]
    fn test_settling_dt() {
        let mut bpm = BreakpointManager::new();
        bpm.configure_settling(5, 0.1);

        let normal_dt = 1e-6;
        let min_dt = 1e-12;

        let settling_dt = bpm.settling_dt(normal_dt, min_dt);
        assert!((settling_dt - 1e-7).abs() < 1e-15); // 10% of 1us
    }

    // ========================================================================
    // Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_pulse() {
        let spec = "PULSE(0 5 1u 10n 10n 5u 10u)";
        let pulse = parse_pulse(spec).unwrap();
        assert!((pulse.v1 - 0.0).abs() < 1e-10);
        assert!((pulse.v2 - 5.0).abs() < 1e-10);
        assert!((pulse.td - 1e-6).abs() < 1e-15);
        assert!((pulse.tr - 10e-9).abs() < 1e-15);
        assert!((pulse.tf - 10e-9).abs() < 1e-15);
        assert!((pulse.pw - 5e-6).abs() < 1e-15);
        assert!((pulse.per - 10e-6).abs() < 1e-15);
    }

    #[test]
    fn test_parse_pwl() {
        let spec = "PWL(0 0 1u 5 2u 3)";
        let pwl = parse_pwl(spec).unwrap();
        assert_eq!(pwl.points.len(), 3);
        assert!((pwl.points[0].0 - 0.0).abs() < 1e-15);
        assert!((pwl.points[0].1 - 0.0).abs() < 1e-10);
        assert!((pwl.points[1].0 - 1e-6).abs() < 1e-15);
        assert!((pwl.points[1].1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_number_with_suffix() {
        assert!((parse_number_with_suffix("1k").unwrap() - 1e3).abs() < 1e-10);
        assert!((parse_number_with_suffix("1m").unwrap() - 1e-3).abs() < 1e-10);
        assert!((parse_number_with_suffix("1u").unwrap() - 1e-6).abs() < 1e-15);
        assert!((parse_number_with_suffix("1n").unwrap() - 1e-9).abs() < 1e-15);
        assert!((parse_number_with_suffix("1p").unwrap() - 1e-12).abs() < 1e-15);
        assert!((parse_number_with_suffix("1meg").unwrap() - 1e6).abs() < 1e-10);
        assert!((parse_number_with_suffix("3.3").unwrap() - 3.3).abs() < 1e-10);
    }

    // ========================================================================
    // Integration Tests
    // ========================================================================

    #[test]
    fn test_full_breakpoint_workflow() {
        // Create sources
        let sources = vec![
            TransientSource {
                name: "V1".to_string(),
                waveform: WaveformSpec::Pulse(PulseParams::new(
                    0.0, 5.0, 1e-6, 100e-9, 100e-9, 5e-6, 0.0,
                )),
            },
            TransientSource {
                name: "V2".to_string(),
                waveform: WaveformSpec::Pwl(PwlParams::new(vec![
                    (0.0, 0.0),
                    (2e-6, 3.3),
                    (4e-6, 0.0),
                ])),
            },
        ];

        // Extract breakpoints
        let mut bpm = BreakpointManager::new();
        bpm.extract_from_sources(&sources, 10e-6);

        // Should have multiple breakpoints
        assert!(bpm.breakpoint_count() > 4);

        // Simulate time stepping with breakpoint limiting
        let mut t = 0.0;
        let tstop = 10e-6;
        let mut steps = 0;

        while t < tstop && steps < 1000 {
            let proposed_dt = 1e-6; // Large step
            let dt = bpm.limit_dt(t, proposed_dt, 1e-15);

            let t_new = (t + dt).min(tstop);
            let _ = bpm.update_settling(t, t_new);
            t = t_new;
            steps += 1;
        }

        // Should complete with reasonable number of steps
        assert!(steps < 100);
        assert!((t - tstop).abs() < 1e-15);
    }

    // ========================================================================
    // SIN/EXP Parsing Tests
    // ========================================================================

    #[test]
    fn test_parse_sin() {
        let spec = "SIN(0 5 1meg 1u 0)";
        let sin = parse_sin(spec).unwrap();
        assert!((sin.vo - 0.0).abs() < 1e-10);
        assert!((sin.va - 5.0).abs() < 1e-10);
        assert!((sin.freq - 1e6).abs() < 1e-10);
        assert!((sin.td - 1e-6).abs() < 1e-15);
        assert!((sin.theta - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_parse_sin_minimal() {
        let spec = "sin(1 2)";
        let sin = parse_sin(spec).unwrap();
        assert!((sin.vo - 1.0).abs() < 1e-10);
        assert!((sin.va - 2.0).abs() < 1e-10);
        assert!((sin.freq - 1e6).abs() < 1e-10); // default
    }

    #[test]
    fn test_parse_exp() {
        let spec = "EXP(0 5 1u 100n 10u 200n)";
        let exp = parse_exp(spec).unwrap();
        assert!((exp.v1 - 0.0).abs() < 1e-10);
        assert!((exp.v2 - 5.0).abs() < 1e-10);
        assert!((exp.td1 - 1e-6).abs() < 1e-15);
        assert!((exp.tau1 - 100e-9).abs() < 1e-15);
        assert!((exp.td2 - 10e-6).abs() < 1e-15);
        assert!((exp.tau2 - 200e-9).abs() < 1e-15);
    }

    #[test]
    fn test_parse_exp_minimal() {
        let spec = "exp(0 3.3)";
        let exp = parse_exp(spec).unwrap();
        assert!((exp.v1 - 0.0).abs() < 1e-10);
        assert!((exp.v2 - 3.3).abs() < 1e-10);
    }

    #[test]
    fn test_parse_source_value_dc() {
        let spec = "5";
        let waveform = parse_source_value(spec).unwrap();
        match waveform {
            WaveformSpec::Dc(v) => assert!((v - 5.0).abs() < 1e-10),
            _ => panic!("Expected DC waveform"),
        }
    }

    #[test]
    fn test_parse_source_value_pulse() {
        let spec = "PULSE(0 5 0 1n 1n 10n 20n)";
        let waveform = parse_source_value(spec).unwrap();
        match waveform {
            WaveformSpec::Pulse(p) => {
                assert!((p.v1 - 0.0).abs() < 1e-10);
                assert!((p.v2 - 5.0).abs() < 1e-10);
            }
            _ => panic!("Expected PULSE waveform"),
        }
    }

    #[test]
    fn test_parse_source_value_pwl() {
        let spec = "PWL(0 0 1u 5)";
        let waveform = parse_source_value(spec).unwrap();
        match waveform {
            WaveformSpec::Pwl(p) => {
                assert_eq!(p.points.len(), 2);
            }
            _ => panic!("Expected PWL waveform"),
        }
    }

    #[test]
    fn test_parse_source_value_sin() {
        let spec = "SIN(0 1 1k)";
        let waveform = parse_source_value(spec).unwrap();
        match waveform {
            WaveformSpec::Sin(s) => {
                assert!((s.vo - 0.0).abs() < 1e-10);
                assert!((s.va - 1.0).abs() < 1e-10);
                assert!((s.freq - 1e3).abs() < 1e-10);
            }
            _ => panic!("Expected SIN waveform"),
        }
    }

    #[test]
    fn test_parse_source_value_exp() {
        let spec = "EXP(0 5 0 1u)";
        let waveform = parse_source_value(spec).unwrap();
        match waveform {
            WaveformSpec::Exp(e) => {
                assert!((e.v1 - 0.0).abs() < 1e-10);
                assert!((e.v2 - 5.0).abs() < 1e-10);
            }
            _ => panic!("Expected EXP waveform"),
        }
    }

    #[test]
    fn test_evaluate_source_at_time() {
        // DC
        assert!((evaluate_source_at_time("5", 0.0).unwrap() - 5.0).abs() < 1e-10);
        assert!((evaluate_source_at_time("5", 1e-6).unwrap() - 5.0).abs() < 1e-10);

        // PULSE at high level
        let pulse_val = evaluate_source_at_time("PULSE(0 5 0 0 0 10u 20u)", 5e-6).unwrap();
        assert!((pulse_val - 5.0).abs() < 1e-10);

        // PULSE at low level
        let pulse_val = evaluate_source_at_time("PULSE(0 5 0 0 0 10u 20u)", 15e-6).unwrap();
        assert!((pulse_val - 0.0).abs() < 1e-10);

        // PWL interpolation
        let pwl_val = evaluate_source_at_time("PWL(0 0 1u 10)", 0.5e-6).unwrap();
        assert!((pwl_val - 5.0).abs() < 1e-10);

        // SIN at peak
        let sin_val = evaluate_source_at_time("SIN(0 1 1meg)", 0.25e-6).unwrap();
        assert!((sin_val - 1.0).abs() < 1e-10);
    }
}
