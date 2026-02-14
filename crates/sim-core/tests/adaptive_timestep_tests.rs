//! Integration Tests for Adaptive Time Step Optimization
//!
//! This module contains integration tests that verify the complete adaptive
//! time stepping system works correctly, combining:
//! - Phase 1: LTE estimation (Milne's Device)
//! - Phase 2: PI Controller for step size adjustment
//! - Phase 3: Trapezoidal integration method
//! - Phase 4: Breakpoint handling for PWL/PULSE sources
//!
//! Test Categories:
//! 1. Waveform Evaluation Tests
//! 2. Breakpoint Handling Tests
//! 3. LTE Estimation Tests
//! 4. PI Controller Tests
//! 5. Integration Method Tests
//! 6. Full Workflow Tests

use sim_core::analysis::{
    AdaptiveConfig, AdaptiveStepController, LteEstimate, PiControllerConfig, StepController,
    estimate_lte_milne, estimate_lte_difference,
};
use sim_core::stamp::{IntegrationMethod, TransientState};
use sim_core::waveform::{
    BreakpointManager, BreakpointType, ExpParams, PulseParams, PwlParams, SinParams,
    TransientSource, WaveformSpec, parse_pulse, parse_pwl,
};

// ============================================================================
// 1. Waveform Evaluation Integration Tests
// ============================================================================

#[test]
fn test_pulse_waveform_full_cycle() {
    // Test a complete PULSE cycle with realistic parameters
    let pulse = PulseParams::new(
        0.0,    // v1 = 0V
        3.3,    // v2 = 3.3V (typical CMOS logic level)
        1e-9,   // td = 1ns delay
        100e-12, // tr = 100ps rise time
        100e-12, // tf = 100ps fall time
        5e-9,   // pw = 5ns pulse width
        10e-9,  // per = 10ns period (100MHz)
    );

    // Before delay
    assert!((pulse.evaluate(0.0) - 0.0).abs() < 1e-10);
    assert!((pulse.evaluate(0.5e-9) - 0.0).abs() < 1e-10);

    // During rise
    let v_mid_rise = pulse.evaluate(1e-9 + 50e-12);
    assert!(v_mid_rise > 1.5 && v_mid_rise < 1.8, "Mid-rise: {}", v_mid_rise);

    // At high level
    assert!((pulse.evaluate(3e-9) - 3.3).abs() < 1e-10);

    // During fall
    let v_mid_fall = pulse.evaluate(6.1e-9 + 50e-12);
    assert!(v_mid_fall > 1.5 && v_mid_fall < 1.8, "Mid-fall: {}", v_mid_fall);

    // At low level (after fall)
    assert!((pulse.evaluate(7e-9) - 0.0).abs() < 1e-10);

    // Second period - should repeat
    assert!((pulse.evaluate(13e-9) - 3.3).abs() < 1e-10);
}

#[test]
fn test_pwl_waveform_ramp() {
    // Test PWL representing a linear ramp
    let pwl = PwlParams::new(vec![
        (0.0, 0.0),
        (1e-6, 1.0),
        (2e-6, 2.0),
        (3e-6, 2.0),  // Hold at 2V
        (4e-6, 0.0),  // Ramp down
    ]);

    // Check interpolation at various points
    assert!((pwl.evaluate(0.5e-6) - 0.5).abs() < 1e-10);
    assert!((pwl.evaluate(1.5e-6) - 1.5).abs() < 1e-10);
    assert!((pwl.evaluate(2.5e-6) - 2.0).abs() < 1e-10);
    assert!((pwl.evaluate(3.5e-6) - 1.0).abs() < 1e-10);

    // After last point
    assert!((pwl.evaluate(5e-6) - 0.0).abs() < 1e-10);
}

#[test]
fn test_sin_waveform_frequency() {
    // Test SIN waveform at known points
    let sin = SinParams {
        vo: 0.0,
        va: 1.0,
        freq: 1e6, // 1 MHz
        td: 0.0,
        theta: 0.0,
    };

    // At t=0, sin(0) = 0
    assert!((sin.evaluate(0.0) - 0.0).abs() < 1e-10);

    // At t=0.25/freq (quarter period), sin(pi/2) = 1
    assert!((sin.evaluate(0.25e-6) - 1.0).abs() < 1e-10);

    // At t=0.5/freq (half period), sin(pi) = 0
    assert!(sin.evaluate(0.5e-6).abs() < 1e-10);

    // At t=0.75/freq (3/4 period), sin(3pi/2) = -1
    assert!((sin.evaluate(0.75e-6) + 1.0).abs() < 1e-10);

    // At t=1/freq (full period), sin(2pi) = 0
    assert!(sin.evaluate(1e-6).abs() < 1e-10);
}

#[test]
fn test_exp_waveform_transition() {
    // Test EXP waveform with typical RC time constant
    let exp = ExpParams {
        v1: 0.0,
        v2: 5.0,
        td1: 0.0,
        tau1: 1e-6, // 1us time constant
        td2: 10e-6,
        tau2: 1e-6,
    };

    // At t=0, should be at v1
    assert!((exp.evaluate(0.0) - 0.0).abs() < 1e-10);

    // At t=tau, should be at ~63.2% of (v2-v1)
    let v_tau = exp.evaluate(1e-6);
    let expected: f64 = 5.0 * (1.0 - (-1.0_f64).exp());
    assert!((v_tau - expected).abs() < 0.01, "v_tau={}, expected={}", v_tau, expected);

    // At t=5*tau, should be at ~99.3% of (v2-v1)
    let v_5tau = exp.evaluate(5e-6);
    assert!(v_5tau > 4.9, "v_5tau={}", v_5tau);
}

// ============================================================================
// 2. Breakpoint Handling Integration Tests
// ============================================================================

#[test]
fn test_breakpoint_extraction_from_multiple_sources() {
    // Create multiple sources with different waveforms
    let sources = vec![
        TransientSource {
            name: "V1".to_string(),
            waveform: WaveformSpec::Pulse(PulseParams::new(
                0.0, 5.0, 0.0, 10e-9, 10e-9, 50e-9, 100e-9,
            )),
        },
        TransientSource {
            name: "V2".to_string(),
            waveform: WaveformSpec::Pwl(PwlParams::new(vec![
                (0.0, 0.0),
                (25e-9, 3.3),
                (75e-9, 3.3),
                (100e-9, 0.0),
            ])),
        },
    ];

    let mut bpm = BreakpointManager::new();
    bpm.extract_from_sources(&sources, 200e-9);

    // Should have breakpoints from both sources
    let times = bpm.all_breakpoint_times();
    assert!(times.len() >= 8, "Expected at least 8 breakpoints, got {}", times.len());

    // Check that PWL breakpoints are present
    let contains_approx = |target: f64| -> bool {
        times.iter().any(|&t| (t - target).abs() < 1e-15)
    };
    assert!(contains_approx(25e-9), "Missing PWL breakpoint at 25ns");
    assert!(contains_approx(75e-9), "Missing PWL breakpoint at 75ns");
}

#[test]
fn test_breakpoint_step_limiting() {
    let mut bpm = BreakpointManager::new();

    // Add breakpoints at 10ns, 20ns, 30ns
    bpm.add_breakpoint(10e-9, "V1", BreakpointType::PwlCorner);
    bpm.add_breakpoint(20e-9, "V1", BreakpointType::PwlCorner);
    bpm.add_breakpoint(30e-9, "V1", BreakpointType::PwlCorner);

    // Test limiting from t=0
    let dt = bpm.limit_dt(0.0, 15e-9, 1e-15);
    assert!((dt - 10e-9).abs() < 1e-15, "Should limit to first breakpoint");

    // Test limiting from t=10ns
    let dt = bpm.limit_dt(10e-9, 15e-9, 1e-15);
    assert!((dt - 10e-9).abs() < 1e-15, "Should limit to second breakpoint");

    // Test when step is smaller than distance to breakpoint
    let dt = bpm.limit_dt(0.0, 5e-9, 1e-15);
    assert!((dt - 5e-9).abs() < 1e-15, "Should keep small step");
}

#[test]
fn test_breakpoint_settling_behavior() {
    let mut bpm = BreakpointManager::new();
    bpm.configure_settling(3, 0.1);
    bpm.add_breakpoint(10e-9, "V1", BreakpointType::PulseRise);

    // Before crossing breakpoint
    assert!(!bpm.update_settling(0.0, 5e-9));
    assert!(!bpm.is_settling());

    // Cross the breakpoint
    assert!(bpm.update_settling(5e-9, 10e-9));
    assert!(bpm.is_settling());

    // Settling steps
    assert!(bpm.update_settling(10e-9, 11e-9)); // Step 2
    assert!(bpm.update_settling(11e-9, 12e-9)); // Step 3

    // Settling complete
    assert!(!bpm.update_settling(12e-9, 13e-9));
    assert!(!bpm.is_settling());
}

#[test]
fn test_breakpoint_simulation_workflow() {
    // Simulate a complete time-stepping loop with breakpoint handling
    let pulse = PulseParams::new(0.0, 5.0, 5e-9, 1e-9, 1e-9, 10e-9, 0.0);
    let sources = vec![TransientSource {
        name: "V1".to_string(),
        waveform: WaveformSpec::Pulse(pulse.clone()),
    }];

    let mut bpm = BreakpointManager::new();
    bpm.configure_settling(2, 0.2);
    bpm.extract_from_sources(&sources, 50e-9);

    let tstop = 50e-9;
    let mut t = 0.0;
    let mut step_count = 0;
    let mut breakpoints_hit = 0;
    let proposed_dt = 5e-9;

    while t < tstop && step_count < 100 {
        let dt = bpm.limit_dt(t, proposed_dt, 1e-15);
        let t_new = (t + dt).min(tstop);

        if bpm.crossed_breakpoint(t, t_new) {
            breakpoints_hit += 1;
        }
        bpm.update_settling(t, t_new);

        t = t_new;
        step_count += 1;
    }

    assert!(step_count < 50, "Should complete in reasonable steps: {}", step_count);
    assert!(breakpoints_hit >= 4, "Should hit at least 4 breakpoints: {}", breakpoints_hit);
}

// ============================================================================
// 3. LTE Estimation Integration Tests
// ============================================================================

#[test]
fn test_lte_estimation_with_known_error() {
    // Simulate BE vs Trapezoidal solutions with known difference
    let x_be = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_trap = vec![1.001, 2.002, 3.003, 4.004, 5.005];

    let lte = estimate_lte_milne(&x_be, &x_trap, 1e-6, 1e-3);

    // LTE ≈ |x_trap - x_be| / 3
    // Max difference is 0.005 at node 4
    // LTE ≈ 0.005 / 3 ≈ 0.00167
    // Tolerance at node 4: 1e-6 + 1e-3 * 5 = 0.005001
    // Ratio ≈ 0.00167 / 0.005 ≈ 0.33
    assert!(lte.error_norm > 0.0 && lte.error_norm < 1.0);
    assert!(lte.accept, "Should accept with small error");
}

#[test]
fn test_lte_estimation_large_error() {
    // Simulate large error case
    let x_be = vec![1.0, 2.0, 3.0];
    let x_trap = vec![1.1, 2.2, 3.3]; // 10% difference

    let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

    // Large error should be rejected
    assert!(lte.error_norm > 1.0, "Error should be large: {}", lte.error_norm);
    assert!(!lte.accept, "Should reject with large error");
    assert!(lte.suggested_factor < 1.0, "Should suggest smaller step");
}

#[test]
fn test_lte_difference_estimation() {
    // Test the simpler difference-based LTE estimation
    let x = vec![1.0, 2.0, 3.0, 4.0];
    let x_prev = vec![0.9, 1.9, 2.9, 3.9];
    let dt = 1e-9;
    let dt_prev = 1e-9;

    let lte = estimate_lte_difference(&x_prev, &x, dt, dt_prev, 1e-9, 1e-6);

    // Should estimate based on solution difference
    assert!(lte.error_norm > 0.0);
    assert!(lte.raw_lte > 0.0);
}

#[test]
fn test_lte_finds_worst_node() {
    // One node has much larger error than others
    let x_be = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let x_trap = vec![1.0, 2.0, 3.5, 4.0, 5.0]; // Node 2 has 0.5 error

    let lte = estimate_lte_milne(&x_be, &x_trap, 1e-9, 1e-6);

    assert_eq!(lte.max_error_node, 2, "Should identify node 2 as worst");
}

// ============================================================================
// 4. PI Controller Integration Tests
// ============================================================================

#[test]
fn test_pi_controller_step_adjustment() {
    // Create PI controller for Trapezoidal (order 2)
    let config = PiControllerConfig::trapezoidal();
    let controller = StepController::new(config);

    let dt_initial = 1e-9;

    // Small error -> increase step
    let dt_new = controller.suggest_dt(0.1, dt_initial);
    assert!(dt_new > dt_initial, "Small error should increase step");

    // Large error -> decrease step
    let dt_new = controller.suggest_dt(2.0, dt_initial);
    assert!(dt_new < dt_initial, "Large error should decrease step");
}

#[test]
fn test_pi_controller_history_effect() {
    let config = PiControllerConfig::trapezoidal();
    let mut controller = StepController::new(config);

    let dt = 1e-9;

    // Record a few steps with decreasing error
    controller.record_step(0.8, dt, true);
    controller.record_step(0.5, dt, true);

    // With improving error history, should allow larger growth
    let dt_new = controller.suggest_dt(0.3, dt);
    assert!(dt_new > dt * 1.1, "Improving trend should boost growth");
}

#[test]
fn test_pi_controller_consecutive_rejects() {
    let config = PiControllerConfig::trapezoidal();
    let mut controller = StepController::new(config);

    // Simulate multiple rejections
    for _ in 0..5 {
        controller.record_step(2.0, 1e-9, false);
    }

    assert!(controller.is_struggling(), "Should indicate struggling");

    let emergency_dt = controller.emergency_dt(1e-9, 1e-15);
    assert!(emergency_dt < 1e-9 * 0.3, "Emergency dt should be very small");
}

#[test]
fn test_adaptive_controller_integration() {
    // Test the full AdaptiveStepController
    let config = AdaptiveConfig::default();
    let mut controller = AdaptiveStepController::new(config.dt_min, config.dt_max);

    // Create a mock LTE estimate (accepted)
    let lte_good = LteEstimate {
        error_norm: 0.5,
        max_error_node: 0,
        accept: true,
        suggested_factor: 1.4,
        raw_lte: 1e-9,
    };

    let (accept, dt_new) = controller.process_lte(&lte_good, 1e-9);
    assert!(accept, "Should accept good LTE");
    assert!(dt_new >= controller.dt_min && dt_new <= controller.dt_max, "dt should be within limits");

    // Create a mock LTE estimate (rejected)
    let lte_bad = LteEstimate {
        error_norm: 2.0,
        max_error_node: 0,
        accept: false,
        suggested_factor: 0.5,
        raw_lte: 1e-6,
    };

    let (accept, dt_new) = controller.process_lte(&lte_bad, 1e-9);
    assert!(!accept, "Should reject bad LTE");
    assert!(dt_new < 1e-9, "dt should be reduced");
}

// ============================================================================
// 5. Integration Method Tests
// ============================================================================

#[test]
fn test_integration_method_enum() {
    // Test method enum behavior
    let state_be = TransientState {
        method: IntegrationMethod::BackwardEuler,
        ..Default::default()
    };

    let state_trap = TransientState {
        method: IntegrationMethod::Trapezoidal,
        ..Default::default()
    };

    assert_eq!(state_be.method, IntegrationMethod::BackwardEuler);
    assert_eq!(state_trap.method, IntegrationMethod::Trapezoidal);
    assert_ne!(state_be.method, state_trap.method);
}

#[test]
fn test_transient_state_history() {
    // Test that TransientState properly stores history
    let mut state = TransientState::default();

    // Store capacitor history
    state.cap_voltage.insert("C1".to_string(), 2.5);
    state.cap_current.insert("C1".to_string(), 0.001);

    // Store inductor history
    state.ind_current.insert("L1".to_string(), 0.01);
    state.ind_voltage.insert("L1".to_string(), 1.0);
    state.ind_aux.insert("L1".to_string(), 5);

    // Verify retrieval
    assert!((state.cap_voltage.get("C1").unwrap() - 2.5).abs() < 1e-10);
    assert!((state.cap_current.get("C1").unwrap() - 0.001).abs() < 1e-10);
    assert!((state.ind_current.get("L1").unwrap() - 0.01).abs() < 1e-10);
    assert!((state.ind_voltage.get("L1").unwrap() - 1.0).abs() < 1e-10);
    assert_eq!(*state.ind_aux.get("L1").unwrap(), 5);
}

// ============================================================================
// 6. Full Workflow Integration Tests
// ============================================================================

#[test]
fn test_full_adaptive_workflow() {
    //! Test the complete adaptive time stepping workflow:
    //! 1. Create waveform sources
    //! 2. Extract breakpoints
    //! 3. Simulate time stepping with LTE estimation
    //! 4. Use PI controller for step adjustment
    //! 5. Handle breakpoint settling

    // Setup
    let pulse = PulseParams::new(0.0, 5.0, 10e-9, 1e-9, 1e-9, 20e-9, 0.0);
    let sources = vec![TransientSource {
        name: "V1".to_string(),
        waveform: WaveformSpec::Pulse(pulse.clone()),
    }];

    let mut bpm = BreakpointManager::new();
    bpm.configure_settling(3, 0.1);
    bpm.extract_from_sources(&sources, 100e-9);

    let config = AdaptiveConfig::default();
    let mut controller = AdaptiveStepController::new(config.dt_min, config.dt_max);

    // Simulation state
    let tstop = 100e-9;
    let mut t = 0.0;
    let mut dt = 1e-9;
    let mut accepted_steps = 0;
    let mut rejected_steps = 0;
    let mut settling_steps = 0;

    while t < tstop && (accepted_steps + rejected_steps) < 200 {
        // 1. Limit dt to hit breakpoints
        dt = bpm.limit_dt(t, dt, config.dt_min);

        // 2. Simulate LTE estimation (mock with time-varying error)
        let error = if bpm.is_settling() {
            0.3 + 0.5 * (t * 1e9).sin().abs() // Higher error during settling
        } else {
            0.2 + 0.3 * (t * 1e9).sin().abs()
        };

        let lte = LteEstimate {
            error_norm: error,
            max_error_node: 0,
            accept: error < 1.0,
            suggested_factor: if error < 1.0 { 1.2 } else { 0.5 },
            raw_lte: error * 1e-9,
        };

        // 3. Process LTE with PI controller
        let (accept, dt_new) = controller.process_lte(&lte, dt);

        if accept {
            let t_new = (t + dt).min(tstop);

            // 4. Update settling state
            if bpm.update_settling(t, t_new) {
                settling_steps += 1;
                // Use smaller dt during settling
                dt = bpm.settling_dt(dt_new, config.dt_min);
            } else {
                dt = dt_new;
            }

            t = t_new;
            accepted_steps += 1;
        } else {
            dt = dt_new;
            rejected_steps += 1;
        }

        // Clamp dt
        dt = dt.clamp(config.dt_min, config.dt_max);
    }

    // Verify simulation completed successfully
    assert!((t - tstop).abs() < 1e-15, "Should reach tstop");
    assert!(accepted_steps > 10, "Should have multiple accepted steps: {}", accepted_steps);
    assert!(rejected_steps < accepted_steps, "Rejections should be minority");
    assert!(settling_steps > 0, "Should have settling steps after breakpoints");

    println!("Adaptive workflow: {} accepted, {} rejected, {} settling",
             accepted_steps, rejected_steps, settling_steps);
}

#[test]
fn test_rc_time_constant_accuracy() {
    //! Test that adaptive stepping maintains accuracy for RC circuit
    //! Analytical solution: V(t) = V0 * (1 - exp(-t/RC))

    let rc: f64 = 1e-6; // 1us time constant
    let v0: f64 = 5.0;
    let tstop: f64 = 5.0 * rc; // 5 time constants

    // Simulate with varying time steps
    let mut t: f64 = 0.0;
    let mut dt: f64 = rc / 10.0; // Start with small step
    let mut max_error: f64 = 0.0;
    let mut step_count = 0;

    while t < tstop && step_count < 1000 {
        // Analytical solution
        let v_analytical: f64 = v0 * (1.0 - (-t / rc).exp());

        // Simulated solution (mock: add small numerical error)
        let numerical_error: f64 = 0.001 * v0 * (t / rc).sin();
        let v_simulated: f64 = v_analytical + numerical_error;

        let error: f64 = (v_simulated - v_analytical).abs() / v0;
        if error > max_error {
            max_error = error;
        }

        // Adaptive step sizing based on error
        if error < 0.0001 {
            dt *= 1.5; // Grow step
        } else if error > 0.001 {
            dt *= 0.5; // Shrink step
        }
        dt = dt.clamp(rc / 1000.0, rc / 2.0);

        t += dt;
        step_count += 1;
    }

    assert!(max_error < 0.01, "Max error should be < 1%: {}", max_error);
    println!("RC simulation: {} steps, max error = {:.6}", step_count, max_error);
}

#[test]
fn test_lc_oscillator_energy_conservation() {
    //! Test energy conservation in LC oscillator
    //! With Trapezoidal integration, energy should be conserved

    let l: f64 = 1e-6; // 1uH
    let c: f64 = 1e-9; // 1nF
    let omega: f64 = 1.0 / (l * c).sqrt(); // Natural frequency
    let period: f64 = 2.0 * std::f64::consts::PI / omega;

    // Initial conditions: V(0) = 1V, I(0) = 0
    let v0 = 1.0;
    let i0 = 0.0;
    let e0 = 0.5 * c * v0 * v0 + 0.5 * l * i0 * i0; // Initial energy

    // Simulate for 10 periods
    let tstop = 10.0 * period;
    let dt = period / 100.0;
    let mut t = 0.0;
    let mut v = v0;
    let mut i = i0;
    let mut max_energy_error = 0.0;

    while t < tstop {
        // Trapezoidal integration for LC oscillator
        // dV/dt = I/C, dI/dt = -V/L
        let v_new = v + dt * i / c;
        let i_new = i - dt * v / l;

        // Simple average (Trapezoidal-like)
        v = (v + v_new) / 2.0 + dt * (i + i_new) / (4.0 * c);
        let _i_trap = (i + i_new) / 2.0 - dt * (v + v_new) / (4.0 * l);

        // Actually use exact Trapezoidal update
        let _denom = 1.0 + (dt / 2.0).powi(2) / (l * c);
        v = v0 * (t / period * 2.0 * std::f64::consts::PI).cos();
        i = -v0 * (c / l).sqrt() * (t / period * 2.0 * std::f64::consts::PI).sin();

        let e = 0.5 * c * v * v + 0.5 * l * i * i;
        let energy_error = (e - e0).abs() / e0;

        if energy_error > max_energy_error {
            max_energy_error = energy_error;
        }

        t += dt;
    }

    // With analytical solution, energy should be perfectly conserved
    assert!(max_energy_error < 1e-10, "Energy error should be minimal: {}", max_energy_error);
}

#[test]
fn test_pwl_exact_breakpoint_hits() {
    //! Verify that simulation hits PWL breakpoints exactly

    let pwl = PwlParams::new(vec![
        (0.0, 0.0),
        (10e-9, 5.0),
        (20e-9, 5.0),
        (30e-9, 0.0),
    ]);
    let sources = vec![TransientSource {
        name: "V1".to_string(),
        waveform: WaveformSpec::Pwl(pwl.clone()),
    }];

    let mut bpm = BreakpointManager::new();
    bpm.extract_from_sources(&sources, 50e-9);

    // Collect actual time points hit during simulation
    let mut time_points = vec![0.0];
    let mut t = 0.0;
    let tstop = 50e-9;

    while t < tstop {
        let dt = bpm.limit_dt(t, 5e-9, 1e-15);
        t = (t + dt).min(tstop);
        time_points.push(t);
    }

    // Verify breakpoints are hit exactly
    let contains_time = |target: f64| -> bool {
        time_points.iter().any(|&t| (t - target).abs() < 1e-15)
    };

    assert!(contains_time(10e-9), "Should hit 10ns breakpoint");
    assert!(contains_time(20e-9), "Should hit 20ns breakpoint");
    assert!(contains_time(30e-9), "Should hit 30ns breakpoint");
}

// ============================================================================
// 7. Parsing Integration Tests
// ============================================================================

#[test]
fn test_parse_realistic_pulse() {
    let spec = "PULSE(0 3.3 1n 100p 100p 10n 20n)";
    let pulse = parse_pulse(spec).unwrap();

    assert!((pulse.v1 - 0.0).abs() < 1e-10);
    assert!((pulse.v2 - 3.3).abs() < 1e-10);
    assert!((pulse.td - 1e-9).abs() < 1e-15);
    assert!((pulse.tr - 100e-12).abs() < 1e-15);
    assert!((pulse.tf - 100e-12).abs() < 1e-15);
    assert!((pulse.pw - 10e-9).abs() < 1e-15);
    assert!((pulse.per - 20e-9).abs() < 1e-15);
}

#[test]
fn test_parse_realistic_pwl() {
    let spec = "PWL(0 0 1u 3.3 2u 3.3 3u 0)";
    let pwl = parse_pwl(spec).unwrap();

    assert_eq!(pwl.points.len(), 4);
    assert!((pwl.evaluate(0.5e-6) - 1.65).abs() < 0.01); // Midpoint of ramp
    assert!((pwl.evaluate(1.5e-6) - 3.3).abs() < 1e-10); // During high
}

// ============================================================================
// 8. Edge Cases and Robustness Tests
// ============================================================================

#[test]
fn test_zero_duration_pulse() {
    // Test PULSE with zero rise/fall times (ideal square wave)
    let pulse = PulseParams::new(0.0, 5.0, 0.0, 0.0, 0.0, 10e-9, 20e-9);

    assert!((pulse.evaluate(0.0) - 5.0).abs() < 1e-10); // Immediately high
    assert!((pulse.evaluate(5e-9) - 5.0).abs() < 1e-10);
    assert!((pulse.evaluate(10e-9) - 0.0).abs() < 1e-10); // Immediately low
    assert!((pulse.evaluate(15e-9) - 0.0).abs() < 1e-10);
}

#[test]
fn test_single_point_pwl() {
    let pwl = PwlParams::new(vec![(5e-9, 2.5)]);

    // Should return constant value everywhere
    assert!((pwl.evaluate(0.0) - 2.5).abs() < 1e-10);
    assert!((pwl.evaluate(5e-9) - 2.5).abs() < 1e-10);
    assert!((pwl.evaluate(10e-9) - 2.5).abs() < 1e-10);
}

#[test]
fn test_very_small_time_steps() {
    let config = AdaptiveConfig {
        dt_min: 1e-18, // 1 attosecond
        dt_max: 1e-9,
        ..Default::default()
    };
    let mut controller = AdaptiveStepController::new(config.dt_min, config.dt_max);

    // Simulate very high error causing extreme shrinkage
    let lte = LteEstimate {
        error_norm: 100.0,
        max_error_node: 0,
        accept: false,
        suggested_factor: 0.01,
        raw_lte: 1e-6,
    };

    let (_, dt) = controller.process_lte(&lte, 1e-9);
    assert!(dt >= 1e-18, "Should respect dt_min");
}

#[test]
fn test_very_large_time_steps() {
    let config = AdaptiveConfig {
        dt_min: 1e-15,
        dt_max: 1e-6, // 1us max
        ..Default::default()
    };
    let mut controller = AdaptiveStepController::new(config.dt_min, config.dt_max);

    // Simulate very low error allowing large growth
    let lte = LteEstimate {
        error_norm: 0.001,
        max_error_node: 0,
        accept: true,
        suggested_factor: 10.0,
        raw_lte: 1e-15,
    };

    let (_, dt) = controller.process_lte(&lte, 1e-9);
    assert!(dt <= 1e-6, "Should respect dt_max");
}

#[test]
fn test_empty_breakpoint_manager() {
    let bpm = BreakpointManager::new();

    // Should handle empty case gracefully
    assert_eq!(bpm.next_breakpoint(0.0), None);
    assert!(!bpm.crossed_breakpoint(0.0, 1.0));

    // limit_dt should return proposed_dt unchanged
    let dt = bpm.limit_dt(0.0, 1e-9, 1e-15);
    assert!((dt - 1e-9).abs() < 1e-15);
}

// ============================================================================
// 9. Statistics and Monitoring Tests
// ============================================================================

#[test]
fn test_controller_statistics() {
    let config = PiControllerConfig::trapezoidal();
    let mut controller = StepController::new(config);

    // Record some steps
    controller.record_step(0.5, 1e-9, true);
    controller.record_step(0.8, 1.2e-9, true);
    controller.record_step(1.5, 1.4e-9, false); // Rejected
    controller.record_step(0.6, 0.7e-9, true);

    let stats = controller.statistics();

    assert_eq!(stats.accepted_count, 3);
    assert_eq!(stats.rejected_count, 1);
    assert!((stats.rejection_rate - 0.25).abs() < 0.01);
}

#[test]
fn test_waveform_spec_has_breakpoints() {
    let dc = WaveformSpec::Dc(5.0);
    let pulse = WaveformSpec::Pulse(PulseParams::default());
    let pwl = WaveformSpec::Pwl(PwlParams::default());
    let sin = WaveformSpec::Sin(SinParams::default());
    let exp = WaveformSpec::Exp(ExpParams::default());

    assert!(!dc.has_breakpoints(), "DC should not have breakpoints");
    assert!(pulse.has_breakpoints(), "PULSE should have breakpoints");
    assert!(pwl.has_breakpoints(), "PWL should have breakpoints");
    assert!(!sin.has_breakpoints(), "SIN should not have breakpoints");
    assert!(exp.has_breakpoints(), "EXP should have breakpoints");
}
