use crate::circuit::{DeviceKind, Instance, PolySpec};
use crate::complex_mna::ComplexStampContext;
use crate::mna::StampContext;
use crate::waveform::evaluate_source_at_time;
use num_complex::Complex64;
use std::collections::HashMap;

/// Evaluate a POLY(n) polynomial and its derivatives
///
/// For POLY(1): f(x) = c0 + c1*x + c2*x² + c3*x³ + ...
/// For POLY(2): f(x1,x2) = c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1² + c5*x2² + ...
///
/// Returns (value, partial_derivatives)
fn evaluate_poly(poly: &PolySpec, inputs: &[f64]) -> (f64, Vec<f64>) {
    let n = poly.degree;
    let coeffs = &poly.coeffs;

    if n == 0 || inputs.len() < n {
        return (coeffs.first().copied().unwrap_or(0.0), vec![0.0; n]);
    }

    match n {
        1 => {
            // POLY(1): f(x) = c0 + c1*x + c2*x² + c3*x³ + ...
            let x = inputs[0];
            let mut value = 0.0;
            let mut x_power = 1.0;

            // Compute value
            for &c in coeffs.iter() {
                value += c * x_power;
                x_power *= x;
            }

            // Compute derivative: df/dx = c1 + 2*c2*x + 3*c3*x² + ...
            let mut deriv = 0.0;
            let mut x_power = 1.0;
            for (i, &c) in coeffs.iter().enumerate().skip(1) {
                deriv += c * (i as f64) * x_power;
                x_power *= x;
            }

            (value, vec![deriv])
        }
        2 => {
            // POLY(2): f(x1,x2) = c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1² + c5*x2² + c6*x1²*x2 + ...
            // Simplified: only use up to cross term for now
            let x1 = inputs[0];
            let x2 = inputs[1];

            let c0 = coeffs.get(0).copied().unwrap_or(0.0);
            let c1 = coeffs.get(1).copied().unwrap_or(0.0);
            let c2 = coeffs.get(2).copied().unwrap_or(0.0);
            let c3 = coeffs.get(3).copied().unwrap_or(0.0);
            let c4 = coeffs.get(4).copied().unwrap_or(0.0);
            let c5 = coeffs.get(5).copied().unwrap_or(0.0);

            // f = c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1² + c5*x2²
            let value = c0 + c1*x1 + c2*x2 + c3*x1*x2 + c4*x1*x1 + c5*x2*x2;

            // df/dx1 = c1 + c3*x2 + 2*c4*x1
            let deriv1 = c1 + c3*x2 + 2.0*c4*x1;

            // df/dx2 = c2 + c3*x1 + 2*c5*x2
            let deriv2 = c2 + c3*x1 + 2.0*c5*x2;

            (value, vec![deriv1, deriv2])
        }
        _ => {
            // General case: linear terms only for simplicity
            // f = c0 + c1*x1 + c2*x2 + c3*x3 + ...
            let c0 = coeffs.get(0).copied().unwrap_or(0.0);
            let mut value = c0;
            let mut derivs = Vec::with_capacity(n);

            for i in 0..n {
                let c = coeffs.get(i + 1).copied().unwrap_or(0.0);
                let x = inputs.get(i).copied().unwrap_or(0.0);
                value += c * x;
                derivs.push(c);
            }

            (value, derivs)
        }
    }
}

#[derive(Debug, Clone)]
pub enum StampError {
    MissingValue,
    InvalidNodes,
}

pub trait DeviceStamp {
    fn stamp_dc(&self, ctx: &mut StampContext, x: Option<&[f64]>) -> Result<(), StampError>;
    fn stamp_tran(
        &self,
        ctx: &mut StampContext,
        x: Option<&[f64]>,
        dt: f64,
        state: &mut TransientState,
    ) -> Result<(), StampError>;
    /// Stamp for transient analysis with time-varying sources
    ///
    /// This method evaluates source waveforms (PULSE, PWL, SIN, EXP) at time `t`
    /// for accurate transient simulation. For non-source devices, this delegates
    /// to `stamp_tran`.
    fn stamp_tran_at_time(
        &self,
        ctx: &mut StampContext,
        x: Option<&[f64]>,
        t: f64,
        dt: f64,
        state: &mut TransientState,
    ) -> Result<(), StampError>;
    fn stamp_ac(
        &self,
        ctx: &mut ComplexStampContext,
        dc_solution: &[f64],
    ) -> Result<(), StampError>;
}

#[derive(Debug, Clone)]
pub struct InstanceStamp {
    pub instance: Instance,
}

impl DeviceStamp for InstanceStamp {
    fn stamp_dc(&self, ctx: &mut StampContext, x: Option<&[f64]>) -> Result<(), StampError> {
        match self.instance.kind {
            DeviceKind::R => stamp_resistor(ctx, &self.instance),
            DeviceKind::I => stamp_current(ctx, &self.instance),
            DeviceKind::V => stamp_voltage(ctx, &self.instance),
            DeviceKind::D => stamp_diode(ctx, &self.instance, x),
            DeviceKind::M => stamp_mos(ctx, &self.instance, x),
            DeviceKind::L => stamp_inductor_dc(ctx, &self.instance),
            DeviceKind::C => Ok(()), // Capacitor is open circuit in DC
            DeviceKind::E => stamp_vcvs(ctx, &self.instance, x),
            DeviceKind::G => stamp_vccs(ctx, &self.instance, x),
            DeviceKind::F => stamp_cccs(ctx, &self.instance, x),
            DeviceKind::H => stamp_ccvs(ctx, &self.instance, x),
            DeviceKind::X => Ok(()), // Subcircuit instances are already expanded
        }
    }

    fn stamp_tran(
        &self,
        ctx: &mut StampContext,
        x: Option<&[f64]>,
        dt: f64,
        state: &mut TransientState,
    ) -> Result<(), StampError> {
        match self.instance.kind {
            DeviceKind::C => {
                // Use selected integration method
                match state.method {
                    IntegrationMethod::BackwardEuler => {
                        stamp_capacitor_tran(ctx, &self.instance, x, dt, state)
                    }
                    IntegrationMethod::Trapezoidal => {
                        stamp_capacitor_trap(ctx, &self.instance, x, dt, state)
                    }
                }
            }
            DeviceKind::L => {
                // Use selected integration method
                match state.method {
                    IntegrationMethod::BackwardEuler => {
                        stamp_inductor_tran(ctx, &self.instance, x, dt, state)
                    }
                    IntegrationMethod::Trapezoidal => {
                        stamp_inductor_trap(ctx, &self.instance, x, dt, state)
                    }
                }
            }
            _ => self.stamp_dc(ctx, x),
        }
    }

    fn stamp_tran_at_time(
        &self,
        ctx: &mut StampContext,
        x: Option<&[f64]>,
        t: f64,
        dt: f64,
        state: &mut TransientState,
    ) -> Result<(), StampError> {
        match self.instance.kind {
            // Time-varying sources
            DeviceKind::V => stamp_voltage_at_time(ctx, &self.instance, t),
            DeviceKind::I => stamp_current_at_time(ctx, &self.instance, t),
            // Capacitors and inductors use integration method
            DeviceKind::C => {
                match state.method {
                    IntegrationMethod::BackwardEuler => {
                        stamp_capacitor_tran(ctx, &self.instance, x, dt, state)
                    }
                    IntegrationMethod::Trapezoidal => {
                        stamp_capacitor_trap(ctx, &self.instance, x, dt, state)
                    }
                }
            }
            DeviceKind::L => {
                match state.method {
                    IntegrationMethod::BackwardEuler => {
                        stamp_inductor_tran(ctx, &self.instance, x, dt, state)
                    }
                    IntegrationMethod::Trapezoidal => {
                        stamp_inductor_trap(ctx, &self.instance, x, dt, state)
                    }
                }
            }
            // Other devices use DC stamping (no time dependence)
            _ => self.stamp_dc(ctx, x),
        }
    }

    fn stamp_ac(
        &self,
        ctx: &mut ComplexStampContext,
        dc_solution: &[f64],
    ) -> Result<(), StampError> {
        match self.instance.kind {
            DeviceKind::R => stamp_resistor_ac(ctx, &self.instance),
            DeviceKind::C => stamp_capacitor_ac(ctx, &self.instance),
            DeviceKind::L => stamp_inductor_ac(ctx, &self.instance),
            DeviceKind::V => stamp_voltage_ac(ctx, &self.instance),
            DeviceKind::I => stamp_current_ac(ctx, &self.instance),
            DeviceKind::D => stamp_diode_ac(ctx, &self.instance, dc_solution),
            DeviceKind::M => stamp_mos_ac(ctx, &self.instance, dc_solution),
            DeviceKind::E => stamp_vcvs_ac(ctx, &self.instance, dc_solution),
            DeviceKind::G => stamp_vccs_ac(ctx, &self.instance, dc_solution),
            DeviceKind::F => stamp_cccs_ac(ctx, &self.instance, dc_solution),
            DeviceKind::H => stamp_ccvs_ac(ctx, &self.instance, dc_solution),
            DeviceKind::X => Ok(()), // Subcircuit instances are already expanded
        }
    }
}

fn stamp_resistor(ctx: &mut StampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let value = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let g = 1.0 / value;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add(a, a, g);
    ctx.add(b, b, g);
    ctx.add(a, b, -g);
    ctx.add(b, a, -g);
    Ok(())
}

fn stamp_current(ctx: &mut StampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let value = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let value = value * ctx.source_scale;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add_rhs(a, -value);
    ctx.add_rhs(b, value);
    Ok(())
}

fn stamp_voltage(ctx: &mut StampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let value = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let value = value * ctx.source_scale;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    let k = ctx.allocate_aux(&inst.name);
    ctx.add(a, k, 1.0);
    ctx.add(b, k, -1.0);
    ctx.add(k, a, 1.0);
    ctx.add(k, b, -1.0);
    ctx.add_rhs(k, value);
    Ok(())
}

/// Stamp voltage source with time-varying waveform
///
/// Evaluates PULSE, PWL, SIN, or EXP waveforms at time `t`.
/// Falls back to DC value if waveform parsing fails.
fn stamp_voltage_at_time(ctx: &mut StampContext, inst: &Instance, t: f64) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }

    // Evaluate waveform at time t
    let value = inst
        .value
        .as_deref()
        .and_then(|s| evaluate_source_at_time(s, t))
        .ok_or(StampError::MissingValue)?;

    let value = value * ctx.source_scale;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    let k = ctx.allocate_aux(&inst.name);
    ctx.add(a, k, 1.0);
    ctx.add(b, k, -1.0);
    ctx.add(k, a, 1.0);
    ctx.add(k, b, -1.0);
    ctx.add_rhs(k, value);
    Ok(())
}

/// Stamp current source with time-varying waveform
///
/// Evaluates PULSE, PWL, SIN, or EXP waveforms at time `t`.
/// Falls back to DC value if waveform parsing fails.
fn stamp_current_at_time(ctx: &mut StampContext, inst: &Instance, t: f64) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }

    // Evaluate waveform at time t
    let value = inst
        .value
        .as_deref()
        .and_then(|s| evaluate_source_at_time(s, t))
        .ok_or(StampError::MissingValue)?;

    let value = value * ctx.source_scale;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add_rhs(a, -value);
    ctx.add_rhs(b, value);
    Ok(())
}

fn stamp_diode(
    ctx: &mut StampContext,
    inst: &Instance,
    x: Option<&[f64]>,
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    let gmin = if ctx.gmin > 0.0 { ctx.gmin } else { 1e-12 };
    let isat = param_value(&inst.params, &["is"]).unwrap_or(1e-14);
    let emission = param_value(&inst.params, &["n", "nj"]).unwrap_or(1.0);
    let vt = 0.02585 * emission;
    if let Some(x) = x {
        let va = x.get(a).copied().unwrap_or(0.0);
        let vb = x.get(b).copied().unwrap_or(0.0);
        let vd = va - vb;
        let exp_vd = (vd / vt).exp();
        let id = isat * (exp_vd - 1.0);
        let gd = (isat / vt) * exp_vd;
        let g = gd.max(gmin);
        let ieq = id - gd * vd;
        ctx.add(a, a, g);
        ctx.add(b, b, g);
        ctx.add(a, b, -g);
        ctx.add(b, a, -g);
        ctx.add_rhs(a, -ieq);
        ctx.add_rhs(b, ieq);
        return Ok(());
    }
    ctx.add(a, a, gmin);
    ctx.add(b, b, gmin);
    ctx.add(a, b, -gmin);
    ctx.add(b, a, -gmin);
    Ok(())
}

fn stamp_mos(ctx: &mut StampContext, inst: &Instance, x: Option<&[f64]>) -> Result<(), StampError> {
    if inst.nodes.len() < 4 {
        return Err(StampError::InvalidNodes);
    }
    let drain = inst.nodes[0].0;
    let gate = inst.nodes[1].0;
    let source = inst.nodes[2].0;
    let bulk = inst.nodes[3].0;
    let gmin = if ctx.gmin > 0.0 { ctx.gmin } else { 1e-12 };

    // Parse model level (default to 49 for BSIM3)
    let level = param_value(&inst.params, &["level"]).unwrap_or(49.0) as u32;

    // Determine NMOS/PMOS from model type
    let is_pmos = if let Some(t) = inst.params.get("type") {
        let t_lower = t.to_ascii_lowercase();
        t_lower.contains("pmos") || t_lower == "p"
    } else if inst.params.contains_key("pmos") {
        true
    } else {
        false
    };

    // Build BSIM parameters from instance params
    let params = sim_devices::bsim::build_bsim_params(&inst.params, level, is_pmos);

    // Get device dimensions
    let w = param_value(&inst.params, &["w"]).unwrap_or(1e-6);
    let l = param_value(&inst.params, &["l"]).unwrap_or(1e-6);

    // Temperature (default 27C = 300.15K)
    let temp = param_value(&inst.params, &["temp"]).unwrap_or(300.15);

    // BSIM4: Stress parameters (SA/SB distance to STI)
    let sa = param_value(&inst.params, &["sa"]).unwrap_or(0.0);
    let sb = param_value(&inst.params, &["sb"]).unwrap_or(0.0);

    if let Some(x) = x {
        let vd = x.get(drain).copied().unwrap_or(0.0);
        let vg = x.get(gate).copied().unwrap_or(0.0);
        let vs = x.get(source).copied().unwrap_or(0.0);
        let vb = x.get(bulk).copied().unwrap_or(0.0);

        // Use BSIM4 evaluator for Level 54, BSIM3 for others
        if level == 54 {
            // BSIM4: Full evaluation with stress and additional currents
            let output = sim_devices::bsim::evaluate_mos_bsim4(
                &params, w, l, vd, vg, vs, vb, temp, sa, sb
            );

            let gm = output.base.gm;
            let gds = output.base.gds.max(gmin);
            let gmbs = output.base.gmbs;
            let ieq = output.base.ieq;

            // Stamp gds (output conductance between drain and source)
            ctx.add(drain, drain, gds);
            ctx.add(source, source, gds);
            ctx.add(drain, source, -gds);
            ctx.add(source, drain, -gds);

            // Stamp gm (transconductance: current controlled by Vgs)
            ctx.add(drain, gate, gm);
            ctx.add(drain, source, -gm);
            ctx.add(source, gate, -gm);
            ctx.add(source, source, gm);

            // Stamp gmbs (body transconductance: current controlled by Vbs)
            if gmbs.abs() > gmin * 0.01 {
                ctx.add(drain, bulk, gmbs);
                ctx.add(drain, source, -gmbs);
                ctx.add(source, bulk, -gmbs);
                ctx.add(source, source, gmbs);
            }

            // Stamp equivalent current source for Ids
            ctx.add_rhs(drain, -ieq);
            ctx.add_rhs(source, ieq);

            // BSIM4: Substrate current (impact ionization)
            // Isub flows from drain to bulk
            if output.isub.abs() > gmin && output.gsub > gmin * 0.01 {
                // Stamp gsub (substrate conductance)
                ctx.add(drain, drain, output.gsub);
                ctx.add(bulk, bulk, output.gsub);
                ctx.add(drain, bulk, -output.gsub);
                ctx.add(bulk, drain, -output.gsub);

                // Equivalent current for Isub
                let isub_eq = output.isub - output.gsub * (vd - vb);
                ctx.add_rhs(drain, -isub_eq);
                ctx.add_rhs(bulk, isub_eq);
            }

            // BSIM4: Gate tunneling currents
            // Igs flows from gate to source
            if output.igs.abs() > gmin && output.gigs > gmin * 0.01 {
                ctx.add(gate, gate, output.gigs);
                ctx.add(source, source, output.gigs);
                ctx.add(gate, source, -output.gigs);
                ctx.add(source, gate, -output.gigs);

                let igs_eq = output.igs - output.gigs * (vg - vs);
                ctx.add_rhs(gate, -igs_eq);
                ctx.add_rhs(source, igs_eq);
            }

            // Igd flows from gate to drain
            if output.igd.abs() > gmin && output.gigd > gmin * 0.01 {
                ctx.add(gate, gate, output.gigd);
                ctx.add(drain, drain, output.gigd);
                ctx.add(gate, drain, -output.gigd);
                ctx.add(drain, gate, -output.gigd);

                let igd_eq = output.igd - output.gigd * (vg - vd);
                ctx.add_rhs(gate, -igd_eq);
                ctx.add_rhs(drain, igd_eq);
            }

            return Ok(());
        }

        // BSIM3 or Level 1: Use standard evaluator
        let output = sim_devices::bsim::evaluate_mos(
            &params, w, l, vd, vg, vs, vb, temp
        );

        let gm = output.gm;
        let gds = output.gds.max(gmin);
        let gmbs = output.gmbs;
        let ieq = output.ieq;

        // Stamp gds (output conductance between drain and source)
        ctx.add(drain, drain, gds);
        ctx.add(source, source, gds);
        ctx.add(drain, source, -gds);
        ctx.add(source, drain, -gds);

        // Stamp gm (transconductance: current controlled by Vgs)
        ctx.add(drain, gate, gm);
        ctx.add(drain, source, -gm);
        ctx.add(source, gate, -gm);
        ctx.add(source, source, gm);

        // Stamp gmbs (body transconductance: current controlled by Vbs)
        if gmbs.abs() > gmin * 0.01 {
            ctx.add(drain, bulk, gmbs);
            ctx.add(drain, source, -gmbs);
            ctx.add(source, bulk, -gmbs);
            ctx.add(source, source, gmbs);
        }

        // Stamp equivalent current source
        ctx.add_rhs(drain, -ieq);
        ctx.add_rhs(source, ieq);
        return Ok(());
    }

    // Initial guess: add small conductance for convergence
    ctx.add(drain, drain, gmin);
    ctx.add(source, source, gmin);
    ctx.add(drain, source, -gmin);
    ctx.add(source, drain, -gmin);
    Ok(())
}

pub fn debug_dump_stamp(instance: &Instance) {
    println!(
        "stamp: name={} kind={:?} nodes={} value={:?}",
        instance.name,
        instance.kind,
        instance.nodes.len(),
        instance.value
    );
}

/// Update transient state after an accepted time step.
///
/// This function stores the current solution values as history
/// for the next time step's integration.
///
/// For Backward Euler, only voltage/current history is needed.
/// For Trapezoidal, both voltage and current history are needed.
pub fn update_transient_state(instances: &[Instance], x: &[f64], state: &mut TransientState) {
    for inst in instances {
        match inst.kind {
            DeviceKind::C => {
                if inst.nodes.len() == 2 {
                    let a = inst.nodes[0].0;
                    let b = inst.nodes[1].0;
                    let va = x.get(a).copied().unwrap_or(0.0);
                    let vb = x.get(b).copied().unwrap_or(0.0);
                    state.cap_voltage.insert(inst.name.clone(), va - vb);
                }
            }
            DeviceKind::L => {
                if let Some(aux) = state.ind_aux.get(&inst.name) {
                    if let Some(current) = x.get(*aux).copied() {
                        state.ind_current.insert(inst.name.clone(), current);
                    }
                }
                // Store voltage for Trapezoidal
                if inst.nodes.len() == 2 {
                    let a = inst.nodes[0].0;
                    let b = inst.nodes[1].0;
                    let va = x.get(a).copied().unwrap_or(0.0);
                    let vb = x.get(b).copied().unwrap_or(0.0);
                    state.ind_voltage.insert(inst.name.clone(), va - vb);
                }
            }
            _ => {}
        }
    }
}

/// Update transient state with current computation for Trapezoidal method.
///
/// This extended version also computes and stores the capacitor currents,
/// which are needed for Trapezoidal integration.
///
/// # Arguments
/// * `instances` - Circuit instances
/// * `x` - Current solution vector
/// * `x_prev` - Previous solution vector (for current computation)
/// * `dt` - Time step used
/// * `state` - Transient state to update
pub fn update_transient_state_full(
    instances: &[Instance],
    x: &[f64],
    _x_prev: &[f64],
    dt: f64,
    state: &mut TransientState,
) {
    for inst in instances {
        match inst.kind {
            DeviceKind::C => {
                if inst.nodes.len() == 2 {
                    let a = inst.nodes[0].0;
                    let b = inst.nodes[1].0;
                    let va = x.get(a).copied().unwrap_or(0.0);
                    let vb = x.get(b).copied().unwrap_or(0.0);
                    let v = va - vb;

                    // Get capacitance
                    let c = inst
                        .value
                        .as_deref()
                        .and_then(parse_number_with_suffix)
                        .unwrap_or(1e-12);

                    // Compute current: I = C * dV/dt
                    let v_prev = *state.cap_voltage.get(&inst.name).unwrap_or(&v);
                    let i = c * (v - v_prev) / dt;

                    // Update state
                    state.cap_voltage.insert(inst.name.clone(), v);
                    state.cap_current.insert(inst.name.clone(), i);
                }
            }
            DeviceKind::L => {
                // Store current from auxiliary variable
                if let Some(aux) = state.ind_aux.get(&inst.name) {
                    if let Some(current) = x.get(*aux).copied() {
                        state.ind_current.insert(inst.name.clone(), current);
                    }
                }

                // Store voltage for Trapezoidal
                if inst.nodes.len() == 2 {
                    let a = inst.nodes[0].0;
                    let b = inst.nodes[1].0;
                    let va = x.get(a).copied().unwrap_or(0.0);
                    let vb = x.get(b).copied().unwrap_or(0.0);
                    state.ind_voltage.insert(inst.name.clone(), va - vb);
                }
            }
            _ => {}
        }
    }
}

// ============================================================================
// Integration Method Selection for Transient Analysis
// ============================================================================

/// Integration method for transient analysis.
///
/// # Backward Euler (BE) - 1st Order
/// - Unconditionally stable (A-stable)
/// - Introduces numerical damping (good for stiff systems)
/// - LTE ∝ dt² (lower accuracy)
///
/// # Trapezoidal Rule (TRAP) - 2nd Order
/// - Also A-stable
/// - No numerical damping (preserves oscillations)
/// - LTE ∝ dt³ (higher accuracy)
/// - Can cause numerical ringing on discontinuities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IntegrationMethod {
    /// Backward Euler (1st order, more stable)
    #[default]
    BackwardEuler,
    /// Trapezoidal rule (2nd order, more accurate)
    Trapezoidal,
}

/// Transient state for time-domain simulation.
///
/// Stores history values needed for numerical integration of
/// capacitors and inductors.
#[derive(Debug, Default, Clone)]
pub struct TransientState {
    // === Capacitor state ===
    /// Previous voltage across each capacitor: V_{n-1}
    pub cap_voltage: HashMap<String, f64>,
    /// Previous current through each capacitor: I_{n-1} (for Trapezoidal)
    pub cap_current: HashMap<String, f64>,

    // === Inductor state ===
    /// Previous current through each inductor: I_{n-1}
    pub ind_current: HashMap<String, f64>,
    /// Previous voltage across each inductor: V_{n-1} (for Trapezoidal)
    pub ind_voltage: HashMap<String, f64>,
    /// Auxiliary variable indices for inductor currents
    pub ind_aux: HashMap<String, usize>,

    // === Integration method ===
    /// Current integration method
    pub method: IntegrationMethod,
}

fn parse_number_with_suffix(token: &str) -> Option<f64> {
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

    if let Ok(num) = num_str.parse::<f64>() {
        Some(num * multiplier)
    } else {
        None
    }
}

fn param_value(params: &HashMap<String, String>, keys: &[&str]) -> Option<f64> {
    for key in keys {
        let key = key.to_ascii_lowercase();
        if let Some(value) = params.get(&key) {
            if let Some(num) = parse_number_with_suffix(value).or_else(|| value.parse().ok()) {
                return Some(num);
            }
        }
    }
    None
}

fn stamp_capacitor_tran(
    ctx: &mut StampContext,
    inst: &Instance,
    x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let c = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let g = c / dt;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    let v_prev = *state.cap_voltage.get(&inst.name).unwrap_or(&0.0);
    let ieq = g * v_prev;
    ctx.add(a, a, g);
    ctx.add(b, b, g);
    ctx.add(a, b, -g);
    ctx.add(b, a, -g);
    ctx.add_rhs(a, -ieq);
    ctx.add_rhs(b, ieq);
    let _ = x;
    Ok(())
}

fn stamp_inductor_tran(
    ctx: &mut StampContext,
    inst: &Instance,
    x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let l = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    let k = *state
        .ind_aux
        .entry(inst.name.clone())
        .or_insert_with(|| ctx.allocate_aux(&inst.name));
    let g = -(l / dt);
    let i_prev = *state.ind_current.get(&inst.name).unwrap_or(&0.0);
    ctx.add(a, k, 1.0);
    ctx.add(b, k, -1.0);
    ctx.add(k, a, 1.0);
    ctx.add(k, b, -1.0);
    ctx.add(k, k, g);
    ctx.add_rhs(k, g * i_prev);
    let _ = x;
    Ok(())
}

fn stamp_inductor_dc(ctx: &mut StampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let gshort = 1e9;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add(a, a, gshort);
    ctx.add(b, b, gshort);
    ctx.add(a, b, -gshort);
    ctx.add(b, a, -gshort);
    Ok(())
}

// ============================================================================
// Trapezoidal Integration Stamp Functions
// ============================================================================

/// Capacitor stamp using Trapezoidal integration (2nd order).
///
/// # Derivation
///
/// The capacitor constitutive equation is:
/// ```text
/// I = C * dV/dt
/// ```
///
/// Trapezoidal approximation:
/// ```text
/// (V_n - V_{n-1}) / dt = (I_n + I_{n-1}) / (2C)
/// ```
///
/// Rearranging:
/// ```text
/// I_n = (2C/dt) * V_n - (2C/dt) * V_{n-1} - I_{n-1}
/// ```
///
/// This gives an equivalent circuit of:
/// - Conductance: g = 2C/dt (parallel resistor)
/// - Equivalent current: I_eq = g * V_{n-1} + I_{n-1}
///
/// # Comparison with Backward Euler
///
/// | Method | Conductance | History Term |
/// |--------|-------------|--------------|
/// | BE     | g = C/dt    | I_eq = g * V_{n-1} |
/// | Trap   | g = 2C/dt   | I_eq = g * V_{n-1} + I_{n-1} |
///
/// Trapezoidal has 2x the conductance and includes current history.
fn stamp_capacitor_trap(
    ctx: &mut StampContext,
    inst: &Instance,
    _x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }

    let c = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    // Trapezoidal: g = 2C/dt
    let g = 2.0 * c / dt;

    // Get history values
    let v_prev = *state.cap_voltage.get(&inst.name).unwrap_or(&0.0);
    let i_prev = *state.cap_current.get(&inst.name).unwrap_or(&0.0);

    // Equivalent current: I_eq = g * V_{n-1} + I_{n-1}
    let ieq = g * v_prev + i_prev;

    // Stamp conductance matrix (same pattern as resistor)
    ctx.add(a, a, g);
    ctx.add(b, b, g);
    ctx.add(a, b, -g);
    ctx.add(b, a, -g);

    // Stamp RHS with history term
    // Current flows from a to b, so:
    // KCL at a: -I_cap = -g*(Va-Vb) + I_eq  =>  add I_eq to RHS_a
    // KCL at b: +I_cap = +g*(Va-Vb) - I_eq  =>  add -I_eq to RHS_b
    ctx.add_rhs(a, ieq);
    ctx.add_rhs(b, -ieq);

    Ok(())
}

/// Inductor stamp using Trapezoidal integration (2nd order).
///
/// # Derivation
///
/// The inductor constitutive equation is:
/// ```text
/// V = L * dI/dt
/// ```
///
/// Trapezoidal approximation:
/// ```text
/// (I_n - I_{n-1}) / dt = (V_n + V_{n-1}) / (2L)
/// ```
///
/// Rearranging:
/// ```text
/// I_n = (dt/2L) * V_n + (dt/2L) * V_{n-1} + I_{n-1}
/// ```
///
/// Using auxiliary variable k for inductor current:
/// ```text
/// V_a - V_b = (2L/dt) * I_k - (2L/dt) * I_{n-1} - V_{n-1}
/// ```
///
/// # Equivalent Circuit
/// - Resistance term: R_eq = 2L/dt
/// - Voltage source: V_eq = R_eq * I_{n-1} + V_{n-1}
fn stamp_inductor_trap(
    ctx: &mut StampContext,
    inst: &Instance,
    _x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }

    let l = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    // Get or allocate auxiliary variable for inductor current
    let k = *state
        .ind_aux
        .entry(inst.name.clone())
        .or_insert_with(|| ctx.allocate_aux(&inst.name));

    // Trapezoidal: R_eq = 2L/dt, g_eq = dt/(2L)
    let r_eq = 2.0 * l / dt;

    // Get history values
    let i_prev = *state.ind_current.get(&inst.name).unwrap_or(&0.0);
    let v_prev = *state.ind_voltage.get(&inst.name).unwrap_or(&0.0);

    // Voltage source equation: V_a - V_b - R_eq * I_k = -R_eq * I_{n-1} - V_{n-1}
    // Rearranged: V_a - V_b - R_eq * I_k = -(R_eq * I_{n-1} + V_{n-1})

    // KCL at nodes: current flows from a to b through inductor
    ctx.add(a, k, 1.0);
    ctx.add(b, k, -1.0);

    // Constitutive relation row
    ctx.add(k, a, 1.0);
    ctx.add(k, b, -1.0);
    ctx.add(k, k, -r_eq);

    // RHS: -(R_eq * I_{n-1} + V_{n-1}) = -R_eq * I_{n-1} - V_{n-1}
    let rhs = -r_eq * i_prev - v_prev;
    ctx.add_rhs(k, rhs);

    Ok(())
}

/// Stamp capacitor for transient analysis using selected integration method.
pub fn stamp_capacitor_tran_method(
    ctx: &mut StampContext,
    inst: &Instance,
    x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
    method: IntegrationMethod,
) -> Result<(), StampError> {
    match method {
        IntegrationMethod::BackwardEuler => stamp_capacitor_tran(ctx, inst, x, dt, state),
        IntegrationMethod::Trapezoidal => stamp_capacitor_trap(ctx, inst, x, dt, state),
    }
}

/// Stamp inductor for transient analysis using selected integration method.
pub fn stamp_inductor_tran_method(
    ctx: &mut StampContext,
    inst: &Instance,
    x: Option<&[f64]>,
    dt: f64,
    state: &mut TransientState,
    method: IntegrationMethod,
) -> Result<(), StampError> {
    match method {
        IntegrationMethod::BackwardEuler => stamp_inductor_tran(ctx, inst, x, dt, state),
        IntegrationMethod::Trapezoidal => stamp_inductor_trap(ctx, inst, x, dt, state),
    }
}

/// Voltage Controlled Voltage Source (VCVS)
/// Vout = E * Vin where E is the gain
/// nodes: [out+, out-, in+, in-] for simple case
/// nodes: [out+, out-] with POLY for polynomial case
fn stamp_vcvs(ctx: &mut StampContext, inst: &Instance, x: Option<&[f64]>) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_vcvs_poly(ctx, inst, poly, x);
    }

    // Simple linear case: Vout = gain * Vin
    if inst.nodes.len() != 4 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let in_p = inst.nodes[2].0;
    let in_n = inst.nodes[3].0;

    // Allocate auxiliary variable for output current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at output nodes: I flows from out+ to out-
    ctx.add(out_p, k, 1.0);
    ctx.add(out_n, k, -1.0);

    // Constitutive relation: V(out+) - V(out-) = E * (V(in+) - V(in-))
    ctx.add(k, out_p, 1.0);
    ctx.add(k, out_n, -1.0);
    ctx.add(k, in_p, -gain);
    ctx.add(k, in_n, gain);

    Ok(())
}

/// VCVS with POLY syntax
/// Vout = f(Vin1, Vin2, ...) where f is a polynomial
fn stamp_vcvs_poly(
    ctx: &mut StampContext,
    inst: &Instance,
    poly: &PolySpec,
    x: Option<&[f64]>,
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Allocate auxiliary variable for output current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at output nodes
    ctx.add(out_p, k, 1.0);
    ctx.add(out_n, k, -1.0);

    // Get control voltages from solution vector
    let mut control_voltages: Vec<f64> = Vec::with_capacity(poly.degree);
    if let Some(x) = x {
        for &(pos, neg) in &poly.control_nodes {
            let v_pos = x.get(pos).copied().unwrap_or(0.0);
            let v_neg = x.get(neg).copied().unwrap_or(0.0);
            control_voltages.push(v_pos - v_neg);
        }
    } else {
        control_voltages.resize(poly.degree, 0.0);
    }

    // Evaluate polynomial and derivatives
    let (f_value, derivs) = evaluate_poly(poly, &control_voltages);

    // Check if purely linear (only c0 and c1..cn terms, no higher order)
    let is_linear = poly.coeffs.len() <= poly.degree + 1;

    if is_linear {
        // Linear case: V(out) = c0 + c1*V1 + c2*V2 + ...
        // Stamp directly into matrix
        ctx.add(k, out_p, 1.0);
        ctx.add(k, out_n, -1.0);

        // Stamp the constant term
        let c0 = poly.coeffs.first().copied().unwrap_or(0.0);
        ctx.add_rhs(k, c0);

        // Stamp the linear coefficients
        for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
            let c = poly.coeffs.get(i + 1).copied().unwrap_or(0.0);
            ctx.add(k, pos, -c);
            ctx.add(k, neg, c);
        }
    } else {
        // Nonlinear case: use Newton-Raphson linearization
        // f(x) ≈ f(x0) + f'(x0) * (x - x0)
        // V(out) = f(V1, V2, ...) is linearized as:
        // V(out) = f(V1_0, V2_0, ...) + df/dV1 * (V1 - V1_0) + df/dV2 * (V2 - V2_0) + ...
        // Rearranging: V(out) - df/dV1 * V1 - df/dV2 * V2 - ... = f(x0) - df/dV1 * V1_0 - ...

        ctx.add(k, out_p, 1.0);
        ctx.add(k, out_n, -1.0);

        // Stamp derivatives as coefficients for control nodes
        for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                ctx.add(k, pos, -deriv);
                ctx.add(k, neg, deriv);
            }
        }

        // RHS: f(x0) - sum(df/dVi * Vi_0)
        let mut rhs = f_value;
        for (i, &v) in control_voltages.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                rhs -= deriv * v;
            }
        }
        ctx.add_rhs(k, rhs);
    }

    Ok(())
}

/// Voltage Controlled Current Source (VCCS)
/// Iout = G * Vin where G is the transconductance
/// nodes: [out+, out-, in+, in-] for simple case
/// nodes: [out+, out-] with POLY for polynomial case
fn stamp_vccs(ctx: &mut StampContext, inst: &Instance, x: Option<&[f64]>) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_vccs_poly(ctx, inst, poly, x);
    }

    // Simple linear case
    if inst.nodes.len() != 4 {
        return Err(StampError::InvalidNodes);
    }
    let gm = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let in_p = inst.nodes[2].0;
    let in_n = inst.nodes[3].0;

    // Current flows from out+ to out-, controlled by V(in+) - V(in-)
    // I = G * (V(in+) - V(in-))
    ctx.add(out_p, in_p, gm);
    ctx.add(out_p, in_n, -gm);
    ctx.add(out_n, in_p, -gm);
    ctx.add(out_n, in_n, gm);

    Ok(())
}

/// VCCS with POLY syntax
/// Iout = f(Vin1, Vin2, ...) where f is a polynomial
fn stamp_vccs_poly(
    ctx: &mut StampContext,
    inst: &Instance,
    poly: &PolySpec,
    x: Option<&[f64]>,
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get control voltages from solution vector
    let mut control_voltages: Vec<f64> = Vec::with_capacity(poly.degree);
    if let Some(x) = x {
        for &(pos, neg) in &poly.control_nodes {
            let v_pos = x.get(pos).copied().unwrap_or(0.0);
            let v_neg = x.get(neg).copied().unwrap_or(0.0);
            control_voltages.push(v_pos - v_neg);
        }
    } else {
        control_voltages.resize(poly.degree, 0.0);
    }

    // Evaluate polynomial and derivatives
    let (f_value, derivs) = evaluate_poly(poly, &control_voltages);

    // Check if purely linear
    let is_linear = poly.coeffs.len() <= poly.degree + 1;

    if is_linear {
        // Linear case: I = c0 + c1*V1 + c2*V2 + ...
        // Stamp constant term as equivalent current source
        let c0 = poly.coeffs.first().copied().unwrap_or(0.0);
        ctx.add_rhs(out_p, -c0);
        ctx.add_rhs(out_n, c0);

        // Stamp the linear coefficients (transconductances)
        for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
            let gm = poly.coeffs.get(i + 1).copied().unwrap_or(0.0);
            ctx.add(out_p, pos, gm);
            ctx.add(out_p, neg, -gm);
            ctx.add(out_n, pos, -gm);
            ctx.add(out_n, neg, gm);
        }
    } else {
        // Nonlinear case: Newton-Raphson linearization
        // I = f(V1, V2, ...) linearized as:
        // I = f(x0) + df/dV1 * (V1 - V1_0) + df/dV2 * (V2 - V2_0) + ...

        // Stamp derivatives as transconductances
        for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                ctx.add(out_p, pos, deriv);
                ctx.add(out_p, neg, -deriv);
                ctx.add(out_n, pos, -deriv);
                ctx.add(out_n, neg, deriv);
            }
        }

        // Equivalent current source: I_eq = f(x0) - sum(df/dVi * Vi_0)
        let mut i_eq = f_value;
        for (i, &v) in control_voltages.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                i_eq -= deriv * v;
            }
        }
        ctx.add_rhs(out_p, -i_eq);
        ctx.add_rhs(out_n, i_eq);
    }

    Ok(())
}

/// Current Controlled Current Source (CCCS)
/// Iout = F * Icontrol where F is the current gain
/// nodes: [out+, out-], control: name of controlling voltage source
fn stamp_cccs(ctx: &mut StampContext, inst: &Instance, x: Option<&[f64]>) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_cccs_poly(ctx, inst, poly, x);
    }

    // Simple linear case
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get the controlling voltage source's auxiliary variable
    let control_name = inst.control.as_ref().ok_or(StampError::MissingValue)?;
    let control_aux = ctx
        .aux
        .name_to_id
        .get(control_name)
        .copied()
        .ok_or(StampError::MissingValue)?;
    let k_control = ctx.node_count + control_aux;

    // Current flows from out+ to out-, controlled by current through controlling source
    // I = F * I_control
    ctx.add(out_p, k_control, gain);
    ctx.add(out_n, k_control, -gain);

    Ok(())
}

/// CCCS with POLY syntax
/// Iout = f(I1, I2, ...) where f is a polynomial of control currents
fn stamp_cccs_poly(
    ctx: &mut StampContext,
    inst: &Instance,
    poly: &PolySpec,
    x: Option<&[f64]>,
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get auxiliary variable indices for control sources
    let mut control_aux_indices: Vec<usize> = Vec::with_capacity(poly.degree);
    for source_name in &poly.control_sources {
        if let Some(&aux_id) = ctx.aux.name_to_id.get(source_name) {
            control_aux_indices.push(ctx.node_count + aux_id);
        } else {
            // Control source not yet stamped - this shouldn't happen if ordering is correct
            return Err(StampError::MissingValue);
        }
    }

    // Get control currents from solution vector
    let mut control_currents: Vec<f64> = Vec::with_capacity(poly.degree);
    if let Some(x) = x {
        for &aux_idx in &control_aux_indices {
            let i = x.get(aux_idx).copied().unwrap_or(0.0);
            control_currents.push(i);
        }
    } else {
        control_currents.resize(poly.degree, 0.0);
    }

    // Evaluate polynomial and derivatives
    let (f_value, derivs) = evaluate_poly(poly, &control_currents);

    // Check if purely linear
    let is_linear = poly.coeffs.len() <= poly.degree + 1;

    if is_linear {
        // Linear case: I = c0 + c1*I1 + c2*I2 + ...
        let c0 = poly.coeffs.first().copied().unwrap_or(0.0);
        ctx.add_rhs(out_p, -c0);
        ctx.add_rhs(out_n, c0);

        // Stamp linear coefficients (current gains)
        for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
            let gain = poly.coeffs.get(i + 1).copied().unwrap_or(0.0);
            ctx.add(out_p, aux_idx, gain);
            ctx.add(out_n, aux_idx, -gain);
        }
    } else {
        // Nonlinear case: Newton-Raphson linearization
        // Stamp derivatives as gains
        for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                ctx.add(out_p, aux_idx, deriv);
                ctx.add(out_n, aux_idx, -deriv);
            }
        }

        // Equivalent current source
        let mut i_eq = f_value;
        for (i, &curr) in control_currents.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                i_eq -= deriv * curr;
            }
        }
        ctx.add_rhs(out_p, -i_eq);
        ctx.add_rhs(out_n, i_eq);
    }

    Ok(())
}

/// Current Controlled Voltage Source (CCVS)
/// Vout = H * Icontrol where H is the transresistance
/// nodes: [out+, out-], control: name of controlling voltage source
fn stamp_ccvs(ctx: &mut StampContext, inst: &Instance, x: Option<&[f64]>) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_ccvs_poly(ctx, inst, poly, x);
    }

    // Simple linear case
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get the controlling voltage source's auxiliary variable
    let control_name = inst.control.as_ref().ok_or(StampError::MissingValue)?;
    let control_aux = ctx
        .aux
        .name_to_id
        .get(control_name)
        .copied()
        .ok_or(StampError::MissingValue)?;
    let k_control = ctx.node_count + control_aux;

    // Allocate auxiliary variable for output current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at output nodes
    ctx.add(out_p, k, 1.0);
    ctx.add(out_n, k, -1.0);

    // Constitutive relation: V(out+) - V(out-) = H * I_control
    ctx.add(k, out_p, 1.0);
    ctx.add(k, out_n, -1.0);
    ctx.add(k, k_control, -gain);

    Ok(())
}

/// CCVS with POLY syntax
/// Vout = f(I1, I2, ...) where f is a polynomial of control currents
fn stamp_ccvs_poly(
    ctx: &mut StampContext,
    inst: &Instance,
    poly: &PolySpec,
    x: Option<&[f64]>,
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Allocate auxiliary variable for output current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at output nodes
    ctx.add(out_p, k, 1.0);
    ctx.add(out_n, k, -1.0);

    // Get auxiliary variable indices for control sources
    let mut control_aux_indices: Vec<usize> = Vec::with_capacity(poly.degree);
    for source_name in &poly.control_sources {
        if let Some(&aux_id) = ctx.aux.name_to_id.get(source_name) {
            control_aux_indices.push(ctx.node_count + aux_id);
        } else {
            return Err(StampError::MissingValue);
        }
    }

    // Get control currents from solution vector
    let mut control_currents: Vec<f64> = Vec::with_capacity(poly.degree);
    if let Some(x) = x {
        for &aux_idx in &control_aux_indices {
            let i = x.get(aux_idx).copied().unwrap_or(0.0);
            control_currents.push(i);
        }
    } else {
        control_currents.resize(poly.degree, 0.0);
    }

    // Evaluate polynomial and derivatives
    let (f_value, derivs) = evaluate_poly(poly, &control_currents);

    // Check if purely linear
    let is_linear = poly.coeffs.len() <= poly.degree + 1;

    if is_linear {
        // Linear case: V(out) = c0 + c1*I1 + c2*I2 + ...
        ctx.add(k, out_p, 1.0);
        ctx.add(k, out_n, -1.0);

        // Stamp constant term
        let c0 = poly.coeffs.first().copied().unwrap_or(0.0);
        ctx.add_rhs(k, c0);

        // Stamp linear coefficients (transresistances)
        for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
            let h = poly.coeffs.get(i + 1).copied().unwrap_or(0.0);
            ctx.add(k, aux_idx, -h);
        }
    } else {
        // Nonlinear case: Newton-Raphson linearization
        ctx.add(k, out_p, 1.0);
        ctx.add(k, out_n, -1.0);

        // Stamp derivatives as transresistances
        for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                ctx.add(k, aux_idx, -deriv);
            }
        }

        // RHS: f(x0) - sum(df/dIi * Ii_0)
        let mut rhs = f_value;
        for (i, &curr) in control_currents.iter().enumerate() {
            if let Some(&deriv) = derivs.get(i) {
                rhs -= deriv * curr;
            }
        }
        ctx.add_rhs(k, rhs);
    }

    Ok(())
}

// ============================================================================
// AC Small-Signal Stamping Functions
// ============================================================================

/// Resistor AC stamping: Y = G = 1/R (real admittance)
fn stamp_resistor_ac(ctx: &mut ComplexStampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let value = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let g = 1.0 / value;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add_real(a, a, g);
    ctx.add_real(b, b, g);
    ctx.add_real(a, b, -g);
    ctx.add_real(b, a, -g);
    Ok(())
}

/// Capacitor AC stamping: Y = jωC (imaginary admittance)
fn stamp_capacitor_ac(ctx: &mut ComplexStampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let c = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let y = ctx.omega * c; // jωC has imaginary part ωC
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;
    ctx.add_imag(a, a, y);
    ctx.add_imag(b, b, y);
    ctx.add_imag(a, b, -y);
    ctx.add_imag(b, a, -y);
    Ok(())
}

/// Inductor AC stamping: Y = 1/(jωL) = -j/(ωL)
/// Uses auxiliary variable for inductor current
fn stamp_inductor_ac(ctx: &mut ComplexStampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let l = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    // Allocate auxiliary variable for inductor current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at nodes: current flows from a to b
    ctx.add_real(a, k, 1.0);
    ctx.add_real(b, k, -1.0);

    // Constitutive relation: V(a) - V(b) = jωL * I
    ctx.add_real(k, a, 1.0);
    ctx.add_real(k, b, -1.0);
    ctx.add_imag(k, k, -ctx.omega * l); // -jωL

    Ok(())
}

/// Voltage source AC stamping with AC magnitude and phase
fn stamp_voltage_ac(ctx: &mut ComplexStampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    // Allocate auxiliary variable for source current
    let k = ctx.allocate_aux(&inst.name);

    // KCL at nodes
    ctx.add_real(a, k, 1.0);
    ctx.add_real(b, k, -1.0);

    // Constitutive relation: V(a) - V(b) = Vac
    ctx.add_real(k, a, 1.0);
    ctx.add_real(k, b, -1.0);

    // AC excitation: Vac = ac_mag * exp(j * ac_phase)
    let ac_mag = inst.ac_mag.unwrap_or(0.0);
    let ac_phase_deg = inst.ac_phase.unwrap_or(0.0);
    let ac_phase_rad = ac_phase_deg * std::f64::consts::PI / 180.0;
    let vac = Complex64::from_polar(ac_mag, ac_phase_rad);
    ctx.add_rhs(k, vac);

    Ok(())
}

/// Current source AC stamping with AC magnitude and phase
fn stamp_current_ac(ctx: &mut ComplexStampContext, inst: &Instance) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    // AC excitation: Iac = ac_mag * exp(j * ac_phase)
    let ac_mag = inst.ac_mag.unwrap_or(0.0);
    let ac_phase_deg = inst.ac_phase.unwrap_or(0.0);
    let ac_phase_rad = ac_phase_deg * std::f64::consts::PI / 180.0;
    let iac = Complex64::from_polar(ac_mag, ac_phase_rad);

    // Current flows from a to b (out of a, into b)
    ctx.add_rhs(a, -iac);
    ctx.add_rhs(b, iac);

    Ok(())
}

/// Diode AC stamping: linearized small-signal conductance from DC operating point
fn stamp_diode_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let a = inst.nodes[0].0;
    let b = inst.nodes[1].0;

    let gmin = 1e-12;
    let isat = param_value(&inst.params, &["is"]).unwrap_or(1e-14);
    let emission = param_value(&inst.params, &["n", "nj"]).unwrap_or(1.0);
    let vt = 0.02585 * emission;

    let va = dc_solution.get(a).copied().unwrap_or(0.0);
    let vb = dc_solution.get(b).copied().unwrap_or(0.0);
    let vd = va - vb;

    // Small-signal conductance gd = dId/dVd = (Is/Vt) * exp(Vd/Vt)
    let exp_vd = (vd / vt).exp();
    let gd = (isat / vt) * exp_vd;
    let g = gd.max(gmin);

    ctx.add_real(a, a, g);
    ctx.add_real(b, b, g);
    ctx.add_real(a, b, -g);
    ctx.add_real(b, a, -g);

    Ok(())
}

/// MOSFET AC stamping: linearized small-signal model from DC operating point
fn stamp_mos_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() < 4 {
        return Err(StampError::InvalidNodes);
    }
    let drain = inst.nodes[0].0;
    let gate = inst.nodes[1].0;
    let source = inst.nodes[2].0;
    let bulk = inst.nodes[3].0;
    let gmin = 1e-12;

    // Parse model level
    let level = param_value(&inst.params, &["level"]).unwrap_or(49.0) as u32;

    // Determine NMOS/PMOS
    let is_pmos = if let Some(t) = inst.params.get("type") {
        let t_lower = t.to_ascii_lowercase();
        t_lower.contains("pmos") || t_lower == "p"
    } else if inst.params.contains_key("pmos") {
        true
    } else {
        false
    };

    // Build BSIM parameters
    let params = sim_devices::bsim::build_bsim_params(&inst.params, level, is_pmos);

    let w = param_value(&inst.params, &["w"]).unwrap_or(1e-6);
    let l = param_value(&inst.params, &["l"]).unwrap_or(1e-6);
    let temp = param_value(&inst.params, &["temp"]).unwrap_or(300.15);

    let vd = dc_solution.get(drain).copied().unwrap_or(0.0);
    let vg = dc_solution.get(gate).copied().unwrap_or(0.0);
    let vs = dc_solution.get(source).copied().unwrap_or(0.0);
    let vb = dc_solution.get(bulk).copied().unwrap_or(0.0);

    // Get small-signal parameters from DC operating point
    let output = sim_devices::bsim::evaluate_mos(&params, w, l, vd, vg, vs, vb, temp);

    let gm = output.gm;
    let gds = output.gds.max(gmin);
    let gmbs = output.gmbs;

    // Stamp gds (output conductance between drain and source)
    ctx.add_real(drain, drain, gds);
    ctx.add_real(source, source, gds);
    ctx.add_real(drain, source, -gds);
    ctx.add_real(source, drain, -gds);

    // Stamp gm (transconductance: current controlled by Vgs)
    ctx.add_real(drain, gate, gm);
    ctx.add_real(drain, source, -gm);
    ctx.add_real(source, gate, -gm);
    ctx.add_real(source, source, gm);

    // Stamp gmbs (body transconductance: current controlled by Vbs)
    if gmbs.abs() > gmin * 0.01 {
        ctx.add_real(drain, bulk, gmbs);
        ctx.add_real(drain, source, -gmbs);
        ctx.add_real(source, bulk, -gmbs);
        ctx.add_real(source, source, gmbs);
    }

    Ok(())
}

/// VCVS AC stamping (frequency-independent)
/// For POLY, uses linearized small-signal model from DC operating point
fn stamp_vcvs_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_vcvs_poly_ac(ctx, inst, poly, dc_solution);
    }

    // Simple linear case
    if inst.nodes.len() != 4 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let in_p = inst.nodes[2].0;
    let in_n = inst.nodes[3].0;

    let k = ctx.allocate_aux(&inst.name);

    ctx.add_real(out_p, k, 1.0);
    ctx.add_real(out_n, k, -1.0);
    ctx.add_real(k, out_p, 1.0);
    ctx.add_real(k, out_n, -1.0);
    ctx.add_real(k, in_p, -gain);
    ctx.add_real(k, in_n, gain);

    Ok(())
}

/// VCVS POLY AC stamping - linearized around DC operating point
fn stamp_vcvs_poly_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    poly: &PolySpec,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let k = ctx.allocate_aux(&inst.name);

    ctx.add_real(out_p, k, 1.0);
    ctx.add_real(out_n, k, -1.0);
    ctx.add_real(k, out_p, 1.0);
    ctx.add_real(k, out_n, -1.0);

    // Get control voltages from DC solution
    let mut control_voltages: Vec<f64> = Vec::with_capacity(poly.degree);
    for &(pos, neg) in &poly.control_nodes {
        let v_pos = dc_solution.get(pos).copied().unwrap_or(0.0);
        let v_neg = dc_solution.get(neg).copied().unwrap_or(0.0);
        control_voltages.push(v_pos - v_neg);
    }

    // Evaluate derivatives at DC operating point
    let (_, derivs) = evaluate_poly(poly, &control_voltages);

    // Stamp the linearized gains
    for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
        let gain = derivs.get(i).copied().unwrap_or(0.0);
        ctx.add_real(k, pos, -gain);
        ctx.add_real(k, neg, gain);
    }

    Ok(())
}

/// VCCS AC stamping (frequency-independent)
fn stamp_vccs_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_vccs_poly_ac(ctx, inst, poly, dc_solution);
    }

    // Simple linear case
    if inst.nodes.len() != 4 {
        return Err(StampError::InvalidNodes);
    }
    let gm = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let in_p = inst.nodes[2].0;
    let in_n = inst.nodes[3].0;

    ctx.add_real(out_p, in_p, gm);
    ctx.add_real(out_p, in_n, -gm);
    ctx.add_real(out_n, in_p, -gm);
    ctx.add_real(out_n, in_n, gm);

    Ok(())
}

/// VCCS POLY AC stamping - linearized around DC operating point
fn stamp_vccs_poly_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    poly: &PolySpec,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get control voltages from DC solution
    let mut control_voltages: Vec<f64> = Vec::with_capacity(poly.degree);
    for &(pos, neg) in &poly.control_nodes {
        let v_pos = dc_solution.get(pos).copied().unwrap_or(0.0);
        let v_neg = dc_solution.get(neg).copied().unwrap_or(0.0);
        control_voltages.push(v_pos - v_neg);
    }

    // Evaluate derivatives at DC operating point
    let (_, derivs) = evaluate_poly(poly, &control_voltages);

    // Stamp the linearized transconductances
    for (i, &(pos, neg)) in poly.control_nodes.iter().enumerate() {
        let gm = derivs.get(i).copied().unwrap_or(0.0);
        ctx.add_real(out_p, pos, gm);
        ctx.add_real(out_p, neg, -gm);
        ctx.add_real(out_n, pos, -gm);
        ctx.add_real(out_n, neg, gm);
    }

    Ok(())
}

/// CCCS AC stamping (frequency-independent)
fn stamp_cccs_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_cccs_poly_ac(ctx, inst, poly, dc_solution);
    }

    // Simple linear case
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    let control_name = inst.control.as_ref().ok_or(StampError::MissingValue)?;
    let control_aux = ctx
        .aux
        .name_to_id
        .get(control_name)
        .copied()
        .ok_or(StampError::MissingValue)?;
    let k_control = ctx.node_count + control_aux;

    ctx.add_real(out_p, k_control, gain);
    ctx.add_real(out_n, k_control, -gain);

    Ok(())
}

/// CCCS POLY AC stamping - linearized around DC operating point
fn stamp_cccs_poly_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    poly: &PolySpec,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    // Get auxiliary variable indices for control sources
    let mut control_aux_indices: Vec<usize> = Vec::with_capacity(poly.degree);
    for source_name in &poly.control_sources {
        if let Some(&aux_id) = ctx.aux.name_to_id.get(source_name) {
            control_aux_indices.push(ctx.node_count + aux_id);
        } else {
            return Err(StampError::MissingValue);
        }
    }

    // Get control currents from DC solution
    let mut control_currents: Vec<f64> = Vec::with_capacity(poly.degree);
    for &aux_idx in &control_aux_indices {
        let i = dc_solution.get(aux_idx).copied().unwrap_or(0.0);
        control_currents.push(i);
    }

    // Evaluate derivatives at DC operating point
    let (_, derivs) = evaluate_poly(poly, &control_currents);

    // Stamp the linearized gains
    for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
        let gain = derivs.get(i).copied().unwrap_or(0.0);
        ctx.add_real(out_p, aux_idx, gain);
        ctx.add_real(out_n, aux_idx, -gain);
    }

    Ok(())
}

/// CCVS AC stamping (frequency-independent)
fn stamp_ccvs_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    // Check for POLY syntax
    if let Some(ref poly) = inst.poly {
        return stamp_ccvs_poly_ac(ctx, inst, poly, dc_solution);
    }

    // Simple linear case
    if inst.nodes.len() != 2 {
        return Err(StampError::InvalidNodes);
    }
    let gain = inst
        .value
        .as_deref()
        .and_then(parse_number_with_suffix)
        .ok_or(StampError::MissingValue)?;

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;

    let control_name = inst.control.as_ref().ok_or(StampError::MissingValue)?;
    let control_aux = ctx
        .aux
        .name_to_id
        .get(control_name)
        .copied()
        .ok_or(StampError::MissingValue)?;
    let k_control = ctx.node_count + control_aux;

    let k = ctx.allocate_aux(&inst.name);

    ctx.add_real(out_p, k, 1.0);
    ctx.add_real(out_n, k, -1.0);
    ctx.add_real(k, out_p, 1.0);
    ctx.add_real(k, out_n, -1.0);
    ctx.add_real(k, k_control, -gain);

    Ok(())
}

/// CCVS POLY AC stamping - linearized around DC operating point
fn stamp_ccvs_poly_ac(
    ctx: &mut ComplexStampContext,
    inst: &Instance,
    poly: &PolySpec,
    dc_solution: &[f64],
) -> Result<(), StampError> {
    if inst.nodes.len() < 2 {
        return Err(StampError::InvalidNodes);
    }

    let out_p = inst.nodes[0].0;
    let out_n = inst.nodes[1].0;
    let k = ctx.allocate_aux(&inst.name);

    ctx.add_real(out_p, k, 1.0);
    ctx.add_real(out_n, k, -1.0);
    ctx.add_real(k, out_p, 1.0);
    ctx.add_real(k, out_n, -1.0);

    // Get auxiliary variable indices for control sources
    let mut control_aux_indices: Vec<usize> = Vec::with_capacity(poly.degree);
    for source_name in &poly.control_sources {
        if let Some(&aux_id) = ctx.aux.name_to_id.get(source_name) {
            control_aux_indices.push(ctx.node_count + aux_id);
        } else {
            return Err(StampError::MissingValue);
        }
    }

    // Get control currents from DC solution
    let mut control_currents: Vec<f64> = Vec::with_capacity(poly.degree);
    for &aux_idx in &control_aux_indices {
        let i = dc_solution.get(aux_idx).copied().unwrap_or(0.0);
        control_currents.push(i);
    }

    // Evaluate derivatives at DC operating point
    let (_, derivs) = evaluate_poly(poly, &control_currents);

    // Stamp the linearized transresistances
    for (i, &aux_idx) in control_aux_indices.iter().enumerate() {
        let h = derivs.get(i).copied().unwrap_or(0.0);
        ctx.add_real(k, aux_idx, -h);
    }

    Ok(())
}

// ============================================================================
// Unit Tests for Integration Methods
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::circuit::NodeId;
    use crate::mna::{AuxVarTable, SparseBuilder, StampContext};

    /// Helper to create a test stamp context with sparse builder
    fn make_test_context(node_count: usize) -> (SparseBuilder, Vec<f64>, AuxVarTable) {
        let size = node_count + 10; // Extra space for aux vars
        let builder = SparseBuilder::new(size);
        let rhs = vec![0.0; size];
        let aux = AuxVarTable::new();
        (builder, rhs, aux)
    }

    /// Extract matrix value from sparse builder (sum of entries at position)
    fn get_matrix_value(builder: &SparseBuilder, row: usize, col: usize) -> f64 {
        builder
            .col_entries
            .get(col)
            .map(|entries| {
                entries
                    .iter()
                    .filter(|(r, _)| *r == row)
                    .map(|(_, v)| v)
                    .sum()
            })
            .unwrap_or(0.0)
    }

    /// Helper to create a capacitor instance
    fn make_capacitor(name: &str, node_a: usize, node_b: usize, capacitance: &str) -> Instance {
        Instance {
            name: name.to_string(),
            kind: DeviceKind::C,
            nodes: vec![NodeId(node_a), NodeId(node_b)],
            model: None,
            value: Some(capacitance.to_string()),
            params: HashMap::new(),
            control: None,
            poly: None,
            ac_mag: None,
            ac_phase: None,
        }
    }

    /// Helper to create an inductor instance
    fn make_inductor(name: &str, node_a: usize, node_b: usize, inductance: &str) -> Instance {
        Instance {
            name: name.to_string(),
            kind: DeviceKind::L,
            nodes: vec![NodeId(node_a), NodeId(node_b)],
            model: None,
            value: Some(inductance.to_string()),
            params: HashMap::new(),
            control: None,
            poly: None,
            ac_mag: None,
            ac_phase: None,
        }
    }

    // ========================================================================
    // Capacitor Backward Euler Tests
    // ========================================================================

    #[test]
    fn test_capacitor_be_conductance() {
        // Test: Capacitor with BE method has conductance g = C/dt
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let cap = make_capacitor("C1", 1, 2, "1u"); // 1µF
        let mut state = TransientState::default();
        state.method = IntegrationMethod::BackwardEuler;

        let dt = 1e-6; // 1µs
        stamp_capacitor_tran(&mut ctx, &cap, None, dt, &mut state).unwrap();

        // Expected: g = C/dt = 1e-6 / 1e-6 = 1.0
        let expected_g = 1.0;
        let g11 = get_matrix_value(&builder, 1, 1);
        let g22 = get_matrix_value(&builder, 2, 2);
        let g12 = get_matrix_value(&builder, 1, 2);
        let g21 = get_matrix_value(&builder, 2, 1);

        assert!((g11 - expected_g).abs() < 1e-10, "G(1,1) = {}", g11);
        assert!((g22 - expected_g).abs() < 1e-10, "G(2,2) = {}", g22);
        assert!((g12 + expected_g).abs() < 1e-10, "G(1,2) = {}", g12);
        assert!((g21 + expected_g).abs() < 1e-10, "G(2,1) = {}", g21);
    }

    #[test]
    fn test_capacitor_be_history() {
        // Test: Capacitor BE uses correct history term I_eq = g * V_{n-1}
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let cap = make_capacitor("C1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::BackwardEuler;
        state.cap_voltage.insert("C1".to_string(), 5.0); // V_{n-1} = 5V

        let dt = 1e-6;
        stamp_capacitor_tran(&mut ctx, &cap, None, dt, &mut state).unwrap();

        // Expected: g = 1.0, I_eq = g * V_{n-1} = 1.0 * 5.0 = 5.0
        // RHS: add -I_eq to node a, +I_eq to node b
        let expected_ieq = 5.0;
        assert!((rhs[1] + expected_ieq).abs() < 1e-10, "RHS[1] = {}", rhs[1]);
        assert!((rhs[2] - expected_ieq).abs() < 1e-10, "RHS[2] = {}", rhs[2]);
    }

    // ========================================================================
    // Capacitor Trapezoidal Tests
    // ========================================================================

    #[test]
    fn test_capacitor_trap_conductance() {
        // Test: Capacitor with TRAP method has conductance g = 2C/dt (2x BE)
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let cap = make_capacitor("C1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::Trapezoidal;

        let dt = 1e-6;
        stamp_capacitor_trap(&mut ctx, &cap, None, dt, &mut state).unwrap();

        // Expected: g = 2C/dt = 2 * 1e-6 / 1e-6 = 2.0
        let expected_g = 2.0;
        let g11 = get_matrix_value(&builder, 1, 1);
        let g22 = get_matrix_value(&builder, 2, 2);
        let g12 = get_matrix_value(&builder, 1, 2);
        let g21 = get_matrix_value(&builder, 2, 1);

        assert!((g11 - expected_g).abs() < 1e-10, "G(1,1) = {}", g11);
        assert!((g22 - expected_g).abs() < 1e-10, "G(2,2) = {}", g22);
        assert!((g12 + expected_g).abs() < 1e-10, "G(1,2) = {}", g12);
        assert!((g21 + expected_g).abs() < 1e-10, "G(2,1) = {}", g21);
    }

    #[test]
    fn test_capacitor_trap_history() {
        // Test: Capacitor TRAP uses I_eq = g * V_{n-1} + I_{n-1}
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let cap = make_capacitor("C1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::Trapezoidal;
        state.cap_voltage.insert("C1".to_string(), 5.0); // V_{n-1} = 5V
        state.cap_current.insert("C1".to_string(), 0.5);  // I_{n-1} = 0.5A

        let dt = 1e-6;
        stamp_capacitor_trap(&mut ctx, &cap, None, dt, &mut state).unwrap();

        // Expected: g = 2.0, I_eq = g * V_{n-1} + I_{n-1} = 2.0 * 5.0 + 0.5 = 10.5
        // RHS: add I_eq to node a, -I_eq to node b
        let expected_ieq = 10.5;
        assert!((rhs[1] - expected_ieq).abs() < 1e-10, "RHS[1] = {}", rhs[1]);
        assert!((rhs[2] + expected_ieq).abs() < 1e-10, "RHS[2] = {}", rhs[2]);
    }

    #[test]
    fn test_capacitor_trap_vs_be_ratio() {
        // Test: TRAP conductance is exactly 2x BE conductance
        let (mut builder_be, mut rhs_be, mut aux_be) = make_test_context(3);
        let (mut builder_trap, mut rhs_trap, mut aux_trap) = make_test_context(3);

        let cap = make_capacitor("C1", 1, 2, "10n"); // 10nF
        let dt = 10e-9; // 10ns

        let mut state_be = TransientState::default();
        state_be.method = IntegrationMethod::BackwardEuler;

        let mut state_trap = TransientState::default();
        state_trap.method = IntegrationMethod::Trapezoidal;

        {
            let mut ctx_be = StampContext {
                builder: &mut builder_be,
                rhs: &mut rhs_be,
                aux: &mut aux_be,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_capacitor_tran(&mut ctx_be, &cap, None, dt, &mut state_be).unwrap();
        }

        {
            let mut ctx_trap = StampContext {
                builder: &mut builder_trap,
                rhs: &mut rhs_trap,
                aux: &mut aux_trap,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_capacitor_trap(&mut ctx_trap, &cap, None, dt, &mut state_trap).unwrap();
        }

        // TRAP conductance should be exactly 2x BE
        let g_be = get_matrix_value(&builder_be, 1, 1);
        let g_trap = get_matrix_value(&builder_trap, 1, 1);
        let ratio = g_trap / g_be;
        assert!((ratio - 2.0).abs() < 1e-10, "TRAP/BE ratio = {}", ratio);
    }

    // ========================================================================
    // Inductor Backward Euler Tests
    // ========================================================================

    #[test]
    fn test_inductor_be_stamp() {
        // Test: Inductor BE stamps correct resistance term
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let ind = make_inductor("L1", 1, 2, "1u"); // 1µH
        let mut state = TransientState::default();
        state.method = IntegrationMethod::BackwardEuler;

        let dt = 1e-6;
        stamp_inductor_tran(&mut ctx, &ind, None, dt, &mut state).unwrap();

        // Aux variable is at index 3 (node_count = 3)
        let k = 3;

        // Check KCL stamps
        let g1k = get_matrix_value(&builder, 1, k);
        let g2k = get_matrix_value(&builder, 2, k);
        assert!((g1k - 1.0).abs() < 1e-10, "G(1,k) = {}", g1k);
        assert!((g2k + 1.0).abs() < 1e-10, "G(2,k) = {}", g2k);

        // Check constitutive relation: V_a - V_b - (L/dt)*I_k = -(L/dt)*I_{n-1}
        // g = -L/dt = -1e-6/1e-6 = -1.0
        let expected_g = -1.0;
        let gk1 = get_matrix_value(&builder, k, 1);
        let gk2 = get_matrix_value(&builder, k, 2);
        let gkk = get_matrix_value(&builder, k, k);
        assert!((gk1 - 1.0).abs() < 1e-10, "G(k,1) = {}", gk1);
        assert!((gk2 + 1.0).abs() < 1e-10, "G(k,2) = {}", gk2);
        assert!((gkk - expected_g).abs() < 1e-10, "G(k,k) = {}", gkk);
    }

    #[test]
    fn test_inductor_be_history() {
        // Test: Inductor BE uses correct history term
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let ind = make_inductor("L1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::BackwardEuler;
        state.ind_current.insert("L1".to_string(), 2.0); // I_{n-1} = 2A

        let dt = 1e-6;
        stamp_inductor_tran(&mut ctx, &ind, None, dt, &mut state).unwrap();

        // g = -L/dt = -1.0, RHS = g * I_{n-1} = -1.0 * 2.0 = -2.0
        let k = 3;
        let expected_rhs = -2.0;
        assert!((rhs[k] - expected_rhs).abs() < 1e-10, "RHS[k] = {}", rhs[k]);
    }

    // ========================================================================
    // Inductor Trapezoidal Tests
    // ========================================================================

    #[test]
    fn test_inductor_trap_stamp() {
        // Test: Inductor TRAP has R_eq = 2L/dt
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let ind = make_inductor("L1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::Trapezoidal;

        let dt = 1e-6;
        stamp_inductor_trap(&mut ctx, &ind, None, dt, &mut state).unwrap();

        // Aux variable at index 3
        let k = 3;

        // R_eq = 2L/dt = 2 * 1e-6 / 1e-6 = 2.0
        // Stamp: -R_eq in (k,k) position
        let expected_r_eq = -2.0;
        let gkk = get_matrix_value(&builder, k, k);
        assert!((gkk - expected_r_eq).abs() < 1e-10, "G(k,k) = {}", gkk);
    }

    #[test]
    fn test_inductor_trap_history() {
        // Test: Inductor TRAP uses RHS = -R_eq * I_{n-1} - V_{n-1}
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut ctx = StampContext {
            builder: &mut builder,
            rhs: &mut rhs,
            aux: &mut aux,
            node_count: 3,
            source_scale: 1.0,
            gmin: 1e-12,
        };

        let ind = make_inductor("L1", 1, 2, "1u");
        let mut state = TransientState::default();
        state.method = IntegrationMethod::Trapezoidal;
        state.ind_current.insert("L1".to_string(), 2.0); // I_{n-1} = 2A
        state.ind_voltage.insert("L1".to_string(), 3.0);  // V_{n-1} = 3V

        let dt = 1e-6;
        stamp_inductor_trap(&mut ctx, &ind, None, dt, &mut state).unwrap();

        // R_eq = 2.0, RHS = -R_eq * I_{n-1} - V_{n-1} = -2.0 * 2.0 - 3.0 = -7.0
        let k = 3;
        let expected_rhs = -7.0;
        assert!((rhs[k] - expected_rhs).abs() < 1e-10, "RHS[k] = {}", rhs[k]);
    }

    #[test]
    fn test_inductor_trap_vs_be_ratio() {
        // Test: TRAP resistance term is exactly 2x BE
        let (mut builder_be, mut rhs_be, mut aux_be) = make_test_context(3);
        let (mut builder_trap, mut rhs_trap, mut aux_trap) = make_test_context(3);

        let ind = make_inductor("L1", 1, 2, "10n"); // 10nH
        let dt = 10e-9;

        let mut state_be = TransientState::default();
        state_be.method = IntegrationMethod::BackwardEuler;

        let mut state_trap = TransientState::default();
        state_trap.method = IntegrationMethod::Trapezoidal;

        {
            let mut ctx_be = StampContext {
                builder: &mut builder_be,
                rhs: &mut rhs_be,
                aux: &mut aux_be,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_inductor_tran(&mut ctx_be, &ind, None, dt, &mut state_be).unwrap();
        }

        {
            let mut ctx_trap = StampContext {
                builder: &mut builder_trap,
                rhs: &mut rhs_trap,
                aux: &mut aux_trap,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_inductor_trap(&mut ctx_trap, &ind, None, dt, &mut state_trap).unwrap();
        }

        // Both have aux var at index 3
        let k = 3;
        // TRAP R_eq should be exactly 2x BE (both are negative)
        let g_be = get_matrix_value(&builder_be, k, k);
        let g_trap = get_matrix_value(&builder_trap, k, k);
        let ratio = g_trap / g_be;
        assert!((ratio - 2.0).abs() < 1e-10, "TRAP/BE ratio = {}", ratio);
    }

    // ========================================================================
    // Integration Method Selection Tests
    // ========================================================================

    #[test]
    fn test_method_selection_default() {
        // Test: Default method is BackwardEuler
        let state = TransientState::default();
        assert_eq!(state.method, IntegrationMethod::BackwardEuler);
    }

    #[test]
    fn test_method_selection_capacitor() {
        // Test: stamp_tran uses state.method for capacitor
        let cap = make_capacitor("C1", 1, 2, "1u");
        let inst_stamp = InstanceStamp { instance: cap };
        let dt = 1e-6;

        // Test with BE
        let (mut builder_be, mut rhs_be, mut aux_be) = make_test_context(3);
        let mut state_be = TransientState::default();
        state_be.method = IntegrationMethod::BackwardEuler;
        {
            let mut ctx = StampContext {
                builder: &mut builder_be,
                rhs: &mut rhs_be,
                aux: &mut aux_be,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            inst_stamp.stamp_tran(&mut ctx, None, dt, &mut state_be).unwrap();
        }

        // Test with TRAP
        let (mut builder_trap, mut rhs_trap, mut aux_trap) = make_test_context(3);
        let mut state_trap = TransientState::default();
        state_trap.method = IntegrationMethod::Trapezoidal;
        {
            let mut ctx = StampContext {
                builder: &mut builder_trap,
                rhs: &mut rhs_trap,
                aux: &mut aux_trap,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            inst_stamp.stamp_tran(&mut ctx, None, dt, &mut state_trap).unwrap();
        }

        // BE: g = C/dt = 1.0, TRAP: g = 2C/dt = 2.0
        let g_be = get_matrix_value(&builder_be, 1, 1);
        let g_trap = get_matrix_value(&builder_trap, 1, 1);
        assert!((g_be - 1.0).abs() < 1e-10, "BE g = {}", g_be);
        assert!((g_trap - 2.0).abs() < 1e-10, "TRAP g = {}", g_trap);
    }

    #[test]
    fn test_method_selection_inductor() {
        // Test: stamp_tran uses state.method for inductor
        let ind = make_inductor("L1", 1, 2, "1u");
        let inst_stamp = InstanceStamp { instance: ind };
        let dt = 1e-6;

        // Test with BE
        let (mut builder_be, mut rhs_be, mut aux_be) = make_test_context(3);
        let mut state_be = TransientState::default();
        state_be.method = IntegrationMethod::BackwardEuler;
        {
            let mut ctx = StampContext {
                builder: &mut builder_be,
                rhs: &mut rhs_be,
                aux: &mut aux_be,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            inst_stamp.stamp_tran(&mut ctx, None, dt, &mut state_be).unwrap();
        }

        // Test with TRAP
        let (mut builder_trap, mut rhs_trap, mut aux_trap) = make_test_context(3);
        let mut state_trap = TransientState::default();
        state_trap.method = IntegrationMethod::Trapezoidal;
        {
            let mut ctx = StampContext {
                builder: &mut builder_trap,
                rhs: &mut rhs_trap,
                aux: &mut aux_trap,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            inst_stamp.stamp_tran(&mut ctx, None, dt, &mut state_trap).unwrap();
        }

        // BE: g = -L/dt = -1.0, TRAP: -R_eq = -2L/dt = -2.0
        let k = 3;
        let g_be = get_matrix_value(&builder_be, k, k);
        let g_trap = get_matrix_value(&builder_trap, k, k);
        assert!((g_be + 1.0).abs() < 1e-10, "BE g = {}", g_be);
        assert!((g_trap + 2.0).abs() < 1e-10, "TRAP g = {}", g_trap);
    }

    // ========================================================================
    // Transient State Update Tests
    // ========================================================================

    #[test]
    fn test_update_transient_state_capacitor() {
        // Test: update_transient_state stores capacitor voltage
        let cap = make_capacitor("C1", 1, 2, "1u");
        let instances = vec![cap];
        let x = vec![0.0, 5.0, 2.0]; // V1 = 5V, V2 = 2V
        let mut state = TransientState::default();

        update_transient_state(&instances, &x, &mut state);

        // V_cap = V1 - V2 = 3V
        let v_cap = state.cap_voltage.get("C1").copied().unwrap_or(0.0);
        assert!((v_cap - 3.0).abs() < 1e-10, "V_cap = {}", v_cap);
    }

    #[test]
    fn test_update_transient_state_full_capacitor() {
        // Test: update_transient_state_full computes and stores capacitor current
        let cap = make_capacitor("C1", 1, 2, "1u");
        let instances = vec![cap];

        let x_prev = vec![0.0, 3.0, 1.0]; // V_prev = 2V
        let x = vec![0.0, 5.0, 2.0];      // V_now = 3V
        let dt = 1e-6;

        let mut state = TransientState::default();
        state.cap_voltage.insert("C1".to_string(), 2.0); // Previous voltage

        update_transient_state_full(&instances, &x, &x_prev, dt, &mut state);

        // V_now = 3V, V_prev = 2V, dV = 1V
        // I = C * dV/dt = 1e-6 * 1 / 1e-6 = 1.0A
        let i_cap = state.cap_current.get("C1").copied().unwrap_or(0.0);
        assert!((i_cap - 1.0).abs() < 1e-10, "I_cap = {}", i_cap);

        // Voltage should be updated
        let v_cap = state.cap_voltage.get("C1").copied().unwrap_or(0.0);
        assert!((v_cap - 3.0).abs() < 1e-10, "V_cap = {}", v_cap);
    }

    #[test]
    fn test_update_transient_state_inductor() {
        // Test: update_transient_state stores inductor current and voltage
        let ind = make_inductor("L1", 1, 2, "1u");
        let instances = vec![ind];

        let mut state = TransientState::default();
        // ind_aux stores the actual solution vector index, not the aux_id offset
        state.ind_aux.insert("L1".to_string(), 3); // Aux var at actual index 3

        let x = vec![0.0, 5.0, 2.0, 1.5]; // V0=0, V1=5, V2=2, I_L=1.5A at index 3

        update_transient_state(&instances, &x, &mut state);

        // Current from aux variable at index 3
        let i_ind = state.ind_current.get("L1").copied().unwrap_or(0.0);
        assert!((i_ind - 1.5).abs() < 1e-10, "I_ind = {}", i_ind);

        // Voltage stored for TRAP: V_L = V1 - V2 = 5 - 2 = 3V
        let v_ind = state.ind_voltage.get("L1").copied().unwrap_or(0.0);
        assert!((v_ind - 3.0).abs() < 1e-10, "V_ind = {}", v_ind);
    }

    // ========================================================================
    // Helper Function Method Tests
    // ========================================================================

    #[test]
    fn test_stamp_capacitor_tran_method() {
        // Test: stamp_capacitor_tran_method dispatches correctly
        let cap = make_capacitor("C1", 1, 2, "1u");
        let dt = 1e-6;

        // BE via helper
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut state = TransientState::default();
        {
            let mut ctx = StampContext {
                builder: &mut builder,
                rhs: &mut rhs,
                aux: &mut aux,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_capacitor_tran_method(&mut ctx, &cap, None, dt, &mut state, IntegrationMethod::BackwardEuler).unwrap();
        }
        let g_be = get_matrix_value(&builder, 1, 1);
        assert!((g_be - 1.0).abs() < 1e-10);

        // TRAP via helper
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut state = TransientState::default();
        {
            let mut ctx = StampContext {
                builder: &mut builder,
                rhs: &mut rhs,
                aux: &mut aux,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_capacitor_tran_method(&mut ctx, &cap, None, dt, &mut state, IntegrationMethod::Trapezoidal).unwrap();
        }
        let g_trap = get_matrix_value(&builder, 1, 1);
        assert!((g_trap - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_stamp_inductor_tran_method() {
        // Test: stamp_inductor_tran_method dispatches correctly
        let ind = make_inductor("L1", 1, 2, "1u");
        let dt = 1e-6;
        let k = 3;

        // BE via helper
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut state = TransientState::default();
        {
            let mut ctx = StampContext {
                builder: &mut builder,
                rhs: &mut rhs,
                aux: &mut aux,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_inductor_tran_method(&mut ctx, &ind, None, dt, &mut state, IntegrationMethod::BackwardEuler).unwrap();
        }
        let g_be = get_matrix_value(&builder, k, k);
        assert!((g_be + 1.0).abs() < 1e-10);

        // TRAP via helper
        let (mut builder, mut rhs, mut aux) = make_test_context(3);
        let mut state = TransientState::default();
        {
            let mut ctx = StampContext {
                builder: &mut builder,
                rhs: &mut rhs,
                aux: &mut aux,
                node_count: 3,
                source_scale: 1.0,
                gmin: 1e-12,
            };
            stamp_inductor_tran_method(&mut ctx, &ind, None, dt, &mut state, IntegrationMethod::Trapezoidal).unwrap();
        }
        let g_trap = get_matrix_value(&builder, k, k);
        assert!((g_trap + 2.0).abs() < 1e-10);
    }
}
