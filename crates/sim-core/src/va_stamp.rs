//! Stamp bridge: maps OSDI evaluation results into RustSpice's MNA system.
//!
//! This module implements the critical interface between OSDI-compiled device
//! models and the RustSpice simulation engine. After OSDI `eval()` computes
//! device equations, these functions extract residuals and Jacobian entries
//! and inject them into the MNA matrix via `StampContext`.
//!
//! # OSDI <-> Newton-Raphson Compatibility
//!
//! OSDI provides dedicated SPICE-format loading functions:
//! - `load_spice_rhs_dc`: computes J*x - f(x) for DC Newton iteration
//! - `load_spice_rhs_tran`: same with reactive terms scaled by alpha=1/dt
//! - `load_jacobian_resist`: loads G matrix entries into instance data
//! - `load_jacobian_tran`: loads G + alpha*C combined entries
//!
//! These match exactly what RustSpice's Newton iteration expects.

use sim_va::osdi_types::*;
use sim_va::OsdiInstance;

use crate::complex_mna::ComplexStampContext;
use crate::mna::StampContext;

/// Stamp the device's DC contributions into the MNA matrix and RHS.
pub fn va_stamp_dc(
    inst: &mut OsdiInstance,
    ctx: &mut StampContext,
    x: Option<&[f64]>,
    rhs_buf: &mut Vec<f64>,
) -> Result<(), String> {
    let solution = match x {
        Some(s) => s,
        None => return Ok(()),
    };

    // Phase 1: Mutable operations (evaluate + load Jacobian)
    let flags = CALC_RESIST_RESIDUAL | CALC_RESIST_JACOBIAN | ANALYSIS_DC;
    inst.evaluate(solution, None, flags)
        .map_err(|e| e.to_string())?;
    inst.load_jacobian_resist();

    // Phase 2: Read-only stamping (borrow descriptor after mutable calls)
    let num_jac = inst.model.descriptor.jacobian_entries.len();
    for k in 0..num_jac {
        let entry = &inst.model.descriptor.jacobian_entries[k];
        let row_mna = inst.node_mapping.get(entry.row_node as usize).copied().unwrap_or(-1);
        let col_mna = inst.node_mapping.get(entry.col_node as usize).copied().unwrap_or(-1);
        if row_mna < 0 || col_mna < 0 {
            continue;
        }

        let jac_value = inst.read_jacobian_resist(k);
        if jac_value != 0.0 {
            ctx.add(row_mna as usize, col_mna as usize, jac_value);
        }
    }

    // Phase 3: RHS loading and stamping
    let matrix_size = ctx.node_count + ctx.aux.id_to_name.len();
    rhs_buf.resize(matrix_size, 0.0);
    rhs_buf.fill(0.0);

    let mut prev_solve = vec![0.0; matrix_size];
    for (i, val) in prev_solve.iter_mut().enumerate() {
        *val = solution.get(i).copied().unwrap_or(0.0);
    }

    inst.load_spice_rhs_dc(rhs_buf, &mut prev_solve);

    let node_mapping_len = inst.node_mapping.len();
    for osdi_idx in 0..node_mapping_len {
        let mna_idx = inst.node_mapping[osdi_idx];
        if mna_idx < 0 {
            continue;
        }
        let idx = mna_idx as usize;
        if let Some(&rhs_val) = rhs_buf.get(idx) {
            if rhs_val != 0.0 {
                ctx.add_rhs(idx, rhs_val);
            }
        }
    }

    Ok(())
}

/// Stamp device contributions for transient analysis.
///
/// Includes reactive (capacitive/inductive) contributions scaled by `alpha`:
/// - Backward Euler: alpha = 1/dt
/// - Trapezoidal: alpha = 2/dt
pub fn va_stamp_tran(
    inst: &mut OsdiInstance,
    ctx: &mut StampContext,
    x: Option<&[f64]>,
    alpha: f64,
    rhs_buf: &mut Vec<f64>,
) -> Result<(), String> {
    let solution = match x {
        Some(s) => s,
        None => return Ok(()),
    };

    // Phase 1: Mutable operations
    let flags = CALC_RESIST_RESIDUAL
        | CALC_REACT_RESIDUAL
        | CALC_RESIST_JACOBIAN
        | CALC_REACT_JACOBIAN
        | ANALYSIS_TRAN;
    inst.evaluate(solution, None, flags)
        .map_err(|e| e.to_string())?;
    inst.load_jacobian_tran(alpha);

    // Phase 2: Read-only stamping
    let num_jac = inst.model.descriptor.jacobian_entries.len();
    for k in 0..num_jac {
        let entry = &inst.model.descriptor.jacobian_entries[k];
        let row_mna = inst.node_mapping.get(entry.row_node as usize).copied().unwrap_or(-1);
        let col_mna = inst.node_mapping.get(entry.col_node as usize).copied().unwrap_or(-1);
        if row_mna < 0 || col_mna < 0 {
            continue;
        }

        let jac_value = inst.read_jacobian_resist(k);
        if jac_value != 0.0 {
            ctx.add(row_mna as usize, col_mna as usize, jac_value);
        }
    }

    // Phase 3: RHS
    let matrix_size = ctx.node_count + ctx.aux.id_to_name.len();
    rhs_buf.resize(matrix_size, 0.0);
    rhs_buf.fill(0.0);

    let mut prev_solve = vec![0.0; matrix_size];
    for (i, val) in prev_solve.iter_mut().enumerate() {
        *val = solution.get(i).copied().unwrap_or(0.0);
    }

    inst.load_spice_rhs_tran(rhs_buf, &mut prev_solve, alpha);

    let node_mapping_len = inst.node_mapping.len();
    for osdi_idx in 0..node_mapping_len {
        let mna_idx = inst.node_mapping[osdi_idx];
        if mna_idx < 0 {
            continue;
        }
        let idx = mna_idx as usize;
        if let Some(&rhs_val) = rhs_buf.get(idx) {
            if rhs_val != 0.0 {
                ctx.add_rhs(idx, rhs_val);
            }
        }
    }

    Ok(())
}

/// Stamp device contributions for AC small-signal analysis.
///
/// Evaluates at the DC operating point and stamps the complex admittance
/// Y(jw) = G + jw*C into the complex MNA matrix.
pub fn va_stamp_ac(
    inst: &mut OsdiInstance,
    ctx: &mut ComplexStampContext,
    dc_solution: &[f64],
) -> Result<(), String> {
    // Phase 1: Mutable operations
    let flags = CALC_RESIST_RESIDUAL
        | CALC_REACT_RESIDUAL
        | CALC_RESIST_JACOBIAN
        | CALC_REACT_JACOBIAN
        | ANALYSIS_AC;
    inst.evaluate(dc_solution, None, flags)
        .map_err(|e| e.to_string())?;
    inst.load_jacobian_resist();

    // Phase 2: Read-only stamping
    let omega = ctx.omega;
    let num_jac = inst.model.descriptor.jacobian_entries.len();

    for k in 0..num_jac {
        let entry = &inst.model.descriptor.jacobian_entries[k];
        let row_mna = inst.node_mapping.get(entry.row_node as usize).copied().unwrap_or(-1);
        let col_mna = inst.node_mapping.get(entry.col_node as usize).copied().unwrap_or(-1);
        if row_mna < 0 || col_mna < 0 {
            continue;
        }

        let g = inst.read_jacobian_resist(k);
        let c = inst.read_jacobian_react(k);

        let row = row_mna as usize;
        let col = col_mna as usize;

        if g != 0.0 {
            ctx.add_real(row, col, g);
        }
        if c != 0.0 {
            ctx.add_imag(row, col, omega * c);
        }
    }

    Ok(())
}
