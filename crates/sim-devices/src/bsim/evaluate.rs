//! BSIM3/BSIM4 DC Evaluation
//!
//! Main entry points for BSIM MOSFET DC analysis.
//! Computes drain current (Ids) and small-signal parameters (gm, gds, gmbs)
//! using the modular physics functions.
//!
//! ## DC Current Model
//!
//! The drain current is calculated differently depending on operating region:
//!
//! **Cutoff (Vgs < Vth)**:
//! - Ids ≈ 0 (subthreshold leakage in full model)
//!
//! **Linear (Vds < Vdsat)**:
//! - Ids = W/L * ueff * Cox * [(Vgs-Vth)*Vds - Vds^2/2]
//!
//! **Saturation (Vds >= Vdsat)**:
//! - Ids = W/L * ueff * Cox * Vdsat^2/2 * CLM_factor
//!
//! ## Small-Signal Parameters
//!
//! - gm = dIds/dVgs (transconductance)
//! - gds = dIds/dVds (output conductance)
//! - gmbs = dIds/dVbs (body transconductance)
//!
//! ## BSIM4 Extensions
//!
//! BSIM4 adds:
//! - Layout-dependent stress effects on mobility and Vth
//! - Substrate current from impact ionization
//! - Gate tunneling currents (source and drain sides)
//!
//! ## MNA Stamping
//!
//! For Newton-Raphson iteration, the linearized current is:
//! ```text
//! i_ds = gm*(vgs-VGS) + gds*(vds-VDS) + gmbs*(vbs-VBS) + IDS
//!      = gm*vgs + gds*vds + gmbs*vbs + (IDS - gm*VGS - gds*VDS - gmbs*VBS)
//!      = gm*vgs + gds*vds + gmbs*vbs + ieq
//! ```

use super::params::{BsimParams, EPSILON_OX, K_BOLTZMANN, Q_ELECTRON};
use super::types::{BsimOutput, Bsim4Output, MosRegion, MosType};
use super::threshold::calculate_vth;
use super::mobility::calculate_mobility;
use super::channel::{calculate_vdsat, calculate_clm_factor, calculate_rds};
use super::bsim4::{calculate_isub, calculate_stress_effects, calculate_gate_tunneling};

/// Minimum conductance for numerical stability [S]
const GMIN: f64 = 1e-12;

/// Main BSIM DC evaluation function
///
/// Computes drain current and all small-signal parameters needed for
/// MNA matrix stamping.
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `w` - Device width [m]
/// * `l` - Device length [m]
/// * `vd` - Drain voltage [V]
/// * `vg` - Gate voltage [V]
/// * `vs` - Source voltage [V]
/// * `vb` - Bulk/body voltage [V]
/// * `temp` - Temperature [K]
///
/// # Returns
/// * `BsimOutput` containing Ids, gm, gds, gmbs, ieq, region, vth_eff
///
/// # Example
/// ```ignore
/// let params = BsimParams::nmos_default();
/// let output = evaluate_bsim_dc(&params, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
/// println!("Ids = {} A", output.ids);
/// println!("gm = {} S", output.gm);
/// ```
pub fn evaluate_bsim_dc(
    params: &BsimParams,
    w: f64,
    l: f64,
    vd: f64,
    vg: f64,
    vs: f64,
    vb: f64,
    temp: f64,
) -> BsimOutput {
    // Handle PMOS by flipping voltage signs
    let (vd_int, vg_int, vs_int, vb_int, sign) = match params.mos_type {
        MosType::Nmos => (vd, vg, vs, vb, 1.0),
        MosType::Pmos => (-vs, -vg, -vd, -vb, -1.0), // Swap D/S and negate
    };

    // Terminal voltages (internal, after PMOS flip)
    let mut vgs = vg_int - vs_int;
    let mut vds = vd_int - vs_int;
    let vbs = vb_int - vs_int;

    // Source/drain swap for negative Vds (reverse mode)
    let reversed = vds < 0.0;
    if reversed {
        vds = -vds;
        vgs = vg_int - vd_int; // Vgd becomes effective Vgs
    }

    // Effective dimensions
    let leff = params.leff(l);
    let weff = params.weff(w);

    // Oxide capacitance per unit area
    let cox = EPSILON_OX / params.tox;

    // Thermal voltage
    let vt = K_BOLTZMANN * temp / Q_ELECTRON;

    // ========================================
    // Step 1: Threshold Voltage
    // ========================================
    let (vth, dvth_dvbs) = calculate_vth(params, vbs, vds, leff, weff, temp);

    // Gate overdrive
    let vgst = vgs - vth;

    // ========================================
    // Step 2: Operating Region Determination
    // ========================================
    let region;
    let mut ids: f64;
    let mut gm: f64;
    let mut gds: f64;
    let gmbs: f64;

    if vgst <= 0.0 {
        // ========================================
        // Cutoff Region (with subthreshold)
        // ========================================
        region = MosRegion::Cutoff;

        // Subthreshold current (weak inversion)
        // Ids = I0 * exp((Vgs - Vth) / (n * Vt)) * (1 - exp(-Vds/Vt))
        let n = params.nfactor.max(1.0);
        let i0 = weff / leff * params.u0 * 1e-4 * cox * vt * vt * (n - 1.0);

        let exp_vgst = (vgst / (n * vt)).exp();
        let exp_vds = (-vds / vt).exp();

        // Subthreshold current
        ids = i0 * exp_vgst * (1.0 - exp_vds);
        ids = ids.max(0.0);

        // Small-signal parameters in subthreshold
        gm = ids / (n * vt);
        gds = i0 * exp_vgst * exp_vds / vt;
        gmbs = -gm * dvth_dvbs;

        // Ensure minimum conductance
        gds = gds.max(GMIN);
        gm = gm.max(GMIN * 0.01);

    } else {
        // ========================================
        // Step 3: Mobility Calculation
        // ========================================
        let ueff = calculate_mobility(params, vgs, vbs, vth, leff, temp);

        // ========================================
        // Step 4: Saturation Voltage
        // ========================================
        let (vdsat, dvdsat_dvgs) = calculate_vdsat(params, vgs, vth, ueff, leff);

        // ========================================
        // Step 5: Drain Current Calculation
        // ========================================
        // Beta factor: W/L * ueff * Cox
        let ueff_m2 = ueff * 1e-4; // cm^2/V/s to m^2/V/s
        let beta = weff / leff * ueff_m2 * cox;

        if vds < vdsat {
            // ========================================
            // Linear Region
            // ========================================
            region = MosRegion::Linear;

            // Ids = beta * [(Vgst - Vds/2) * Vds]
            ids = beta * (vgst * vds - 0.5 * vds * vds);

            // gm = dIds/dVgs = beta * Vds
            gm = beta * vds;

            // gds = dIds/dVds = beta * (Vgst - Vds)
            gds = beta * (vgst - vds);
            gds = gds.max(GMIN);

            // gmbs = dIds/dVbs = -gm * dVth/dVbs
            gmbs = -gm * dvth_dvbs;

        } else {
            // ========================================
            // Saturation Region
            // ========================================
            region = MosRegion::Saturation;

            // Channel length modulation
            let (clm_factor, dclm_dvds) = calculate_clm_factor(params, vds, vdsat, leff, ueff);

            // Saturation current: Ids = beta * Vdsat^2 / 2 * CLM
            let ids_sat = 0.5 * beta * vdsat * vdsat;
            ids = ids_sat * clm_factor;

            // gm = dIds/dVgs
            // = d/dVgs [beta * Vdsat^2/2 * CLM]
            // = beta * Vdsat * dVdsat/dVgs * CLM
            gm = beta * vdsat * dvdsat_dvgs * clm_factor;

            // gds = dIds/dVds (from CLM)
            // = Ids_sat * dCLM/dVds
            gds = ids_sat * dclm_dvds;
            gds = gds.max(GMIN);

            // DIBL contribution to gds
            // gds_dibl ≈ gm * ETA0
            let gds_dibl = gm * params.eta0;
            gds += gds_dibl;

            // gmbs = dIds/dVbs = -gm * dVth/dVbs (Vdsat depends on Vth)
            // Plus contribution from Vdsat dependence on Vth
            gmbs = -gm * dvth_dvbs;
        }
    }

    // Ensure positive current
    ids = ids.max(0.0);

    // ========================================
    // Source/Drain Series Resistance
    // ========================================
    let rds = calculate_rds(params, weff, temp);
    if rds > 0.0 && ids > 0.0 {
        // Simplified Rds effect: reduce effective gds
        // Full model would iterate on Vds_int
        let v_rds = ids * rds;
        if v_rds < vds * 0.5 {
            // Only apply if Rds drop is small
            gds = gds / (1.0 + rds * gds);
        }
    }

    // ========================================
    // Handle source/drain reversal
    // ========================================
    if reversed {
        // In reverse mode, gm acts on Vgd instead of Vgs
        // For stamping purposes, we keep the same form
        // The caller will handle node mapping
    }

    // Apply sign for PMOS
    ids *= sign;

    // ========================================
    // Calculate equivalent current for MNA
    // ========================================
    // ieq = Ids - gm*Vgs - gds*Vds - gmbs*Vbs
    // This is the DC offset for linearized current source
    let vgs_orig = vg - vs;
    let vds_orig = vd - vs;
    let vbs_orig = vb - vs;

    let ieq = ids - gm * vgs_orig - gds * vds_orig - gmbs * vbs_orig;

    BsimOutput {
        ids,
        gm,
        gds,
        gmbs,
        ieq,
        region,
        vth_eff: vth,
    }
}

/// Simplified Level 1 MOSFET evaluation
///
/// For backward compatibility with simple models.
/// Uses only VTH0, KP (or BETA), and LAMBDA parameters.
pub fn evaluate_level1_dc(
    vth0: f64,
    beta: f64,
    lambda: f64,
    w: f64,
    l: f64,
    vd: f64,
    vg: f64,
    vs: f64,
    _vb: f64,
    is_pmos: bool,
) -> BsimOutput {
    // Handle PMOS
    let (vd_int, vg_int, vs_int, sign) = if is_pmos {
        (-vs, -vg, -vd, -1.0)
    } else {
        (vd, vg, vs, 1.0)
    };

    let mut vgs = vg_int - vs_int;
    let mut vds = vd_int - vs_int;

    // Source/drain swap
    if vds < 0.0 {
        vds = -vds;
        vgs = vg_int - vd_int;
    }

    let vth = if is_pmos { -vth0.abs() } else { vth0.abs() };
    let beta_eff = beta * w / l;

    let region;
    let ids;
    let gm;
    let gds;

    if vgs <= vth {
        // Cutoff
        region = MosRegion::Cutoff;
        ids = 0.0;
        gm = 0.0;
        gds = GMIN;
    } else if vds < vgs - vth {
        // Linear
        region = MosRegion::Linear;
        ids = beta_eff * ((vgs - vth) * vds - 0.5 * vds * vds);
        gm = beta_eff * vds;
        gds = (beta_eff * ((vgs - vth) - vds)).max(GMIN);
    } else {
        // Saturation
        region = MosRegion::Saturation;
        ids = 0.5 * beta_eff * (vgs - vth).powi(2) * (1.0 + lambda * vds);
        gm = beta_eff * (vgs - vth) * (1.0 + lambda * vds);
        gds = (0.5 * beta_eff * (vgs - vth).powi(2) * lambda).max(GMIN);
    }

    let ids_signed = ids * sign;

    let vgs_orig = vg - vs;
    let vds_orig = vd - vs;
    let ieq = ids_signed - gm * vgs_orig - gds * vds_orig;

    BsimOutput {
        ids: ids_signed,
        gm,
        gds,
        gmbs: 0.0, // Level 1 ignores body effect on current
        ieq,
        region,
        vth_eff: vth,
    }
}

/// BSIM4 DC evaluation function
///
/// Enhanced version of BSIM DC evaluation with:
/// - Layout-dependent stress effects (SA/SB parameters)
/// - Substrate current from impact ionization (ALPHA0, BETA0)
/// - Gate tunneling currents (JTSS, JTSD)
///
/// # Arguments
/// * `params` - BSIM parameters (should have level=54 for full BSIM4)
/// * `w` - Device width [m]
/// * `l` - Device length [m]
/// * `vd` - Drain voltage [V]
/// * `vg` - Gate voltage [V]
/// * `vs` - Source voltage [V]
/// * `vb` - Bulk/body voltage [V]
/// * `temp` - Temperature [K]
/// * `sa` - Distance to STI on source side [m] (optional, use 0.0 to disable)
/// * `sb` - Distance to STI on drain side [m] (optional, use 0.0 to disable)
///
/// # Returns
/// * `Bsim4Output` containing all currents and conductances
///
/// # Example
/// ```ignore
/// let params = BsimParams::nmos_default();
/// let output = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);
/// println!("Ids = {} A", output.base.ids);
/// println!("Isub = {} A", output.isub);
/// ```
pub fn evaluate_bsim4_dc(
    params: &BsimParams,
    w: f64,
    l: f64,
    vd: f64,
    vg: f64,
    vs: f64,
    vb: f64,
    temp: f64,
    sa: f64,
    sb: f64,
) -> Bsim4Output {
    // Handle PMOS by flipping voltage signs
    let (vd_int, vg_int, vs_int, vb_int, sign) = match params.mos_type {
        MosType::Nmos => (vd, vg, vs, vb, 1.0),
        MosType::Pmos => (-vs, -vg, -vd, -vb, -1.0),
    };

    // Terminal voltages (internal, after PMOS flip)
    let mut vgs = vg_int - vs_int;
    let mut vds = vd_int - vs_int;
    let vbs = vb_int - vs_int;

    // Source/drain swap for negative Vds (reverse mode)
    let reversed = vds < 0.0;
    if reversed {
        vds = -vds;
        vgs = vg_int - vd_int;
    }

    // Effective dimensions
    let leff = params.leff(l);
    let weff = params.weff(w);

    // Oxide capacitance per unit area
    let cox = EPSILON_OX / params.tox;

    // Thermal voltage
    let vt = K_BOLTZMANN * temp / Q_ELECTRON;

    // ========================================
    // BSIM4: Stress Effects
    // ========================================
    let (u0_stress_mult, vth_stress_shift) = if sa > 0.0 || sb > 0.0 {
        calculate_stress_effects(
            sa.max(params.saref), // Use reference if not specified
            sb.max(params.sbref),
            params.saref,
            params.sbref,
            params.ku0,
            params.kvth0,
            temp,
            params.tnom,
            params.tku0,
        )
    } else {
        (1.0, 0.0)
    };

    // ========================================
    // Step 1: Threshold Voltage (with stress shift)
    // ========================================
    let (vth_base, dvth_dvbs) = calculate_vth(params, vbs, vds, leff, weff, temp);
    let vth = vth_base + vth_stress_shift;

    // Gate overdrive
    let vgst = vgs - vth;

    // ========================================
    // Step 2: Operating Region Determination
    // ========================================
    let region;
    let mut ids: f64;
    let mut gm: f64;
    let mut gds: f64;
    let gmbs: f64;
    let mut ueff: f64 = 0.0;
    let mut vdsat: f64 = 0.0;

    if vgst <= 0.0 {
        // ========================================
        // Cutoff Region (with subthreshold)
        // ========================================
        region = MosRegion::Cutoff;

        let n = params.nfactor.max(1.0);
        let i0 = weff / leff * params.u0 * 1e-4 * u0_stress_mult * cox * vt * vt * (n - 1.0);

        let exp_vgst = (vgst / (n * vt)).exp();
        let exp_vds = (-vds / vt).exp();

        ids = i0 * exp_vgst * (1.0 - exp_vds);
        ids = ids.max(0.0);

        gm = ids / (n * vt);
        gds = i0 * exp_vgst * exp_vds / vt;
        gmbs = -gm * dvth_dvbs;

        gds = gds.max(GMIN);
        gm = gm.max(GMIN * 0.01);

    } else {
        // ========================================
        // Step 3: Mobility (with stress effect)
        // ========================================
        ueff = calculate_mobility(params, vgs, vbs, vth, leff, temp) * u0_stress_mult;

        // ========================================
        // Step 4: Saturation Voltage
        // ========================================
        let (vdsat_calc, dvdsat_dvgs) = calculate_vdsat(params, vgs, vth, ueff, leff);
        vdsat = vdsat_calc;

        // ========================================
        // Step 5: Drain Current Calculation
        // ========================================
        let ueff_m2 = ueff * 1e-4;
        let beta = weff / leff * ueff_m2 * cox;

        if vds < vdsat {
            // Linear Region
            region = MosRegion::Linear;

            ids = beta * (vgst * vds - 0.5 * vds * vds);
            gm = beta * vds;
            gds = (beta * (vgst - vds)).max(GMIN);
            gmbs = -gm * dvth_dvbs;

        } else {
            // Saturation Region
            region = MosRegion::Saturation;

            let (clm_factor, dclm_dvds) = calculate_clm_factor(params, vds, vdsat, leff, ueff);

            let ids_sat = 0.5 * beta * vdsat * vdsat;
            ids = ids_sat * clm_factor;

            gm = beta * vdsat * dvdsat_dvgs * clm_factor;
            gds = (ids_sat * dclm_dvds).max(GMIN);

            // DIBL contribution
            let gds_dibl = gm * params.eta0;
            gds += gds_dibl;

            gmbs = -gm * dvth_dvbs;
        }
    }

    ids = ids.max(0.0);

    // ========================================
    // Source/Drain Series Resistance
    // ========================================
    let rds = calculate_rds(params, weff, temp);
    if rds > 0.0 && ids > 0.0 {
        let v_rds = ids * rds;
        if v_rds < vds * 0.5 {
            gds = gds / (1.0 + rds * gds);
        }
    }

    // ========================================
    // BSIM4: Substrate Current (Impact Ionization)
    // ========================================
    let (isub, gsub) = if params.alpha0 > 0.0 && region == MosRegion::Saturation {
        calculate_isub(
            ids,
            vds,
            vdsat,
            params.alpha0,
            params.alpha1,
            params.beta0,
            params.beta1,
            vbs,
        )
    } else {
        (0.0, 0.0)
    };

    // ========================================
    // BSIM4: Gate Tunneling Currents
    // ========================================
    let vgd = vgs - vds;
    let (igs, igd, gigs, gigd) = if params.jtss > 0.0 || params.jtsd > 0.0 {
        calculate_gate_tunneling(
            vgs,
            vgd,
            weff,
            leff,
            params.jtss,
            params.jtsd,
            params.vtss,
            params.vtsd,
            params.nsti,
        )
    } else {
        (0.0, 0.0, 0.0, 0.0)
    };

    // Apply sign for PMOS
    let ids_signed = ids * sign;
    let isub_signed = isub * sign.abs(); // Substrate current always flows to bulk
    let igs_signed = igs * sign.abs();
    let igd_signed = igd * sign.abs();

    // ========================================
    // Calculate equivalent current for MNA
    // ========================================
    let vgs_orig = vg - vs;
    let vds_orig = vd - vs;
    let vbs_orig = vb - vs;

    let ieq = ids_signed - gm * vgs_orig - gds * vds_orig - gmbs * vbs_orig;

    Bsim4Output {
        base: BsimOutput {
            ids: ids_signed,
            gm,
            gds,
            gmbs,
            ieq,
            region,
            vth_eff: vth,
        },
        isub: isub_signed,
        gsub,
        igs: igs_signed,
        igd: igd_signed,
        gigs,
        gigd,
        ueff,
        vdsat,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nmos_cutoff() {
        let params = BsimParams::nmos_default();
        // Vg = 0.3V is well below Vth0 = 0.7V, should be in cutoff
        // Use longer channel for cleaner behavior
        let output = evaluate_bsim_dc(&params, 10e-6, 10e-6, 1.0, 0.2, 0.0, 0.0, 300.15);
        assert_eq!(output.region, MosRegion::Cutoff, "Expected cutoff, got {:?}, vth_eff={}", output.region, output.vth_eff);
        assert!(output.ids.abs() < 1e-6); // Very small current
    }

    #[test]
    fn test_nmos_linear() {
        let params = BsimParams::nmos_default();
        // Vgs = 1.5V > Vth (~0.7V), Vds = 0.1V < Vgs - Vth
        let output = evaluate_bsim_dc(&params, 1e-6, 1e-6, 0.1, 1.5, 0.0, 0.0, 300.15);
        assert_eq!(output.region, MosRegion::Linear, "Expected linear, got {:?}", output.region);
        assert!(output.ids > 0.0);
    }

    #[test]
    fn test_nmos_saturation() {
        let params = BsimParams::nmos_default();
        // Vgs = 1.2V, Vds = 2.0V >> Vgs - Vth, clearly saturation
        let output = evaluate_bsim_dc(&params, 1e-6, 1e-6, 2.0, 1.2, 0.0, 0.0, 300.15);
        assert_eq!(output.region, MosRegion::Saturation, "Expected saturation, got {:?}, vth_eff={}", output.region, output.vth_eff);
        assert!(output.ids > 0.0);
    }

    #[test]
    fn test_pmos_saturation() {
        let params = BsimParams::pmos_default();
        let output = evaluate_bsim_dc(&params, 1e-6, 1e-6, 0.0, 0.0, 1.8, 1.8, 300.15);
        // PMOS: Vgs = 0 - 1.8 = -1.8V, should be on
        // Vds = 0 - 1.8 = -1.8V
        assert!(output.ids < 0.0); // Current flows out of drain
    }

    #[test]
    fn test_ids_increases_with_vgs() {
        let params = BsimParams::nmos_default();
        let out1 = evaluate_bsim_dc(&params, 1e-6, 1e-6, 1.8, 1.0, 0.0, 0.0, 300.15);
        let out2 = evaluate_bsim_dc(&params, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
        let out3 = evaluate_bsim_dc(&params, 1e-6, 1e-6, 1.8, 2.0, 0.0, 0.0, 300.15);
        assert!(out2.ids > out1.ids);
        assert!(out3.ids > out2.ids);
    }

    #[test]
    fn test_ids_increases_with_width() {
        let params = BsimParams::nmos_default();
        let out1 = evaluate_bsim_dc(&params, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
        let out2 = evaluate_bsim_dc(&params, 2e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
        assert!((out2.ids / out1.ids - 2.0).abs() < 0.2); // Should roughly double
    }

    #[test]
    fn test_level1_compatibility() {
        let out = evaluate_level1_dc(0.7, 1e-3, 0.02, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, false);
        assert_eq!(out.region, MosRegion::Saturation);
        assert!(out.ids > 0.0);
    }

    // ========================================
    // BSIM4 Evaluator Tests
    // ========================================

    #[test]
    fn test_bsim4_basic_nmos() {
        let params = BsimParams::nmos_default();
        let out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);
        assert!(out.base.ids > 0.0, "BSIM4 NMOS should conduct in saturation");
        assert_eq!(out.base.region, MosRegion::Saturation);
    }

    #[test]
    fn test_bsim4_basic_pmos() {
        let params = BsimParams::pmos_default();
        let out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 0.0, 0.0, 1.8, 1.8, 300.15, 0.0, 0.0);
        assert!(out.base.ids < 0.0, "BSIM4 PMOS current should be negative");
    }

    #[test]
    fn test_bsim4_with_substrate_current() {
        let mut params = BsimParams::nmos_default();
        params.alpha0 = 1e-6;  // Enable impact ionization
        params.beta0 = 30.0;

        let out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 2.0, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);

        // Should be in saturation and have substrate current
        assert_eq!(out.base.region, MosRegion::Saturation);
        assert!(out.isub >= 0.0, "Substrate current should be non-negative");
        // Note: isub may be very small or zero depending on vds-vdsat
    }

    #[test]
    fn test_bsim4_with_tunneling() {
        let mut params = BsimParams::nmos_default();
        params.jtss = 1e-10;  // Enable source-side tunneling
        params.jtsd = 1e-10;  // Enable drain-side tunneling
        params.vtss = 10.0;
        params.vtsd = 10.0;
        params.nsti = 1.0;

        let out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);

        // Should have gate tunneling currents
        assert!(out.igs >= 0.0, "Gate-source tunneling should be non-negative for Vgs > 0");
        assert!(out.gigs >= 0.0, "Gate-source conductance should be non-negative");
    }

    #[test]
    fn test_bsim4_with_stress() {
        let mut params = BsimParams::nmos_default();
        params.ku0 = 0.1;      // Mobility stress coefficient
        params.kvth0 = 0.01;   // Vth stress coefficient
        params.saref = 1e-6;
        params.sbref = 1e-6;

        // Without stress (sa=sb=0)
        let out_no_stress = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);

        // With stress (sa=sb closer than reference)
        let out_stress = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.5e-6, 0.5e-6);

        // Currents should differ due to stress effects
        assert!(out_no_stress.base.ids > 0.0);
        assert!(out_stress.base.ids > 0.0);
        // The specific relationship depends on the sign of stress coefficients
    }

    #[test]
    fn test_bsim4_backwards_compatible() {
        // BSIM4 with default params should match BSIM3 behavior
        let params = BsimParams::nmos_default();

        let bsim3_out = evaluate_bsim_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15);
        let bsim4_out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);

        // Base outputs should be similar (within 1% for default params)
        let ids_ratio = bsim4_out.base.ids / bsim3_out.ids;
        assert!(
            (ids_ratio - 1.0).abs() < 0.01,
            "BSIM4 should match BSIM3 for default params, got ratio {}",
            ids_ratio
        );

        // Regions should match
        assert_eq!(bsim4_out.base.region, bsim3_out.region);
    }

    #[test]
    fn test_bsim4_all_regions() {
        let params = BsimParams::nmos_default();

        // Cutoff
        let out_cutoff = evaluate_bsim4_dc(&params, 10e-6, 10e-6, 1.0, 0.2, 0.0, 0.0, 300.15, 0.0, 0.0);
        assert_eq!(out_cutoff.base.region, MosRegion::Cutoff);

        // Linear
        let out_linear = evaluate_bsim4_dc(&params, 1e-6, 1e-6, 0.1, 1.5, 0.0, 0.0, 300.15, 0.0, 0.0);
        assert_eq!(out_linear.base.region, MosRegion::Linear);

        // Saturation
        let out_sat = evaluate_bsim4_dc(&params, 1e-6, 1e-6, 2.0, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);
        assert_eq!(out_sat.base.region, MosRegion::Saturation);
    }

    #[test]
    fn test_bsim4_ueff_vdsat_reported() {
        let params = BsimParams::nmos_default();
        let out = evaluate_bsim4_dc(&params, 1e-6, 100e-9, 1.8, 1.2, 0.0, 0.0, 300.15, 0.0, 0.0);

        // In saturation, ueff and vdsat should be reported
        assert_eq!(out.base.region, MosRegion::Saturation);
        assert!(out.ueff > 0.0, "Effective mobility should be positive");
        assert!(out.vdsat > 0.0, "Saturation voltage should be positive");
    }
}
