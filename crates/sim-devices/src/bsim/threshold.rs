//! BSIM3 Threshold Voltage Calculation
//!
//! The threshold voltage in BSIM3 includes several physical effects:
//!
//! 1. **Zero-bias threshold (VTH0)**: Base value at Vbs=0, Vds=0
//!
//! 2. **Body effect**: Vth increases with reverse body bias (Vbs < 0 for NMOS)
//!    - First order: K1 * (sqrt(PHI - Vbs) - sqrt(PHI))
//!    - Second order: K2 * Vbs
//!
//! 3. **Short-channel effect (SCE)**: Vth roll-off for short channels
//!    - Charge sharing with source/drain depletion regions
//!    - dVth = -DVT0 * exp(-DVT1 * Leff / 2*lt)
//!
//! 4. **DIBL (Drain-Induced Barrier Lowering)**: Vth reduction with Vds
//!    - Source barrier lowered by drain field
//!    - dVth = -ETA0 * Vds
//!
//! ## References
//! - BSIM3v3 Manual, UC Berkeley, Chapter 3
//! - Y. Cheng, C. Hu, "MOSFET Modeling & BSIM3 User's Guide"

use super::params::{BsimParams, EPSILON_SI, Q_ELECTRON, K_BOLTZMANN};

/// Built-in potential for silicon (default PHI) [V]
/// PHI = 2 * Vt * ln(Na / ni) ≈ 0.7V for typical doping
const PHI_DEFAULT: f64 = 0.7;


/// Calculate threshold voltage with all BSIM3 effects
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vbs` - Bulk-source voltage [V]
/// * `vds` - Drain-source voltage [V]
/// * `leff` - Effective channel length [m]
/// * `weff` - Effective channel width [m]
/// * `temp` - Temperature [K]
///
/// # Returns
/// * `(vth, dvth_dvbs)` - Threshold voltage and its derivative w.r.t. Vbs
///
/// # Physics
/// ```text
/// Vth = VTH0 + body_effect + SCE + DIBL
///
/// where:
///   body_effect = K1 * (sqrt(PHI - Vbs) - sqrt(PHI)) + K2 * Vbs
///   SCE = -DVT0 * exp(-DVT1 * Leff / 2*lt) * (1 + DVT2 * Vbs)
///   DIBL = -(ETA0 - DSUB * exp(-DROUT * Leff / lt)) * Vds
/// ```
pub fn calculate_vth(
    params: &BsimParams,
    vbs: f64,
    vds: f64,
    leff: f64,
    weff: f64,
    temp: f64,
) -> (f64, f64) {
    let vt = K_BOLTZMANN * temp / Q_ELECTRON;

    // Surface potential (built-in potential)
    let phi = PHI_DEFAULT;

    // Start with zero-bias threshold
    let mut vth = params.vth0;
    let mut dvth_dvbs = 0.0;

    // ========================================
    // Body Effect (First and Second Order)
    // ========================================
    // First-order: K1 * (sqrt(PHI - Vbs) - sqrt(PHI))
    // Physical: Depletion charge increases with reverse body bias
    //
    // For NMOS: Vbs <= 0 (reverse bias), so PHI - Vbs > PHI
    // For PMOS: Vbs >= 0, but we flip signs in evaluate
    let phi_minus_vbs = (phi - vbs).max(0.01); // Prevent sqrt of negative
    let sqrt_phi = phi.sqrt();
    let sqrt_phi_vbs = phi_minus_vbs.sqrt();

    let body_effect_1 = params.k1 * (sqrt_phi_vbs - sqrt_phi);
    let dbody1_dvbs = -params.k1 / (2.0 * sqrt_phi_vbs);

    // Second-order correction
    let body_effect_2 = params.k2 * vbs;
    let dbody2_dvbs = params.k2;

    vth += body_effect_1 + body_effect_2;
    dvth_dvbs += dbody1_dvbs + dbody2_dvbs;

    // ========================================
    // Short-Channel Effect (SCE)
    // ========================================
    // As channel length decreases, source/drain depletion regions
    // take up more of the channel, reducing the charge controlled
    // by the gate, thus lowering Vth
    //
    // Characteristic length: lt = sqrt(epsilon_si * tox * Xdep / epsilon_ox)
    // where Xdep is depletion depth
    let xdep = (2.0 * EPSILON_SI * phi / (Q_ELECTRON * 1e17)).sqrt(); // Rough estimate
    let lt = (EPSILON_SI * params.tox * xdep / (EPSILON_SI * 3.9 / 11.7)).sqrt();

    // SCE = -DVT0 * exp(-DVT1 * Leff / 2*lt)
    // Only apply SCE for short channels (Leff < 10*lt)
    let lt_safe = lt.max(1e-9);
    let sce_ratio = leff / (2.0 * lt_safe);
    let sce_exp = if sce_ratio > 20.0 {
        0.0 // Long channel, no SCE
    } else {
        (-params.dvt1 * sce_ratio).exp()
    };
    let sce = -params.dvt0 * sce_exp * vt; // Scale by Vt for reasonable magnitude

    // Body bias dependence of SCE
    let sce_vbs = sce * (1.0 + params.dvt2 * vbs);
    let dsce_dvbs = sce * params.dvt2;

    vth += sce_vbs;
    dvth_dvbs += dsce_dvbs;

    // ========================================
    // DIBL (Drain-Induced Barrier Lowering)
    // ========================================
    // High Vds creates a field that lowers the source barrier,
    // reducing the effective threshold voltage
    //
    // DIBL = -(ETA0 - DSUB * exp(...)) * Vds
    let vds_abs = vds.abs();

    // Basic DIBL: linear with Vds
    let dibl_factor = params.eta0;
    let dibl = -dibl_factor * vds_abs;

    // Length dependence through DSUB
    let dsub_term = params.dsub * (-params.drout * leff / lt_safe).exp();
    let dibl_length = dsub_term * vds_abs;

    vth += dibl - dibl_length;
    // Note: dVth/dVbs from DIBL is typically small, ignored here

    // ========================================
    // Narrow Width Effect
    // ========================================
    // For narrow devices, fringing fields from channel edges
    // affect threshold voltage
    if params.nlx > 0.0 && weff > 0.0 {
        let nwe = params.nlx / weff;
        vth += nwe * 0.1; // Simplified narrow width effect
    }

    // ========================================
    // Temperature Dependence
    // ========================================
    // Vth decreases with temperature (carriers gain thermal energy)
    // dVth/dT ≈ -1 to -2 mV/K for typical MOSFETs
    let temp_ratio = temp / params.tnom;
    let _delta_temp = temp - params.tnom;

    // KT1 term: primary temperature coefficient
    let vth_temp = params.kt1 * (temp_ratio - 1.0);

    // KT1L term: length-dependent temperature coefficient
    let vth_temp_l = params.kt1l / leff * (temp_ratio - 1.0);

    // KT2 term: body bias dependent temperature coefficient
    let vth_temp_vbs = params.kt2 * vbs * (temp_ratio - 1.0);

    vth += vth_temp + vth_temp_l + vth_temp_vbs;
    dvth_dvbs += params.kt2 * (temp_ratio - 1.0);

    // For PMOS, threshold is negative
    // (handled in evaluate.rs by sign flipping)

    (vth, dvth_dvbs)
}

/// Simplified Vth calculation for Level 1 compatibility
///
/// Only uses VTH0 and basic body effect with LAMBDA for CLM
pub fn calculate_vth_simple(
    vth0: f64,
    k1: f64,
    vbs: f64,
) -> (f64, f64) {
    let phi = PHI_DEFAULT;
    let phi_minus_vbs = (phi - vbs).max(0.01);
    let sqrt_phi = phi.sqrt();
    let sqrt_phi_vbs = phi_minus_vbs.sqrt();

    let vth = vth0 + k1 * (sqrt_phi_vbs - sqrt_phi);
    let dvth_dvbs = -k1 / (2.0 * sqrt_phi_vbs);

    (vth, dvth_dvbs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vth_zero_bias() {
        let params = BsimParams::nmos_default();
        // Use longer channel (10um) to minimize SCE
        let (vth, _) = calculate_vth(&params, 0.0, 0.0, 10e-6, 10e-6, 300.15);
        // At Vbs=0, Vds=0, Vth should be close to VTH0 for long channel
        assert!((vth - params.vth0).abs() < 0.15, "vth={}, vth0={}", vth, params.vth0);
    }

    #[test]
    fn test_body_effect() {
        let params = BsimParams::nmos_default();
        let (vth_zero, _) = calculate_vth(&params, 0.0, 0.0, 1e-6, 1e-6, 300.15);
        let (vth_reverse, _) = calculate_vth(&params, -1.0, 0.0, 1e-6, 1e-6, 300.15);
        // Reverse body bias should increase Vth
        assert!(vth_reverse > vth_zero);
    }

    #[test]
    fn test_dibl() {
        let params = BsimParams::nmos_default();
        let (vth_low_vds, _) = calculate_vth(&params, 0.0, 0.1, 1e-6, 1e-6, 300.15);
        let (vth_high_vds, _) = calculate_vth(&params, 0.0, 3.0, 1e-6, 1e-6, 300.15);
        // DIBL should decrease Vth with higher Vds
        assert!(vth_high_vds < vth_low_vds);
    }

    #[test]
    fn test_dvth_dvbs_negative() {
        let params = BsimParams::nmos_default();
        let (_, dvth_dvbs) = calculate_vth(&params, 0.0, 0.0, 1e-6, 1e-6, 300.15);
        // dVth/dVbs should be negative (Vth increases as Vbs decreases)
        assert!(dvth_dvbs < 0.0);
    }
}
