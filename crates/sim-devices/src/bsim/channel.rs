//! BSIM3 Channel Current and Output Conductance
//!
//! This module handles:
//!
//! 1. **Vdsat Calculation**: Saturation voltage with velocity saturation
//!    - At Vds = Vdsat, channel pinches off or carriers reach vsat
//!    - Key for determining linear vs saturation region
//!
//! 2. **Channel Length Modulation (CLM)**: Ids increase in saturation
//!    - Pinch-off point moves toward source with higher Vds
//!    - Effective channel length decreases
//!    - gds = dIds/dVds in saturation
//!
//! 3. **DIBL Output Resistance**: Additional Vds dependence
//!    - Threshold voltage decreases with Vds
//!    - Contributes to finite output resistance
//!
//! ## Vdsat Model
//! ```text
//! Simple model: Vdsat = Vgs - Vth (pinch-off)
//! With velocity saturation:
//!   Vdsat = (Vgs - Vth) * Esat*Leff / (Vgs - Vth + Esat*Leff)
//!   where Esat = 2*Vsat/ueff
//! ```
//!
//! ## References
//! - BSIM3v3 Manual, UC Berkeley, Chapters 5-6
//! - Y. Taur, T.H. Ning, "Fundamentals of Modern VLSI Devices"

use super::params::BsimParams;

/// Calculate drain saturation voltage (Vdsat)
///
/// Vdsat determines the boundary between linear and saturation regions.
/// It accounts for velocity saturation which limits current even before
/// classical pinch-off.
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vgs` - Gate-source voltage [V]
/// * `vth` - Threshold voltage [V]
/// * `ueff` - Effective mobility [cm^2/V/s]
/// * `leff` - Effective channel length [m]
///
/// # Returns
/// * (vdsat, dvdsat_dvgs) - Saturation voltage and its derivative
///
/// # Physics
/// ```text
/// Without velocity saturation: Vdsat = Vgs - Vth
///
/// With velocity saturation:
///   Vdsat = Vgst * Esat*L / (Vgst + Esat*L)
///
/// where:
///   Vgst = Vgs - Vth (gate overdrive)
///   Esat = 2 * Vsat / ueff (saturation field)
/// ```
pub fn calculate_vdsat(
    params: &BsimParams,
    vgs: f64,
    vth: f64,
    ueff: f64,
    leff: f64,
) -> (f64, f64) {
    let vgst = (vgs - vth).max(1e-6); // Gate overdrive, prevent zero

    // Saturation field [V/m]
    // Esat = 2 * Vsat / ueff
    // ueff is in cm^2/V/s, convert to m^2/V/s
    let ueff_m2 = ueff * 1e-4;
    let esat = 2.0 * params.vsat / ueff_m2;

    // Esat * Leff product [V]
    let esat_leff = esat * leff;

    // Vdsat with velocity saturation effect
    // Vdsat = Vgst * Esat*L / (Vgst + Esat*L)
    //
    // This smoothly transitions between:
    // - Low Vgst: Vdsat ≈ Vgst (classical pinch-off limited)
    // - High Vgst: Vdsat ≈ Esat*L (velocity saturation limited)
    let denom = vgst + esat_leff;
    let vdsat = vgst * esat_leff / denom;

    // Derivative: dVdsat/dVgs = dVdsat/dVgst
    // = (Esat*L * (Vgst + Esat*L) - Vgst * Esat*L) / (Vgst + Esat*L)^2
    // = (Esat*L)^2 / (Vgst + Esat*L)^2
    let dvdsat_dvgs = (esat_leff * esat_leff) / (denom * denom);

    (vdsat, dvdsat_dvgs)
}

/// Calculate channel length modulation factor
///
/// In saturation, the effective channel length is reduced because
/// the pinch-off point moves toward the source. This increases Ids.
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vds` - Drain-source voltage [V]
/// * `vdsat` - Saturation voltage [V]
/// * `leff` - Effective channel length [m]
///
/// # Returns
/// * (clm_factor, dclm_dvds) - CLM multiplier (>1 in saturation) and derivative
///
/// # Physics
/// ```text
/// In saturation (Vds > Vdsat):
///   Delta_L / L = (Vds - Vdsat) / (L * Esat) * PCLM
///
/// CLM factor = 1 / (1 - Delta_L/L)
///           ≈ 1 + PCLM * (Vds - Vdsat) / (L * Esat)
/// ```
pub fn calculate_clm_factor(
    params: &BsimParams,
    vds: f64,
    vdsat: f64,
    leff: f64,
    ueff: f64,
) -> (f64, f64) {
    // Only applies in saturation
    let vds_excess = (vds - vdsat).max(0.0);

    if vds_excess <= 0.0 {
        return (1.0, 0.0);
    }

    // Saturation field
    let ueff_m2 = ueff * 1e-4;
    let esat = 2.0 * params.vsat / ueff_m2;

    // Early voltage approximation
    // VA ≈ Esat * Leff / PCLM
    // Using a simpler formulation:
    // CLM factor = 1 + PCLM * (Vds - Vdsat) / VA_eff
    let va_eff = esat * leff / params.pclm.max(0.1);

    // Channel length modulation factor
    let delta_l_ratio = vds_excess / va_eff;

    // Clamp to prevent extreme values
    let delta_l_ratio_clamped = delta_l_ratio.min(0.9);

    // CLM multiplier: Ids_sat * (1 + CLM) or 1/(1 - dL/L)
    let clm_factor = 1.0 / (1.0 - delta_l_ratio_clamped);

    // Derivative of CLM factor w.r.t. Vds
    // d/dVds [1/(1 - x)] = 1/(1-x)^2 * dx/dVds
    // where x = (Vds - Vdsat)/VA_eff
    let dclm_dvds = clm_factor * clm_factor / va_eff;

    (clm_factor, dclm_dvds)
}

/// Calculate DIBL contribution to output conductance
///
/// DIBL causes Vth to decrease with Vds, which increases Ids
/// and contributes to finite output resistance.
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vds` - Drain-source voltage [V]
/// * `vgst` - Gate overdrive (Vgs - Vth) [V]
/// * `leff` - Effective channel length [m]
///
/// # Returns
/// * DIBL contribution to gds [S equivalent factor]
pub fn calculate_dibl_conductance(
    params: &BsimParams,
    _vds: f64,
    _vgst: f64,
    leff: f64,
) -> f64 {
    // DIBL effect on output conductance
    // gds_dibl ≈ gm * dVth/dVds
    //
    // From threshold.rs: dVth/dVds ≈ -ETA0
    // In saturation: contribution is proportional to ETA0

    let dibl_factor = params.eta0;

    // Length dependence through PDIBLC parameters
    // Shorter channels have stronger DIBL
    let pdibl_term = params.pdiblc1 * (-params.drout * leff / 1e-6).exp()
        + params.pdiblc2;

    // Total DIBL conductance factor
    // This multiplies the base transconductance
    let gds_dibl = dibl_factor * pdibl_term;

    gds_dibl.max(0.0)
}

/// Calculate source/drain series resistance effect
///
/// Series resistance reduces effective Vgs and Vds at the intrinsic device
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `weff` - Effective width [m]
/// * `temp` - Temperature [K]
///
/// # Returns
/// * Total S/D resistance [ohm]
pub fn calculate_rds(
    params: &BsimParams,
    weff: f64,
    temp: f64,
) -> f64 {
    if params.rdsw <= 0.0 {
        return 0.0;
    }

    // RDSW is in ohm*um, so divide by W in um
    let weff_um = weff * 1e6;
    let rds_base = params.rdsw / weff_um.max(0.1);

    // Temperature correction
    let delta_t = temp - params.tnom;
    let rds = rds_base * (1.0 + params.prt * delta_t);

    rds.max(0.0)
}

/// Calculate subthreshold swing factor
///
/// In weak inversion, current varies exponentially with Vgs
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vgs` - Gate-source voltage [V]
/// * `vth` - Threshold voltage [V]
/// * `vt` - Thermal voltage [V]
///
/// # Returns
/// * Subthreshold factor for current calculation
pub fn calculate_subthreshold_factor(
    params: &BsimParams,
    vgs: f64,
    vth: f64,
    vt: f64,
) -> f64 {
    let vgst = vgs - vth;

    if vgst >= 0.0 {
        // Above threshold: no subthreshold correction needed
        return 1.0;
    }

    // Subthreshold swing factor n ≈ 1 + Cdm/Cox
    // n typically 1.2 to 1.5
    let n = params.nfactor.max(1.0);

    // Subthreshold current: exp(Vgs - Vth) / (n * Vt))
    let subvt_factor = (vgst / (n * vt)).exp();

    subvt_factor.min(1.0)
}

/// Calculate effective Vds for short-channel devices
///
/// Smoothly limits Vds to Vdsat region for numerical stability
pub fn calculate_vds_eff(
    vds: f64,
    vdsat: f64,
) -> f64 {
    // Smooth limiting function
    // Vds_eff = Vdsat - 0.5 * (Vdsat - Vds - delta + sqrt((Vdsat - Vds - delta)^2 + 4*delta*Vdsat))
    // where delta is smoothing parameter
    let delta = 0.01;

    let diff = vdsat - vds - delta;
    let sqrt_term = (diff * diff + 4.0 * delta * vdsat).sqrt();

    let vds_eff = vdsat - 0.5 * (diff + sqrt_term);

    vds_eff.max(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vdsat_positive() {
        let params = BsimParams::nmos_default();
        let (vdsat, _) = calculate_vdsat(&params, 1.5, 0.7, 400.0, 1e-6);
        assert!(vdsat > 0.0);
        assert!(vdsat < 1.5); // Should be less than Vgs
    }

    #[test]
    fn test_vdsat_increases_with_vgs() {
        let params = BsimParams::nmos_default();
        let (vdsat_low, _) = calculate_vdsat(&params, 1.0, 0.7, 400.0, 1e-6);
        let (vdsat_high, _) = calculate_vdsat(&params, 2.0, 0.7, 400.0, 1e-6);
        assert!(vdsat_high > vdsat_low);
    }

    #[test]
    fn test_clm_factor_unity_in_linear() {
        let params = BsimParams::nmos_default();
        let (clm, _) = calculate_clm_factor(&params, 0.1, 0.5, 1e-6, 400.0);
        assert!((clm - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_clm_factor_increases_in_saturation() {
        let params = BsimParams::nmos_default();
        let vdsat = 0.5;
        let (clm_at_vdsat, _) = calculate_clm_factor(&params, vdsat, vdsat, 1e-6, 400.0);
        let (clm_above_vdsat, _) = calculate_clm_factor(&params, 2.0, vdsat, 1e-6, 400.0);
        assert!(clm_above_vdsat > clm_at_vdsat);
    }
}
