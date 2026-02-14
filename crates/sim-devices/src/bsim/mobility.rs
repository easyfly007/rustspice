//! BSIM3 Mobility Model
//!
//! Carrier mobility in a MOSFET is degraded by several mechanisms:
//!
//! 1. **Vertical Field Degradation**: Surface scattering due to gate field
//!    - Carriers pulled against Si-SiO2 interface
//!    - First-order: UA * Eeff
//!    - Second-order: UB * Eeff^2
//!
//! 2. **Body Bias Effect**: UC parameter
//!    - Additional scattering due to depletion charge
//!
//! 3. **Velocity Saturation**: High lateral field limits carrier velocity
//!    - At high Vds, carriers reach VSAT
//!    - Creates current saturation mechanism beyond pinch-off
//!
//! 4. **Temperature Dependence**: Phonon scattering increases with T
//!    - u(T) = u0 * (T/Tnom)^UTE
//!
//! ## Effective Field Calculation
//! ```text
//! Eeff = (Qg + eta*Qb) / epsilon_si
//!
//! Simplified:
//! Eeff ≈ (Vgs - Vth) / (6 * tox) + Vbs / (4 * tox)
//! ```
//!
//! ## References
//! - BSIM3v3 Manual, UC Berkeley, Chapter 4
//! - S. Takagi et al., "On the Universality of Inversion Layer Mobility"

use super::params::{BsimParams, K_BOLTZMANN, Q_ELECTRON};

/// Calculate effective carrier mobility with all degradation effects
///
/// # Arguments
/// * `params` - BSIM3 model parameters
/// * `vgs` - Gate-source voltage [V]
/// * `vbs` - Bulk-source voltage [V]
/// * `vth` - Threshold voltage [V]
/// * `leff` - Effective channel length [m]
/// * `temp` - Temperature [K]
///
/// # Returns
/// * Effective mobility [cm^2/V/s]
///
/// # Physics
/// ```text
/// u_eff = u0 * temp_factor / (1 + UA*Eeff + UB*Eeff^2)
///
/// where:
///   temp_factor = (T/Tnom)^UTE
///   Eeff = vertical effective field
/// ```
pub fn calculate_mobility(
    params: &BsimParams,
    vgs: f64,
    vbs: f64,
    vth: f64,
    _leff: f64,
    temp: f64,
) -> f64 {
    // ========================================
    // Temperature Factor
    // ========================================
    // Lattice scattering increases with temperature
    // For electrons (NMOS): UTE ≈ -1.5
    // For holes (PMOS): UTE ≈ -1.0
    let temp_ratio = temp / params.tnom;
    let temp_factor = temp_ratio.powf(params.ute);

    // Base mobility with temperature effect
    let u0_temp = params.u0 * temp_factor;

    // ========================================
    // Effective Vertical Field
    // ========================================
    // The effective field determines surface scattering
    // Higher field = more carriers at interface = more scattering
    //
    // Eeff = (Vgs - Vth + 2*Vt) / (6 * tox)
    // Plus body bias contribution

    let vt = K_BOLTZMANN * temp / Q_ELECTRON;
    let vgst = (vgs - vth).max(0.0); // Gate overdrive

    // Effective field calculation
    // Simplified from full BSIM3 expression
    let tox_m = params.tox;

    // Vertical field in V/m
    let eeff = (vgst + 2.0 * vt) / (6.0 * tox_m);

    // Convert to MV/cm for typical parameter values
    // (BSIM parameters are often calibrated for MV/cm)
    let _eeff_mv_cm = eeff * 1e-6 / 1e-2; // V/m to MV/cm

    // ========================================
    // Mobility Degradation
    // ========================================
    // Universal mobility model:
    // u_eff = u0 / (1 + theta * Eeff)
    //
    // BSIM3 uses:
    // u_eff = u0 / (1 + UA*Eeff + UB*Eeff^2)
    //
    // UA term: first-order degradation (~linear with field)
    // UB term: second-order degradation (significant at high fields)

    // Note: UA and UB have units that work with Eeff in V/m
    let ua_term = params.ua * eeff;
    let ub_term = params.ub * eeff * eeff;

    // Body bias effect on mobility (UC parameter)
    // Typically small correction
    let uc_term = params.uc * vbs * eeff;

    // Total degradation denominator
    let denom = 1.0 + ua_term + ub_term + uc_term;

    // Effective mobility (ensure positive and bounded)
    let ueff = (u0_temp / denom.max(0.1)).max(1.0);

    ueff
}

/// Calculate mobility degradation factor for output conductance
///
/// Returns the degradation factor (0 to 1) that can be used
/// to modify current calculation
pub fn mobility_degradation_factor(
    params: &BsimParams,
    vgs: f64,
    vth: f64,
    temp: f64,
) -> f64 {
    let vt = K_BOLTZMANN * temp / Q_ELECTRON;
    let vgst = (vgs - vth).max(0.0);

    // Simplified effective field
    let eeff = (vgst + 2.0 * vt) / (6.0 * params.tox);

    // Degradation factor
    let denom = 1.0 + params.ua * eeff + params.ub * eeff * eeff;

    1.0 / denom.max(0.1)
}

/// Calculate mobility with velocity saturation effect included
///
/// At high lateral fields, carrier velocity saturates at VSAT.
/// This creates an additional current limiting mechanism.
///
/// # Returns
/// * (ueff, dueff_dvgs) - Effective mobility and its derivative
pub fn calculate_mobility_with_vsat(
    params: &BsimParams,
    vgs: f64,
    vbs: f64,
    vth: f64,
    vds: f64,
    leff: f64,
    temp: f64,
) -> (f64, f64) {
    // Base mobility calculation
    let ueff = calculate_mobility(params, vgs, vbs, vth, leff, temp);

    // Lateral field: E_lateral ≈ Vds / Leff
    let e_lateral = vds.abs() / leff;

    // Critical field for velocity saturation
    // Ec = 2 * VSAT / ueff (in V/m)
    // Convert ueff from cm^2/V/s to m^2/V/s
    let ueff_m2 = ueff * 1e-4;
    let e_crit = 2.0 * params.vsat / ueff_m2;

    // Velocity saturation reduces effective mobility at high fields
    // u_eff_vsat = ueff / (1 + E_lateral / E_crit)
    let vsat_factor = 1.0 + e_lateral / e_crit.max(1e6);
    let ueff_vsat = ueff / vsat_factor;

    // Derivative calculation (simplified)
    // dueff/dvgs is complex; using numerical approximation
    let delta_vgs = 0.001; // 1mV
    let ueff_plus = calculate_mobility(params, vgs + delta_vgs, vbs, vth, leff, temp);
    let dueff_dvgs = (ueff_plus - ueff) / delta_vgs / vsat_factor;

    (ueff_vsat, dueff_dvgs)
}

/// Simplified mobility for Level 1 compatibility
///
/// Uses only U0 and simple Vgs dependence
pub fn calculate_mobility_simple(
    u0: f64,
    vgs: f64,
    vth: f64,
    _tox: f64,
) -> f64 {
    let vgst = (vgs - vth).max(0.0);

    // Simple theta model: u = u0 / (1 + theta * Vgst)
    // Approximate theta from typical values
    let theta = 0.1; // 1/V typical

    u0 / (1.0 + theta * vgst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mobility_positive() {
        let params = BsimParams::nmos_default();
        let ueff = calculate_mobility(&params, 1.5, 0.0, 0.7, 1e-6, 300.15);
        assert!(ueff > 0.0);
        assert!(ueff < 1000.0); // Reasonable upper bound
    }

    #[test]
    fn test_mobility_degradation_with_field() {
        let params = BsimParams::nmos_default();
        let u_low_field = calculate_mobility(&params, 0.8, 0.0, 0.7, 1e-6, 300.15);
        let u_high_field = calculate_mobility(&params, 3.0, 0.0, 0.7, 1e-6, 300.15);
        // Higher gate voltage = higher field = more degradation
        assert!(u_high_field < u_low_field);
    }

    #[test]
    fn test_temperature_effect() {
        let params = BsimParams::nmos_default();
        let u_room = calculate_mobility(&params, 1.5, 0.0, 0.7, 1e-6, 300.15);
        let u_hot = calculate_mobility(&params, 1.5, 0.0, 0.7, 1e-6, 400.0);
        // Higher temperature = lower mobility (more phonon scattering)
        assert!(u_hot < u_room);
    }
}
