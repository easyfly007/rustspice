//! BSIM4 Substrate Current (Impact Ionization) Model
//!
//! In saturation, hot carriers near the drain can generate
//! electron-hole pairs through impact ionization, creating
//! substrate current that flows from drain to bulk.
//!
//! ## Physics
//!
//! When Vds exceeds Vdsat, carriers gain enough energy to
//! ionize silicon atoms, creating secondary carriers.
//!
//! ```text
//! Isub = ALPHA0 * (Vds - Vdsat) * Ids * exp(-BETA0 / (Vds - Vdsat))
//! ```
//!
//! The substrate current is typically small but important for:
//! - Power dissipation accuracy
//! - Hot carrier reliability analysis
//! - Body effect in analog circuits

/// Calculate substrate current from impact ionization
///
/// # Arguments
/// * `ids` - Drain-source current [A]
/// * `vds` - Drain-source voltage [V]
/// * `vdsat` - Saturation voltage [V]
/// * `alpha0` - Primary impact ionization coefficient [m/V]
/// * `alpha1` - Drain voltage coefficient for impact ionization [1/V]
/// * `beta0` - Impact ionization exponent [V]
/// * `beta1` - Body-bias coefficient for impact ionization [V]
/// * `vbs` - Body-source voltage [V]
///
/// # Returns
/// * `(isub, disub_dvds)` - Substrate current [A] and derivative [S]
///
/// # Physics
/// The impact ionization rate depends exponentially on the
/// inverse of the excess voltage (Vds - Vdsat). The ALPHA
/// parameter controls the magnitude while BETA controls
/// the onset sharpness.
pub fn calculate_isub(
    ids: f64,
    vds: f64,
    vdsat: f64,
    alpha0: f64,
    alpha1: f64,
    beta0: f64,
    beta1: f64,
    vbs: f64,
) -> (f64, f64) {
    // Only applies in saturation with sufficient excess voltage
    let vds_excess = vds - vdsat;

    // No substrate current if not in saturation or alpha0 is zero
    if alpha0 <= 0.0 || vds_excess < 0.01 {
        return (0.0, 0.0);
    }

    // Clamp excess voltage for numerical stability
    let vds_excess = vds_excess.max(0.01);

    // Effective coefficients with body bias and drain voltage dependence
    let alpha_eff = alpha0 * (1.0 + alpha1 * vds);
    let beta_eff = (beta0 + beta1 * vbs).max(1.0);

    // Impact ionization: Isub = alpha * (Vds-Vdsat) * Ids * exp(-beta/(Vds-Vdsat))
    let exp_arg = (-beta_eff / vds_excess).max(-40.0); // Clamp for numerical stability
    let exp_term = exp_arg.exp();

    let isub = alpha_eff * vds_excess * ids.abs() * exp_term;

    // Derivative: dIsub/dVds
    // d/dVds [alpha * (Vds-Vdsat) * Ids * exp(-beta/(Vds-Vdsat))]
    // = alpha * Ids * exp_term * (1 + beta/(Vds-Vdsat))
    let disub_dvds = alpha_eff * ids.abs() * exp_term * (1.0 + beta_eff / vds_excess);

    (isub.max(0.0), disub_dvds.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_impact_ionization_below_vdsat() {
        // When Vds < Vdsat, no substrate current
        let (isub, gsub) = calculate_isub(
            1e-3,  // ids = 1mA
            0.5,   // vds = 0.5V
            0.6,   // vdsat = 0.6V (Vds < Vdsat)
            1e-6,  // alpha0
            0.0,   // alpha1
            30.0,  // beta0
            0.0,   // beta1
            0.0,   // vbs
        );
        assert!(isub.abs() < 1e-20, "Isub should be zero below Vdsat");
        assert!(gsub.abs() < 1e-20);
    }

    #[test]
    fn test_no_impact_ionization_when_alpha_zero() {
        // When alpha0 = 0, no substrate current regardless of bias
        let (isub, _) = calculate_isub(
            1e-3, 2.0, 0.5, 0.0, 0.0, 30.0, 0.0, 0.0,
        );
        assert!(isub.abs() < 1e-20, "Isub should be zero when alpha0=0");
    }

    #[test]
    fn test_impact_ionization_in_saturation() {
        // In saturation with Vds >> Vdsat
        let (isub, gsub) = calculate_isub(
            1e-3,  // ids = 1mA
            2.0,   // vds = 2.0V
            0.5,   // vdsat = 0.5V
            1e-6,  // alpha0
            0.0,   // alpha1
            30.0,  // beta0
            0.0,   // beta1
            0.0,   // vbs
        );
        assert!(isub > 0.0, "Isub should be positive in saturation");
        assert!(gsub > 0.0, "gsub should be positive");
    }

    #[test]
    fn test_isub_increases_with_vds() {
        // Higher Vds should give more substrate current
        let (isub1, _) = calculate_isub(
            1e-3, 1.5, 0.5, 1e-6, 0.0, 30.0, 0.0, 0.0,
        );
        let (isub2, _) = calculate_isub(
            1e-3, 3.0, 0.5, 1e-6, 0.0, 30.0, 0.0, 0.0,
        );
        assert!(isub2 > isub1, "Higher Vds should give more Isub");
    }

    #[test]
    fn test_isub_proportional_to_ids() {
        // Substrate current should scale with drain current
        let (isub1, _) = calculate_isub(
            1e-3, 2.0, 0.5, 1e-6, 0.0, 30.0, 0.0, 0.0,
        );
        let (isub2, _) = calculate_isub(
            2e-3, 2.0, 0.5, 1e-6, 0.0, 30.0, 0.0, 0.0,
        );
        let ratio = isub2 / isub1;
        assert!((ratio - 2.0).abs() < 0.01, "Isub should scale with Ids");
    }
}
