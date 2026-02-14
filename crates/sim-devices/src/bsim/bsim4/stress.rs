//! BSIM4 Stress Effect Model
//!
//! Layout-dependent stress from STI (Shallow Trench Isolation)
//! affects carrier mobility and threshold voltage.
//!
//! ## Physics
//!
//! Mechanical stress from fabrication modifies the silicon
//! band structure, affecting:
//! - Carrier mobility (tensile stress increases electron mobility)
//! - Threshold voltage (stress changes band alignment)
//!
//! Stress depends on the distance to STI edges:
//! - SA: Distance from gate to STI on one side
//! - SB: Distance from gate to STI on other side
//!
//! ## Model
//!
//! ```text
//! stress_factor = 1/SA + 1/SB - 1/SAREF - 1/SBREF
//! Δμ = KU0 * stress_factor
//! ΔVth = KVTH0 * stress_factor
//! ```
//!
//! Devices closer to STI experience more stress and
//! therefore have modified electrical characteristics.

/// Calculate stress effect factors on mobility and threshold
///
/// # Arguments
/// * `sa` - Distance to STI on one side [m]
/// * `sb` - Distance to STI on other side [m]
/// * `saref` - Reference SA for stress normalization [m]
/// * `sbref` - Reference SB for stress normalization [m]
/// * `ku0` - Mobility stress coefficient [dimensionless]
/// * `kvth0` - Threshold voltage stress coefficient [V]
/// * `temp` - Operating temperature [K]
/// * `tnom` - Nominal temperature [K]
/// * `tku0` - Temperature coefficient for KU0 [1/K]
///
/// # Returns
/// * `(u0_mult, delta_vth)` - Mobility multiplier and Vth shift [V]
///
/// # Notes
/// - u0_mult is applied as: ueff = u0 * u0_mult * ...
/// - delta_vth is added to Vth: Vth_eff = Vth + delta_vth
/// - If no stress parameters, returns (1.0, 0.0)
pub fn calculate_stress_effects(
    sa: f64,
    sb: f64,
    saref: f64,
    sbref: f64,
    ku0: f64,
    kvth0: f64,
    temp: f64,
    tnom: f64,
    tku0: f64,
) -> (f64, f64) {
    // If no stress coefficients, return no effect
    if ku0.abs() < 1e-20 && kvth0.abs() < 1e-20 {
        return (1.0, 0.0);
    }

    // Ensure positive distances (clamp to minimum)
    let sa_eff = sa.max(1e-9);
    let sb_eff = sb.max(1e-9);
    let saref_eff = saref.max(1e-9);
    let sbref_eff = sbref.max(1e-9);

    // Stress function: difference of inverse distances
    // Devices closer to STI have smaller SA/SB -> larger 1/SA, 1/SB
    let inv_sa_diff = 1.0 / sa_eff - 1.0 / saref_eff;
    let inv_sb_diff = 1.0 / sb_eff - 1.0 / sbref_eff;

    // Total stress factor (average of both sides)
    let stress_factor = (inv_sa_diff + inv_sb_diff) / 2.0;

    // Temperature dependence of stress effect
    let temp_factor = 1.0 + tku0 * (temp - tnom);

    // Mobility multiplier: u0_eff = u0 * (1 + KU0 * stress * temp_factor)
    let u0_mult = (1.0 + ku0 * stress_factor * temp_factor).max(0.1);

    // Threshold voltage shift: Vth_eff = Vth + KVTH0 * stress
    let delta_vth = kvth0 * stress_factor;

    (u0_mult, delta_vth)
}

/// Calculate stress effect with LOD (Length of Diffusion) model
///
/// Extended model that considers the entire diffusion length,
/// not just the distance to STI.
///
/// # Arguments
/// * `sa` - Distance to STI on source side [m]
/// * `sb` - Distance to STI on drain side [m]
/// * `sd` - Additional diffusion length [m]
/// * `saref` - Reference SA [m]
/// * `sbref` - Reference SB [m]
/// * `wlod` - LOD width parameter [m]
/// * `ku0` - Mobility stress coefficient
/// * `kvth0` - Vth stress coefficient [V]
///
/// # Returns
/// * `(u0_mult, delta_vth)`
pub fn calculate_stress_lod(
    sa: f64,
    sb: f64,
    sd: f64,
    saref: f64,
    sbref: f64,
    wlod: f64,
    ku0: f64,
    kvth0: f64,
) -> (f64, f64) {
    if ku0.abs() < 1e-20 && kvth0.abs() < 1e-20 {
        return (1.0, 0.0);
    }

    // LOD effective distances
    let sa_lod = (sa + wlod).max(1e-9);
    let sb_lod = (sb + wlod).max(1e-9);
    let saref_lod = (saref + wlod).max(1e-9);
    let sbref_lod = (sbref + wlod).max(1e-9);

    // Stress with LOD correction
    let inv_sa_diff = 1.0 / sa_lod - 1.0 / saref_lod;
    let inv_sb_diff = 1.0 / sb_lod - 1.0 / sbref_lod;
    let stress_factor = (inv_sa_diff + inv_sb_diff) / 2.0;

    // Additional SD contribution (simplified)
    let sd_factor = if sd > 0.0 { 1.0 + 0.1 * sd / saref } else { 1.0 };

    let u0_mult = (1.0 + ku0 * stress_factor * sd_factor).max(0.1);
    let delta_vth = kvth0 * stress_factor;

    (u0_mult, delta_vth)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_stress_when_coefficients_zero() {
        let (u0_mult, delta_vth) = calculate_stress_effects(
            1e-6, 1e-6,   // sa, sb
            1e-6, 1e-6,   // saref, sbref
            0.0, 0.0,     // ku0, kvth0 = 0
            300.15, 300.15, 0.0,
        );
        assert!((u0_mult - 1.0).abs() < 1e-10, "u0_mult should be 1.0 when ku0=0");
        assert!(delta_vth.abs() < 1e-10, "delta_vth should be 0 when kvth0=0");
    }

    #[test]
    fn test_no_stress_at_reference_distance() {
        // When SA=SAREF and SB=SBREF, stress should be zero
        let (u0_mult, delta_vth) = calculate_stress_effects(
            1e-6, 1e-6,   // sa = saref, sb = sbref
            1e-6, 1e-6,   // saref, sbref
            0.1, 0.01,    // ku0, kvth0
            300.15, 300.15, 0.0,
        );
        assert!((u0_mult - 1.0).abs() < 1e-6, "No stress at reference distance");
        assert!(delta_vth.abs() < 1e-6);
    }

    #[test]
    fn test_stress_increases_closer_to_sti() {
        // Closer to STI (smaller SA/SB) should give larger stress effect
        let (u0_far, _) = calculate_stress_effects(
            2e-6, 2e-6,   // far from STI
            1e-6, 1e-6,   // reference
            0.1, 0.0,     // ku0 only
            300.15, 300.15, 0.0,
        );
        let (u0_near, _) = calculate_stress_effects(
            0.5e-6, 0.5e-6,  // close to STI
            1e-6, 1e-6,      // reference
            0.1, 0.0,        // ku0 only
            300.15, 300.15, 0.0,
        );
        // Near STI: larger 1/SA difference -> larger stress
        // With positive ku0, this should increase mobility
        assert!(
            (u0_near - 1.0).abs() > (u0_far - 1.0).abs(),
            "Stress effect should be larger near STI"
        );
    }

    #[test]
    fn test_vth_shift_sign() {
        // Positive kvth0 with positive stress should give positive delta_vth
        let (_, delta_vth) = calculate_stress_effects(
            0.5e-6, 0.5e-6,  // closer than ref -> positive stress
            1e-6, 1e-6,
            0.0, 0.01,       // kvth0 = 0.01V
            300.15, 300.15, 0.0,
        );
        assert!(delta_vth > 0.0, "Positive kvth0 with closer distance should give positive Vth shift");
    }

    #[test]
    fn test_temperature_effect_on_stress() {
        // Higher temperature with positive tku0 should increase stress effect
        let (u0_cold, _) = calculate_stress_effects(
            0.5e-6, 0.5e-6,
            1e-6, 1e-6,
            0.1, 0.0,
            250.0, 300.15, 0.001,  // cold, positive tku0
        );
        let (u0_hot, _) = calculate_stress_effects(
            0.5e-6, 0.5e-6,
            1e-6, 1e-6,
            0.1, 0.0,
            350.0, 300.15, 0.001,  // hot, positive tku0
        );
        // With positive tku0, higher temp should amplify stress effect
        assert!(
            (u0_hot - 1.0).abs() > (u0_cold - 1.0).abs(),
            "Higher temperature should amplify stress effect"
        );
    }

    #[test]
    fn test_u0_mult_clamped_positive() {
        // Even with extreme negative stress, u0_mult should stay positive
        let (u0_mult, _) = calculate_stress_effects(
            10e-6, 10e-6,  // very far from STI
            1e-6, 1e-6,
            -10.0, 0.0,    // large negative ku0
            300.15, 300.15, 0.0,
        );
        assert!(u0_mult >= 0.1, "u0_mult should be clamped to minimum 0.1");
    }
}
