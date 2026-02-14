//! BSIM4 Gate Tunneling Current Model
//!
//! For thin gate oxides (<2nm), direct tunneling through
//! the oxide barrier becomes significant.
//!
//! ## Physics
//!
//! Gate tunneling occurs when the oxide is thin enough for
//! quantum mechanical tunneling of carriers:
//! - Source-side tunneling (Igs): Gate to source
//! - Drain-side tunneling (Igd): Gate to drain
//!
//! The tunneling current depends exponentially on the
//! gate voltage and inversely on oxide thickness.
//!
//! ## Model
//!
//! ```text
//! Igs = W * L * JTSS * exp(Vgs / (NSTI * VTSS))
//! Igd = W * L * JTSD * exp(Vgd / (NSTI * VTSD))
//! ```
//!
//! where:
//! - JTSS/JTSD: Saturation current density [A/m²]
//! - VTSS/VTSD: Tunneling voltage [V]
//! - NSTI: Ideality factor
//!
//! ## Impact
//!
//! Gate leakage is critical for:
//! - Static power consumption
//! - Input impedance of analog circuits
//! - Gate voltage divider effects

/// Calculate gate tunneling currents (source and drain sides)
///
/// # Arguments
/// * `vgs` - Gate-source voltage [V]
/// * `vgd` - Gate-drain voltage [V]
/// * `weff` - Effective channel width [m]
/// * `leff` - Effective channel length [m]
/// * `jtss` - Source-side tunneling current density [A/m²]
/// * `jtsd` - Drain-side tunneling current density [A/m²]
/// * `vtss` - Source-side tunneling voltage [V]
/// * `vtsd` - Drain-side tunneling voltage [V]
/// * `nsti` - Tunneling ideality factor [dimensionless]
///
/// # Returns
/// * `(igs, igd, gigs, gigd)` - Currents [A] and conductances [S]
///
/// # Notes
/// - Currents flow from gate to source/drain
/// - Only significant for thin oxides (< 2nm)
/// - Typical JTSS/JTSD values: 1e-12 to 1e-8 A/m²
pub fn calculate_gate_tunneling(
    vgs: f64,
    vgd: f64,
    weff: f64,
    leff: f64,
    jtss: f64,
    jtsd: f64,
    vtss: f64,
    vtsd: f64,
    nsti: f64,
) -> (f64, f64, f64, f64) {
    // Gate area
    let area = weff * leff;

    // Ideality factor (minimum 1.0)
    let n = nsti.max(1.0);

    // Source-side tunneling current
    let (igs, gigs) = if jtss > 0.0 && vtss > 0.0 {
        let vt_eff = vtss * n;
        if vgs > 0.0 {
            // Forward tunneling (gate to source)
            let exp_arg = (vgs / vt_eff).min(40.0);  // Clamp for numerical stability
            let exp_term = exp_arg.exp();
            let igs = area * jtss * exp_term;
            let gigs = igs / vt_eff;
            (igs, gigs)
        } else {
            // Reverse: small leakage
            let exp_arg = (vgs / vt_eff).max(-40.0);
            let exp_term = exp_arg.exp();
            let igs = area * jtss * exp_term;
            let gigs = (igs / vt_eff).abs();
            (igs, gigs)
        }
    } else {
        (0.0, 0.0)
    };

    // Drain-side tunneling current
    let (igd, gigd) = if jtsd > 0.0 && vtsd > 0.0 {
        let vt_eff = vtsd * n;
        if vgd > 0.0 {
            // Forward tunneling (gate to drain)
            let exp_arg = (vgd / vt_eff).min(40.0);
            let exp_term = exp_arg.exp();
            let igd = area * jtsd * exp_term;
            let gigd = igd / vt_eff;
            (igd, gigd)
        } else {
            // Reverse: small leakage
            let exp_arg = (vgd / vt_eff).max(-40.0);
            let exp_term = exp_arg.exp();
            let igd = area * jtsd * exp_term;
            let gigd = (igd / vt_eff).abs();
            (igd, gigd)
        }
    } else {
        (0.0, 0.0)
    };

    (igs, igd, gigs, gigd)
}

/// Calculate gate-induced drain leakage (GIDL)
///
/// GIDL occurs at the drain-gate overlap region when
/// the gate voltage is low and drain voltage is high,
/// causing band-to-band tunneling.
///
/// # Arguments
/// * `vgd` - Gate-drain voltage [V]
/// * `vds` - Drain-source voltage [V]
/// * `weff` - Effective width [m]
/// * `tox` - Oxide thickness [m]
/// * `agidl` - GIDL coefficient [A/V³]
/// * `bgidl` - GIDL voltage [V]
/// * `cgidl` - GIDL bias dependence [1/V]
///
/// # Returns
/// * `(igidl, gigidl)` - GIDL current [A] and conductance [S]
pub fn calculate_gidl(
    vgd: f64,
    vds: f64,
    weff: f64,
    tox: f64,
    agidl: f64,
    bgidl: f64,
    cgidl: f64,
) -> (f64, f64) {
    // GIDL only occurs when Vgd < 0 (gate low, drain high)
    if agidl <= 0.0 || vgd >= 0.0 || vds < 0.1 {
        return (0.0, 0.0);
    }

    let vgd_abs = (-vgd).max(0.01);
    let eox = vgd_abs / tox;

    // GIDL: I = A * W * |Vgd|^3 * exp(-B/Eox) * (1 + C*Vds)
    let exp_arg = (-bgidl / eox).max(-40.0);
    let exp_term = exp_arg.exp();

    let igidl = agidl * weff * vgd_abs.powi(3) * exp_term * (1.0 + cgidl * vds);
    let gigidl = 3.0 * igidl / vgd_abs + igidl * bgidl / (eox * tox);

    (igidl.max(0.0), gigidl.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_tunneling_when_parameters_zero() {
        let (igs, igd, gigs, gigd) = calculate_gate_tunneling(
            1.0, 0.5,      // vgs, vgd
            1e-6, 100e-9,  // weff, leff
            0.0, 0.0,      // jtss, jtsd = 0
            10.0, 10.0,    // vtss, vtsd
            1.0,           // nsti
        );
        assert!(igs.abs() < 1e-30, "No tunneling when jtss=0");
        assert!(igd.abs() < 1e-30, "No tunneling when jtsd=0");
        assert!(gigs.abs() < 1e-30);
        assert!(gigd.abs() < 1e-30);
    }

    #[test]
    fn test_tunneling_current_positive_forward() {
        let (igs, igd, gigs, gigd) = calculate_gate_tunneling(
            1.2, 0.3,      // vgs > 0, vgd > 0
            1e-6, 100e-9,  // weff, leff
            1e-10, 1e-10,  // jtss, jtsd
            10.0, 10.0,    // vtss, vtsd
            1.0,           // nsti
        );
        assert!(igs > 0.0, "Igs should be positive for Vgs > 0");
        assert!(igd > 0.0, "Igd should be positive for Vgd > 0");
        assert!(gigs > 0.0, "gigs should be positive");
        assert!(gigd > 0.0, "gigd should be positive");
    }

    #[test]
    fn test_tunneling_increases_with_voltage() {
        let (igs1, _, _, _) = calculate_gate_tunneling(
            0.5, 0.0, 1e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 1.0,
        );
        let (igs2, _, _, _) = calculate_gate_tunneling(
            1.5, 0.0, 1e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 1.0,
        );
        assert!(igs2 > igs1, "Higher Vgs should give more tunneling current");
    }

    #[test]
    fn test_tunneling_scales_with_area() {
        let (igs1, _, _, _) = calculate_gate_tunneling(
            1.0, 0.0, 1e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 1.0,
        );
        let (igs2, _, _, _) = calculate_gate_tunneling(
            1.0, 0.0, 2e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 1.0,
        );
        let ratio = igs2 / igs1;
        assert!((ratio - 2.0).abs() < 0.01, "Tunneling should scale with width");
    }

    #[test]
    fn test_source_drain_tunneling_separate() {
        // Only source-side enabled
        let (igs, igd, _, _) = calculate_gate_tunneling(
            1.0, 1.0, 1e-6, 100e-9,
            1e-10, 0.0,  // Only jtss
            10.0, 10.0, 1.0,
        );
        assert!(igs > 0.0);
        assert!(igd.abs() < 1e-30, "No drain tunneling when jtsd=0");
    }

    #[test]
    fn test_ideality_factor_effect() {
        // Higher ideality factor should reduce current at same voltage
        let (igs_n1, _, _, _) = calculate_gate_tunneling(
            1.0, 0.0, 1e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 1.0,
        );
        let (igs_n2, _, _, _) = calculate_gate_tunneling(
            1.0, 0.0, 1e-6, 100e-9, 1e-10, 0.0, 10.0, 10.0, 2.0,
        );
        assert!(igs_n2 < igs_n1, "Higher ideality factor should reduce tunneling");
    }

    #[test]
    fn test_gidl_off_when_vgd_positive() {
        let (igidl, _) = calculate_gidl(
            0.5,   // vgd > 0 (no GIDL)
            1.8,   // vds
            1e-6,  // weff
            1.5e-9, // tox
            1e-12, // agidl
            3.0,   // bgidl
            0.0,   // cgidl
        );
        assert!(igidl.abs() < 1e-30, "No GIDL when Vgd > 0");
    }

    #[test]
    fn test_gidl_on_when_vgd_negative() {
        let (igidl, gigidl) = calculate_gidl(
            -1.0,  // vgd < 0 (GIDL condition)
            1.8,   // vds
            1e-6,  // weff
            1.5e-9, // tox
            1e-12, // agidl
            3.0,   // bgidl
            0.0,   // cgidl
        );
        assert!(igidl > 0.0, "GIDL should occur when Vgd < 0");
        assert!(gigidl > 0.0);
    }
}
