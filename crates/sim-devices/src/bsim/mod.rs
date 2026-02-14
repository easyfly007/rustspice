//! BSIM MOSFET Model Implementation
//!
//! This module implements BSIM3 (Level 49) and BSIM4 (Level 54) DC models
//! for SPICE simulation. BSIM3/4 are industry-standard models for CMOS processes.
//!
//! ## Module Structure
//!
//! - `params`: Model parameters (BsimParams) with defaults
//! - `types`: Enums and output structures (MosType, MosRegion, BsimOutput)
//! - `threshold`: Threshold voltage calculation with body effect, SCE, DIBL
//! - `mobility`: Mobility degradation with field and temperature effects
//! - `channel`: Vdsat, CLM, and output conductance calculations
//! - `evaluate`: Main DC evaluation entry point
//! - `bsim4`: BSIM4-specific physics (substrate current, stress, tunneling)
//!
//! ## Usage
//!
//! ```ignore
//! use sim_devices::bsim::{BsimParams, evaluate_bsim_dc, MosType};
//!
//! // Create NMOS parameters
//! let params = BsimParams::nmos_default();
//!
//! // Evaluate DC operating point
//! let output = evaluate_bsim_dc(
//!     &params,
//!     1e-6,    // W = 1um
//!     100e-9,  // L = 100nm
//!     1.8,     // Vd
//!     1.2,     // Vg
//!     0.0,     // Vs
//!     0.0,     // Vb
//!     300.15,  // T = 27C
//! );
//!
//! println!("Ids = {:.3e} A", output.ids);
//! println!("Region: {:?}", output.region);
//! ```
//!
//! ## Supported Model Levels
//!
//! | Level | Model | Status |
//! |-------|-------|--------|
//! | 1 | Level 1 (Shichman-Hodges) | Supported via `evaluate_level1_dc` |
//! | 49 | BSIM3v3 | Core DC supported |
//! | 54 | BSIM4 | Supported (substrate current, stress, tunneling) |
//!
//! ## References
//!
//! - BSIM3v3.3 Manual, UC Berkeley Device Group
//! - BSIM4 Technical Manual, UC Berkeley Device Group
//! - Y. Cheng, C. Hu, "MOSFET Modeling & BSIM3 User's Guide"
//! - W. Liu, "MOSFET Models for SPICE Simulation"

pub mod params;
pub mod types;
pub mod threshold;
pub mod mobility;
pub mod channel;
pub mod evaluate;
pub mod bsim4;

// Re-export commonly used items
pub use params::BsimParams;
pub use types::{MosType, MosRegion, BsimOutput, BsimState, Bsim4Output, Bsim4State};
pub use evaluate::{evaluate_bsim_dc, evaluate_level1_dc, evaluate_bsim4_dc};
pub use bsim4::{calculate_isub, calculate_stress_effects, calculate_gate_tunneling};

use std::collections::HashMap;

/// Build BsimParams from a parameter HashMap
///
/// Extracts BSIM parameters from the netlist parameter map.
/// Uses defaults for any unspecified parameters.
///
/// # Arguments
/// * `params` - HashMap of parameter name -> value string
/// * `level` - Model level (1, 49, 54)
/// * `is_pmos` - True for PMOS device
///
/// # Returns
/// * `BsimParams` with extracted values
pub fn build_bsim_params(
    params: &HashMap<String, String>,
    level: u32,
    is_pmos: bool,
) -> BsimParams {
    let mut p = if is_pmos {
        BsimParams::pmos_default()
    } else {
        BsimParams::nmos_default()
    };

    p.level = level;

    // Helper to parse parameter value
    let get_param = |keys: &[&str]| -> Option<f64> {
        for key in keys {
            let key_lower = key.to_ascii_lowercase();
            if let Some(value) = params.get(&key_lower) {
                if let Some(num) = parse_number(value) {
                    return Some(num);
                }
            }
        }
        None
    };

    // Threshold voltage parameters
    if let Some(v) = get_param(&["vth0", "vto", "vth"]) {
        p.vth0 = v;
    }
    if let Some(v) = get_param(&["k1"]) {
        p.k1 = v;
    }
    if let Some(v) = get_param(&["k2"]) {
        p.k2 = v;
    }
    if let Some(v) = get_param(&["dvt0"]) {
        p.dvt0 = v;
    }
    if let Some(v) = get_param(&["dvt1"]) {
        p.dvt1 = v;
    }
    if let Some(v) = get_param(&["dvt2"]) {
        p.dvt2 = v;
    }
    if let Some(v) = get_param(&["eta0"]) {
        p.eta0 = v;
    }
    if let Some(v) = get_param(&["dsub"]) {
        p.dsub = v;
    }
    if let Some(v) = get_param(&["nlx"]) {
        p.nlx = v;
    }
    if let Some(v) = get_param(&["nfactor"]) {
        p.nfactor = v;
    }

    // Mobility parameters
    if let Some(v) = get_param(&["u0", "uo"]) {
        p.u0 = v;
    }
    if let Some(v) = get_param(&["ua"]) {
        p.ua = v;
    }
    if let Some(v) = get_param(&["ub"]) {
        p.ub = v;
    }
    if let Some(v) = get_param(&["uc"]) {
        p.uc = v;
    }
    if let Some(v) = get_param(&["vsat"]) {
        p.vsat = v;
    }
    if let Some(v) = get_param(&["a0"]) {
        p.a0 = v;
    }
    if let Some(v) = get_param(&["ags"]) {
        p.ags = v;
    }

    // Output conductance parameters
    if let Some(v) = get_param(&["pclm"]) {
        p.pclm = v;
    }
    if let Some(v) = get_param(&["pdiblc1"]) {
        p.pdiblc1 = v;
    }
    if let Some(v) = get_param(&["pdiblc2"]) {
        p.pdiblc2 = v;
    }
    if let Some(v) = get_param(&["pdiblcb"]) {
        p.pdiblcb = v;
    }
    if let Some(v) = get_param(&["drout"]) {
        p.drout = v;
    }

    // Geometry parameters
    if let Some(v) = get_param(&["tox"]) {
        p.tox = v;
    }
    if let Some(v) = get_param(&["lint"]) {
        p.lint = v;
    }
    if let Some(v) = get_param(&["wint"]) {
        p.wint = v;
    }

    // Parasitic resistance
    if let Some(v) = get_param(&["rdsw"]) {
        p.rdsw = v;
    }
    if let Some(v) = get_param(&["rsh"]) {
        p.rsh = v;
    }

    // Temperature parameters
    if let Some(v) = get_param(&["tnom"]) {
        p.tnom = v + 273.15; // Convert C to K if given in C
    }
    if let Some(v) = get_param(&["ute"]) {
        p.ute = v;
    }
    if let Some(v) = get_param(&["kt1"]) {
        p.kt1 = v;
    }
    if let Some(v) = get_param(&["kt1l"]) {
        p.kt1l = v;
    }
    if let Some(v) = get_param(&["kt2"]) {
        p.kt2 = v;
    }

    // Capacitance parameters (for future use)
    if let Some(v) = get_param(&["cgso"]) {
        p.cgso = v;
    }
    if let Some(v) = get_param(&["cgdo"]) {
        p.cgdo = v;
    }
    if let Some(v) = get_param(&["cgbo"]) {
        p.cgbo = v;
    }

    // ============================================================
    // BSIM4-specific parameters (Level 54)
    // ============================================================

    // Width-dependent SCE parameters
    if let Some(v) = get_param(&["dvt0w"]) {
        p.dvt0w = v;
    }
    if let Some(v) = get_param(&["dvt1w"]) {
        p.dvt1w = v;
    }
    if let Some(v) = get_param(&["dvt2w"]) {
        p.dvt2w = v;
    }

    // Subthreshold offset parameters
    if let Some(v) = get_param(&["voff"]) {
        p.voff = v;
    }
    if let Some(v) = get_param(&["voffl"]) {
        p.voffl = v;
    }
    if let Some(v) = get_param(&["minv"]) {
        p.minv = v;
    }

    // Enhanced narrow width parameters
    if let Some(v) = get_param(&["k3"]) {
        p.k3 = v;
    }
    if let Some(v) = get_param(&["k3b"]) {
        p.k3b = v;
    }
    if let Some(v) = get_param(&["w0"]) {
        p.w0 = v;
    }

    // Lateral doping parameters
    if let Some(v) = get_param(&["lpe0"]) {
        p.lpe0 = v;
    }
    if let Some(v) = get_param(&["lpeb"]) {
        p.lpeb = v;
    }
    if let Some(v) = get_param(&["vfb"]) {
        p.vfb = v;
    }

    // BSIM4 mobility parameters
    if let Some(v) = get_param(&["ute0"]) {
        p.ute0 = v;
    }
    if let Some(v) = get_param(&["ute1"]) {
        p.ute1 = v;
    }
    if let Some(v) = get_param(&["pemod"]) {
        p.pemod = v as u32;
    }
    if let Some(v) = get_param(&["up"]) {
        p.up = v;
    }
    if let Some(v) = get_param(&["lp"]) {
        p.lp = v;
    }
    if let Some(v) = get_param(&["ud"]) {
        p.ud = v;
    }
    if let Some(v) = get_param(&["ud1"]) {
        p.ud1 = v;
    }
    if let Some(v) = get_param(&["eu"]) {
        p.eu = v;
    }

    // Enhanced velocity saturation parameters
    if let Some(v) = get_param(&["vs"]) {
        p.vs = v;
    }
    if let Some(v) = get_param(&["vsattemp"]) {
        p.vsattemp = v;
    }
    if let Some(v) = get_param(&["lambda"]) {
        p.lambda = v;
    }
    if let Some(v) = get_param(&["vtl"]) {
        p.vtl = v;
    }
    if let Some(v) = get_param(&["lc"]) {
        p.lc = v;
    }

    // Substrate current (impact ionization) parameters
    if let Some(v) = get_param(&["alpha0"]) {
        p.alpha0 = v;
    }
    if let Some(v) = get_param(&["alpha1"]) {
        p.alpha1 = v;
    }
    if let Some(v) = get_param(&["beta0"]) {
        p.beta0 = v;
    }
    if let Some(v) = get_param(&["beta1"]) {
        p.beta1 = v;
    }

    // Stress effect parameters
    if let Some(v) = get_param(&["saref"]) {
        p.saref = v;
    }
    if let Some(v) = get_param(&["sbref"]) {
        p.sbref = v;
    }
    if let Some(v) = get_param(&["wlod"]) {
        p.wlod = v;
    }
    if let Some(v) = get_param(&["ku0"]) {
        p.ku0 = v;
    }
    if let Some(v) = get_param(&["kvth0"]) {
        p.kvth0 = v;
    }
    if let Some(v) = get_param(&["ku0mult"]) {
        p.ku0mult = v;
    }
    if let Some(v) = get_param(&["tku0"]) {
        p.tku0 = v;
    }

    // Gate tunneling parameters
    if let Some(v) = get_param(&["jtss"]) {
        p.jtss = v;
    }
    if let Some(v) = get_param(&["jtsd"]) {
        p.jtsd = v;
    }
    if let Some(v) = get_param(&["nsti"]) {
        p.nsti = v;
    }
    if let Some(v) = get_param(&["vtss"]) {
        p.vtss = v;
    }
    if let Some(v) = get_param(&["vtsd"]) {
        p.vtsd = v;
    }

    // Output conductance parameters
    if let Some(v) = get_param(&["pvag"]) {
        p.pvag = v;
    }
    if let Some(v) = get_param(&["fprout"]) {
        p.fprout = v;
    }
    if let Some(v) = get_param(&["pdits"]) {
        p.pdits = v;
    }
    if let Some(v) = get_param(&["delta"]) {
        p.delta = v;
    }

    p
}

/// Parse a number with optional SI suffix
fn parse_number(s: &str) -> Option<f64> {
    let lower = s.to_ascii_lowercase();
    let trimmed = lower.trim();

    // Check for SI suffixes
    let (num_str, multiplier) = if trimmed.ends_with("meg") {
        (&trimmed[..trimmed.len() - 3], 1e6)
    } else if trimmed.ends_with("mil") {
        (&trimmed[..trimmed.len() - 3], 25.4e-6)
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

    num_str.parse::<f64>().ok().map(|n| n * multiplier)
        .or_else(|| trimmed.parse::<f64>().ok())
}

/// Route to appropriate model evaluation based on level
///
/// # Arguments
/// * `params` - BSIM parameters
/// * `w` - Device width [m]
/// * `l` - Device length [m]
/// * `vd`, `vg`, `vs`, `vb` - Terminal voltages [V]
/// * `temp` - Temperature [K]
///
/// # Returns
/// * `BsimOutput` from the appropriate model
pub fn evaluate_mos(
    params: &BsimParams,
    w: f64,
    l: f64,
    vd: f64,
    vg: f64,
    vs: f64,
    vb: f64,
    temp: f64,
) -> BsimOutput {
    match params.level {
        1 => {
            // Level 1: Simple Shichman-Hodges model
            let vth0 = params.vth0;
            let lambda = 0.02; // Default CLM for Level 1
            let beta = params.u0 * 1e-4 * params.cox();

            evaluate_level1_dc(
                vth0,
                beta,
                lambda,
                w, l,
                vd, vg, vs, vb,
                params.mos_type == MosType::Pmos,
            )
        }
        49 => {
            // BSIM3 (Level 49)
            evaluate_bsim_dc(params, w, l, vd, vg, vs, vb, temp)
        }
        54 => {
            // BSIM4 (Level 54) - use enhanced model
            // For basic BsimOutput, use BSIM4 with no stress parameters
            // Use evaluate_mos_bsim4() for full BSIM4 output with stress
            let bsim4_out = evaluate_bsim4_dc(params, w, l, vd, vg, vs, vb, temp, 0.0, 0.0);
            bsim4_out.base
        }
        _ => {
            // Default to BSIM3 for unknown levels
            evaluate_bsim_dc(params, w, l, vd, vg, vs, vb, temp)
        }
    }
}

/// Route to BSIM4 model evaluation with full output
///
/// Returns extended BSIM4 output including substrate and gate tunneling currents.
///
/// # Arguments
/// * `params` - BSIM parameters (level should be 54)
/// * `w` - Device width [m]
/// * `l` - Device length [m]
/// * `vd`, `vg`, `vs`, `vb` - Terminal voltages [V]
/// * `temp` - Temperature [K]
/// * `sa` - Distance to STI on source side [m] (0 to disable stress)
/// * `sb` - Distance to STI on drain side [m] (0 to disable stress)
///
/// # Returns
/// * `Bsim4Output` with all currents and conductances
pub fn evaluate_mos_bsim4(
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
    evaluate_bsim4_dc(params, w, l, vd, vg, vs, vb, temp, sa, sb)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_params_empty() {
        let params = HashMap::new();
        let p = build_bsim_params(&params, 49, false);
        assert_eq!(p.level, 49);
        assert_eq!(p.mos_type, MosType::Nmos);
    }

    #[test]
    fn test_build_params_with_values() {
        let mut params = HashMap::new();
        params.insert("vth0".to_string(), "0.5".to_string());
        params.insert("u0".to_string(), "400".to_string());
        params.insert("tox".to_string(), "2n".to_string());

        let p = build_bsim_params(&params, 49, false);
        assert!((p.vth0 - 0.5).abs() < 0.001);
        assert!((p.u0 - 400.0).abs() < 0.1);
        assert!((p.tox - 2e-9).abs() < 1e-12);
    }

    #[test]
    fn test_parse_number_suffixes() {
        assert!((parse_number("1.5").unwrap() - 1.5).abs() < 1e-10);
        assert!((parse_number("1n").unwrap() - 1e-9).abs() < 1e-15);
        assert!((parse_number("1u").unwrap() - 1e-6).abs() < 1e-12);
        assert!((parse_number("10k").unwrap() - 1e4).abs() < 1e-6);
        assert!((parse_number("2.5meg").unwrap() - 2.5e6).abs() < 1.0);
    }

    #[test]
    fn test_evaluate_mos_level1() {
        let params = BsimParams {
            level: 1,
            ..BsimParams::nmos_default()
        };
        let out = evaluate_mos(&params, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
        assert!(out.ids > 0.0);
    }

    #[test]
    fn test_evaluate_mos_level49() {
        let params = BsimParams {
            level: 49,
            ..BsimParams::nmos_default()
        };
        let out = evaluate_mos(&params, 1e-6, 1e-6, 1.8, 1.5, 0.0, 0.0, 300.15);
        assert!(out.ids > 0.0);
    }

    #[test]
    fn test_build_params_bsim4() {
        let mut params = HashMap::new();
        // BSIM4-specific parameters
        params.insert("alpha0".to_string(), "1e-6".to_string());
        params.insert("beta0".to_string(), "30".to_string());
        params.insert("ku0".to_string(), "0.1".to_string());
        params.insert("kvth0".to_string(), "0.01".to_string());
        params.insert("jtss".to_string(), "1e-10".to_string());

        let p = build_bsim_params(&params, 54, false);
        assert_eq!(p.level, 54);
        assert!((p.alpha0 - 1e-6).abs() < 1e-12);
        assert!((p.beta0 - 30.0).abs() < 0.1);
        assert!((p.ku0 - 0.1).abs() < 0.001);
        assert!((p.kvth0 - 0.01).abs() < 0.001);
        assert!((p.jtss - 1e-10).abs() < 1e-15);
    }

    #[test]
    fn test_evaluate_mos_level54() {
        // BSIM4 should fall back to BSIM3 evaluation for now
        let params = BsimParams {
            level: 54,
            ..BsimParams::nmos_default()
        };
        let out = evaluate_mos(&params, 1e-6, 100e-9, 1.8, 1.5, 0.0, 0.0, 300.15);
        assert!(out.ids > 0.0);
    }
}
