//! BSIM MOSFET type definitions
//!
//! Contains enums and output structures for BSIM model evaluation.

/// MOSFET device type (NMOS or PMOS)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosType {
    Nmos,
    Pmos,
}

impl Default for MosType {
    fn default() -> Self {
        MosType::Nmos
    }
}

/// Operating region of the MOSFET
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MosRegion {
    /// Cutoff: Vgs < Vth
    Cutoff,
    /// Linear/Triode: Vgs > Vth, Vds < Vdsat
    Linear,
    /// Saturation: Vgs > Vth, Vds >= Vdsat
    Saturation,
}

impl Default for MosRegion {
    fn default() -> Self {
        MosRegion::Cutoff
    }
}

/// Output from BSIM DC evaluation
///
/// Contains all values needed for MNA matrix stamping:
/// - DC current and small-signal conductances
/// - Equivalent current source for linearization
/// - Operating region for debugging
#[derive(Debug, Clone, Default)]
pub struct BsimOutput {
    /// Drain-to-source current [A]
    pub ids: f64,
    /// Transconductance dIds/dVgs [S]
    pub gm: f64,
    /// Output conductance dIds/dVds [S]
    pub gds: f64,
    /// Body transconductance dIds/dVbs [S]
    pub gmbs: f64,
    /// Equivalent current for MNA linearization [A]
    /// ieq = ids - gm*vgs - gds*vds - gmbs*vbs
    pub ieq: f64,
    /// Operating region
    pub region: MosRegion,
    /// Effective threshold voltage [V]
    pub vth_eff: f64,
}

/// Internal state for BSIM calculations
#[derive(Debug, Clone, Default)]
pub struct BsimState {
    /// Effective channel length [m]
    pub leff: f64,
    /// Effective channel width [m]
    pub weff: f64,
    /// Effective mobility [cm^2/V/s]
    pub ueff: f64,
    /// Drain saturation voltage [V]
    pub vdsat: f64,
    /// Effective threshold voltage [V]
    pub vth: f64,
    /// Derivative of Vth w.r.t Vbs [V/V]
    pub dvth_dvbs: f64,
    /// Channel length modulation factor
    pub clm_factor: f64,
}

// ============================================================
// BSIM4 Extended Types
// ============================================================

/// Extended output from BSIM4 DC evaluation
///
/// Includes additional currents for substrate and gate tunneling.
#[derive(Debug, Clone, Default)]
pub struct Bsim4Output {
    /// Base BSIM output (Ids, gm, gds, gmbs, ieq, region, vth_eff)
    pub base: BsimOutput,

    /// Substrate current from impact ionization [A]
    pub isub: f64,
    /// Substrate conductance dIsub/dVds [S]
    pub gsub: f64,

    /// Gate-source tunneling current [A]
    pub igs: f64,
    /// Gate-drain tunneling current [A]
    pub igd: f64,
    /// Gate-source tunneling conductance [S]
    pub gigs: f64,
    /// Gate-drain tunneling conductance [S]
    pub gigd: f64,

    /// Effective mobility [cm^2/V/s]
    pub ueff: f64,
    /// Saturation voltage [V]
    pub vdsat: f64,
}

/// Extended internal state for BSIM4 calculations
#[derive(Debug, Clone, Default)]
pub struct Bsim4State {
    /// Base BSIM state
    pub base: BsimState,

    /// Stress effect on mobility [multiplier]
    pub stress_ueff: f64,
    /// Stress effect on Vth [V shift]
    pub stress_vth: f64,

    /// Width-dependent SCE contribution [V]
    pub sce_width: f64,

    /// Impact ionization multiplication factor
    pub ii_mult: f64,

    /// Gate tunneling current density [A/m^2]
    pub jtun: f64,
}
