//! BSIM MOSFET model parameters
//!
//! Contains the BsimParams structure with all BSIM3 model parameters
//! and their default values for NMOS and PMOS devices.

use super::types::MosType;

/// Physical constants
pub const EPSILON_SI: f64 = 11.7 * 8.854e-12; // Silicon permittivity [F/m]
pub const EPSILON_OX: f64 = 3.9 * 8.854e-12; // Oxide permittivity [F/m]
pub const Q_ELECTRON: f64 = 1.602e-19; // Electron charge [C]
pub const K_BOLTZMANN: f64 = 1.381e-23; // Boltzmann constant [J/K]
pub const T_NOMINAL: f64 = 300.15; // Nominal temperature [K] (27C)

/// BSIM3 Model Parameters
///
/// This structure contains the key parameters for BSIM3 (Level 49) model.
/// Parameters are grouped by their physical function:
///
/// - Model selection: level, mos_type
/// - Threshold voltage: vth0, k1, k2, dvt0, dvt1, dvt2, eta0, dsub
/// - Mobility: u0, ua, ub, uc, vsat
/// - Short-channel effects: pclm, pdiblc1, pdiblc2
/// - Geometry: tox, lint, wint
/// - Parasitic: rdsw
/// - Temperature: tnom, ute, kt1
#[derive(Debug, Clone)]
pub struct BsimParams {
    // ============ Model Selection ============
    /// Model level: 1=Level1, 49=BSIM3, 54=BSIM4
    pub level: u32,
    /// Device type: NMOS or PMOS
    pub mos_type: MosType,

    // ============ Threshold Voltage ============
    /// Zero-bias threshold voltage [V]
    /// Physical meaning: Gate voltage needed to create inversion layer at Vbs=0
    pub vth0: f64,
    /// First-order body effect coefficient [V^0.5]
    /// Physical meaning: How Vth increases with reverse body bias (sqrt dependence)
    pub k1: f64,
    /// Second-order body effect coefficient [dimensionless]
    /// Physical meaning: Correction to first-order body effect
    pub k2: f64,
    /// Short-channel effect coefficient for Vth [dimensionless]
    /// Physical meaning: Controls Vth roll-off with channel length
    pub dvt0: f64,
    /// Short-channel effect exponent [dimensionless]
    /// Physical meaning: Exponential decay rate of SCE with length
    pub dvt1: f64,
    /// Body-bias coefficient for short-channel effect [1/V]
    /// Physical meaning: How body bias affects SCE
    pub dvt2: f64,
    /// DIBL (Drain-Induced Barrier Lowering) coefficient [dimensionless]
    /// Physical meaning: Vth reduction due to Vds
    pub eta0: f64,
    /// DIBL exponent [dimensionless]
    /// Physical meaning: Length dependence of DIBL effect
    pub dsub: f64,
    /// Narrow width effect coefficient [dimensionless]
    pub nlx: f64,
    /// Subthreshold swing coefficient [dimensionless]
    pub nfactor: f64,

    // ============ Mobility ============
    /// Low-field mobility [cm^2/V/s]
    /// Physical meaning: Carrier mobility without field degradation
    pub u0: f64,
    /// First-order mobility degradation coefficient [m/V]
    /// Physical meaning: Linear reduction of mobility with vertical field
    pub ua: f64,
    /// Second-order mobility degradation coefficient [(m/V)^2]
    /// Physical meaning: Quadratic mobility degradation with vertical field
    pub ub: f64,
    /// Body-bias mobility degradation coefficient [m/V^2]
    /// Physical meaning: How body bias affects mobility degradation
    pub uc: f64,
    /// Saturation velocity [m/s]
    /// Physical meaning: Maximum carrier velocity under high lateral field
    pub vsat: f64,
    /// Mobility reduction factor due to Rds [dimensionless]
    pub a0: f64,
    /// Gate-bias dependent Rds parameter [dimensionless]
    pub ags: f64,
    /// Source/drain resistance gate bias coefficient [1/V]
    pub prwg: f64,
    /// Source/drain resistance body bias coefficient [1/V^0.5]
    pub prwb: f64,

    // ============ Short-channel/Output Conductance ============
    /// Channel length modulation coefficient [dimensionless]
    /// Physical meaning: Controls increase of Ids with Vds in saturation
    pub pclm: f64,
    /// DIBL output resistance coefficient 1 [dimensionless]
    pub pdiblc1: f64,
    /// DIBL output resistance coefficient 2 [dimensionless]
    pub pdiblc2: f64,
    /// DIBL body bias coefficient [1/V]
    pub pdiblcb: f64,
    /// Drain-induced threshold shift coefficient [dimensionless]
    pub drout: f64,
    /// Subthreshold output conductance parameter [dimensionless]
    pub pscbe1: f64,
    /// Subthreshold output conductance exponent [V/m]
    pub pscbe2: f64,
    /// Substrate current body effect coefficient [1/V]
    pub alpha0: f64,
    /// Substrate current DIBL coefficient [V]
    pub beta0: f64,

    // ============ Geometry ============
    /// Gate oxide thickness [m]
    pub tox: f64,
    /// Channel length offset for Leff calculation [m]
    /// Leff = L - 2*LINT
    pub lint: f64,
    /// Channel width offset for Weff calculation [m]
    /// Weff = W - 2*WINT
    pub wint: f64,
    /// Minimum channel length for model validity [m]
    pub lmin: f64,
    /// Minimum channel width for model validity [m]
    pub wmin: f64,
    /// Effective length scaling parameter [dimensionless]
    pub lln: f64,
    /// Effective length scaling reference [m]
    pub lw: f64,
    /// Effective length scaling parameter [dimensionless]
    pub lwn: f64,
    /// Effective width scaling parameter [dimensionless]
    pub wln: f64,
    /// Effective width scaling reference [m]
    pub ww: f64,
    /// Effective width scaling parameter [dimensionless]
    pub wwn: f64,

    // ============ Parasitic Resistance ============
    /// Source/drain sheet resistance per unit width [ohm*um]
    /// Total Rds = RDSW / Weff
    pub rdsw: f64,
    /// Gate resistance per unit width [ohm*um]
    pub rsh: f64,

    // ============ Temperature ============
    /// Nominal temperature for parameter extraction [K]
    pub tnom: f64,
    /// Mobility temperature exponent [dimensionless]
    /// u(T) = u0 * (T/Tnom)^UTE
    pub ute: f64,
    /// Vth temperature coefficient [V]
    /// Vth(T) = Vth0 + KT1 * (T/Tnom - 1)
    pub kt1: f64,
    /// Vth temperature coefficient (length dependence) [V*m]
    pub kt1l: f64,
    /// Vth temperature coefficient (body bias) [V]
    pub kt2: f64,
    /// Saturation velocity temperature coefficient [m/s/K]
    pub at: f64,
    /// RDSW temperature coefficient [1/K]
    pub prt: f64,

    // ============ Capacitance (for future AC/transient) ============
    /// Gate-source overlap capacitance per unit width [F/m]
    pub cgso: f64,
    /// Gate-drain overlap capacitance per unit width [F/m]
    pub cgdo: f64,
    /// Gate-bulk overlap capacitance per unit width [F/m]
    pub cgbo: f64,
    /// Junction capacitance parameter [F/m^2]
    pub cj: f64,
    /// Junction sidewall capacitance [F/m]
    pub cjsw: f64,
    /// Junction built-in potential [V]
    pub pb: f64,
    /// Junction sidewall built-in potential [V]
    pub pbsw: f64,
    /// Junction grading coefficient [dimensionless]
    pub mj: f64,
    /// Junction sidewall grading coefficient [dimensionless]
    pub mjsw: f64,

    // ============ Flicker Noise (for future noise analysis) ============
    /// Flicker noise coefficient A [dimensionless]
    pub kf: f64,
    /// Flicker noise exponent [dimensionless]
    pub af: f64,
    /// Flicker noise frequency exponent [dimensionless]
    pub ef: f64,

    // ============================================================
    // BSIM4 Enhanced Parameters (Level 54)
    // ============================================================

    // ============ BSIM4 Enhanced Threshold Voltage ============
    /// Width-dependent short-channel effect coefficient [1/m]
    pub dvt0w: f64,
    /// Width-dependent short-channel effect exponent [1/m]
    pub dvt1w: f64,
    /// Body-bias coefficient for width-dependent SCE [1/V]
    pub dvt2w: f64,
    /// Subthreshold offset voltage [V]
    pub voff: f64,
    /// Length-dependent VOFF [V*m]
    pub voffl: f64,
    /// Minimum inversion factor [dimensionless]
    pub minv: f64,
    /// Narrow width effect coefficient [dimensionless]
    pub k3: f64,
    /// Body-bias coefficient for K3 [1/V]
    pub k3b: f64,
    /// Narrow width reference [m]
    pub w0: f64,
    /// Lateral non-uniform doping effect [m]
    pub lpe0: f64,
    /// Body-bias dependence of LPE [m]
    pub lpeb: f64,
    /// Flat-band voltage [V]
    pub vfb: f64,

    // ============ BSIM4 Enhanced Mobility ============
    /// Primary temperature exponent for mobility [dimensionless]
    pub ute0: f64,
    /// Secondary temperature exponent [dimensionless]
    pub ute1: f64,
    /// Phonon scattering model selector (0-3)
    pub pemod: u32,
    /// Channel length uniformity parameter [dimensionless]
    pub up: f64,
    /// Reference length for UP [m]
    pub lp: f64,
    /// Drain bias mobility coefficient [1/V]
    pub ud: f64,
    /// Secondary drain mobility coefficient [dimensionless]
    pub ud1: f64,
    /// Field effect mobility exponent [dimensionless]
    pub eu: f64,

    // ============ BSIM4 Enhanced Velocity Saturation ============
    /// Alternative saturation velocity [m/s] (0 = use VSAT)
    pub vs: f64,
    /// Temperature coefficient of VSAT [m/s/K]
    pub vsattemp: f64,
    /// Velocity saturation exponent [dimensionless]
    pub lambda: f64,
    /// Thermal velocity [m/s]
    pub vtl: f64,
    /// Characteristic length for velocity [m]
    pub lc: f64,

    // ============ BSIM4 Substrate Current (Impact Ionization) ============
    /// Impact ionization drain voltage coefficient [1/V]
    pub alpha1: f64,
    /// Impact ionization body-bias coefficient [V]
    pub beta1: f64,

    // ============ BSIM4 Stress Effects ============
    /// Reference SA for stress calculation [m]
    pub saref: f64,
    /// Reference SB for stress calculation [m]
    pub sbref: f64,
    /// LOD width [m]
    pub wlod: f64,
    /// Mobility stress coefficient [dimensionless]
    pub ku0: f64,
    /// Threshold voltage stress coefficient [V]
    pub kvth0: f64,
    /// U0 stress multiplier [dimensionless]
    pub ku0mult: f64,
    /// Temperature coefficient for KU0 [1/K]
    pub tku0: f64,

    // ============ BSIM4 Gate Tunneling ============
    /// Source-side gate tunneling current density [A/m^2]
    pub jtss: f64,
    /// Drain-side gate tunneling current density [A/m^2]
    pub jtsd: f64,
    /// Gate tunneling ideality factor [dimensionless]
    pub nsti: f64,
    /// Source-side tunneling voltage [V]
    pub vtss: f64,
    /// Drain-side tunneling voltage [V]
    pub vtsd: f64,

    // ============ BSIM4 Enhanced Output Conductance ============
    /// Gate voltage PDIBL parameter [dimensionless]
    pub pvag: f64,
    /// PDIBL output resistance factor [dimensionless]
    pub fprout: f64,
    /// Drain impact on output conductance [dimensionless]
    pub pdits: f64,
    /// Effective Vds transition smoothing [V]
    pub delta: f64,
}

impl Default for BsimParams {
    fn default() -> Self {
        Self::nmos_default()
    }
}

impl BsimParams {
    /// Create NMOS default parameters
    pub fn nmos_default() -> Self {
        BsimParams {
            // Model Selection
            level: 49,
            mos_type: MosType::Nmos,

            // Threshold Voltage
            vth0: 0.7,
            k1: 0.5,
            k2: 0.0,
            dvt0: 2.2,
            dvt1: 0.53,
            dvt2: -0.032,
            eta0: 0.08,
            dsub: 0.56,
            nlx: 1.74e-7,
            nfactor: 1.0,

            // Mobility
            u0: 500.0,    // cm^2/V/s for NMOS
            ua: 2.25e-9,  // m/V
            ub: 5.87e-19, // (m/V)^2
            uc: -4.65e-11,
            vsat: 1.5e5,  // m/s
            a0: 1.0,
            ags: 0.2,
            prwg: 0.0,
            prwb: 0.0,

            // Short-channel/Output Conductance
            pclm: 1.3,
            pdiblc1: 0.39,
            pdiblc2: 0.0086,
            pdiblcb: -0.1,
            drout: 0.56,
            pscbe1: 4.24e8,
            pscbe2: 1.0e-5,
            alpha0: 0.0,
            beta0: 30.0,

            // Geometry
            tox: 1.5e-8,  // 15nm
            lint: 0.0,
            wint: 0.0,
            lmin: 0.0,
            wmin: 0.0,
            lln: 1.0,
            lw: 0.0,
            lwn: 1.0,
            wln: 1.0,
            ww: 0.0,
            wwn: 1.0,

            // Parasitic
            rdsw: 0.0,
            rsh: 0.0,

            // Temperature
            tnom: T_NOMINAL,
            ute: -1.5,
            kt1: -0.11,
            kt1l: 0.0,
            kt2: 0.022,
            at: 3.3e4,
            prt: 0.0,

            // Capacitance
            cgso: 0.0,
            cgdo: 0.0,
            cgbo: 0.0,
            cj: 5.0e-4,
            cjsw: 5.0e-10,
            pb: 1.0,
            pbsw: 1.0,
            mj: 0.5,
            mjsw: 0.33,

            // Noise
            kf: 0.0,
            af: 1.0,
            ef: 1.0,

            // ============ BSIM4 Parameters ============
            // Enhanced Threshold Voltage
            dvt0w: 0.0,
            dvt1w: 5.3e6,
            dvt2w: -0.032,
            voff: -0.1,
            voffl: 0.0,
            minv: 0.0,
            k3: 80.0,
            k3b: 0.0,
            w0: 0.0,
            lpe0: 1.74e-7,
            lpeb: 0.0,
            vfb: -1.0,

            // Enhanced Mobility
            ute0: -1.5,
            ute1: 0.0,
            pemod: 0,
            up: 0.0,
            lp: 1e-5,
            ud: 0.0,
            ud1: 0.0,
            eu: 1.67,

            // Enhanced Velocity Saturation
            vs: 0.0,
            vsattemp: 0.0,
            lambda: 0.0,
            vtl: 2.0e5,
            lc: 5e-9,

            // Substrate Current (Impact Ionization)
            alpha1: 0.0,
            beta1: 0.0,

            // Stress Effects
            saref: 1e-6,
            sbref: 1e-6,
            wlod: 0.0,
            ku0: 0.0,
            kvth0: 0.0,
            ku0mult: 1.0,
            tku0: 0.0,

            // Gate Tunneling
            jtss: 0.0,
            jtsd: 0.0,
            nsti: 1.0,
            vtss: 10.0,
            vtsd: 10.0,

            // Enhanced Output Conductance
            pvag: 0.0,
            fprout: 0.0,
            pdits: 0.0,
            delta: 0.01,
        }
    }

    /// Create PMOS default parameters
    pub fn pmos_default() -> Self {
        let mut params = Self::nmos_default();
        params.mos_type = MosType::Pmos;
        params.vth0 = -0.7;      // Negative for PMOS
        params.u0 = 150.0;       // Lower mobility for holes
        params.ute = -1.0;       // Different temp coefficient
        params.kt1 = -0.08;
        // BSIM4 PMOS adjustments
        params.voff = 0.1;       // Positive for PMOS
        params.ute0 = -1.0;      // Different temp coefficient for PMOS
        params
    }

    /// Calculate oxide capacitance per unit area [F/m^2]
    pub fn cox(&self) -> f64 {
        EPSILON_OX / self.tox
    }

    /// Calculate effective channel length [m]
    pub fn leff(&self, l: f64) -> f64 {
        (l - 2.0 * self.lint).max(1e-9)
    }

    /// Calculate effective channel width [m]
    pub fn weff(&self, w: f64) -> f64 {
        (w - 2.0 * self.wint).max(1e-9)
    }

    /// Calculate thermal voltage at given temperature [V]
    pub fn vt(&self, temp: f64) -> f64 {
        K_BOLTZMANN * temp / Q_ELECTRON
    }
}
