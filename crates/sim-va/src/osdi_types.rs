//! OSDI (Open Source Device Interface) v0.3 C ABI type definitions.
//!
//! These `#[repr(C)]` structs must match the binary layout produced by OpenVAF.
//! Reference: ngspice src/osdi/osdi.h and OSDI v0.3 specification.

use std::os::raw::{c_char, c_void};

// ---------------------------------------------------------------------------
// Version constants
// ---------------------------------------------------------------------------
pub const OSDI_VERSION_MAJOR: u32 = 0;
pub const OSDI_VERSION_MINOR: u32 = 3;

// ---------------------------------------------------------------------------
// Parameter type masks
// ---------------------------------------------------------------------------
pub const PARA_TY_MASK: u32 = 3;
pub const PARA_TY_REAL: u32 = 0;
pub const PARA_TY_INT: u32 = 1;
pub const PARA_TY_STR: u32 = 2;

pub const PARA_KIND_MASK: u32 = 3 << 30;
pub const PARA_KIND_MODEL: u32 = 0 << 30;
pub const PARA_KIND_INST: u32 = 1 << 30;
pub const PARA_KIND_OPVAR: u32 = 2 << 30;

// ---------------------------------------------------------------------------
// Access flags
// ---------------------------------------------------------------------------
pub const ACCESS_FLAG_READ: u32 = 0;
pub const ACCESS_FLAG_SET: u32 = 1;
pub const ACCESS_FLAG_INSTANCE: u32 = 4;

// ---------------------------------------------------------------------------
// Jacobian entry flags
// ---------------------------------------------------------------------------
pub const JACOBIAN_ENTRY_RESIST_CONST: u32 = 1;
pub const JACOBIAN_ENTRY_REACT_CONST: u32 = 2;
pub const JACOBIAN_ENTRY_RESIST: u32 = 4;
pub const JACOBIAN_ENTRY_REACT: u32 = 8;

// ---------------------------------------------------------------------------
// Calculation / analysis flags (passed to eval)
// ---------------------------------------------------------------------------
pub const CALC_RESIST_RESIDUAL: u32 = 1;
pub const CALC_REACT_RESIDUAL: u32 = 2;
pub const CALC_RESIST_JACOBIAN: u32 = 4;
pub const CALC_REACT_JACOBIAN: u32 = 8;
pub const CALC_NOISE: u32 = 16;
pub const CALC_OP: u32 = 32;
pub const CALC_RESIST_LIM_RHS: u32 = 64;
pub const CALC_REACT_LIM_RHS: u32 = 128;
pub const ENABLE_LIM: u32 = 256;
pub const INIT_LIM: u32 = 512;

pub const ANALYSIS_NOISE: u32 = 1024;
pub const ANALYSIS_DC: u32 = 2048;
pub const ANALYSIS_AC: u32 = 4096;
pub const ANALYSIS_TRAN: u32 = 8192;
pub const ANALYSIS_IC: u32 = 16384;
pub const ANALYSIS_STATIC: u32 = 32768;
pub const ANALYSIS_NODESET: u32 = 65536;

// ---------------------------------------------------------------------------
// Eval return flags
// ---------------------------------------------------------------------------
pub const EVAL_RET_FLAG_LIM: u32 = 1;
pub const EVAL_RET_FLAG_FATAL: u32 = 2;
pub const EVAL_RET_FLAG_FINISH: u32 = 4;
pub const EVAL_RET_FLAG_STOP: u32 = 8;

// ---------------------------------------------------------------------------
// Init error codes
// ---------------------------------------------------------------------------
pub const INIT_ERR_OUT_OF_BOUNDS: u32 = 1;

// ---------------------------------------------------------------------------
// Structs
// ---------------------------------------------------------------------------

/// Limiting function descriptor.
#[repr(C)]
pub struct OsdiLimFunction {
    pub name: *const c_char,
    pub num_args: u32,
    pub func_ptr: *const c_void,
}

/// Simulator parameter arrays passed during setup/eval.
#[repr(C)]
pub struct OsdiSimParas {
    pub names: *const *const c_char,
    pub vals: *const f64,
    pub names_str: *const *const c_char,
    pub vals_str: *const *const c_char,
}

/// Simulation info passed to the eval function.
#[repr(C)]
pub struct OsdiSimInfo {
    pub paras: OsdiSimParas,
    pub abstime: f64,
    pub prev_solve: *const f64,
    pub prev_state: *const f64,
    pub next_state: *mut f64,
    pub flags: u32,
}

/// Payload for initialization errors (union in C, we use the largest member).
#[repr(C)]
#[derive(Clone, Copy)]
pub union OsdiInitErrorPayload {
    pub parameter_id: u32,
}

/// A single initialization error.
#[repr(C)]
pub struct OsdiInitError {
    pub code: u32,
    pub payload: OsdiInitErrorPayload,
}

/// Result of setup_model / setup_instance calls.
#[repr(C)]
pub struct OsdiInitInfo {
    pub flags: u32,
    pub num_errors: u32,
    pub errors: *const OsdiInitError,
}

/// A pair of node indices (used for collapsible pairs and noise sources).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OsdiNodePair {
    pub node_1: u32,
    pub node_2: u32,
}

/// A Jacobian matrix entry descriptor.
#[repr(C)]
pub struct OsdiJacobianEntry {
    pub nodes: OsdiNodePair,
    /// Offset into instance data for the reactive Jacobian pointer.
    pub react_ptr_off: u32,
    /// Flags indicating which components exist (JACOBIAN_ENTRY_RESIST, etc).
    pub flags: u32,
}

/// A circuit node descriptor.
#[repr(C)]
pub struct OsdiNode {
    pub name: *const c_char,
    pub units: *const c_char,
    pub residual_units: *const c_char,
    /// Offset in instance data for resistive residual.
    pub resist_residual_off: u32,
    /// Offset in instance data for reactive residual.
    pub react_residual_off: u32,
    /// Offset in instance data for resistive limit RHS.
    pub resist_limit_rhs_off: u32,
    /// Offset in instance data for reactive limit RHS.
    pub react_limit_rhs_off: u32,
    /// True if this is a flow (current) node rather than a potential (voltage) node.
    pub is_flow: bool,
}

/// A parameter or operating variable descriptor.
#[repr(C)]
pub struct OsdiParamOpvar {
    /// Pointer to array of name strings (primary name + aliases).
    pub name: *const *const c_char,
    pub num_alias: u32,
    pub description: *const c_char,
    pub units: *const c_char,
    pub flags: u32,
    /// 0 for scalar, >0 for array length.
    pub len: u32,
}

/// Noise source descriptor.
#[repr(C)]
pub struct OsdiNoiseSource {
    pub name: *const c_char,
    pub nodes: OsdiNodePair,
}

// ---------------------------------------------------------------------------
// Function pointer types
// ---------------------------------------------------------------------------

/// Access function: returns a pointer to a parameter value within model/instance data.
pub type OsdiAccessFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    id: u32,
    flags: u32,
) -> *mut c_void;

/// Setup model: initializes model-level data and validates parameters.
pub type OsdiSetupModelFn = unsafe extern "C" fn(
    handle: *mut c_void,
    model: *mut c_void,
    sim_params: *const OsdiSimParas,
    res: *mut OsdiInitInfo,
);

/// Setup instance: initializes instance-level data, handles node collapsing.
pub type OsdiSetupInstanceFn = unsafe extern "C" fn(
    handle: *mut c_void,
    inst: *mut c_void,
    model: *mut c_void,
    temperature: f64,
    num_terminals: u32,
    sim_params: *const OsdiSimParas,
    res: *mut OsdiInitInfo,
);

/// Eval function: evaluates device equations at given operating point.
/// Returns flags (EVAL_RET_FLAG_*).
pub type OsdiEvalFn = unsafe extern "C" fn(
    handle: *mut c_void,
    inst: *mut c_void,
    model: *const c_void,
    info: *const OsdiSimInfo,
) -> u32;

/// Load noise density values.
pub type OsdiLoadNoiseFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    freq: f64,
    noise_dens: *mut f64,
);

/// Load residual values (resistive or reactive) into destination array.
pub type OsdiLoadResidualFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    dst: *mut f64,
);

/// Load SPICE RHS for DC analysis.
pub type OsdiLoadSpiceRhsDcFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    dst: *mut f64,
    prev_solve: *mut f64,
);

/// Load SPICE RHS for transient analysis (with integration factor alpha).
pub type OsdiLoadSpiceRhsTranFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    dst: *mut f64,
    prev_solve: *mut f64,
    alpha: f64,
);

/// Load Jacobian entries (resistive).
pub type OsdiLoadJacobianFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
);

/// Load Jacobian entries (reactive or transient combined, with alpha = 1/dt).
pub type OsdiLoadJacobianAlphaFn = unsafe extern "C" fn(
    inst: *mut c_void,
    model: *mut c_void,
    alpha: f64,
);

// ---------------------------------------------------------------------------
// Main descriptor struct
// ---------------------------------------------------------------------------

/// The top-level OSDI device descriptor.
///
/// One `.osdi` shared library may contain multiple descriptors (one per
/// Verilog-A module). The library exports:
/// - `OSDI_NUM_DESCRIPTORS: u32` - number of descriptors
/// - `OSDI_DESCRIPTORS: *const *const OsdiDescriptor` - array of pointers
/// - `OSDI_VERSION_MAJOR: u32`
/// - `OSDI_VERSION_MINOR: u32`
#[repr(C)]
pub struct OsdiDescriptor {
    pub name: *const c_char,

    // Nodes
    pub num_nodes: u32,
    pub num_terminals: u32,
    pub nodes: *const OsdiNode,

    // Jacobian
    pub num_jacobian_entries: u32,
    pub jacobian_entries: *const OsdiJacobianEntry,

    // Collapsible node pairs
    pub num_collapsible: u32,
    pub collapsible: *const OsdiNodePair,
    pub collapsed_offset: u32,

    // Noise
    pub noise_sources: *const OsdiNoiseSource,
    pub num_noise_src: u32,

    // Parameters and operating variables
    pub num_params: u32,
    pub num_instance_params: u32,
    pub num_opvars: u32,
    pub param_opvar: *const OsdiParamOpvar,

    // Instance data layout offsets
    pub node_mapping_offset: u32,
    pub jacobian_ptr_resist_offset: u32,

    // State variables (for transient)
    pub num_states: u32,
    pub state_idx_off: u32,

    // Bound step
    pub bound_step_offset: u32,

    // Allocation sizes
    pub instance_size: u32,
    pub model_size: u32,

    // Function pointers
    pub access: OsdiAccessFn,
    pub setup_model: OsdiSetupModelFn,
    pub setup_instance: OsdiSetupInstanceFn,
    pub eval: OsdiEvalFn,
    pub load_noise: OsdiLoadNoiseFn,
    pub load_residual_resist: OsdiLoadResidualFn,
    pub load_residual_react: OsdiLoadResidualFn,
    pub load_limit_rhs_resist: OsdiLoadResidualFn,
    pub load_limit_rhs_react: OsdiLoadResidualFn,
    pub load_spice_rhs_dc: OsdiLoadSpiceRhsDcFn,
    pub load_spice_rhs_tran: OsdiLoadSpiceRhsTranFn,
    pub load_jacobian_resist: OsdiLoadJacobianFn,
    pub load_jacobian_react: OsdiLoadJacobianAlphaFn,
    pub load_jacobian_tran: OsdiLoadJacobianAlphaFn,
}
