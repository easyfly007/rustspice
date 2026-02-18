use sim_va::osdi_types::*;
use sim_va::osdi_loader::ParamInfo;

// ---------------------------------------------------------------------------
// ParamInfo type classification
// ---------------------------------------------------------------------------

fn make_param(flags: u32) -> ParamInfo {
    ParamInfo {
        name: "test_param".into(),
        aliases: vec![],
        description: "test".into(),
        units: "V".into(),
        flags,
        len: 0,
        index: 0,
    }
}

#[test]
fn param_info_is_model_param() {
    let p = make_param(PARA_KIND_MODEL | PARA_TY_REAL);
    assert!(p.is_model_param());
    assert!(!p.is_instance_param());
    assert!(!p.is_opvar());
    assert!(p.is_real());
}

#[test]
fn param_info_is_instance_param() {
    let p = make_param(PARA_KIND_INST | PARA_TY_REAL);
    assert!(!p.is_model_param());
    assert!(p.is_instance_param());
    assert!(!p.is_opvar());
}

#[test]
fn param_info_is_opvar() {
    let p = make_param(PARA_KIND_OPVAR | PARA_TY_REAL);
    assert!(!p.is_model_param());
    assert!(!p.is_instance_param());
    assert!(p.is_opvar());
}

#[test]
fn param_info_type_real() {
    let p = make_param(PARA_TY_REAL);
    assert_eq!(p.param_type(), PARA_TY_REAL);
    assert!(p.is_real());
}

#[test]
fn param_info_type_int() {
    let p = make_param(PARA_TY_INT);
    assert_eq!(p.param_type(), PARA_TY_INT);
    assert!(!p.is_real());
}

#[test]
fn param_info_type_str() {
    let p = make_param(PARA_TY_STR);
    assert_eq!(p.param_type(), PARA_TY_STR);
    assert!(!p.is_real());
}

#[test]
fn param_info_combined_flags() {
    // Model-level integer parameter
    let p = make_param(PARA_KIND_MODEL | PARA_TY_INT);
    assert!(p.is_model_param());
    assert!(!p.is_real());
    assert_eq!(p.param_type(), PARA_TY_INT);
}

// ---------------------------------------------------------------------------
// OSDI type constants
// ---------------------------------------------------------------------------

#[test]
fn osdi_version_constants() {
    assert_eq!(OSDI_VERSION_MAJOR, 0);
    assert_eq!(OSDI_VERSION_MINOR, 3);
}

#[test]
fn parameter_type_masks() {
    // PARA_TY_MASK should extract lower 2 bits
    assert_eq!(PARA_TY_REAL & PARA_TY_MASK, PARA_TY_REAL);
    assert_eq!(PARA_TY_INT & PARA_TY_MASK, PARA_TY_INT);
    assert_eq!(PARA_TY_STR & PARA_TY_MASK, PARA_TY_STR);
}

#[test]
fn parameter_kind_masks() {
    // PARA_KIND_MASK should extract top 2 bits (bits 30-31)
    assert_eq!(PARA_KIND_MODEL & PARA_KIND_MASK, PARA_KIND_MODEL);
    assert_eq!(PARA_KIND_INST & PARA_KIND_MASK, PARA_KIND_INST);
    assert_eq!(PARA_KIND_OPVAR & PARA_KIND_MASK, PARA_KIND_OPVAR);

    // Kind and type should not overlap
    assert_eq!(PARA_KIND_MODEL & PARA_TY_MASK, 0);
    assert_eq!(PARA_KIND_INST & PARA_TY_MASK, 0);
}

#[test]
fn calc_flags_are_distinct_powers_of_two() {
    let flags = [
        CALC_RESIST_RESIDUAL,
        CALC_REACT_RESIDUAL,
        CALC_RESIST_JACOBIAN,
        CALC_REACT_JACOBIAN,
        CALC_NOISE,
        CALC_OP,
        CALC_RESIST_LIM_RHS,
        CALC_REACT_LIM_RHS,
        ENABLE_LIM,
        INIT_LIM,
    ];
    // Each flag should be a power of 2
    for &flag in &flags {
        assert!(flag.is_power_of_two(), "Flag {} is not a power of 2", flag);
    }
    // All flags should be distinct
    for i in 0..flags.len() {
        for j in (i + 1)..flags.len() {
            assert_ne!(flags[i], flags[j], "Flags at {} and {} collide", i, j);
        }
    }
}

#[test]
fn analysis_flags_are_distinct_powers_of_two() {
    let flags = [
        ANALYSIS_NOISE,
        ANALYSIS_DC,
        ANALYSIS_AC,
        ANALYSIS_TRAN,
        ANALYSIS_IC,
        ANALYSIS_STATIC,
        ANALYSIS_NODESET,
    ];
    for &flag in &flags {
        assert!(flag.is_power_of_two(), "Flag {} is not a power of 2", flag);
    }
    for i in 0..flags.len() {
        for j in (i + 1)..flags.len() {
            assert_ne!(flags[i], flags[j]);
        }
    }
}

#[test]
fn eval_return_flags() {
    assert_eq!(EVAL_RET_FLAG_LIM, 1);
    assert_eq!(EVAL_RET_FLAG_FATAL, 2);
    assert_eq!(EVAL_RET_FLAG_FINISH, 4);
    assert_eq!(EVAL_RET_FLAG_STOP, 8);
}

#[test]
fn access_flags() {
    assert_eq!(ACCESS_FLAG_READ, 0);
    assert_eq!(ACCESS_FLAG_SET, 1);
    assert_eq!(ACCESS_FLAG_INSTANCE, 4);
    // SET and INSTANCE can be combined
    let combined = ACCESS_FLAG_SET | ACCESS_FLAG_INSTANCE;
    assert_eq!(combined, 5);
}

#[test]
fn jacobian_entry_flags() {
    assert_ne!(JACOBIAN_ENTRY_RESIST_CONST, JACOBIAN_ENTRY_REACT_CONST);
    assert_ne!(JACOBIAN_ENTRY_RESIST, JACOBIAN_ENTRY_REACT);
    // RESIST and RESIST_CONST should be different bits
    assert_ne!(JACOBIAN_ENTRY_RESIST, JACOBIAN_ENTRY_RESIST_CONST);
}

// ---------------------------------------------------------------------------
// SafeDescriptor (with aliases)
// ---------------------------------------------------------------------------

#[test]
fn param_info_with_aliases() {
    let p = ParamInfo {
        name: "vth0".into(),
        aliases: vec!["vth".into(), "vtho".into()],
        description: "threshold voltage".into(),
        units: "V".into(),
        flags: PARA_KIND_MODEL | PARA_TY_REAL,
        len: 0,
        index: 3,
    };
    assert_eq!(p.name, "vth0");
    assert_eq!(p.aliases.len(), 2);
    assert_eq!(p.index, 3);
    assert!(p.is_model_param());
    assert!(p.is_real());
}

// ---------------------------------------------------------------------------
// OsdiNodePair struct
// ---------------------------------------------------------------------------

#[test]
fn node_pair_clone_and_debug() {
    let pair = OsdiNodePair {
        node_1: 0,
        node_2: 1,
    };
    let pair2 = pair;
    assert_eq!(pair2.node_1, 0);
    assert_eq!(pair2.node_2, 1);
    let debug = format!("{:?}", pair2);
    assert!(debug.contains("0"));
    assert!(debug.contains("1"));
}

// ---------------------------------------------------------------------------
// JacobianEntryInfo
// ---------------------------------------------------------------------------

#[test]
fn jacobian_entry_info_clone_and_debug() {
    use sim_va::osdi_loader::JacobianEntryInfo;

    let entry = JacobianEntryInfo {
        row_node: 0,
        col_node: 1,
        react_ptr_off: 128,
        flags: JACOBIAN_ENTRY_RESIST | JACOBIAN_ENTRY_REACT,
    };
    let cloned = entry.clone();
    assert_eq!(cloned.row_node, 0);
    assert_eq!(cloned.col_node, 1);
    assert_eq!(cloned.react_ptr_off, 128);

    let debug = format!("{:?}", cloned);
    assert!(debug.contains("row_node"));
}
