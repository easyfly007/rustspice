use sim_va::VaError;
use std::path::PathBuf;

#[test]
fn display_openvaf_not_found() {
    let err = VaError::OpenvafNotFound;
    let msg = err.to_string();
    assert!(msg.contains("openvaf not found"), "Got: {}", msg);
    assert!(msg.contains("OpenVAF") || msg.contains("OPENVAF_PATH"));
}

#[test]
fn display_openvaf_failed() {
    let err = VaError::OpenvafFailed {
        stderr: "syntax error on line 42".into(),
        exit_code: 1,
    };
    let msg = err.to_string();
    assert!(msg.contains("failed"), "Got: {}", msg);
    assert!(msg.contains("syntax error on line 42"));
    assert!(msg.contains("1")); // exit code
}

#[test]
fn display_va_file_not_found() {
    let err = VaError::VaFileNotFound(PathBuf::from("/path/to/model.va"));
    let msg = err.to_string();
    assert!(msg.contains("not found"), "Got: {}", msg);
    assert!(msg.contains("model.va"));
}

#[test]
fn display_cache_error() {
    let io_err = std::io::Error::new(std::io::ErrorKind::PermissionDenied, "access denied");
    let err = VaError::CacheError(io_err);
    let msg = err.to_string();
    assert!(msg.contains("cache") || msg.contains("Cache"), "Got: {}", msg);
    assert!(msg.contains("access denied"));
}

#[test]
fn display_load_failed() {
    let err = VaError::LoadFailed {
        path: PathBuf::from("broken.osdi"),
        cause: "invalid ELF header".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("load") || msg.contains("failed"), "Got: {}", msg);
    assert!(msg.contains("broken.osdi"));
    assert!(msg.contains("invalid ELF header"));
}

#[test]
fn display_missing_symbol() {
    let err = VaError::MissingSymbol("OSDI_DESCRIPTORS".into());
    let msg = err.to_string();
    assert!(msg.contains("OSDI_DESCRIPTORS"), "Got: {}", msg);
}

#[test]
fn display_version_mismatch() {
    let err = VaError::VersionMismatch {
        expected: "0.3".into(),
        found: "1.0".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("0.3"), "Got: {}", msg);
    assert!(msg.contains("1.0"));
    assert!(msg.contains("mismatch") || msg.contains("version"));
}

#[test]
fn display_module_not_found() {
    let err = VaError::ModuleNotFound("ekv".into());
    let msg = err.to_string();
    assert!(msg.contains("ekv"), "Got: {}", msg);
}

#[test]
fn display_invalid_descriptor() {
    let err = VaError::InvalidDescriptor("null module name".into());
    let msg = err.to_string();
    assert!(msg.contains("null module name"), "Got: {}", msg);
}

#[test]
fn display_setup_model_failed() {
    let err = VaError::SetupModelFailed(vec![
        "error code=1, parameter_id=5".into(),
        "error code=1, parameter_id=8".into(),
    ]);
    let msg = err.to_string();
    assert!(msg.contains("setup_model"), "Got: {}", msg);
    assert!(msg.contains("parameter_id=5"));
}

#[test]
fn display_setup_instance_failed() {
    let err = VaError::SetupInstanceFailed(vec!["node mismatch".into()]);
    let msg = err.to_string();
    assert!(msg.contains("setup_instance"), "Got: {}", msg);
}

#[test]
fn display_eval_failed() {
    let err = VaError::EvalFailed(2);
    let msg = err.to_string();
    assert!(msg.contains("eval") || msg.contains("Eval"), "Got: {}", msg);
    assert!(msg.contains("2"));
}

#[test]
fn display_node_mapping_error() {
    let err = VaError::NodeMappingError("module 'psp' expects 4 terminals, got 3".into());
    let msg = err.to_string();
    assert!(msg.contains("4 terminals"), "Got: {}", msg);
}

#[test]
fn display_parameter_error() {
    let err = VaError::ParameterError {
        name: "vto".into(),
        cause: "invalid numeric value: abc".into(),
    };
    let msg = err.to_string();
    assert!(msg.contains("vto"), "Got: {}", msg);
    assert!(msg.contains("abc"));
}

#[test]
fn error_is_std_error() {
    let err = VaError::OpenvafNotFound;
    let _: &dyn std::error::Error = &err;
}

#[test]
fn cache_error_has_source() {
    let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "missing");
    let err = VaError::CacheError(io_err);
    let source = std::error::Error::source(&err);
    assert!(source.is_some());
}

#[test]
fn non_cache_error_has_no_source() {
    let err = VaError::OpenvafNotFound;
    let source = std::error::Error::source(&err);
    assert!(source.is_none());
}

#[test]
fn error_debug_format() {
    let err = VaError::ModuleNotFound("bsimcmg".into());
    let debug = format!("{:?}", err);
    assert!(debug.contains("ModuleNotFound"));
    assert!(debug.contains("bsimcmg"));
}
