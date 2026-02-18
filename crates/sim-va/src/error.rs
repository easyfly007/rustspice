use std::fmt;
use std::path::PathBuf;

/// Errors that can occur during Verilog-A model compilation, loading, or evaluation.
#[derive(Debug)]
pub enum VaError {
    // Compiler errors
    /// OpenVAF binary not found in PATH
    OpenvafNotFound,
    /// OpenVAF compilation failed
    OpenvafFailed { stderr: String, exit_code: i32 },
    /// Verilog-A source file not found
    VaFileNotFound(PathBuf),
    /// Cache directory I/O error
    CacheError(std::io::Error),

    // Loader errors
    /// Failed to dlopen the .osdi shared library
    LoadFailed { path: PathBuf, cause: String },
    /// Required OSDI symbol not found in library
    MissingSymbol(String),
    /// OSDI version mismatch
    VersionMismatch { expected: String, found: String },
    /// Module name not found in loaded library
    ModuleNotFound(String),
    /// Descriptor data is invalid or corrupt
    InvalidDescriptor(String),

    // Runtime errors
    /// OSDI setup_model() returned errors
    SetupModelFailed(Vec<String>),
    /// OSDI setup_instance() returned errors
    SetupInstanceFailed(Vec<String>),
    /// OSDI eval() returned non-zero
    EvalFailed(i32),
    /// Node mapping between OSDI and MNA failed
    NodeMappingError(String),
    /// Parameter name not found or value invalid
    ParameterError { name: String, cause: String },
}

impl fmt::Display for VaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VaError::OpenvafNotFound => write!(
                f,
                "openvaf not found. Install from https://github.com/pascalkuthe/OpenVAF or set OPENVAF_PATH"
            ),
            VaError::OpenvafFailed { stderr, exit_code } => {
                write!(f, "openvaf failed (exit {}): {}", exit_code, stderr)
            }
            VaError::VaFileNotFound(path) => write!(f, "Verilog-A file not found: {}", path.display()),
            VaError::CacheError(e) => write!(f, "VA cache error: {}", e),
            VaError::LoadFailed { path, cause } => {
                write!(f, "failed to load OSDI library {}: {}", path.display(), cause)
            }
            VaError::MissingSymbol(sym) => write!(f, "missing OSDI symbol: {}", sym),
            VaError::VersionMismatch { expected, found } => {
                write!(f, "OSDI version mismatch: expected {}, found {}", expected, found)
            }
            VaError::ModuleNotFound(name) => write!(f, "OSDI module not found: {}", name),
            VaError::InvalidDescriptor(msg) => write!(f, "invalid OSDI descriptor: {}", msg),
            VaError::SetupModelFailed(errors) => {
                write!(f, "OSDI setup_model failed: {}", errors.join("; "))
            }
            VaError::SetupInstanceFailed(errors) => {
                write!(f, "OSDI setup_instance failed: {}", errors.join("; "))
            }
            VaError::EvalFailed(rc) => write!(f, "OSDI eval failed with code {}", rc),
            VaError::NodeMappingError(msg) => write!(f, "VA node mapping error: {}", msg),
            VaError::ParameterError { name, cause } => {
                write!(f, "VA parameter '{}' error: {}", name, cause)
            }
        }
    }
}

impl std::error::Error for VaError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            VaError::CacheError(e) => Some(e),
            _ => None,
        }
    }
}
