//! Verilog-A compiler wrapper.
//!
//! Invokes OpenVAF as a subprocess to compile `.va` files to `.osdi` shared
//! libraries. Compiled artifacts are cached by content hash so recompilation
//! only happens when the source changes.

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use sha2::{Digest, Sha256};

use crate::error::VaError;

/// Wrapper around the OpenVAF compiler.
pub struct VaCompiler {
    /// Directory for cached `.osdi` files.
    cache_dir: PathBuf,
    /// Path or name of the `openvaf` binary.
    openvaf_path: String,
}

impl VaCompiler {
    /// Create a compiler with default settings.
    ///
    /// Cache directory: `$RUSTSPICE_VA_CACHE`, or `~/.cache/rustspice/va/`.
    /// OpenVAF binary: `$OPENVAF_PATH`, or `"openvaf"` (found via PATH).
    pub fn new() -> Self {
        let cache_dir = std::env::var("RUSTSPICE_VA_CACHE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
                PathBuf::from(home).join(".cache").join("rustspice").join("va")
            });

        let openvaf_path = std::env::var("OPENVAF_PATH").unwrap_or_else(|_| "openvaf".into());

        Self {
            cache_dir,
            openvaf_path,
        }
    }

    /// Create a compiler with explicit paths.
    pub fn with_paths(cache_dir: PathBuf, openvaf_path: String) -> Self {
        Self {
            cache_dir,
            openvaf_path,
        }
    }

    /// Check that OpenVAF is installed and return its version string.
    pub fn check_openvaf(&self) -> Result<String, VaError> {
        let output = Command::new(&self.openvaf_path)
            .arg("--version")
            .output()
            .map_err(|_| VaError::OpenvafNotFound)?;

        if !output.status.success() {
            return Err(VaError::OpenvafNotFound);
        }

        let version = String::from_utf8_lossy(&output.stdout).trim().to_string();
        Ok(version)
    }

    /// Compile a `.va` file to `.osdi`, returning the path to the compiled library.
    ///
    /// Uses content-hash caching: if the `.va` file content has not changed
    /// since the last compilation, the cached `.osdi` is returned directly.
    pub fn compile(&self, va_path: &Path) -> Result<PathBuf, VaError> {
        // Verify source file exists
        if !va_path.exists() {
            return Err(VaError::VaFileNotFound(va_path.to_path_buf()));
        }

        // Read source and compute content hash
        let content = fs::read(va_path).map_err(VaError::CacheError)?;
        let hash = {
            let mut hasher = Sha256::new();
            hasher.update(&content);
            format!("{:x}", hasher.finalize())
        };

        // Derive the stem from the original filename
        let stem = va_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("model");
        let osdi_name = format!("{}_{}.osdi", stem, &hash[..16]);

        // Ensure cache directory exists
        fs::create_dir_all(&self.cache_dir).map_err(VaError::CacheError)?;

        let osdi_path = self.cache_dir.join(&osdi_name);

        // Return cached result if it exists
        if osdi_path.exists() {
            return Ok(osdi_path);
        }

        // Compile with OpenVAF
        let output = Command::new(&self.openvaf_path)
            .arg(va_path)
            .arg("-o")
            .arg(&osdi_path)
            .output()
            .map_err(|_| VaError::OpenvafNotFound)?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr).to_string();
            let exit_code = output.status.code().unwrap_or(-1);

            // Clean up partial output
            let _ = fs::remove_file(&osdi_path);

            return Err(VaError::OpenvafFailed { stderr, exit_code });
        }

        // Verify the .osdi file was actually created
        if !osdi_path.exists() {
            return Err(VaError::OpenvafFailed {
                stderr: "openvaf produced no output file".into(),
                exit_code: 0,
            });
        }

        Ok(osdi_path)
    }

    /// Compile a `.va` file, or if the path ends in `.osdi`, just return it.
    ///
    /// This is a convenience method for handling both `.hdl` and `.osdi`
    /// directives uniformly.
    pub fn compile_or_passthrough(&self, path: &Path) -> Result<PathBuf, VaError> {
        match path.extension().and_then(|e| e.to_str()) {
            Some("osdi") | Some("so") | Some("dll") | Some("dylib") => {
                if !path.exists() {
                    return Err(VaError::LoadFailed {
                        path: path.to_path_buf(),
                        cause: "file not found".into(),
                    });
                }
                Ok(path.to_path_buf())
            }
            _ => self.compile(path),
        }
    }
}

impl Default for VaCompiler {
    fn default() -> Self {
        Self::new()
    }
}
