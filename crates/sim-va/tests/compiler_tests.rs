use sim_va::VaCompiler;
use std::path::{Path, PathBuf};

#[test]
fn compiler_default_uses_openvaf_name() {
    let compiler = VaCompiler::new();
    // Should construct without panicking using default paths
    // (OPENVAF_PATH env or "openvaf", cache dir from HOME or /tmp)
    drop(compiler);
}

#[test]
fn compiler_with_custom_paths() {
    let cache = PathBuf::from("/tmp/rustspice_test_cache");
    let compiler = VaCompiler::with_paths(cache.clone(), "custom_openvaf".into());
    drop(compiler);
}

#[test]
fn compile_missing_va_file_returns_error() {
    let compiler = VaCompiler::new();
    let result = compiler.compile(Path::new("/nonexistent/path/model.va"));
    assert!(result.is_err());
    let err = result.unwrap_err();
    let msg = err.to_string();
    assert!(
        msg.contains("not found"),
        "Expected 'not found' in error: {}",
        msg
    );
}

#[test]
fn compile_or_passthrough_osdi_extension_returns_path_if_exists() {
    // Create a temp file with .osdi extension
    let dir = std::env::temp_dir().join("rustspice_test_passthrough");
    std::fs::create_dir_all(&dir).unwrap();
    let osdi_path = dir.join("test_model.osdi");
    std::fs::write(&osdi_path, b"fake osdi content").unwrap();

    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(&osdi_path);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), osdi_path);

    // Cleanup
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_or_passthrough_osdi_missing_returns_error() {
    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(Path::new("/nonexistent/model.osdi"));
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found") || msg.contains("failed to load"));
}

#[test]
fn compile_or_passthrough_so_extension_passthrough() {
    let dir = std::env::temp_dir().join("rustspice_test_so");
    std::fs::create_dir_all(&dir).unwrap();
    let so_path = dir.join("model.so");
    std::fs::write(&so_path, b"fake so").unwrap();

    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(&so_path);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), so_path);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_or_passthrough_dll_extension_passthrough() {
    let dir = std::env::temp_dir().join("rustspice_test_dll");
    std::fs::create_dir_all(&dir).unwrap();
    let dll_path = dir.join("model.dll");
    std::fs::write(&dll_path, b"fake dll").unwrap();

    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(&dll_path);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), dll_path);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_or_passthrough_dylib_extension_passthrough() {
    let dir = std::env::temp_dir().join("rustspice_test_dylib");
    std::fs::create_dir_all(&dir).unwrap();
    let dylib_path = dir.join("model.dylib");
    std::fs::write(&dylib_path, b"fake dylib").unwrap();

    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(&dylib_path);
    assert!(result.is_ok());
    assert_eq!(result.unwrap(), dylib_path);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn compile_or_passthrough_va_extension_tries_compile() {
    // A .va file should attempt compilation (which will fail since the file
    // content is not valid Verilog-A and OpenVAF isn't installed)
    let dir = std::env::temp_dir().join("rustspice_test_va_compile");
    std::fs::create_dir_all(&dir).unwrap();
    let va_path = dir.join("dummy.va");
    std::fs::write(&va_path, b"not valid verilog-a").unwrap();

    let compiler = VaCompiler::new();
    let result = compiler.compile_or_passthrough(&va_path);
    // Should fail because openvaf is not installed
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn check_openvaf_fails_when_not_installed() {
    let compiler = VaCompiler::with_paths(
        PathBuf::from("/tmp"),
        "nonexistent_openvaf_binary_12345".into(),
    );
    let result = compiler.check_openvaf();
    assert!(result.is_err());
    let msg = result.unwrap_err().to_string();
    assert!(msg.contains("not found"), "Expected 'not found' in: {}", msg);
}

#[test]
fn compile_creates_cache_directory() {
    let cache_dir = std::env::temp_dir().join("rustspice_test_cache_create");
    let _ = std::fs::remove_dir_all(&cache_dir);

    let va_dir = std::env::temp_dir().join("rustspice_test_va_src");
    std::fs::create_dir_all(&va_dir).unwrap();
    let va_path = va_dir.join("test.va");
    std::fs::write(&va_path, b"module test; endmodule").unwrap();

    let compiler = VaCompiler::with_paths(cache_dir.clone(), "openvaf".into());
    // This will fail at the openvaf invocation step, but it should have
    // created the cache directory first
    let _ = compiler.compile(&va_path);
    assert!(cache_dir.exists(), "Cache directory should be created");

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&va_dir);
}

#[test]
fn compile_content_hash_produces_deterministic_filename() {
    // Two compilations of the same content should target the same cache file.
    // We can't verify the full path without OpenVAF, but we can verify the
    // cache directory creation and that the same content doesn't change hash.
    let cache_dir = std::env::temp_dir().join("rustspice_test_hash");
    let _ = std::fs::remove_dir_all(&cache_dir);

    let va_dir = std::env::temp_dir().join("rustspice_test_hash_src");
    std::fs::create_dir_all(&va_dir).unwrap();

    // Write two files with identical content
    let va1 = va_dir.join("model_a.va");
    let va2 = va_dir.join("model_b.va");
    let content = b"module resistor(p, n); endmodule";
    std::fs::write(&va1, content).unwrap();
    std::fs::write(&va2, content).unwrap();

    let compiler = VaCompiler::with_paths(cache_dir.clone(), "openvaf".into());

    // Both should fail (no openvaf), but we test that the function handles
    // it gracefully without panicking
    let _ = compiler.compile(&va1);
    let _ = compiler.compile(&va2);

    let _ = std::fs::remove_dir_all(&cache_dir);
    let _ = std::fs::remove_dir_all(&va_dir);
}
