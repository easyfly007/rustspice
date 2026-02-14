//! Build script for sim-core
//!
//! This script handles linking to the SuiteSparse KLU library when the
//! `klu` feature is enabled.
//!
//! # Environment Variables
//!
//! The following environment variables control KLU linking:
//!
//! - `SUITESPARSE_DIR`: Root directory of SuiteSparse installation
//!   (expects `lib/` and `include/` subdirectories)
//! - `KLU_LIB_DIR`: Directory containing KLU libraries (overrides SUITESPARSE_DIR)
//! - `KLU_INCLUDE_DIR`: Directory containing KLU headers (overrides SUITESPARSE_DIR)
//! - `KLU_STATIC`: Set to "1" to prefer static linking
//!
//! # Platform-Specific Notes
//!
//! ## Linux
//! ```bash
//! # Install from package manager (Debian/Ubuntu)
//! sudo apt-get install libsuitesparse-dev
//!
//! # Or build from source
//! export SUITESPARSE_DIR=/usr/local
//! cargo build --features klu
//! ```
//!
//! ## macOS
//! ```bash
//! # Install via Homebrew
//! brew install suite-sparse
//! export SUITESPARSE_DIR=$(brew --prefix suite-sparse)
//! cargo build --features klu
//! ```
//!
//! ## Windows (MSVC)
//! ```powershell
//! # Build SuiteSparse with CMake or use vcpkg
//! vcpkg install suitesparse:x64-windows
//! $env:SUITESPARSE_DIR = "C:\vcpkg\installed\x64-windows"
//! cargo build --features klu
//! ```

use std::env;
use std::path::PathBuf;

fn main() {
    // Only run KLU linking when feature is enabled
    if env::var_os("CARGO_FEATURE_KLU").is_none() {
        return;
    }

    println!("cargo:rerun-if-env-changed=SUITESPARSE_DIR");
    println!("cargo:rerun-if-env-changed=KLU_LIB_DIR");
    println!("cargo:rerun-if-env-changed=KLU_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=KLU_STATIC");

    // Determine library and include directories
    let suitesparse_dir = env::var_os("SUITESPARSE_DIR").map(PathBuf::from);

    let lib_dir = env::var_os("KLU_LIB_DIR")
        .map(PathBuf::from)
        .or_else(|| suitesparse_dir.as_ref().map(|p| p.join("lib")));

    let include_dir = env::var_os("KLU_INCLUDE_DIR")
        .map(PathBuf::from)
        .or_else(|| suitesparse_dir.as_ref().map(|p| p.join("include")));

    // Check for static linking preference
    let prefer_static = env::var("KLU_STATIC")
        .map(|v| v == "1" || v.to_lowercase() == "true")
        .unwrap_or(false);

    // Platform-specific library search paths
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Add library search path
    if let Some(ref dir) = lib_dir {
        println!("cargo:rustc-link-search=native={}", dir.display());

        // On Windows, also check for lib64
        if target_os == "windows" {
            let lib64 = dir.parent().map(|p| p.join("lib64"));
            if let Some(ref lib64_dir) = lib64 {
                if lib64_dir.exists() {
                    println!("cargo:rustc-link-search=native={}", lib64_dir.display());
                }
            }
        }
    } else {
        // Try common system paths
        match target_os.as_str() {
            "linux" => {
                println!("cargo:rustc-link-search=native=/usr/lib");
                println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
                println!("cargo:rustc-link-search=native=/usr/local/lib");
            }
            "macos" => {
                println!("cargo:rustc-link-search=native=/usr/local/lib");
                println!("cargo:rustc-link-search=native=/opt/homebrew/lib");
                // Homebrew on Intel Mac
                println!("cargo:rustc-link-search=native=/usr/local/opt/suite-sparse/lib");
                // Homebrew on Apple Silicon
                println!("cargo:rustc-link-search=native=/opt/homebrew/opt/suite-sparse/lib");
            }
            "windows" => {
                println!("cargo:warning=KLU_LIB_DIR or SUITESPARSE_DIR not set");
                println!("cargo:warning=Please set environment variable to SuiteSparse location");
            }
            _ => {}
        }
    }

    // Emit include path for downstream crates if needed
    if let Some(ref dir) = include_dir {
        println!("cargo:include={}", dir.display());
    }

    // Link type is determined by prefer_static flag below

    // Link KLU and its dependencies
    // The order matters for static linking: dependents before dependencies

    // KLU - the main library
    if prefer_static {
        println!("cargo:rustc-link-lib=static=klu");
    } else {
        println!("cargo:rustc-link-lib=klu");
    }

    // BTF - Block Triangular Form (used by KLU)
    if prefer_static {
        println!("cargo:rustc-link-lib=static=btf");
    } else {
        println!("cargo:rustc-link-lib=btf");
    }

    // AMD - Approximate Minimum Degree ordering
    if prefer_static {
        println!("cargo:rustc-link-lib=static=amd");
    } else {
        println!("cargo:rustc-link-lib=amd");
    }

    // COLAMD - Column Approximate Minimum Degree
    if prefer_static {
        println!("cargo:rustc-link-lib=static=colamd");
    } else {
        println!("cargo:rustc-link-lib=colamd");
    }

    // SuiteSparse_config - common configuration
    if prefer_static {
        println!("cargo:rustc-link-lib=static=suitesparseconfig");
    } else {
        println!("cargo:rustc-link-lib=suitesparseconfig");
    }

    // Platform-specific dependencies
    match target_os.as_str() {
        "linux" => {
            // May need libm on some systems
            println!("cargo:rustc-link-lib=m");
        }
        "windows" if target_env == "msvc" => {
            // MSVC runtime is linked automatically
        }
        "windows" if target_env == "gnu" => {
            // MinGW may need additional libraries
            println!("cargo:rustc-link-lib=m");
        }
        "macos" => {
            // Usually no additional libs needed
        }
        _ => {}
    }

    // Print configuration summary
    println!(
        "cargo:warning=KLU linking: lib_dir={:?}, static={}",
        lib_dir, prefer_static
    );
}
