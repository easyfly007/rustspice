//! Ngspice raw file format output support.
//!
//! This module provides functions to write simulation results in the ngspice raw format,
//! which is compatible with ngspice, ltspice, gwave, and other SPICE waveform viewers.

use crate::result_store::RunResult;
use std::f64::consts::PI;
use std::fs;
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Write operating point analysis result to ngspice raw format.
pub fn write_raw_op(run: &RunResult, path: &Path, precision: usize) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = run.node_names.iter().filter(|n| *n != "0").collect();
    let filtered_values: Vec<_> = run
        .node_names
        .iter()
        .enumerate()
        .filter(|(_, n)| *n != "0")
        .map(|(i, _)| run.solution.get(i).copied().unwrap_or(0.0))
        .collect();

    write_raw_header(
        &mut out,
        "Operating Point",
        "real",
        filtered_names.len(),
        1,
    );

    // Variables section
    out.push_str("Variables:\n");
    for (idx, name) in filtered_names.iter().enumerate() {
        // Check for current variable names: I(source) or #branch patterns
        let is_current = name.starts_with("I(") || name.starts_with("i(") || name.contains("#branch");
        let var_type = if is_current { "current" } else { "voltage" };
        out.push_str(&format!("\t{}\tv({})\t{}\n", idx, name, var_type));
    }

    // Values section
    out.push_str("Values:\n");
    out.push_str(&format!(" 0\t{}\n", format_real(filtered_values[0], precision)));
    for value in filtered_values.iter().skip(1) {
        out.push_str(&format!("\t{}\n", format_real(*value, precision)));
    }

    fs::write(path, out)
}

/// Write DC sweep results to ngspice raw format.
pub fn write_raw_sweep(
    source: &str,
    sweep_values: &[f64],
    node_names: &[String],
    sweep_results: &[Vec<f64>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node from node_names
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();
    let num_vars = 1 + filtered_names.len(); // sweep variable + node voltages

    write_raw_header(
        &mut out,
        "DC transfer characteristic",
        "real",
        num_vars,
        sweep_values.len(),
    );

    // Variables section
    out.push_str("Variables:\n");
    out.push_str(&format!("\t0\t{}\tvoltage\n", source.to_lowercase()));
    for (idx, name) in filtered_names.iter().enumerate() {
        out.push_str(&format!("\t{}\tv({})\tvoltage\n", idx + 1, name));
    }

    // Values section
    out.push_str("Values:\n");
    for (point_idx, sweep_val) in sweep_values.iter().enumerate() {
        // First line has point index and sweep value
        out.push_str(&format!(" {}\t{}\n", point_idx, format_real(*sweep_val, precision)));

        // Subsequent lines have node values
        if let Some(solution) = sweep_results.get(point_idx) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!("\t{}\n", format_real(val, precision)));
                }
            }
        }
    }

    fs::write(path, out)
}

/// Write transient analysis results to ngspice raw format.
pub fn write_raw_tran(
    times: &[f64],
    node_names: &[String],
    solutions: &[Vec<f64>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();
    let num_vars = 1 + filtered_names.len(); // time + node voltages

    write_raw_header(
        &mut out,
        "Transient Analysis",
        "real",
        num_vars,
        times.len(),
    );

    // Variables section
    out.push_str("Variables:\n");
    out.push_str("\t0\ttime\ttime\n");
    for (idx, name) in filtered_names.iter().enumerate() {
        out.push_str(&format!("\t{}\tv({})\tvoltage\n", idx + 1, name));
    }

    // Values section
    out.push_str("Values:\n");
    for (point_idx, time) in times.iter().enumerate() {
        // First line has point index and time value
        out.push_str(&format!(" {}\t{}\n", point_idx, format_real(*time, precision)));

        // Subsequent lines have node values
        if let Some(solution) = solutions.get(point_idx) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!("\t{}\n", format_real(val, precision)));
                }
            }
        }
    }

    fs::write(path, out)
}

/// Write AC analysis results to ngspice raw format.
///
/// The ac_solutions contain (magnitude_dB, phase_degrees) tuples which are converted
/// to complex numbers for the raw format.
pub fn write_raw_ac(
    frequencies: &[f64],
    node_names: &[String],
    ac_solutions: &[Vec<(f64, f64)>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();
    let num_vars = 1 + filtered_names.len(); // frequency + node voltages

    write_raw_header(
        &mut out,
        "AC Analysis",
        "complex",
        num_vars,
        frequencies.len(),
    );

    // Variables section
    out.push_str("Variables:\n");
    out.push_str("\t0\tfrequency\tfrequency\n");
    for (idx, name) in filtered_names.iter().enumerate() {
        out.push_str(&format!("\t{}\tv({})\tvoltage\n", idx + 1, name));
    }

    // Values section
    out.push_str("Values:\n");
    for (point_idx, freq) in frequencies.iter().enumerate() {
        // First line has point index and frequency (frequency is real, imag=0)
        out.push_str(&format!(
            " {}\t{}\n",
            point_idx,
            format_complex(*freq, 0.0, precision)
        ));

        // Subsequent lines have node values as complex
        if let Some(solution) = ac_solutions.get(point_idx) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    if let Some(&(mag_db, phase_deg)) = solution.get(node_idx) {
                        let (real, imag) = db_phase_to_complex(mag_db, phase_deg);
                        out.push_str(&format!("\t{}\n", format_complex(real, imag, precision)));
                    }
                }
            }
        }
    }

    fs::write(path, out)
}

/// Write the raw file header.
fn write_raw_header(
    out: &mut String,
    plotname: &str,
    flags: &str,
    num_vars: usize,
    num_points: usize,
) {
    out.push_str(&format!("Title: RustSpice v{} Simulation\n", VERSION));
    out.push_str(&format!("Date: {}\n", chrono_lite_now()));
    out.push_str(&format!("Plotname: {}\n", plotname));
    out.push_str(&format!("Flags: {}\n", flags));
    out.push_str(&format!("No. Variables: {}\n", num_vars));
    out.push_str(&format!("No. Points: {}\n", num_points));
}

/// Format a real value in scientific notation.
fn format_real(value: f64, precision: usize) -> String {
    format!("{:.*e}", precision, value)
}

/// Format a complex value as "real,imag".
fn format_complex(real: f64, imag: f64, precision: usize) -> String {
    format!("{:.*e},{:.*e}", precision, real, precision, imag)
}

/// Convert magnitude (dB) and phase (degrees) to complex representation.
fn db_phase_to_complex(mag_db: f64, phase_deg: f64) -> (f64, f64) {
    let magnitude = 10.0_f64.powf(mag_db / 20.0);
    let phase_rad = phase_deg * PI / 180.0;
    let real = magnitude * phase_rad.cos();
    let imag = magnitude * phase_rad.sin();
    (real, imag)
}

/// Simple timestamp function without external dependency.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    // Convert to date/time components (simplified, UTC)
    let days_since_epoch = secs / 86400;
    let time_of_day = secs % 86400;

    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate year, month, day from days since epoch (1970-01-01)
    let (year, month, day) = days_to_ymd(days_since_epoch as i64);

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        year, month, day, hours, minutes, seconds
    )
}

/// Convert days since Unix epoch to year/month/day.
fn days_to_ymd(days: i64) -> (i64, u32, u32) {
    let mut remaining = days;
    let mut year = 1970i64;

    loop {
        let days_in_year = if is_leap_year(year) { 366 } else { 365 };
        if remaining < days_in_year {
            break;
        }
        remaining -= days_in_year;
        year += 1;
    }

    let leap = is_leap_year(year);
    let days_in_months: [i64; 12] = if leap {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };

    let mut month = 1u32;
    for &days_in_month in &days_in_months {
        if remaining < days_in_month {
            break;
        }
        remaining -= days_in_month;
        month += 1;
    }

    let day = remaining as u32 + 1;
    (year, month, day)
}

fn is_leap_year(year: i64) -> bool {
    (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_real() {
        assert_eq!(format_real(1.0, 6), "1.000000e0");
        assert_eq!(format_real(0.001, 6), "1.000000e-3");
    }

    #[test]
    fn test_format_complex() {
        let result = format_complex(1.0, -0.5, 6);
        assert!(result.contains(","));
        assert!(result.starts_with("1.000000e0,"));
    }

    #[test]
    fn test_db_phase_to_complex() {
        // 0 dB, 0 degrees should give (1, 0)
        let (real, imag) = db_phase_to_complex(0.0, 0.0);
        assert!((real - 1.0).abs() < 1e-10);
        assert!(imag.abs() < 1e-10);

        // 0 dB, 90 degrees should give (0, 1)
        let (real, imag) = db_phase_to_complex(0.0, 90.0);
        assert!(real.abs() < 1e-10);
        assert!((imag - 1.0).abs() < 1e-10);

        // -20 dB should give magnitude of 0.1
        let (real, imag) = db_phase_to_complex(-20.0, 0.0);
        assert!((real - 0.1).abs() < 1e-10);
        assert!(imag.abs() < 1e-10);
    }
}
