//! CSV file format output support.
//!
//! This module provides functions to write simulation results in CSV format,
//! which is easy to import into spreadsheet applications and data analysis tools.

use crate::result_store::RunResult;
use std::fs;
use std::path::Path;

/// Write operating point analysis result to CSV format.
pub fn write_csv_op(run: &RunResult, path: &Path, precision: usize) -> std::io::Result<()> {
    let mut out = String::new();

    // Header row
    out.push_str("node,type,value\n");

    // Data rows - filter out ground node
    for (idx, name) in run.node_names.iter().enumerate() {
        if name == "0" {
            continue;
        }
        let value = run.solution.get(idx).copied().unwrap_or(0.0);
        let is_current = name.starts_with("I(") || name.starts_with("i(") || name.contains("#branch");
        let var_type = if is_current { "current" } else { "voltage" };
        out.push_str(&format!("{},{},{:.prec$e}\n", name, var_type, value, prec = precision));
    }

    fs::write(path, out)
}

/// Write DC sweep results to CSV format.
pub fn write_csv_sweep(
    source: &str,
    sweep_values: &[f64],
    node_names: &[String],
    sweep_results: &[Vec<f64>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();

    // Header row
    out.push_str(source);
    for name in &filtered_names {
        out.push_str(&format!(",V({})", name));
    }
    out.push('\n');

    // Data rows
    for (i, sweep_val) in sweep_values.iter().enumerate() {
        out.push_str(&format!("{:.prec$e}", sweep_val, prec = precision));

        if let Some(solution) = sweep_results.get(i) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!(",{:.prec$e}", val, prec = precision));
                }
            }
        }
        out.push('\n');
    }

    fs::write(path, out)
}

/// Write transient analysis results to CSV format.
pub fn write_csv_tran(
    times: &[f64],
    node_names: &[String],
    solutions: &[Vec<f64>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();

    // Header row
    out.push_str("time");
    for name in &filtered_names {
        out.push_str(&format!(",V({})", name));
    }
    out.push('\n');

    // Data rows
    for (i, time) in times.iter().enumerate() {
        out.push_str(&format!("{:.prec$e}", time, prec = precision));

        if let Some(solution) = solutions.get(i) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!(",{:.prec$e}", val, prec = precision));
                }
            }
        }
        out.push('\n');
    }

    fs::write(path, out)
}

/// Write AC analysis results to CSV format.
/// ac_solutions contains (magnitude_dB, phase_degrees) tuples for each node at each frequency.
/// For each node, outputs magnitude (dB) and phase (degrees) columns.
pub fn write_csv_ac(
    frequencies: &[f64],
    node_names: &[String],
    ac_solutions: &[Vec<(f64, f64)>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();

    // Header row - frequency, then mag_dB/phase_deg pairs for each node
    out.push_str("frequency");
    for name in &filtered_names {
        out.push_str(&format!(",|V({})|_dB,phase({})", name, name));
    }
    out.push('\n');

    // Data rows
    for (i, freq) in frequencies.iter().enumerate() {
        out.push_str(&format!("{:.prec$e}", freq, prec = precision));

        if let Some(solution) = ac_solutions.get(i) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let (mag_db, phase_deg) = solution.get(node_idx).copied().unwrap_or((0.0, 0.0));
                    out.push_str(&format!(",{:.prec$e},{:.prec$e}", mag_db, phase_deg, prec = precision));
                }
            }
        }
        out.push('\n');
    }

    fs::write(path, out)
}
