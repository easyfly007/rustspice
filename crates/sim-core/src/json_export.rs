//! JSON file format output support.
//!
//! This module provides functions to write simulation results in JSON format,
//! which is easy to parse and integrate with other tools and programming languages.

use crate::result_store::RunResult;
use std::fs;
use std::path::Path;

const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Write operating point analysis result to JSON format.
pub fn write_json_op(run: &RunResult, path: &Path, precision: usize) -> std::io::Result<()> {
    let mut out = String::new();

    out.push_str("{\n");
    out.push_str(&format!("  \"format\": \"myspice-json\",\n"));
    out.push_str(&format!("  \"version\": \"{}\",\n", VERSION));
    out.push_str(&format!("  \"analysis\": \"{:?}\",\n", run.analysis));
    out.push_str(&format!("  \"status\": \"{:?}\",\n", run.status));
    out.push_str(&format!("  \"iterations\": {},\n", run.iterations));

    // Variables array
    out.push_str("  \"variables\": [\n");
    let filtered: Vec<_> = run
        .node_names
        .iter()
        .enumerate()
        .filter(|(_, n)| *n != "0")
        .collect();

    for (i, (idx, name)) in filtered.iter().enumerate() {
        let value = run.solution.get(*idx).copied().unwrap_or(0.0);
        let is_current = name.starts_with("I(") || name.starts_with("i(") || name.contains("#branch");
        let var_type = if is_current { "current" } else { "voltage" };
        let comma = if i < filtered.len() - 1 { "," } else { "" };
        out.push_str(&format!(
            "    {{\"name\": \"{}\", \"type\": \"{}\", \"value\": {:.prec$e}}}{}\n",
            name,
            var_type,
            value,
            comma,
            prec = precision
        ));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    fs::write(path, out)
}

/// Write DC sweep results to JSON format.
pub fn write_json_sweep(
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

    out.push_str("{\n");
    out.push_str(&format!("  \"format\": \"myspice-json\",\n"));
    out.push_str(&format!("  \"version\": \"{}\",\n", VERSION));
    out.push_str("  \"analysis\": \"Dc\",\n");
    out.push_str(&format!("  \"sweep_source\": \"{}\",\n", source));
    out.push_str(&format!("  \"points\": {},\n", sweep_values.len()));

    // Variables definition
    out.push_str("  \"variables\": [\n");
    out.push_str(&format!("    {{\"name\": \"{}\", \"type\": \"sweep\"}}", source));
    for name in &filtered_names {
        out.push_str(&format!(",\n    {{\"name\": \"{}\", \"type\": \"voltage\"}}", name));
    }
    out.push_str("\n  ],\n");

    // Data array
    out.push_str("  \"data\": [\n");
    for (i, sweep_val) in sweep_values.iter().enumerate() {
        out.push_str("    [");
        out.push_str(&format!("{:.prec$e}", sweep_val, prec = precision));

        if let Some(solution) = sweep_results.get(i) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!(", {:.prec$e}", val, prec = precision));
                }
            }
        }

        let comma = if i < sweep_values.len() - 1 { "," } else { "" };
        out.push_str(&format!("]{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    fs::write(path, out)
}

/// Write transient analysis results to JSON format.
pub fn write_json_tran(
    times: &[f64],
    node_names: &[String],
    solutions: &[Vec<f64>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();

    out.push_str("{\n");
    out.push_str(&format!("  \"format\": \"myspice-json\",\n"));
    out.push_str(&format!("  \"version\": \"{}\",\n", VERSION));
    out.push_str("  \"analysis\": \"Tran\",\n");
    out.push_str(&format!("  \"points\": {},\n", times.len()));

    // Variables definition
    out.push_str("  \"variables\": [\n");
    out.push_str("    {\"name\": \"time\", \"type\": \"time\"}");
    for name in &filtered_names {
        out.push_str(&format!(",\n    {{\"name\": \"{}\", \"type\": \"voltage\"}}", name));
    }
    out.push_str("\n  ],\n");

    // Data array
    out.push_str("  \"data\": [\n");
    for (i, time) in times.iter().enumerate() {
        out.push_str("    [");
        out.push_str(&format!("{:.prec$e}", time, prec = precision));

        if let Some(solution) = solutions.get(i) {
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let val = solution.get(node_idx).copied().unwrap_or(0.0);
                    out.push_str(&format!(", {:.prec$e}", val, prec = precision));
                }
            }
        }

        let comma = if i < times.len() - 1 { "," } else { "" };
        out.push_str(&format!("]{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    fs::write(path, out)
}

/// Write AC analysis results to JSON format.
/// ac_solutions contains (magnitude_dB, phase_degrees) tuples for each node at each frequency.
pub fn write_json_ac(
    frequencies: &[f64],
    node_names: &[String],
    ac_solutions: &[Vec<(f64, f64)>],
    path: &Path,
    precision: usize,
) -> std::io::Result<()> {
    let mut out = String::new();

    // Filter out ground node
    let filtered_names: Vec<_> = node_names.iter().filter(|n| *n != "0").collect();

    out.push_str("{\n");
    out.push_str(&format!("  \"format\": \"myspice-json\",\n"));
    out.push_str(&format!("  \"version\": \"{}\",\n", VERSION));
    out.push_str("  \"analysis\": \"Ac\",\n");
    out.push_str(&format!("  \"points\": {},\n", frequencies.len()));

    // Variables definition - each node has magnitude and phase
    out.push_str("  \"variables\": [\n");
    out.push_str("    {\"name\": \"frequency\", \"type\": \"frequency\"}");
    for name in &filtered_names {
        out.push_str(&format!(",\n    {{\"name\": \"{}\", \"type\": \"ac\"}}", name));
    }
    out.push_str("\n  ],\n");

    // Data array - frequency, then magnitude_dB and phase_deg for each node
    out.push_str("  \"data\": [\n");
    for (i, freq) in frequencies.iter().enumerate() {
        out.push_str("    {");
        out.push_str(&format!("\"frequency\": {:.prec$e}", freq, prec = precision));

        if let Some(solution) = ac_solutions.get(i) {
            out.push_str(", \"values\": [");
            let mut first = true;
            for (node_idx, name) in node_names.iter().enumerate() {
                if name != "0" {
                    let (mag_db, phase_deg) = solution.get(node_idx).copied().unwrap_or((0.0, 0.0));
                    if !first {
                        out.push_str(", ");
                    }
                    first = false;
                    out.push_str(&format!(
                        "{{\"mag_db\": {:.prec$e}, \"phase_deg\": {:.prec$e}}}",
                        mag_db,
                        phase_deg,
                        prec = precision
                    ));
                }
            }
            out.push_str("]");
        }

        let comma = if i < frequencies.len() - 1 { "," } else { "" };
        out.push_str(&format!("}}{}\n", comma));
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    fs::write(path, out)
}
