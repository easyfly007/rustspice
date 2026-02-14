//! Tests for ngspice raw file format output.

use sim_core::raw;
use sim_core::result_store::{AnalysisType, RunId, RunResult, RunStatus};

fn make_op_result() -> RunResult {
    RunResult {
        id: RunId(0),
        analysis: AnalysisType::Op,
        status: RunStatus::Converged,
        iterations: 1,
        node_names: vec!["0".to_string(), "in".to_string(), "out".to_string()],
        solution: vec![0.0, 1.0, 0.5],
        message: None,
        sweep_var: None,
        sweep_values: Vec::new(),
        sweep_solutions: Vec::new(),
        tran_times: Vec::new(),
        tran_solutions: Vec::new(),
        ac_frequencies: Vec::new(),
        ac_solutions: Vec::new(),
    }
}

#[test]
fn raw_op_format() {
    let run = make_op_result();

    let mut path = std::env::temp_dir();
    path.push("rustspice_raw_op_test.raw");
    raw::write_raw_op(&run, &path, 6).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Check header
    assert!(content.contains("Title: RustSpice"));
    assert!(content.contains("Plotname: Operating Point"));
    assert!(content.contains("Flags: real"));
    assert!(content.contains("No. Variables: 2")); // in, out (ground filtered)
    assert!(content.contains("No. Points: 1"));

    // Check variables section
    assert!(content.contains("Variables:"));
    assert!(content.contains("v(in)"));
    assert!(content.contains("v(out)"));
    assert!(content.contains("voltage"));

    // Check values section
    assert!(content.contains("Values:"));
    assert!(content.contains("1.000000e0")); // v(in) = 1.0
    assert!(content.contains("5.000000e-1")); // v(out) = 0.5
}

#[test]
fn raw_dc_sweep_format() {
    let source = "V1";
    let sweep_values = vec![0.0, 0.5, 1.0];
    let node_names = vec!["0".to_string(), "in".to_string(), "out".to_string()];
    let sweep_results = vec![
        vec![0.0, 0.0, 0.0],
        vec![0.0, 0.5, 0.25],
        vec![0.0, 1.0, 0.5],
    ];

    let mut path = std::env::temp_dir();
    path.push("rustspice_raw_dc_test.raw");
    raw::write_raw_sweep(source, &sweep_values, &node_names, &sweep_results, &path, 6).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Check header
    assert!(content.contains("Plotname: DC transfer characteristic"));
    assert!(content.contains("Flags: real"));
    assert!(content.contains("No. Variables: 3")); // sweep + in + out
    assert!(content.contains("No. Points: 3"));

    // Check variables section
    assert!(content.contains("v1")); // source name lowercase
    assert!(content.contains("v(in)"));
    assert!(content.contains("v(out)"));

    // Check values section
    assert!(content.contains("Values:"));
    // Point indices should be present
    assert!(content.contains(" 0\t"));
    assert!(content.contains(" 1\t"));
    assert!(content.contains(" 2\t"));
}

#[test]
fn raw_tran_format() {
    let times = vec![0.0, 1e-6, 2e-6];
    let node_names = vec!["0".to_string(), "in".to_string(), "out".to_string()];
    let solutions = vec![
        vec![0.0, 1.0, 0.0],
        vec![0.0, 1.0, 0.5],
        vec![0.0, 1.0, 0.8],
    ];

    let mut path = std::env::temp_dir();
    path.push("rustspice_raw_tran_test.raw");
    raw::write_raw_tran(&times, &node_names, &solutions, &path, 6).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Check header
    assert!(content.contains("Plotname: Transient Analysis"));
    assert!(content.contains("Flags: real"));
    assert!(content.contains("No. Variables: 3")); // time + in + out
    assert!(content.contains("No. Points: 3"));

    // Check variables section
    assert!(content.contains("time\ttime"));
    assert!(content.contains("v(in)"));
    assert!(content.contains("v(out)"));

    // Check values section
    assert!(content.contains("Values:"));
    assert!(content.contains("0.000000e0")); // time = 0
    assert!(content.contains("1.000000e-6")); // time = 1us
}

#[test]
fn raw_ac_format() {
    let frequencies = vec![1.0, 10.0, 100.0];
    let node_names = vec!["0".to_string(), "in".to_string(), "out".to_string()];
    // AC solutions are (magnitude_dB, phase_degrees)
    let ac_solutions = vec![
        vec![(0.0, 0.0), (0.0, 0.0), (-3.0, -45.0)], // at 1 Hz
        vec![(0.0, 0.0), (0.0, 0.0), (-6.0, -60.0)], // at 10 Hz
        vec![(0.0, 0.0), (0.0, 0.0), (-20.0, -90.0)], // at 100 Hz
    ];

    let mut path = std::env::temp_dir();
    path.push("rustspice_raw_ac_test.raw");
    raw::write_raw_ac(&frequencies, &node_names, &ac_solutions, &path, 6).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Check header
    assert!(content.contains("Plotname: AC Analysis"));
    assert!(content.contains("Flags: complex"));
    assert!(content.contains("No. Variables: 3")); // frequency + in + out
    assert!(content.contains("No. Points: 3"));

    // Check variables section
    assert!(content.contains("frequency\tfrequency"));
    assert!(content.contains("v(in)"));
    assert!(content.contains("v(out)"));

    // Check values section - should contain complex format with comma
    assert!(content.contains("Values:"));
    assert!(content.contains(",")); // Complex values have comma separator
    assert!(content.contains("1.000000e0,0.000000e0")); // frequency 1 Hz as complex
}

#[test]
fn raw_format_filters_ground_node() {
    let run = make_op_result();

    let mut path = std::env::temp_dir();
    path.push("rustspice_raw_ground_test.raw");
    raw::write_raw_op(&run, &path, 6).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Ground node "0" should not appear in variables
    assert!(!content.contains("v(0)"));
    // But other nodes should
    assert!(content.contains("v(in)"));
    assert!(content.contains("v(out)"));
}
