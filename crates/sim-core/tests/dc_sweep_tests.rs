use std::path::PathBuf;

use sim_core::analysis::AnalysisPlan;
use sim_core::circuit::AnalysisCmd;
use sim_core::engine::Engine;
use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist_file};
use sim_core::result_store::{ResultStore, RunStatus};

/// Test basic DC sweep functionality
/// Circuit: V1 -> R1(1k) -> out -> R2(2k) -> GND
/// Sweep V1 from 0V to 5V in 1V steps
/// Expected: Vout = Vin * R2/(R1+R2) = Vin * 2/3
#[test]
fn dc_sweep_resistor_divider() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("netlists")
        .join("dc_sweep.cir");

    let ast = parse_netlist_file(&path);
    assert!(ast.errors.is_empty(), "parse errors: {:?}", ast.errors);

    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.error_count, 0, "elaboration errors: {}", elab.error_count);

    let circuit = build_circuit(&ast, &elab);

    // Verify DC analysis command was parsed
    let dc_cmd = circuit.analysis.iter().find(|cmd| {
        matches!(cmd, AnalysisCmd::Dc { .. })
    });
    assert!(dc_cmd.is_some(), "DC sweep command not found in circuit");

    if let Some(AnalysisCmd::Dc { source, start, stop, step }) = dc_cmd {
        assert_eq!(source.to_lowercase(), "v1");
        assert!((start - 0.0).abs() < 1e-9);
        assert!((stop - 5.0).abs() < 1e-9);
        assert!((step - 1.0).abs() < 1e-9);
    }

    let mut engine = Engine::new_default(circuit);
    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Dc {
            source: "V1".to_string(),
            start: 0.0,
            stop: 5.0,
            step: 1.0,
        },
    };
    let mut store = ResultStore::new();
    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    assert!(
        matches!(run.status, RunStatus::Converged),
        "DC sweep failed: {:?}",
        run
    );

    // Verify sweep data exists
    assert!(run.sweep_var.is_some(), "sweep_var should be set for DC sweep");
    assert!(!run.sweep_values.is_empty(), "sweep_values should not be empty");
    assert!(!run.sweep_solutions.is_empty(), "sweep_solutions should not be empty");

    // Verify sweep points: 0, 1, 2, 3, 4, 5 (6 points)
    assert_eq!(run.sweep_values.len(), 6, "expected 6 sweep points");
    assert_eq!(run.sweep_solutions.len(), 6, "expected 6 solution vectors");

    // Find output node index
    let out_idx = run.node_names.iter()
        .position(|name| name == "out")
        .expect("node 'out' not found");

    // Verify Vout = Vin * 2/3 at each sweep point
    let divider_ratio = 2.0 / 3.0;
    for (i, &vin) in run.sweep_values.iter().enumerate() {
        let vout = run.sweep_solutions[i][out_idx];
        let expected_vout = vin * divider_ratio;
        assert!(
            (vout - expected_vout).abs() < 1e-6,
            "At Vin={}: expected Vout={}, got Vout={}",
            vin, expected_vout, vout
        );
    }
}

/// Test DC sweep with negative voltage range
#[test]
fn dc_sweep_negative_range() {
    let netlist = r#"
* DC sweep with negative range
V1 in 0 DC 0
R1 in out 1k
R2 out 0 1k
.dc V1 -2 2 1
.end
"#;

    let ast = sim_core::netlist::parse_netlist(netlist);
    assert!(ast.errors.is_empty(), "parse errors: {:?}", ast.errors);

    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    let mut engine = Engine::new_default(circuit);
    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Dc {
            source: "V1".to_string(),
            start: -2.0,
            stop: 2.0,
            step: 1.0,
        },
    };
    let mut store = ResultStore::new();
    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    assert!(matches!(run.status, RunStatus::Converged));

    // Sweep points: -2, -1, 0, 1, 2 (5 points)
    assert_eq!(run.sweep_values.len(), 5, "expected 5 sweep points");

    // Verify sweep values
    let expected_values = [-2.0, -1.0, 0.0, 1.0, 2.0];
    for (i, &expected) in expected_values.iter().enumerate() {
        assert!(
            (run.sweep_values[i] - expected).abs() < 1e-9,
            "sweep point {}: expected {}, got {}",
            i, expected, run.sweep_values[i]
        );
    }
}

/// Test DC sweep with fine step size
#[test]
fn dc_sweep_fine_step() {
    let netlist = r#"
* DC sweep with fine step
V1 in 0 DC 0
R1 in 0 1k
.dc V1 0 1 0.1
.end
"#;

    let ast = sim_core::netlist::parse_netlist(netlist);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    let mut engine = Engine::new_default(circuit);
    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Dc {
            source: "V1".to_string(),
            start: 0.0,
            stop: 1.0,
            step: 0.1,
        },
    };
    let mut store = ResultStore::new();
    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    assert!(matches!(run.status, RunStatus::Converged));

    // Sweep points: 0.0, 0.1, 0.2, ..., 1.0 (11 points)
    assert_eq!(run.sweep_values.len(), 11, "expected 11 sweep points, got {}", run.sweep_values.len());
}

/// Test that single point sweep works (start == stop)
#[test]
fn dc_sweep_single_point() {
    let netlist = r#"
* Single point DC sweep
V1 in 0 DC 0
R1 in 0 1k
.dc V1 1 1 0.1
.end
"#;

    let ast = sim_core::netlist::parse_netlist(netlist);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    let mut engine = Engine::new_default(circuit);
    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Dc {
            source: "V1".to_string(),
            start: 1.0,
            stop: 1.0,
            step: 0.1,
        },
    };
    let mut store = ResultStore::new();
    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    assert!(matches!(run.status, RunStatus::Converged));
    assert_eq!(run.sweep_values.len(), 1, "single point sweep should have 1 point");
    assert!((run.sweep_values[0] - 1.0).abs() < 1e-9);
}
