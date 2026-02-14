use sim_core::analysis::AnalysisPlan;
use sim_core::circuit::AnalysisCmd;
use sim_core::engine::Engine;
use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist};
use sim_core::result_store::{AnalysisType, ResultStore};

fn parse_and_build(netlist: &str) -> sim_core::circuit::Circuit {
    let ast = parse_netlist(netlist);
    assert!(ast.errors.is_empty(), "parse errors: {:?}", ast.errors);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.error_count, 0, "elaboration errors");
    build_circuit(&ast, &elab)
}

#[test]
fn tran_waveform_stores_multiple_time_points() {
    // Simple resistive circuit (no reactive elements for faster simulation)
    let netlist = r#"
V1 in 0 DC 1
R1 in out 1k
R2 out 0 1k
.tran 1u 10u
.end
"#;
    let circuit = parse_and_build(netlist);
    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Tran {
            tstep: 1e-6,
            tstop: 1e-5,
            tstart: 0.0,
            tmax: 1e-5,
        },
    };

    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    // Verify TRAN analysis type
    assert!(matches!(run.analysis, AnalysisType::Tran));

    // Verify we have multiple time points stored
    assert!(run.tran_times.len() > 1, "Expected multiple time points, got {}", run.tran_times.len());
    assert_eq!(run.tran_times.len(), run.tran_solutions.len(), "Times and solutions should have same length");

    // Verify time starts at 0
    assert!((run.tran_times[0] - 0.0).abs() < 1e-15, "First time point should be 0");

    // Verify time increases monotonically
    for i in 1..run.tran_times.len() {
        assert!(run.tran_times[i] > run.tran_times[i-1],
            "Time should increase: t[{}]={} not > t[{}]={}",
            i, run.tran_times[i], i-1, run.tran_times[i-1]);
    }
}

#[test]
fn tran_waveform_solution_has_correct_nodes() {
    let netlist = r#"
V1 in 0 DC 1
R1 in out 1k
R2 out 0 1k
.tran 1u 5u
.end
"#;
    let circuit = parse_and_build(netlist);
    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Tran {
            tstep: 1e-6,
            tstop: 1e-5,
            tstart: 0.0,
            tmax: 1e-5,
        },
    };

    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    // Each solution vector should be non-empty and consistent in size
    assert!(!run.tran_solutions.is_empty(), "Should have at least one solution");
    let first_len = run.tran_solutions[0].len();
    assert!(first_len > 0, "Solutions should not be empty");

    for (i, sol) in run.tran_solutions.iter().enumerate() {
        assert_eq!(sol.len(), first_len,
            "Solution at time {} should have {} elements, got {}",
            run.tran_times[i], first_len, sol.len());
    }
}

#[test]
fn tran_psf_output_format() {
    let netlist = r#"
V1 in 0 DC 1
R1 in out 1k
R2 out 0 1k
.tran 1u 5u
.end
"#;
    let circuit = parse_and_build(netlist);
    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Tran {
            tstep: 1e-6,
            tstop: 1e-5,
            tstart: 0.0,
            tmax: 1e-5,
        },
    };

    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    // Test PSF output
    let mut path = std::env::temp_dir();
    path.push("myspice_tran_test.psf");

    sim_core::psf::write_psf_tran(
        &run.tran_times,
        &run.node_names,
        &run.tran_solutions,
        &path,
        6,
    ).unwrap();

    let content = std::fs::read_to_string(&path).unwrap();

    // Verify PSF content
    assert!(content.contains("PSF_TEXT"), "Should have PSF header");
    assert!(content.contains("[Transient Analysis]"), "Should have TRAN section");
    assert!(content.contains("points ="), "Should have points count");
    assert!(content.contains("[Signals]"), "Should have signals section");
    assert!(content.contains("time"), "Should have time signal");
    assert!(content.contains("[Data]"), "Should have data section");

    // Clean up
    std::fs::remove_file(&path).ok();
}

#[test]
fn tran_with_initial_conditions() {
    // Simple resistive circuit with initial condition on one node
    // Note: .ic values are used as initial guess for Newton solver
    let netlist = r#"
V1 in 0 DC 5
R1 in out 1k
R2 out 0 1k
.ic v(out)=2.5
.tran 1u 10u
.end
"#;
    let circuit = parse_and_build(netlist);

    // Verify initial condition was parsed
    assert_eq!(circuit.initial_conditions.len(), 1, "Should have one initial condition");
    let out_node_id = circuit.nodes.name_to_id.get("out").expect("out node not found");
    let ic_value = circuit.initial_conditions.get(out_node_id).expect("IC for out not found");
    assert!((ic_value - 2.5).abs() < 1e-10, "IC value should be 2.5, got {}", ic_value);

    // Run transient analysis
    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Tran {
            tstep: 1e-6,
            tstop: 10e-6,
            tstart: 0.0,
            tmax: 10e-6,
        },
    };

    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    // Find the out node index in the solution
    let out_idx = run.node_names.iter().position(|n| n == "out").expect("out not in node_names");

    // Verify we have time points
    assert!(!run.tran_times.is_empty(), "Should have time points");

    // With V1=5V DC and resistor divider, the DC operating point is 2.5V
    // The .ic value matches this, so it helps Newton converge
    let initial_out_voltage = run.tran_solutions[0][out_idx];
    assert!((initial_out_voltage - 2.5).abs() < 0.1,
        "Out voltage should be near 2.5V, got {}", initial_out_voltage);
}

#[test]
fn tran_with_multiple_initial_conditions() {
    // Multiple nodes with initial conditions
    let netlist = r#"
V1 in 0 DC 0
R1 in n1 1k
R2 n1 n2 1k
R3 n2 0 1k
.ic v(n1)=3.3 v(n2)=1.8
.tran 1n 10n
.end
"#;
    let circuit = parse_and_build(netlist);

    // Verify both initial conditions were parsed
    assert_eq!(circuit.initial_conditions.len(), 2, "Should have two initial conditions");

    let n1_id = circuit.nodes.name_to_id.get("n1").expect("n1 node not found");
    let n2_id = circuit.nodes.name_to_id.get("n2").expect("n2 node not found");

    let n1_ic = circuit.initial_conditions.get(n1_id).expect("IC for n1 not found");
    let n2_ic = circuit.initial_conditions.get(n2_id).expect("IC for n2 not found");

    assert!((n1_ic - 3.3).abs() < 1e-10, "n1 IC should be 3.3, got {}", n1_ic);
    assert!((n2_ic - 1.8).abs() < 1e-10, "n2 IC should be 1.8, got {}", n2_ic);

    // Run transient analysis
    let mut engine = Engine::new_default(circuit);
    let mut store = ResultStore::new();

    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Tran {
            tstep: 1e-9,
            tstop: 10e-9,
            tstart: 0.0,
            tmax: 10e-9,
        },
    };

    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    // Verify transient analysis completed
    assert!(run.tran_times.len() > 1, "Should have multiple time points");
}
