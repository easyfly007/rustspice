use std::path::PathBuf;

use sim_core::analysis::AnalysisPlan;
use sim_core::circuit::AnalysisCmd;
use sim_core::engine::Engine;
use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist_file};
use sim_core::result_store::{ResultStore, RunStatus};

#[test]
fn dc_basic_end_to_end() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("netlists")
        .join("basic_dc.cir");
    let ast = parse_netlist_file(&path);
    assert!(ast.errors.is_empty(), "ast errors: {:?}", ast.errors);

    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.error_count, 0, "elab errors: {}", elab.error_count);

    let circuit = build_circuit(&ast, &elab);
    let mut engine = Engine::new_default(circuit);
    let plan = AnalysisPlan {
        cmd: AnalysisCmd::Op,
    };
    let mut store = ResultStore::new();
    let run_id = engine.run_with_store(&plan, &mut store);
    let run = &store.runs[run_id.0];

    assert!(
        matches!(run.status, RunStatus::Converged),
        "run failed: {:?}",
        run
    );
    let in_idx = run
        .node_names
        .iter()
        .position(|name| name == "in")
        .expect("node in not found");
    let out_idx = run
        .node_names
        .iter()
        .position(|name| name == "out")
        .expect("node out not found");
    let vin = run.solution[in_idx];
    let vout = run.solution[out_idx];
    assert!((vin - 1.0).abs() < 1e-6, "vin={}", vin);
    assert!((vout - (2.0 / 3.0)).abs() < 1e-6, "vout={}", vout);
}
