use sim_core::result_store::{AnalysisType, ResultStore, RunResult, RunStatus};

#[test]
fn result_store_adds_run() {
    let mut store = ResultStore::new();
    let run = RunResult {
        id: sim_core::result_store::RunId(0),
        analysis: AnalysisType::Op,
        status: RunStatus::Converged,
        iterations: 3,
        node_names: vec!["0".to_string(), "n1".to_string()],
        solution: vec![0.0, 1.0],
        message: None,
        sweep_var: None,
        sweep_values: Vec::new(),
        sweep_solutions: Vec::new(),
        tran_times: Vec::new(),
        tran_solutions: Vec::new(),
        ac_frequencies: Vec::new(),
        ac_solutions: Vec::new(),
    };
    let id = store.add_run(run);
    assert_eq!(id.0, 0);
    assert_eq!(store.runs.len(), 1);
}
