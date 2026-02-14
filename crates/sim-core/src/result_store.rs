#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RunId(pub usize);

#[derive(Debug, Clone, Copy)]
pub enum AnalysisType {
    Op,
    Dc,
    Tran,
    Ac,
}

#[derive(Debug, Clone, Copy)]
pub enum RunStatus {
    Converged,
    MaxIters,
    Failed,
}

#[derive(Debug, Clone)]
pub struct RunResult {
    pub id: RunId,
    pub analysis: AnalysisType,
    pub status: RunStatus,
    pub iterations: usize,
    pub node_names: Vec<String>,
    /// Solution vector for single-point analysis (OP)
    pub solution: Vec<f64>,
    pub message: Option<String>,
    /// DC sweep: name of swept variable (e.g., "V1")
    pub sweep_var: Option<String>,
    /// DC sweep: values at each sweep point
    pub sweep_values: Vec<f64>,
    /// DC sweep: solution vectors at each sweep point
    pub sweep_solutions: Vec<Vec<f64>>,
    /// TRAN analysis: time points
    pub tran_times: Vec<f64>,
    /// TRAN analysis: solution vectors at each time point
    pub tran_solutions: Vec<Vec<f64>>,
    /// AC analysis: frequency points
    pub ac_frequencies: Vec<f64>,
    /// AC analysis: complex solutions at each frequency point
    /// Each inner Vec contains (magnitude_dB, phase_deg) pairs for each node
    pub ac_solutions: Vec<Vec<(f64, f64)>>,
}

#[derive(Debug, Clone)]
pub struct ResultStore {
    pub runs: Vec<RunResult>,
}

impl ResultStore {
    pub fn new() -> Self {
        Self { runs: Vec::new() }
    }

    pub fn add_run(&mut self, mut run: RunResult) -> RunId {
        let id = RunId(self.runs.len());
        run.id = id;
        self.runs.push(run);
        id
    }

    pub fn write_psf_text(&self, id: RunId, path: &std::path::Path, precision: usize) -> std::io::Result<()> {
        let run = self
            .runs
            .get(id.0)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::NotFound, "run not found"))?;
        crate::psf::write_psf_text(run, path, precision)
    }
}

pub fn debug_dump_result_store(store: &ResultStore) {
    println!("result_store: runs={}", store.runs.len());
}
