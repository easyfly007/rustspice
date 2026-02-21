use std::collections::HashMap;
use std::path::PathBuf;

use crate::options::SimOptions;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub usize);

/// POLY specification for controlled sources
/// Represents a polynomial relationship: output = sum of polynomial terms
#[derive(Debug, Clone)]
pub struct PolySpec {
    /// Number of control inputs (n in POLY(n))
    pub degree: usize,
    /// Polynomial coefficients as parsed values
    pub coeffs: Vec<f64>,
    /// Control node indices for E/G (voltage controlled)
    /// Each pair (pos, neg) represents one control voltage
    pub control_nodes: Vec<(usize, usize)>,
    /// Control source names for F/H (current controlled)
    pub control_sources: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ModelId(pub usize);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceId(pub usize);

#[derive(Debug, Clone)]
pub struct NodeTable {
    pub name_to_id: HashMap<String, NodeId>,
    pub id_to_name: Vec<String>,
    pub gnd_id: NodeId,
}

impl NodeTable {
    pub fn new() -> Self {
        let mut table = Self {
            name_to_id: HashMap::new(),
            id_to_name: Vec::new(),
            gnd_id: NodeId(0),
        };
        table.ensure_node("0");
        table
    }

    pub fn ensure_node(&mut self, name: &str) -> NodeId {
        if let Some(id) = self.name_to_id.get(name) {
            return *id;
        }
        let id = NodeId(self.id_to_name.len());
        self.name_to_id.insert(name.to_string(), id);
        self.id_to_name.push(name.to_string());
        id
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub name: String,
    pub model_type: String,
    pub params: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub struct ModelTable {
    pub models: Vec<Model>,
    pub name_to_id: HashMap<String, ModelId>,
}

impl ModelTable {
    pub fn new() -> Self {
        Self {
            models: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn insert(&mut self, model: Model) -> ModelId {
        let id = ModelId(self.models.len());
        self.name_to_id.insert(model.name.clone(), id);
        self.models.push(model);
        id
    }
}

#[derive(Debug, Clone)]
pub enum DeviceKind {
    R,
    C,
    L,
    V,
    I,
    D,
    M,
    E,
    G,
    F,
    H,
    X,
    /// Verilog-A device loaded via OSDI.
    VA { module_name: String },
}

#[derive(Debug, Clone)]
pub struct Instance {
    pub name: String,
    pub kind: DeviceKind,
    pub nodes: Vec<NodeId>,
    pub model: Option<ModelId>,
    pub params: HashMap<String, String>,
    pub value: Option<String>,
    pub control: Option<String>,
    /// AC analysis magnitude (for voltage/current sources)
    pub ac_mag: Option<f64>,
    /// AC analysis phase in degrees (for voltage/current sources)
    pub ac_phase: Option<f64>,
    /// POLY specification for controlled sources (E/G/F/H)
    pub poly: Option<PolySpec>,
}

#[derive(Debug, Clone)]
pub struct InstanceTable {
    pub instances: Vec<Instance>,
    pub name_to_id: HashMap<String, InstanceId>,
}

impl InstanceTable {
    pub fn new() -> Self {
        Self {
            instances: Vec::new(),
            name_to_id: HashMap::new(),
        }
    }

    pub fn insert(&mut self, instance: Instance) -> InstanceId {
        let id = InstanceId(self.instances.len());
        self.name_to_id.insert(instance.name.clone(), id);
        self.instances.push(instance);
        id
    }
}

/// AC sweep type for frequency analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcSweepType {
    /// Logarithmic sweep with N points per decade
    Dec,
    /// Logarithmic sweep with N points per octave
    Oct,
    /// Linear sweep with N total points
    Lin,
}

#[derive(Debug, Clone)]
pub enum AnalysisCmd {
    Op,
    Dc {
        source: String,
        start: f64,
        stop: f64,
        step: f64,
    },
    Tran {
        tstep: f64,
        tstop: f64,
        tstart: f64,
        tmax: f64,
    },
    Ac {
        sweep_type: AcSweepType,
        points: usize,
        fstart: f64,
        fstop: f64,
    },
}

#[derive(Debug, Clone)]
pub struct Circuit {
    pub nodes: NodeTable,
    pub models: ModelTable,
    pub instances: InstanceTable,
    pub analysis: Vec<AnalysisCmd>,
    /// Initial conditions for transient analysis (.ic directive)
    /// Maps node ID to initial voltage
    pub initial_conditions: HashMap<NodeId, f64>,
    /// Verilog-A / OSDI file paths from .hdl and .osdi directives
    pub va_files: Vec<PathBuf>,
    /// Simulator options from .option directives
    pub options: SimOptions,
}

impl Circuit {
    pub fn new() -> Self {
        Self {
            nodes: NodeTable::new(),
            models: ModelTable::new(),
            instances: InstanceTable::new(),
            analysis: Vec::new(),
            initial_conditions: HashMap::new(),
            va_files: Vec::new(),
            options: SimOptions::new(),
        }
    }
}

pub fn debug_dump_circuit(circuit: &Circuit) {
    println!(
        "circuit: nodes={} models={} instances={} analyses={}",
        circuit.nodes.id_to_name.len(),
        circuit.models.models.len(),
        circuit.instances.instances.len(),
        circuit.analysis.len()
    );
}
