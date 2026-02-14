use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct Summary {
    pub node_count: usize,
    pub device_count: usize,
    pub model_count: usize,
}
