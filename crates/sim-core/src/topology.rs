#[derive(Debug, Clone)]
pub struct TopologyGraph {
    pub node_count: usize,
    pub device_count: usize,
}

pub fn debug_dump_topology(graph: &TopologyGraph) {
    println!(
        "topology: nodes={} devices={}",
        graph.node_count, graph.device_count
    );
}
