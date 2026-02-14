use sim_core::circuit::{Circuit, DeviceKind, Instance, Model};
use std::collections::HashMap;

#[test]
fn circuit_tables_accept_entries() {
    let mut circuit = Circuit::new();
    let n1 = circuit.nodes.ensure_node("n1");
    let n2 = circuit.nodes.ensure_node("n2");

    let model_id = circuit.models.insert(Model {
        name: "dmod".to_string(),
        model_type: "diode".to_string(),
        params: HashMap::new(),
    });

    let mut params = HashMap::new();
    params.insert("r".to_string(), "1k".to_string());

    circuit.instances.insert(Instance {
        name: "R1".to_string(),
        kind: DeviceKind::R,
        nodes: vec![n1, n2],
        model: Some(model_id),
        params,
        value: Some("1k".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    });

    assert_eq!(circuit.nodes.id_to_name.len(), 3);
    assert_eq!(circuit.models.models.len(), 1);
    assert_eq!(circuit.instances.instances.len(), 1);
}
