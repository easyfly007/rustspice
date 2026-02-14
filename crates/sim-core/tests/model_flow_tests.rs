use sim_core::netlist::{build_circuit, elaborate_netlist, parse_netlist};

#[test]
fn model_flow_maps_instance_to_model() {
    let input = ".model DIO D IS=1e-12\nD1 in 0 DIO\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty());
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.error_count, 0);

    let circuit = build_circuit(&ast, &elab);
    assert_eq!(circuit.models.models.len(), 1);
    assert_eq!(circuit.instances.instances.len(), 1);

    let model_id = circuit.instances.instances[0]
        .model
        .expect("model id missing");
    let model = &circuit.models.models[model_id.0];
    assert_eq!(model.name, "dio");
    assert_eq!(model.model_type, "d");
    assert!(model.params.contains_key("is"));
    assert!(circuit.instances.instances[0].params.contains_key("is"));
}
