use sim_core::netlist::parse_netlist;
use sim_core::netlist::{elaborate_netlist, parse_netlist_file, ControlKind, Stmt};
use std::path::PathBuf;

#[test]
fn netlist_parser_skeleton_runs() {
    let input = "* comment\nR1 in out 1k\n.op\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty());
    assert!(ast.statements.len() >= 2);
}

#[test]
fn netlist_parser_extracts_nodes_and_value() {
    let input = "R1 in out 10k\n.end\n";
    let ast = parse_netlist(input);
    let device = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Device(dev) => Some(dev),
            _ => None,
        })
        .expect("device not found");
    assert_eq!(device.nodes, vec!["in".to_string(), "out".to_string()]);
    assert_eq!(device.value, Some("10k".to_string()));
}

#[test]
fn netlist_parser_recognizes_model_and_subckt() {
    let input = ".model nmos bsim4 vth0=0.4\n.subckt inv in out vdd vss\n.ends\n";
    let ast = parse_netlist(input);
    let controls: Vec<_> = ast
        .statements
        .iter()
        .filter_map(|stmt| match stmt {
            Stmt::Control(ctrl) => Some(ctrl),
            _ => None,
        })
        .collect();

    let model = controls
        .iter()
        .find(|ctrl| matches!(ctrl.kind, ControlKind::Model))
        .expect("model not found");
    assert_eq!(model.model_name.as_deref(), Some("nmos"));
    assert_eq!(model.model_type.as_deref(), Some("bsim4"));

    let subckt = controls
        .iter()
        .find(|ctrl| matches!(ctrl.kind, ControlKind::Subckt))
        .expect("subckt not found");
    assert_eq!(subckt.subckt_name.as_deref(), Some("inv"));
    assert_eq!(subckt.subckt_ports.len(), 4);
}

#[test]
fn netlist_parser_reports_missing_fields() {
    let input = "R1 in 10k\nM1 d g s nmos\n.end\n";
    let ast = parse_netlist(input);
    assert!(!ast.errors.is_empty());
}

#[test]
fn netlist_elaboration_counts_statements() {
    let ast = parse_netlist("R1 in out 1k\n.op\n.end\n");
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.control_count, 2);
}

#[test]
fn netlist_parser_validates_controlled_sources() {
    let input = "E1 out 0 in 0 2\nF1 out 0 Vctrl 10\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty());
}

#[test]
fn netlist_elaboration_expands_subckt() {
    let input = ".subckt buf in out\nR1 in out 1k\n.ends\nX1 a b buf\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].name, "X1.R1");
    assert_eq!(
        elab.instances[0].nodes,
        vec!["a".to_string(), "b".to_string()]
    );
}

#[test]
fn netlist_elaboration_applies_params() {
    // 参数值会被求值：5k = 5000
    let input = ".param RVAL=5k\nR1 in out RVAL\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("5000"));
}

#[test]
fn netlist_elaboration_applies_subckt_params() {
    let input = ".subckt buf in out RVAL=1k\nR1 in out RVAL\n.ends\nX1 a b buf RVAL=2k\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("2000"));
}

#[test]
fn netlist_parser_expands_include() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("fixtures")
        .join("netlists")
        .join("include_parent.cir");
    let ast = parse_netlist_file(&root);
    assert!(ast.errors.is_empty(), "errors: {:?}", ast.errors);
    let device_count = ast
        .statements
        .iter()
        .filter(|stmt| matches!(stmt, Stmt::Device(_)))
        .count();
    assert!(device_count >= 2);
}

#[test]
fn netlist_param_expression_evaluates() {
    let input = ".param RVAL=1k+1k\nR1 in out RVAL\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("2000"));
}

#[test]
fn netlist_controlled_source_poly_is_accepted() {
    // 标准 POLY 语法: E1 out 0 POLY(n) ctrl+ ctrl- ... coeffs
    let input = "E1 out 0 POLY(1) in 0 1 2\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty(), "errors: {:?}", ast.errors);
}

#[test]
fn netlist_param_expression_functions() {
    let input = ".param RVAL=max(1k,2k)\nR1 in out RVAL\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("2000"));
}

#[test]
fn netlist_param_expression_if() {
    let input = ".param RVAL=if(1,1k,2k)\nR1 in out RVAL\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("1000"));
}

#[test]
fn netlist_poly_is_parsed_into_spec() {
    // 标准 POLY 语法: G1 out 0 POLY(2) ctrl1+ ctrl1- ctrl2+ ctrl2- coeffs
    let input = "G1 out 0 POLY(2) a 0 b 0 1 2 3\n.end\n";
    let ast = parse_netlist(input);
    let device = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Device(dev) => Some(dev),
            _ => None,
        })
        .expect("device not found");
    let poly = device.poly.as_ref().expect("poly not found");
    assert_eq!(poly.degree, 2);
    // coeffs 包含控制节点 + 系数: a 0 b 0 1 2 3 = 7 个
    assert_eq!(poly.coeffs.len(), 7);
}

#[test]
fn netlist_expression_supports_unary_minus_and_pow() {
    // 测试一元负号和指数运算
    // -1k + 2^3 = -1000 + 8 = -992
    let input = ".param RVAL=-1k+2^3\nR1 in out RVAL\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("-992"));
}

#[test]
fn netlist_voltage_source_dc_keyword() {
    let input = "V1 in 0 DC 1.5\n.end\n";
    let ast = parse_netlist(input);
    let device = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Device(dev) => Some(dev),
            _ => None,
        })
        .expect("device not found");
    assert_eq!(device.value.as_deref(), Some("1.5"));
}

#[test]
fn netlist_poly_requires_coeffs() {
    // POLY(2) 需要 4 个控制节点 + 至少 3 个系数，但这里什么都没有
    let input = "E1 out 0 POLY(2)\n.end\n";
    let ast = parse_netlist(input);
    assert!(!ast.errors.is_empty(), "should have errors for missing coeffs");
}

#[test]
fn netlist_poly_controls_are_validated() {
    // 标准 POLY 语法: E1 out 0 POLY(2) ctrl1+ ctrl1- ctrl2+ ctrl2- coeffs
    // POLY(2) 需要 2 对控制节点 (4 个) + 至少 3 个系数
    let ok = "E1 out 0 POLY(2) a 0 b 0 1 2 3\n.end\n";
    let ast_ok = parse_netlist(ok);
    assert!(ast_ok.errors.is_empty(), "errors: {:?}", ast_ok.errors);

    // 只有 2 个控制节点，少于期望的 4 个
    let bad = "E1 out 0 POLY(2) a 0 1 2 3\n.end\n";
    let ast_bad = parse_netlist(bad);
    assert!(!ast_bad.errors.is_empty(), "should have errors for insufficient controls");
}

#[test]
fn netlist_poly_fh_controls_are_validated() {
    // F/H POLY 语法: F1 out 0 POLY(n) Vctrl1 Vctrl2 ... coeffs
    // POLY(2) 需要 2 个控制源 + 至少 3 个系数
    let ok = "F1 out 0 POLY(2) V1 V2 1 2 3\n.end\n";
    let ast_ok = parse_netlist(ok);
    assert!(ast_ok.errors.is_empty(), "errors: {:?}", ast_ok.errors);

    // 只有 1 个控制源，少于期望的 2 个
    let bad = "F1 out 0 POLY(2) V1 1 2 3\n.end\n";
    let ast_bad = parse_netlist(bad);
    assert!(!ast_bad.errors.is_empty(), "should have errors for insufficient controls");
}

#[test]
fn netlist_voltage_source_waveform_is_allowed() {
    let input = "V1 in 0 PULSE(0 1 1n 1n 1n 10n 20n)\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty());
}

#[test]
fn netlist_mos_three_node_is_allowed() {
    let input = "M1 d g s nmos\n.end\n";
    let ast = parse_netlist(input);
    let device = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Device(dev) => Some(dev),
            _ => None,
        })
        .expect("device not found");
    assert_eq!(device.nodes.len(), 4);
    assert_eq!(device.nodes[3], "0");
    assert_eq!(device.model.as_deref(), Some("nmos"));
}

#[test]
fn netlist_waveform_keyword_requires_args() {
    let input = "V1 in 0 SIN\n.end\n";
    let ast = parse_netlist(input);
    assert!(!ast.errors.is_empty());
}

#[test]
fn netlist_waveform_keyword_with_args_is_ok() {
    let input = "V1 in 0 AC 1\n.end\n";
    let ast = parse_netlist(input);
    assert!(ast.errors.is_empty());
}

#[test]
fn netlist_model_params_in_parentheses_are_parsed() {
    let input = ".model DTEST D (IS=1e-14 N=1)\n.end\n";
    let ast = parse_netlist(input);
    let model = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Model) => Some(ctrl),
            _ => None,
        })
        .expect("model not found");
    assert_eq!(model.model_name.as_deref(), Some("DTEST"));
    assert_eq!(model.model_type.as_deref(), Some("D"));
    assert_eq!(model.params.len(), 2);
    assert_eq!(model.params[0].key, "IS");
    assert_eq!(model.params[1].key, "N");
}

#[test]
fn netlist_param_tokens_with_commas_are_split() {
    let input = "R1 in out 1k W=1u,L=2u\n.end\n";
    let ast = parse_netlist(input);
    let device = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Device(dev) => Some(dev),
            _ => None,
        })
        .expect("device not found");
    assert_eq!(device.params.len(), 2);
    assert_eq!(device.params[0].key, "W");
    assert_eq!(device.params[1].key, "L");
}

#[test]
fn netlist_subckt_body_param_is_applied() {
    let input = ".subckt buf in out\n.param RVAL=3k\nR1 in out RVAL\n.ends\nX1 a b buf\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].value.as_deref(), Some("3000"));
}

#[test]
fn netlist_elaboration_expands_nested_subckt() {
    let input = ".subckt leaf a b\nR1 a b 1k\n.ends\n.subckt mid in out\nX1 in out leaf\n.ends\nXtop n1 n2 mid\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    assert_eq!(elab.instances.len(), 1);
    assert_eq!(elab.instances[0].name, "Xtop.X1.R1");
    assert_eq!(
        elab.instances[0].nodes,
        vec!["n1".to_string(), "n2".to_string()]
    );
}

// ============================================================================
// .IC (Initial Condition) Tests
// ============================================================================

use sim_core::netlist::build_circuit;

#[test]
fn netlist_ic_directive_is_recognized() {
    let input = "R1 in out 1k\n.ic v(in)=5\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let ctrl = ast
        .statements
        .iter()
        .find_map(|stmt| match stmt {
            Stmt::Control(ctrl) if matches!(ctrl.kind, ControlKind::Ic) => Some(ctrl),
            _ => None,
        });
    assert!(ctrl.is_some(), ".ic directive should be recognized");
}

#[test]
fn netlist_ic_single_node_is_parsed() {
    let input = "R1 in out 1k\nR2 out 0 1k\n.ic v(in)=5\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    assert_eq!(circuit.initial_conditions.len(), 1);

    // Find the node ID for "in"
    let in_node_id = circuit.nodes.name_to_id.get("in").expect("node 'in' not found");
    let ic_value = circuit.initial_conditions.get(in_node_id).expect("IC for 'in' not found");
    assert!((ic_value - 5.0).abs() < 1e-10, "IC value should be 5.0, got {}", ic_value);
}

#[test]
fn netlist_ic_multiple_nodes_is_parsed() {
    let input = "R1 in mid 1k\nR2 mid out 1k\nR3 out 0 1k\n.ic v(in)=5 v(mid)=2.5 v(out)=1.0\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    assert_eq!(circuit.initial_conditions.len(), 3);

    let in_id = circuit.nodes.name_to_id.get("in").unwrap();
    let mid_id = circuit.nodes.name_to_id.get("mid").unwrap();
    let out_id = circuit.nodes.name_to_id.get("out").unwrap();

    assert!((circuit.initial_conditions[in_id] - 5.0).abs() < 1e-10);
    assert!((circuit.initial_conditions[mid_id] - 2.5).abs() < 1e-10);
    assert!((circuit.initial_conditions[out_id] - 1.0).abs() < 1e-10);
}

#[test]
fn netlist_ic_with_engineering_suffix() {
    let input = "R1 in out 1k\nR2 out 0 1k\n.ic v(in)=3.3 v(out)=1m\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    let in_id = circuit.nodes.name_to_id.get("in").unwrap();
    let out_id = circuit.nodes.name_to_id.get("out").unwrap();

    assert!((circuit.initial_conditions[in_id] - 3.3).abs() < 1e-10);
    assert!((circuit.initial_conditions[out_id] - 0.001).abs() < 1e-10);
}

#[test]
fn netlist_ic_case_insensitive() {
    let input = "R1 in out 1k\n.IC V(IN)=2.5\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    assert_eq!(circuit.initial_conditions.len(), 1);
    // Node name should be lowercase
    let in_id = circuit.nodes.name_to_id.get("in").unwrap();
    assert!((circuit.initial_conditions[in_id] - 2.5).abs() < 1e-10);
}

#[test]
fn netlist_ic_multiple_lines() {
    let input = "R1 in out 1k\nR2 out 0 1k\n.ic v(in)=5\n.ic v(out)=2\n.tran 1n 10n\n.end\n";
    let ast = parse_netlist(input);
    let elab = elaborate_netlist(&ast);
    let circuit = build_circuit(&ast, &elab);

    assert_eq!(circuit.initial_conditions.len(), 2);

    let in_id = circuit.nodes.name_to_id.get("in").unwrap();
    let out_id = circuit.nodes.name_to_id.get("out").unwrap();

    assert!((circuit.initial_conditions[in_id] - 5.0).abs() < 1e-10);
    assert!((circuit.initial_conditions[out_id] - 2.0).abs() < 1e-10);
}
