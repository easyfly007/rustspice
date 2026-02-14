use sim_core::circuit::{DeviceKind, Instance, NodeId};
use sim_core::mna::MnaBuilder;
use sim_core::stamp::{DeviceStamp, InstanceStamp, TransientState};
use std::collections::HashMap;

#[test]
fn diode_stamp_allows_basic_nodes() {
    let mut builder = MnaBuilder::new(2);
    let diode = Instance {
        name: "D1".to_string(),
        kind: DeviceKind::D,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: None,
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: diode }.stamp_dc(&mut ctx, None).unwrap();
}

#[test]
fn mos_stamp_allows_basic_nodes() {
    let mut builder = MnaBuilder::new(4);
    let mos = Instance {
        name: "M1".to_string(),
        kind: DeviceKind::M,
        nodes: vec![NodeId(1), NodeId(2), NodeId(3), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: None,
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: mos }.stamp_dc(&mut ctx, None).unwrap();
}

#[test]
fn capacitor_tran_stamp_basic() {
    let mut builder = MnaBuilder::new(2);
    let cap = Instance {
        name: "C1".to_string(),
        kind: DeviceKind::C,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1u".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    let mut state = TransientState::default();
    InstanceStamp { instance: cap }
        .stamp_tran(&mut ctx, Some(&vec![0.0, 1.0]), 1e-6, &mut state)
        .unwrap();
    assert!(builder.rhs[1].is_finite());
}

#[test]
fn inductor_tran_stamp_basic() {
    let mut builder = MnaBuilder::new(2);
    let ind = Instance {
        name: "L1".to_string(),
        kind: DeviceKind::L,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1m".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    let mut state = TransientState::default();
    InstanceStamp { instance: ind }
        .stamp_tran(&mut ctx, Some(&vec![0.0, 0.0, 0.0]), 1e-6, &mut state)
        .unwrap();
    assert!(builder.builder.n >= 3);
}

#[test]
fn update_transient_state_tracks_cap_voltage() {
    let cap = Instance {
        name: "C1".to_string(),
        kind: DeviceKind::C,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1u".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut state = TransientState::default();
    sim_core::stamp::update_transient_state(&[cap], &[0.0, 2.0], &mut state);
    assert_eq!(state.cap_voltage.get("C1").copied(), Some(2.0));
}

#[test]
fn vcvs_stamp_basic() {
    // E1 out 0 in 0 2.0 (gain=2)
    let mut builder = MnaBuilder::new(3); // nodes: 0=gnd, 1=out, 2=in
    let vcvs = Instance {
        name: "E1".to_string(),
        kind: DeviceKind::E,
        nodes: vec![NodeId(1), NodeId(0), NodeId(2), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("2.0".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: vcvs }.stamp_dc(&mut ctx, None).unwrap();
    // Should have allocated an auxiliary variable
    assert!(builder.builder.n >= 4);
}

#[test]
fn vccs_stamp_basic() {
    // G1 out 0 in 0 0.001 (gm=1mS)
    let mut builder = MnaBuilder::new(3); // nodes: 0=gnd, 1=out, 2=in
    let vccs = Instance {
        name: "G1".to_string(),
        kind: DeviceKind::G,
        nodes: vec![NodeId(1), NodeId(0), NodeId(2), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1m".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: vccs }.stamp_dc(&mut ctx, None).unwrap();
    // VCCS doesn't need auxiliary variable, size should stay same
    assert_eq!(builder.builder.n, 3);
}

#[test]
fn cccs_stamp_requires_control_source() {
    // F1 out 0 Vctrl 2.0 (gain=2, controlled by Vctrl)
    let mut builder = MnaBuilder::new(3);

    // First, stamp a voltage source to create the controlling current
    let vsrc = Instance {
        name: "Vctrl".to_string(),
        kind: DeviceKind::V,
        nodes: vec![NodeId(2), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1.0".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: vsrc }.stamp_dc(&mut ctx, None).unwrap();

    // Now stamp the CCCS
    let cccs = Instance {
        name: "F1".to_string(),
        kind: DeviceKind::F,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("2.0".to_string()),
        control: Some("Vctrl".to_string()),
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: cccs }.stamp_dc(&mut ctx, None).unwrap();
}

#[test]
fn ccvs_stamp_requires_control_source() {
    // H1 out 0 Vctrl 1000 (transresistance=1kOhm)
    let mut builder = MnaBuilder::new(3);

    // First, stamp a voltage source to create the controlling current
    let vsrc = Instance {
        name: "Vctrl".to_string(),
        kind: DeviceKind::V,
        nodes: vec![NodeId(2), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1.0".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: vsrc }.stamp_dc(&mut ctx, None).unwrap();

    // Now stamp the CCVS
    let ccvs = Instance {
        name: "H1".to_string(),
        kind: DeviceKind::H,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1k".to_string()),
        control: Some("Vctrl".to_string()),
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    InstanceStamp { instance: ccvs }.stamp_dc(&mut ctx, None).unwrap();
    // CCVS allocates its own auxiliary variable, so size should increase
    assert!(builder.builder.n >= 5);
}

#[test]
fn subcircuit_instance_stamp_is_noop() {
    let mut builder = MnaBuilder::new(3);
    let xinst = Instance {
        name: "X1".to_string(),
        kind: DeviceKind::X,
        nodes: vec![NodeId(1), NodeId(2), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: None,
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context();
    // Should succeed without doing anything (subcircuits are already expanded)
    InstanceStamp { instance: xinst }.stamp_dc(&mut ctx, None).unwrap();
}
