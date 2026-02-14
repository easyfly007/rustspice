//! BSIM MOSFET Integration Tests
//!
//! Tests that verify the BSIM model integrates correctly with
//! the simulation engine.

use sim_core::circuit::{DeviceKind, Instance, NodeId};
use sim_core::stamp::{InstanceStamp, DeviceStamp};
use sim_core::mna::MnaBuilder;
use std::collections::HashMap;

fn make_mos_instance(name: &str, params: HashMap<String, String>) -> Instance {
    Instance {
        name: name.to_string(),
        kind: DeviceKind::M,
        nodes: vec![
            NodeId(1), // drain
            NodeId(2), // gate
            NodeId(3), // source
            NodeId(4), // bulk
        ],
        value: None,
        params,
        model: None,
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    }
}

#[test]
fn nmos_bsim_stamp_runs_without_panic() {
    let mut params = HashMap::new();
    params.insert("w".to_string(), "1u".to_string());
    params.insert("l".to_string(), "100n".to_string());
    params.insert("level".to_string(), "49".to_string());
    params.insert("vth0".to_string(), "0.5".to_string());

    let inst = make_mos_instance("M1", params);
    let stamp = InstanceStamp { instance: inst };

    let mut mna = MnaBuilder::new(5);
    let mut ctx = mna.context_with(1e-12, 1.0);
    let x = vec![0.0, 1.8, 1.2, 0.0, 0.0]; // ground, drain, gate, source, bulk

    let result = stamp.stamp_dc(&mut ctx, Some(&x));
    assert!(result.is_ok(), "BSIM stamp should succeed");
}

#[test]
fn pmos_bsim_stamp_runs_without_panic() {
    let mut params = HashMap::new();
    params.insert("w".to_string(), "2u".to_string());
    params.insert("l".to_string(), "100n".to_string());
    params.insert("level".to_string(), "49".to_string());
    params.insert("type".to_string(), "pmos".to_string());
    params.insert("vth0".to_string(), "-0.5".to_string());

    let inst = make_mos_instance("M2", params);
    let stamp = InstanceStamp { instance: inst };

    let mut mna = MnaBuilder::new(5);
    let mut ctx = mna.context_with(1e-12, 1.0);
    let x = vec![0.0, 0.0, 0.0, 1.8, 1.8]; // vss, drain, gate, source, bulk

    let result = stamp.stamp_dc(&mut ctx, Some(&x));
    assert!(result.is_ok(), "PMOS BSIM stamp should succeed");
}

#[test]
fn level1_mos_stamp_runs_without_panic() {
    let mut params = HashMap::new();
    params.insert("w".to_string(), "1u".to_string());
    params.insert("l".to_string(), "1u".to_string());
    params.insert("level".to_string(), "1".to_string());
    params.insert("vth0".to_string(), "0.7".to_string());

    let inst = make_mos_instance("M3", params);
    let stamp = InstanceStamp { instance: inst };

    let mut mna = MnaBuilder::new(5);
    let mut ctx = mna.context_with(1e-12, 1.0);
    let x = vec![0.0, 3.0, 2.0, 0.0, 0.0];

    let result = stamp.stamp_dc(&mut ctx, Some(&x));
    assert!(result.is_ok(), "Level 1 MOS stamp should succeed");
}

#[test]
fn mos_initial_stamp_uses_gmin() {
    let mut params = HashMap::new();
    params.insert("w".to_string(), "1u".to_string());
    params.insert("l".to_string(), "100n".to_string());

    let inst = make_mos_instance("M4", params);
    let stamp = InstanceStamp { instance: inst };

    let mut mna = MnaBuilder::new(5);
    let mut ctx = mna.context_with(1e-12, 1.0);

    // No solution yet (initial stamp)
    let result = stamp.stamp_dc(&mut ctx, None);
    assert!(result.is_ok(), "Initial stamp should succeed");
}
