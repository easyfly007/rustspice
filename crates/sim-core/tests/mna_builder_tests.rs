use sim_core::mna::{AuxVarTable, SparseBuilder};
use sim_core::mna::MnaBuilder;
use sim_core::stamp::{DeviceStamp, InstanceStamp};
use sim_core::circuit::{DeviceKind, Instance, NodeId};
use std::collections::HashMap;

#[test]
fn aux_var_table_allocates_unique_ids() {
    let mut table = AuxVarTable::new();
    let id1 = table.allocate("V1");
    let id2 = table.allocate("V2");
    let id1_again = table.allocate("V1");
    assert_eq!(id1, id1_again);
    assert_ne!(id1, id2);
}

#[test]
fn sparse_builder_accepts_inserts() {
    let mut builder = SparseBuilder::new(3);
    builder.insert(0, 0, 1.0);
    builder.insert(1, 0, -1.0);
    assert_eq!(builder.col_entries[0].len(), 1);
    assert_eq!(builder.col_entries[1].len(), 1);
}

#[test]
fn mna_builder_allocates_aux_for_voltage() {
    let mut builder = MnaBuilder::new(2);
    let instance = Instance {
        name: "V1".to_string(),
        kind: DeviceKind::V,
        nodes: vec![NodeId(0), NodeId(1)],
        model: None,
        params: HashMap::new(),
        value: Some("1".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let stamp = InstanceStamp { instance };
    let mut ctx = builder.context();
    stamp.stamp_dc(&mut ctx, None).unwrap();
    assert_eq!(builder.builder.n, 3);
    assert_eq!(builder.rhs.len(), 3);
}

#[test]
fn dc_op_mna_entries_for_r_and_i() {
    let mut builder = MnaBuilder::new(2);

    let r1 = Instance {
        name: "R1".to_string(),
        kind: DeviceKind::R,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1k".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let i1 = Instance {
        name: "I1".to_string(),
        kind: DeviceKind::I,
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
    InstanceStamp { instance: r1 }.stamp_dc(&mut ctx, None).unwrap();
    InstanceStamp { instance: i1 }.stamp_dc(&mut ctx, None).unwrap();

    let g = 1.0 / 1000.0;
    assert_eq!(sum_entry(&builder.builder, 1, 1), g);
    assert_eq!(sum_entry(&builder.builder, 0, 0), g);
    assert_eq!(sum_entry(&builder.builder, 1, 0), -g);
    assert_eq!(sum_entry(&builder.builder, 0, 1), -g);
    assert!((builder.rhs[1] + 0.001).abs() < 1e-12);
    assert!((builder.rhs[0] - 0.001).abs() < 1e-12);
}

#[test]
fn inductor_dc_stamp_as_short() {
    let mut builder = MnaBuilder::new(2);
    let l1 = Instance {
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
    InstanceStamp { instance: l1 }.stamp_dc(&mut ctx, None).unwrap();
    assert!(sum_entry(&builder.builder, 1, 1) > 0.0);
}

#[test]
fn source_scale_applies_to_current() {
    let mut builder = MnaBuilder::new(2);
    let i1 = Instance {
        name: "I1".to_string(),
        kind: DeviceKind::I,
        nodes: vec![NodeId(1), NodeId(0)],
        model: None,
        params: HashMap::new(),
        value: Some("1m".to_string()),
        control: None,
        ac_mag: None,
        ac_phase: None,
        poly: None,
    };
    let mut ctx = builder.context_with(0.0, 0.5);
    InstanceStamp { instance: i1 }.stamp_dc(&mut ctx, None).unwrap();
    assert!((builder.rhs[1] + 0.0005).abs() < 1e-12);
    assert!((builder.rhs[0] - 0.0005).abs() < 1e-12);
}

#[test]
fn gmin_applies_to_diode_stamp() {
    let mut builder = MnaBuilder::new(2);
    let d1 = Instance {
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
    let mut ctx = builder.context_with(1e-6, 1.0);
    InstanceStamp { instance: d1 }.stamp_dc(&mut ctx, None).unwrap();
    assert_eq!(sum_entry(&builder.builder, 1, 1), 1e-6);
    assert_eq!(sum_entry(&builder.builder, 0, 0), 1e-6);
}

#[test]
fn diode_stamp_uses_solution_when_provided() {
    let mut builder = MnaBuilder::new(2);
    let d1 = Instance {
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
    let mut ctx = builder.context_with(1e-12, 1.0);
    let x = vec![0.0, 0.7];
    InstanceStamp { instance: d1 }.stamp_dc(&mut ctx, Some(&x)).unwrap();
    assert!(sum_entry(&builder.builder, 1, 1) > 1e-12);
}

fn sum_entry(builder: &SparseBuilder, row: usize, col: usize) -> f64 {
    builder.col_entries[col]
        .iter()
        .filter(|(r, _)| *r == row)
        .map(|(_, v)| *v)
        .sum()
}

// ============================================================================
// SparseBuilder Pattern Caching Tests
// ============================================================================

#[test]
fn sparse_builder_finalize_merged_combines_duplicates() {
    let mut builder = SparseBuilder::new(3);

    // Insert duplicate entries at same position
    builder.insert(0, 0, 1.0);
    builder.insert(0, 0, 2.0);  // Should be merged with above
    builder.insert(0, 1, 3.0);
    builder.insert(1, 1, 4.0);

    let (ap, ai, ax) = builder.finalize_merged();

    // Column 0: has two entries (row 0 merged, row 1)
    assert_eq!(ap[0], 0);
    assert_eq!(ap[1], 2);  // 2 unique entries in column 0

    // Row 0 value should be 1.0 + 2.0 = 3.0
    assert_eq!(ai[0], 0);
    assert_eq!(ax[0], 3.0);

    // Row 1 value should be 3.0
    assert_eq!(ai[1], 1);
    assert_eq!(ax[1], 3.0);

    // Column 1: has one entry
    assert_eq!(ap[2], 3);
    assert_eq!(ai[2], 1);
    assert_eq!(ax[2], 4.0);

    // Stats should show 1 duplicate merged
    assert_eq!(builder.stats().duplicates_merged, 1);
}

#[test]
fn sparse_builder_freeze_pattern_enables_fast_updates() {
    let mut builder = SparseBuilder::new(3);

    // Build initial pattern
    builder.insert(0, 0, 1.0);
    builder.insert(0, 1, 2.0);
    builder.insert(1, 1, 3.0);
    builder.insert(2, 2, 4.0);

    // Freeze the pattern
    builder.freeze_pattern();
    assert!(builder.is_frozen());
    assert_eq!(builder.stats().freezes, 1);

    // Update values using O(1) lookup
    assert!(builder.update(0, 0, 10.0));
    assert!(builder.update(1, 1, 30.0));

    // Try to update non-existent position (should fail)
    assert!(!builder.update(0, 2, 999.0));

    // Verify values
    let (_, _, ax) = builder.finalize();
    assert_eq!(ax[0], 10.0);  // Updated
    assert_eq!(ax[1], 2.0);   // Unchanged
    assert_eq!(ax[2], 30.0);  // Updated
    assert_eq!(ax[3], 4.0);   // Unchanged
}

#[test]
fn sparse_builder_accumulate_in_frozen_mode() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(0, 1, 2.0);
    builder.freeze_pattern();

    // Accumulate adds to existing value
    assert!(builder.accumulate(0, 0, 5.0));
    assert!(builder.accumulate(0, 0, 3.0));

    let (_, _, ax) = builder.finalize();
    assert_eq!(ax[0], 9.0);  // 1.0 + 5.0 + 3.0
}

#[test]
fn sparse_builder_clear_values_preserves_pattern() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(0, 1, 2.0);
    builder.insert(1, 1, 3.0);
    builder.freeze_pattern();

    // Clear values
    builder.clear_values();

    // Pattern should be preserved
    let (ap, ai, ax) = builder.finalize();
    assert_eq!(ap.len(), 3);  // n+1 = 3
    assert_eq!(ai.len(), 3);  // 3 non-zeros
    assert!(ax.iter().all(|&v| v == 0.0));

    // Can still update
    assert!(builder.update(0, 0, 99.0));
    let values = builder.get_values().unwrap();
    assert_eq!(values[0], 99.0);
}

#[test]
fn sparse_builder_unfreeze_allows_new_entries() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.freeze_pattern();
    assert!(builder.is_frozen());

    builder.unfreeze();
    assert!(!builder.is_frozen());
    assert_eq!(builder.stats().unfreezes, 1);

    // Can now add new entries
    builder.insert(0, 1, 2.0);
    builder.insert(1, 1, 3.0);

    let (ap, _, _) = builder.finalize_merged();
    assert_eq!(ap[1] - ap[0], 2);  // Column 0 now has 2 entries
}

#[test]
fn sparse_builder_insert_in_frozen_mode_uses_index_map() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(0, 1, 2.0);
    builder.freeze_pattern();

    // Insert in frozen mode should accumulate
    builder.insert(0, 0, 5.0);
    builder.insert(0, 1, 3.0);

    let (_, _, ax) = builder.finalize();
    assert_eq!(ax[0], 6.0);  // 1.0 + 5.0
    assert_eq!(ax[1], 5.0);  // 2.0 + 3.0
}

#[test]
fn sparse_builder_get_csc_returns_cached_arrays() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(1, 1, 2.0);

    // Before freeze: get_csc returns None
    assert!(builder.get_csc().is_none());

    builder.freeze_pattern();

    // After freeze: get_csc returns cached arrays
    let (ap, ai, ax) = builder.get_csc().unwrap();
    assert_eq!(ap.len(), 3);
    assert_eq!(ai.len(), 2);
    assert_eq!(ax.len(), 2);
}

#[test]
fn sparse_builder_resize_unfreezes_pattern() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.freeze_pattern();
    assert!(builder.is_frozen());

    // Resize should unfreeze
    builder.resize(3);
    assert!(!builder.is_frozen());

    // Can add entries to new dimension
    builder.insert(2, 2, 5.0);

    let (ap, _, _) = builder.finalize_merged();
    assert_eq!(ap.len(), 4);  // n+1 = 4
}

#[test]
fn sparse_builder_stats_track_operations() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(0, 0, 2.0);
    builder.insert(0, 1, 3.0);

    assert_eq!(builder.stats().inserts, 3);

    builder.finalize_merged();
    assert_eq!(builder.stats().duplicates_merged, 1);

    builder.freeze_pattern();
    builder.update(0, 0, 10.0);
    builder.update(0, 1, 20.0);

    assert_eq!(builder.stats().freezes, 1);
    assert_eq!(builder.stats().updates, 2);

    builder.reset_stats();
    assert_eq!(builder.stats().inserts, 0);
}

#[test]
fn sparse_builder_nnz_returns_correct_count() {
    let mut builder = SparseBuilder::new(3);

    builder.insert(0, 0, 1.0);
    builder.insert(0, 0, 2.0);  // Duplicate
    builder.insert(1, 1, 3.0);
    builder.insert(2, 2, 4.0);

    // Before freeze: nnz includes duplicates
    assert_eq!(builder.nnz(), 4);

    builder.freeze_pattern();

    // After freeze: nnz is the actual count (duplicates merged)
    assert_eq!(builder.nnz(), 3);
}

#[test]
fn sparse_builder_clear_all_resets_everything() {
    let mut builder = SparseBuilder::new(2);

    builder.insert(0, 0, 1.0);
    builder.insert(1, 1, 2.0);
    builder.freeze_pattern();

    builder.clear_all();

    assert!(!builder.is_frozen());
    assert_eq!(builder.nnz(), 0);
    assert!(builder.col_entries.iter().all(|c| c.is_empty()));
}
