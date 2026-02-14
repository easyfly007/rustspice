#[test]
fn parse_basic_netlist_fixture_is_available() {
    let netlist = include_str!("../../../tests/fixtures/netlists/basic_dc.cir");
    assert!(netlist.contains("R1"));
}
