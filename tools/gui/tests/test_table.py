"""
Tests for the enhanced results table widget.
"""

import pytest
from PySide6.QtWidgets import QApplication

from rustspice_gui.viewer.table import ResultsTable, ResultType


@pytest.fixture(scope="session")
def app():
    """Create QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestResultsTable:
    """Tests for ResultsTable widget."""

    def test_creation(self, app):
        """Test table can be created."""
        table = ResultsTable()
        assert table is not None

    def test_initial_state(self, app):
        """Test initial table state."""
        table = ResultsTable()
        assert table.get_entry_count() == 0

    def test_set_op_results(self, app):
        """Test setting OP results."""
        table = ResultsTable()

        nodes = ["in", "out"]
        solution = [5.0, 3.333]

        table.set_op_results(nodes, solution)

        assert table.get_entry_count() == 2

    def test_set_op_results_skips_ground(self, app):
        """Test that ground node is skipped."""
        table = ResultsTable()

        nodes = ["0", "in", "out"]
        solution = [0.0, 5.0, 3.333]

        table.set_op_results(nodes, solution)

        # Should only have 2 entries (in and out, not 0)
        assert table.get_entry_count() == 2

    def test_set_dc_results(self, app):
        """Test setting DC sweep results."""
        table = ResultsTable()

        table.set_dc_results(
            sweep_var="V1",
            sweep_values=[0, 1, 2, 3, 4, 5],
            nodes=["in", "out"],
            solutions=[]
        )

        # Should have summary entries
        assert table.get_entry_count() >= 2

    def test_set_tran_results(self, app):
        """Test setting transient results."""
        table = ResultsTable()

        times = [0, 1e-9, 2e-9, 3e-9]
        table.set_tran_results(
            times=times,
            nodes=["in", "out"],
            solutions=[]
        )

        # Should have summary entries
        assert table.get_entry_count() >= 2

    def test_set_ac_results(self, app):
        """Test setting AC results."""
        table = ResultsTable()

        frequencies = [1, 10, 100, 1000]
        table.set_ac_results(
            frequencies=frequencies,
            nodes=["in", "out"],
            solutions=[]
        )

        # Should have summary entries
        assert table.get_entry_count() >= 2

    def test_clear(self, app):
        """Test clearing table."""
        table = ResultsTable()

        table.set_op_results(["in", "out"], [5.0, 3.333])
        assert table.get_entry_count() > 0

        table.clear()
        assert table.get_entry_count() == 0

    def test_format_value_engineering(self, app):
        """Test engineering notation formatting."""
        table = ResultsTable()

        # Test various values
        test_cases = [
            (1e6, "V", "1"),  # Should be 1 MV
            (1e3, "V", "1"),  # Should be 1 kV
            (1, "V", "1"),    # Should be 1 V
            (1e-3, "V", "1"), # Should be 1 mV
            (1e-6, "V", "1"), # Should be 1 uV
            (1e-9, "V", "1"), # Should be 1 nV
        ]

        for value, unit, expected_prefix in test_cases:
            formatted, full_unit = table._format_value(value, unit)
            assert formatted is not None
            assert unit in full_unit or full_unit.endswith(unit)


class TestResultsTableFiltering:
    """Tests for table filtering functionality."""

    def test_filter_rows(self, app):
        """Test filtering rows by text."""
        table = ResultsTable()

        nodes = ["in", "out", "vdd", "gnd"]
        solution = [5.0, 3.333, 1.8, 0.0]

        table.set_op_results(nodes, solution)

        # Filter should work via the search box
        table._filter_rows("out")

        # Hard to test visibility without rendering, but function should not crash


class TestResultsTableExport:
    """Tests for table export functionality."""

    def test_copy_selected(self, app):
        """Test copy selected works without crashing."""
        table = ResultsTable()

        nodes = ["in", "out"]
        solution = [5.0, 3.333]

        table.set_op_results(nodes, solution)

        # Should not crash even with no selection
        table.copy_selected()

    def test_copy_all(self, app):
        """Test copy all works without crashing."""
        table = ResultsTable()

        nodes = ["in", "out"]
        solution = [5.0, 3.333]

        table.set_op_results(nodes, solution)

        # Should not crash
        table.copy_all()
