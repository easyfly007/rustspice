"""Tests for the console widget."""

import pytest
from PySide6.QtWidgets import QApplication

from rustspice_gui.console import ConsoleWidget, MessageLevel


# Create QApplication for tests (required for Qt widgets)
@pytest.fixture(scope="module")
def qapp():
    """Create QApplication for the test session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestMessageLevel:
    """Tests for MessageLevel enum."""

    def test_values(self):
        """Test message level values."""
        assert MessageLevel.INFO.value == "info"
        assert MessageLevel.SUCCESS.value == "success"
        assert MessageLevel.WARNING.value == "warning"
        assert MessageLevel.ERROR.value == "error"
        assert MessageLevel.DEBUG.value == "debug"


class TestConsoleWidget:
    """Tests for ConsoleWidget."""

    def test_creation(self, qapp):
        """Test widget creation."""
        console = ConsoleWidget()
        assert console is not None

    def test_log_info(self, qapp):
        """Test logging info message."""
        console = ConsoleWidget()
        console.info("Test info message")

        text = console.get_text()
        assert "Test info message" in text

    def test_log_success(self, qapp):
        """Test logging success message."""
        console = ConsoleWidget()
        console.success("Operation completed")

        text = console.get_text()
        assert "Operation completed" in text

    def test_log_warning(self, qapp):
        """Test logging warning message."""
        console = ConsoleWidget()
        console.warning("This is a warning")

        text = console.get_text()
        assert "This is a warning" in text

    def test_log_error(self, qapp):
        """Test logging error message."""
        console = ConsoleWidget()
        console.error("An error occurred")

        text = console.get_text()
        assert "An error occurred" in text

    def test_log_debug(self, qapp):
        """Test logging debug message."""
        console = ConsoleWidget()
        console.debug("Debug info")

        text = console.get_text()
        assert "Debug info" in text

    def test_clear(self, qapp):
        """Test clearing console."""
        console = ConsoleWidget()
        console.info("Message 1")
        console.info("Message 2")
        console.clear()

        text = console.get_text()
        assert text == ""

    def test_message_count(self, qapp):
        """Test message counting."""
        console = ConsoleWidget()
        console.info("Message 1")
        console.info("Message 2")
        console.info("Message 3")

        assert console._message_count == 3

    def test_clear_resets_count(self, qapp):
        """Test that clear resets message count."""
        console = ConsoleWidget()
        console.info("Message 1")
        console.info("Message 2")
        console.clear()

        assert console._message_count == 0

    def test_log_without_timestamp(self, qapp):
        """Test logging without timestamp."""
        console = ConsoleWidget()
        console.log("No timestamp", timestamp=False)

        text = console.get_text()
        assert "> No timestamp" in text

    def test_max_messages_limit(self, qapp):
        """Test that messages are trimmed when limit exceeded."""
        console = ConsoleWidget()
        console.set_max_messages(100)

        # Add more than max messages
        for i in range(150):
            console.info(f"Message {i}")

        # Should have trimmed some messages
        assert console._message_count <= 100
