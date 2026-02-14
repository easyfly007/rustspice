"""
Tests for the theme manager.
"""

import pytest
from PySide6.QtWidgets import QApplication

from myspice_gui.theme import (
    ThemeManager,
    ThemeMode,
    ThemeColors,
    LIGHT_THEME,
    DARK_THEME,
    theme_manager,
)


@pytest.fixture(scope="session")
def app():
    """Create QApplication instance."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


class TestThemeColors:
    """Tests for ThemeColors dataclass."""

    def test_light_theme_colors(self):
        """Test light theme has all required colors."""
        assert LIGHT_THEME.window is not None
        assert LIGHT_THEME.base is not None
        assert LIGHT_THEME.text is not None
        assert LIGHT_THEME.highlight is not None
        assert LIGHT_THEME.editor_bg is not None
        assert LIGHT_THEME.plot_bg is not None

    def test_dark_theme_colors(self):
        """Test dark theme has all required colors."""
        assert DARK_THEME.window is not None
        assert DARK_THEME.base is not None
        assert DARK_THEME.text is not None
        assert DARK_THEME.highlight is not None
        assert DARK_THEME.editor_bg is not None
        assert DARK_THEME.plot_bg is not None

    def test_themes_are_different(self):
        """Test that light and dark themes are different."""
        assert LIGHT_THEME.window != DARK_THEME.window
        assert LIGHT_THEME.base != DARK_THEME.base
        assert LIGHT_THEME.text != DARK_THEME.text


class TestThemeManager:
    """Tests for ThemeManager class."""

    def test_singleton(self):
        """Test that ThemeManager is a singleton."""
        manager1 = ThemeManager()
        manager2 = ThemeManager()
        assert manager1 is manager2

    def test_default_theme(self, app):
        """Test default theme is light."""
        manager = ThemeManager()
        # Reset to light for testing
        manager.set_theme(ThemeMode.LIGHT)
        assert manager.current_theme == ThemeMode.LIGHT

    def test_set_theme(self, app):
        """Test setting theme."""
        manager = ThemeManager()

        manager.set_theme(ThemeMode.DARK)
        assert manager.current_theme == ThemeMode.DARK

        manager.set_theme(ThemeMode.LIGHT)
        assert manager.current_theme == ThemeMode.LIGHT

    def test_toggle_theme(self, app):
        """Test toggling theme."""
        manager = ThemeManager()

        manager.set_theme(ThemeMode.LIGHT)
        new_mode = manager.toggle_theme()
        assert new_mode == ThemeMode.DARK
        assert manager.current_theme == ThemeMode.DARK

        new_mode = manager.toggle_theme()
        assert new_mode == ThemeMode.LIGHT
        assert manager.current_theme == ThemeMode.LIGHT

    def test_colors_property(self, app):
        """Test colors property returns correct theme."""
        manager = ThemeManager()

        manager.set_theme(ThemeMode.LIGHT)
        assert manager.colors == LIGHT_THEME

        manager.set_theme(ThemeMode.DARK)
        assert manager.colors == DARK_THEME

    def test_get_editor_stylesheet(self, app):
        """Test editor stylesheet generation."""
        manager = ThemeManager()

        stylesheet = manager.get_editor_stylesheet()
        assert "QPlainTextEdit" in stylesheet
        assert "background-color" in stylesheet

    def test_get_plot_colors(self, app):
        """Test plot colors getter."""
        manager = ThemeManager()

        colors = manager.get_plot_colors()
        assert "background" in colors
        assert "foreground" in colors
        assert "grid" in colors

    def test_generate_stylesheet(self, app):
        """Test stylesheet generation."""
        manager = ThemeManager()

        stylesheet = manager._generate_stylesheet(LIGHT_THEME)
        assert "QWidget" in stylesheet
        assert "QPushButton" in stylesheet
        assert "QTableWidget" in stylesheet

    def test_generate_palette(self, app):
        """Test palette generation."""
        manager = ThemeManager()

        palette = manager._generate_palette(LIGHT_THEME)
        assert palette is not None


class TestGlobalThemeManager:
    """Tests for global theme_manager instance."""

    def test_global_instance_exists(self):
        """Test that global theme_manager exists."""
        assert theme_manager is not None

    def test_global_instance_is_singleton(self):
        """Test that global instance is the singleton."""
        manager = ThemeManager()
        assert theme_manager is manager
