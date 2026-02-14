"""
Theme manager for RustSpice GUI.

Provides dark and light themes with consistent styling across all widgets.
"""

from enum import Enum
from typing import Optional
from dataclasses import dataclass

from PySide6.QtWidgets import QApplication, QWidget
from PySide6.QtGui import QPalette, QColor
from PySide6.QtCore import QSettings


class ThemeMode(Enum):
    """Available theme modes."""
    LIGHT = "light"
    DARK = "dark"
    SYSTEM = "system"


@dataclass
class ThemeColors:
    """Color scheme for a theme."""
    # Background colors
    window: str
    base: str
    alternate_base: str

    # Text colors
    text: str
    text_disabled: str
    placeholder: str

    # Accent colors
    highlight: str
    highlight_text: str

    # Button colors
    button: str
    button_text: str

    # Border colors
    border: str
    border_light: str

    # Editor colors
    editor_bg: str
    editor_line_number_bg: str
    editor_current_line: str

    # Plot colors
    plot_bg: str
    plot_grid: str

    # Status colors
    success: str
    warning: str
    error: str


# Light theme colors
LIGHT_THEME = ThemeColors(
    window="#f5f5f5",
    base="#ffffff",
    alternate_base="#f9f9f9",
    text="#333333",
    text_disabled="#999999",
    placeholder="#aaaaaa",
    highlight="#0078d4",
    highlight_text="#ffffff",
    button="#e0e0e0",
    button_text="#333333",
    border="#cccccc",
    border_light="#e5e5e5",
    editor_bg="#fafafa",
    editor_line_number_bg="#f0f0f0",
    editor_current_line="#fffacd",
    plot_bg="#ffffff",
    plot_grid="#e0e0e0",
    success="#28a745",
    warning="#ffc107",
    error="#dc3545",
)

# Dark theme colors
DARK_THEME = ThemeColors(
    window="#1e1e1e",
    base="#252526",
    alternate_base="#2d2d2d",
    text="#cccccc",
    text_disabled="#666666",
    placeholder="#666666",
    highlight="#0078d4",
    highlight_text="#ffffff",
    button="#3c3c3c",
    button_text="#cccccc",
    border="#3c3c3c",
    border_light="#4a4a4a",
    editor_bg="#1e1e1e",
    editor_line_number_bg="#252526",
    editor_current_line="#264f78",
    plot_bg="#1e1e1e",
    plot_grid="#3c3c3c",
    success="#4ec9b0",
    warning="#dcdcaa",
    error="#f14c4c",
)


class ThemeManager:
    """
    Manages application themes.

    Usage:
        theme_manager = ThemeManager()
        theme_manager.set_theme(ThemeMode.DARK)
        theme_manager.apply_to_app(app)
    """

    _instance: Optional["ThemeManager"] = None
    _current_theme: ThemeMode = ThemeMode.LIGHT
    _settings_key = "theme/mode"

    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._settings = QSettings("RustSpice", "GUI")

    @property
    def current_theme(self) -> ThemeMode:
        """Get current theme mode."""
        return self._current_theme

    @property
    def colors(self) -> ThemeColors:
        """Get current theme colors."""
        if self._current_theme == ThemeMode.DARK:
            return DARK_THEME
        return LIGHT_THEME

    def load_saved_theme(self) -> ThemeMode:
        """Load saved theme from settings."""
        saved = self._settings.value(self._settings_key, ThemeMode.LIGHT.value)
        try:
            self._current_theme = ThemeMode(saved)
        except ValueError:
            self._current_theme = ThemeMode.LIGHT
        return self._current_theme

    def save_theme(self):
        """Save current theme to settings."""
        self._settings.setValue(self._settings_key, self._current_theme.value)

    def set_theme(self, mode: ThemeMode):
        """Set the theme mode."""
        self._current_theme = mode
        self.save_theme()

    def toggle_theme(self) -> ThemeMode:
        """Toggle between light and dark themes."""
        if self._current_theme == ThemeMode.LIGHT:
            self._current_theme = ThemeMode.DARK
        else:
            self._current_theme = ThemeMode.LIGHT
        self.save_theme()
        return self._current_theme

    def apply_to_app(self, app: QApplication):
        """Apply current theme to the application."""
        colors = self.colors
        stylesheet = self._generate_stylesheet(colors)
        app.setStyleSheet(stylesheet)

        # Also set the palette
        palette = self._generate_palette(colors)
        app.setPalette(palette)

    def _generate_palette(self, colors: ThemeColors) -> QPalette:
        """Generate QPalette from theme colors."""
        palette = QPalette()

        palette.setColor(QPalette.ColorRole.Window, QColor(colors.window))
        palette.setColor(QPalette.ColorRole.WindowText, QColor(colors.text))
        palette.setColor(QPalette.ColorRole.Base, QColor(colors.base))
        palette.setColor(QPalette.ColorRole.AlternateBase, QColor(colors.alternate_base))
        palette.setColor(QPalette.ColorRole.Text, QColor(colors.text))
        palette.setColor(QPalette.ColorRole.Button, QColor(colors.button))
        palette.setColor(QPalette.ColorRole.ButtonText, QColor(colors.button_text))
        palette.setColor(QPalette.ColorRole.Highlight, QColor(colors.highlight))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(colors.highlight_text))
        palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(colors.placeholder))

        # Disabled colors
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(colors.text_disabled))
        palette.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor(colors.text_disabled))

        return palette

    def _generate_stylesheet(self, colors: ThemeColors) -> str:
        """Generate stylesheet from theme colors."""
        return f"""
            /* Global styles */
            QWidget {{
                background-color: {colors.window};
                color: {colors.text};
            }}

            /* Main window */
            QMainWindow {{
                background-color: {colors.window};
            }}

            /* Menu bar */
            QMenuBar {{
                background-color: {colors.window};
                border-bottom: 1px solid {colors.border};
            }}
            QMenuBar::item:selected {{
                background-color: {colors.highlight};
                color: {colors.highlight_text};
            }}

            /* Menus */
            QMenu {{
                background-color: {colors.base};
                border: 1px solid {colors.border};
            }}
            QMenu::item:selected {{
                background-color: {colors.highlight};
                color: {colors.highlight_text};
            }}

            /* Toolbar */
            QToolBar {{
                background-color: {colors.window};
                border: none;
                spacing: 4px;
            }}

            /* Dock widgets */
            QDockWidget {{
                titlebar-close-icon: url(close.png);
                titlebar-normal-icon: url(float.png);
            }}
            QDockWidget::title {{
                background-color: {colors.alternate_base};
                padding: 4px;
                border-bottom: 1px solid {colors.border};
            }}

            /* Tab widgets */
            QTabWidget::pane {{
                border: 1px solid {colors.border};
                background-color: {colors.base};
            }}
            QTabBar::tab {{
                background-color: {colors.alternate_base};
                border: 1px solid {colors.border};
                padding: 6px 12px;
                margin-right: 2px;
            }}
            QTabBar::tab:selected {{
                background-color: {colors.base};
                border-bottom: none;
            }}
            QTabBar::tab:hover {{
                background-color: {colors.button};
            }}

            /* Push buttons */
            QPushButton {{
                background-color: {colors.button};
                border: 1px solid {colors.border};
                border-radius: 4px;
                padding: 6px 12px;
                min-width: 60px;
            }}
            QPushButton:hover {{
                background-color: {colors.border_light};
            }}
            QPushButton:pressed {{
                background-color: {colors.border};
            }}
            QPushButton:disabled {{
                color: {colors.text_disabled};
            }}

            /* Input fields */
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
                background-color: {colors.base};
                border: 1px solid {colors.border};
                border-radius: 4px;
                padding: 4px 8px;
            }}
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
                border-color: {colors.highlight};
            }}

            /* Combo box dropdown */
            QComboBox::drop-down {{
                border: none;
                padding-right: 8px;
            }}
            QComboBox QAbstractItemView {{
                background-color: {colors.base};
                border: 1px solid {colors.border};
                selection-background-color: {colors.highlight};
            }}

            /* Tables */
            QTableWidget, QTableView {{
                background-color: {colors.base};
                alternate-background-color: {colors.alternate_base};
                gridline-color: {colors.border_light};
                border: 1px solid {colors.border};
            }}
            QTableWidget::item:selected, QTableView::item:selected {{
                background-color: {colors.highlight};
                color: {colors.highlight_text};
            }}
            QHeaderView::section {{
                background-color: {colors.alternate_base};
                border: 1px solid {colors.border};
                padding: 4px;
            }}

            /* Scroll bars */
            QScrollBar:vertical {{
                background-color: {colors.alternate_base};
                width: 12px;
                border: none;
            }}
            QScrollBar::handle:vertical {{
                background-color: {colors.border};
                border-radius: 4px;
                min-height: 20px;
                margin: 2px;
            }}
            QScrollBar::handle:vertical:hover {{
                background-color: {colors.text_disabled};
            }}
            QScrollBar:horizontal {{
                background-color: {colors.alternate_base};
                height: 12px;
                border: none;
            }}
            QScrollBar::handle:horizontal {{
                background-color: {colors.border};
                border-radius: 4px;
                min-width: 20px;
                margin: 2px;
            }}
            QScrollBar::add-line, QScrollBar::sub-line {{
                width: 0px;
                height: 0px;
            }}

            /* Text edit / Plain text edit */
            QTextEdit, QPlainTextEdit {{
                background-color: {colors.editor_bg};
                border: 1px solid {colors.border};
            }}

            /* Splitter */
            QSplitter::handle {{
                background-color: {colors.border};
            }}
            QSplitter::handle:horizontal {{
                width: 4px;
            }}
            QSplitter::handle:vertical {{
                height: 4px;
            }}

            /* Status bar */
            QStatusBar {{
                background-color: {colors.alternate_base};
                border-top: 1px solid {colors.border};
            }}

            /* Group box */
            QGroupBox {{
                border: 1px solid {colors.border};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 4px;
            }}

            /* Progress bar */
            QProgressBar {{
                background-color: {colors.alternate_base};
                border: 1px solid {colors.border};
                border-radius: 4px;
                text-align: center;
            }}
            QProgressBar::chunk {{
                background-color: {colors.highlight};
                border-radius: 3px;
            }}

            /* Tool tips */
            QToolTip {{
                background-color: {colors.base};
                color: {colors.text};
                border: 1px solid {colors.border};
                padding: 4px;
            }}
        """

    def get_editor_stylesheet(self) -> str:
        """Get stylesheet specifically for the code editor."""
        colors = self.colors
        return f"""
            QPlainTextEdit {{
                background-color: {colors.editor_bg};
                color: {colors.text};
                border: 1px solid {colors.border};
                border-radius: 2px;
                selection-background-color: {colors.highlight};
                selection-color: {colors.highlight_text};
            }}
        """

    def get_current_line_color(self) -> str:
        """Get the current line highlight color."""
        return self.colors.editor_current_line

    def get_plot_background(self) -> str:
        """Get the plot background color."""
        return self.colors.plot_bg

    def get_plot_colors(self) -> dict:
        """Get colors for plot widgets."""
        colors = self.colors
        return {
            "background": colors.plot_bg,
            "foreground": colors.text,
            "grid": colors.plot_grid,
        }


# Global theme manager instance
theme_manager = ThemeManager()
