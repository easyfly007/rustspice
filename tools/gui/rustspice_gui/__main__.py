"""
Entry point for RustSpice GUI application.

Usage:
    python -m rustspice_gui
    rustspice-gui  # if installed via pip
    rustspice-gui --server http://localhost:3000
"""

import argparse
import sys

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont

from .main_window import MainWindow


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="RustSpice GUI - Circuit Simulator Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    rustspice-gui                           # Connect to default server
    rustspice-gui --server http://host:3000 # Connect to specific server
    rustspice-gui circuit.cir               # Open a netlist file
        """,
    )
    parser.add_argument(
        "--server",
        "-s",
        default="http://127.0.0.1:3000",
        help="sim-api server URL (default: http://127.0.0.1:3000)",
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Netlist file to open on startup",
    )
    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version="%(prog)s 0.1.0",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create Qt application
    app = QApplication(sys.argv)
    app.setApplicationName("RustSpice")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("RustSpice")

    # Set default font
    font = QFont("Consolas", 10)
    if not font.exactMatch():
        font = QFont("Monospace", 10)
    app.setFont(font)

    # Enable high DPI scaling
    app.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    # Create and show main window
    window = MainWindow(server_url=args.server)
    window.show()

    # Open file if specified
    if args.file:
        window.open_file(args.file)

    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
