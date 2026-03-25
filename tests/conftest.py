"""
conftest.py
-----------
Pytest session-wide fixtures and import patches.

NASim optionally imports tkinter (for its GUI utilities). GitHub Actions
Ubuntu runners do not ship tkinter by default, which causes an ImportError
before any test code runs. We mock the affected modules here so NASim can
be imported cleanly in headless CI environments.
"""

import sys
from unittest.mock import MagicMock

# Mock tkinter and all submodules nasim probes before anything imports nasim
for _mod in (
    "tkinter",
    "tkinter.filedialog",
    "tkinter.font",
    "tkinter.messagebox",
    "tkinter.ttk",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()
