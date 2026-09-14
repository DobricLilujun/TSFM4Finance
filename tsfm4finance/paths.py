"""Central path resolution for tsfm4finance.

All user inputs and generated artifacts live under assets/ at the project root.
This module gives every other module a single source of truth for those paths.
"""
from __future__ import annotations
from pathlib import Path

# package dir  -> .../tsfm4finance
PACKAGE_DIR = Path(__file__).resolve().parent
# project root -> .../TSFM4Finance
ROOT = PACKAGE_DIR.parent

# All datasets (open, closed, raw, bis) live under assets/data.
DATA = ROOT / "assets" / "data"
# Leaderboard and evaluation outputs are also under assets/.
LEADERBOARD = ROOT / "assets" / "leaderboard"
OUTPUTS = ROOT / "assets" / "outputs"

REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
