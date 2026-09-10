"""Central path resolution for tsfm4finance.

The data directory lives at the project root (one level above this package),
shared by the core dataset builder and the backend service.
"""
from __future__ import annotations
from pathlib import Path

# package dir  -> .../tsfm4finance
PACKAGE_DIR = Path(__file__).resolve().parent
# project root -> .../TSFM4Finance
ROOT = PACKAGE_DIR.parent
DATA = ROOT / "data"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"
