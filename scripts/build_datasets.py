"""Build/regenerate all open and closed datasets under assets/data.

This script is a thin wrapper around the dataset generators in
``tsfm4finance.core.datasets`` and the unified ``tsfm_eval`` data registry.
"""
from __future__ import annotations

from tsfm4finance.core.datasets import build_all


def main():
    build_all()


if __name__ == "__main__":
    main()
