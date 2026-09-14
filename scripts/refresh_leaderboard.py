"""Refresh the leaderboard by re-running official benchmark evaluations.

This script regenerates the leaderboard cache stored at
``assets/leaderboard/leaderboard.json``. It is intended to be run manually or
from a scheduled GitHub Actions workflow.
"""
from __future__ import annotations

from tsfm4finance.core.seed_leaderboard import main

if __name__ == "__main__":
    main()
