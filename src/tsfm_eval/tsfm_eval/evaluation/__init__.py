"""Evaluation dispatcher and benchmark runner."""
from __future__ import annotations

from tsfm_eval.evaluation.evaluator import evaluate
from tsfm_eval.evaluation.benchmark import run_benchmark

__all__ = ["evaluate", "run_benchmark"]
