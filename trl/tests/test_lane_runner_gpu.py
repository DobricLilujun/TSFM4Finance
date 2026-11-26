"""The lane scheduler must not read an unknown GPU memory as zero free memory.

The DGX Spark (GB10) answers `[N/A]` for memory.free/memory.total while CUDA works
normally. Before this, trux_lane_runner parsed that row away, `gpu_free()` returned an
empty pool, and every lane quietly fell back to CPU on a machine with a working GPU — the
same class of failure as the DGX family envs that silently lost CUDA, only noisier to
debug because the rows still get written.
"""

import importlib.util
import pathlib

import pytest

MODULE = pathlib.Path(__file__).resolve().parents[1] / "probes" / "trux" / "trux_lane_runner.py"


@pytest.fixture()
def runner():
    spec = importlib.util.spec_from_file_location("trux_lane_runner", MODULE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_na_rows_are_dropped_not_read_as_zero_free(runner):
    # this is the shape gpu_free() actually asks for: index, memory.free, memory.total
    out = (
        "0, [N/A], [N/A]\n"            # GB10: the card exists, its memory is unknown
        "1, 41000, 48000\n"            # a card nvidia-smi can size
        "2, 12000, 24000, 3 %\n"       # extra trailing column: still a usable row
    )
    rows = runner.parse_gpu_rows(out)
    assert [r["index"] for r in rows] == [1, 2], "an N/A card must not become '0 MiB free'"
    assert rows[0]["free_mib"] == 41000
    assert rows[0]["reported_by"] == "nvidia-smi"


def test_gpu_free_asks_torch_when_nvidia_smi_cannot_answer(runner, monkeypatch):
    # nvidia-smi present but every row N/A: the pool must come from torch, not be empty
    monkeypatch.setattr(runner.shutil, "which", lambda _exe: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(runner, "sh", lambda *_a, **_k: "0, [N/A], [N/A]")
    monkeypatch.setattr(
        runner, "gpu_free_torch",
        lambda: [{"index": 0, "free_mib": 120000, "total_mib": 130000, "reported_by": "torch"}],
    )
    rows = runner.gpu_free()
    assert rows and rows[0]["reported_by"] == "torch"
    assert rows[0]["free_mib"] > 0, "a GB10 lane must still be schedulable onto CUDA"


def test_gpu_free_reports_nothing_rather_than_guessing(runner, monkeypatch):
    # no nvidia-smi and no torch: [] means the scheduler will say "no usable card",
    # which is honest; inventing a pool would mislabel rows as CUDA
    monkeypatch.setattr(runner.shutil, "which", lambda _exe: None)
    monkeypatch.setattr(runner, "gpu_free_torch", lambda: [])
    assert runner.gpu_free() == []
