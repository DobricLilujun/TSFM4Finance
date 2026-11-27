"""Regression: a row must be able to say that the head it scored was never trained.

Measured on trux (probes/trux/README.md), not assumed: the released MOMENT-1 checkpoint
contains exactly one head tensor — `head.linear` (8, d_model), the reconstruction head.
`from_pretrained(task_name="forecasting")` loads that head unchanged, and `init()` — the
call the model card tells you to make — replaces it with a (forecast_horizon, d_model*64)
tensor that is randomly initialised. transformers still reports "0 missing keys" because
the new head has different *shapes*. So a MOMENT "zero-shot forecast" measures a random
linear map, and the compatibility matrix cannot see it: the call returns fine, and the
benchmark rows come out at MASE 7–19 where a naive forecast scores 2–8.

The fix is not to hide the row (the benchmark keeps what ran) but to let the row state
what it measured, and make the merged report surface that state.
"""
import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def _sweep():
    spec = importlib.util.spec_from_file_location(
        "sweep_evaluate", ROOT / "scripts" / "sweep_evaluate.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _real_row():
    """A row from a committed sweep JSONL, so this test breaks when the schema changes
    instead of quietly testing a fixture that stopped looking like production."""
    src = ROOT / "reports" / "pass2_merged" / "lane_cuda-a.jsonl"
    for line in src.read_text().splitlines():
        rec = json.loads(line)
        if rec.get("rows"):
            return dict(rec["rows"][0])
    raise AssertionError(f"no scored rows in {src}")


def test_forecasting_head_is_not_in_the_checkpoint():
    from tsfm_trl.model.adapters.moment import head_weights_in_checkpoint as f
    ckpt = {"head.linear.weight": [8, 512], "head.linear.bias": [8]}   # reconstruction
    forecasting = {"head.linear.weight": [21, 32768], "head.linear.bias": [21]}
    assert f(ckpt, forecasting) is False, "a (21,32768) head cannot come from a (8,512) ckpt"


def test_reconstruction_head_is_in_the_checkpoint():
    from tsfm_trl.model.adapters.moment import head_weights_in_checkpoint as f
    ckpt = {"head.linear.weight": [8, 512], "head.linear.bias": [8]}
    assert f(ckpt, dict(ckpt)) is True


def test_a_bigger_model_does_not_rescue_an_untrained_head():
    """MOMENT-1-large: d_model grows to 1024, the checkpoint still has one 8-row head."""
    from tsfm_trl.model.adapters.moment import head_weights_in_checkpoint as f
    assert f({"head.linear.weight": [8, 1024]}, {"head.linear.weight": [21, 65536]}) is False


def test_no_head_tensors_means_nothing_was_loaded():
    from tsfm_trl.model.adapters.moment import head_weights_in_checkpoint as f
    assert f({"head.linear.weight": [8, 512]}, {}) is False


def test_checkpoint_reader_reports_unknown_as_unknown(tmp_path):
    """A checkpoint that cannot be inspected must not be guessed at."""
    from tsfm_trl.model.adapters.moment import _checkpoint_heads
    assert _checkpoint_heads(tmp_path / "does-not-exist") is None


def test_checkpoint_reader_reads_only_head_shapes(tmp_path):
    from safetensors.torch import save_file
    import torch
    save_file({"head.linear.weight": torch.zeros(8, 512),
               "model.encoder.blocks.0.attn.q.weight": torch.zeros(4, 4)},
              str(tmp_path / "model.safetensors"))
    from tsfm_trl.model.adapters.moment import _checkpoint_heads
    assert _checkpoint_heads(tmp_path) == {"head.linear.weight": [8, 512]}


def test_pred_meta_reaches_the_row():
    """The plumbing contract: evaluate copies scalar Pred.meta into the row."""
    from tsfm_trl.eval.evaluate import _scalar_meta
    got = _scalar_meta({"heads": ["linear"], "head_from_checkpoint": False,
                        "head_trained_here": False, "junk": np.zeros(3)})
    assert got == {"head_from_checkpoint": False, "head_trained_here": False}, got


def test_pipeline_key_helper_survives_the_adapter_constructor():
    """Regression for a real breakage: the zero-shot sweep lost every MOMENT row as
    `TypeError: 'NoneType' object is not callable`. The adapter's `__init__` set a dead
    `self._key = None` (leftover from an older single-pipeline cache), and the method that
    names a pipeline was also called `_key` — the attribute shadowed the method, and only
    a call at inference time revealed it. Constructing the adapter needs no momentfm, so
    this guard runs in the core environment.
    """
    from tsfm_trl.model.adapters.moment import Moment
    m = Moment()
    assert callable(m._pipe_key), "_pipe_key shadowed by an instance attribute again"
    k = m._pipe_key("forecasting", 21, 2, [np.zeros(256)])
    assert isinstance(k, tuple) and len(k) == 4 and k[0] == "forecasting"
    # the trained/untrained label is only sound if prediction and training name the same
    # pipeline, so the helper must be deterministic for identical inputs
    assert m._pipe_key("forecasting", 21, 2, [np.zeros(256)]) == k


def test_a_different_context_is_a_different_pipeline():
    """MOMENT's head width is fixed by seq_len, so a 256-point fine-tune does not train
    the head a 512-point prediction will use — the label has to reflect that."""
    from tsfm_trl.model.adapters.moment import Moment
    m = Moment()
    k256 = m._pipe_key("forecasting", 21, 2, [np.zeros(256)])
    k512 = m._pipe_key("forecasting", 21, 2, [np.zeros(512)])
    assert k256 != k512, "different context lengths must not share a cached head"


def test_report_surfaces_untrained_head_rows(tmp_path):
    """meta must survive evaluate -> JSONL -> merged markdown, or the label is decorative."""
    sw = _sweep()
    row = _real_row()
    row["meta"] = {"head_from_checkpoint": False, "head_trained_here": False}
    rec = {"model": "moment-1-small", "env": "moment", "tier": "A", "family": "moment",
           "error": "", "interpreter": "/x/moment/bin/python", "rows": [row]}
    f = tmp_path / "lane.jsonl"
    f.write_text(json.dumps({"meta": {"started": "x", "machine": "t", "code_sha": "abc",
                                      "series": [row["series"]], "horizons": [row["horizon"]],
                                      "spans": "all", "max_windows": 8, "ctx": 256,
                                      "batch": 16, "seed": 0, "device": "cuda"}})
                 + "\n" + json.dumps(rec) + "\n")
    out = tmp_path / "r.md"
    sw.report(argparse.Namespace(report=str(f), md=str(out), stdout=False))
    text = out.read_text()
    assert "Flagged: the scored head was never trained" in text
    assert "moment-1-small" in text.split("Flagged:")[1], "the model must be named there"
    # the row is kept in the table, not deleted: a labelled number beats a blank
    before = text.split("Flagged:")[0]
    assert "moment-1-small" in before, "the flagged row must still appear in the table"
    assert row["series"] in before


def test_report_does_not_flag_a_trained_head(tmp_path):
    sw = _sweep()
    row = _real_row()
    row["meta"] = {"head_from_checkpoint": True, "head_trained_here": False}
    rec = {"model": "moment-1-small", "env": "moment", "tier": "A", "family": "moment",
           "error": "", "interpreter": "/x/moment/bin/python", "rows": [row]}
    f = tmp_path / "lane.jsonl"
    f.write_text(json.dumps(rec) + "\n")
    out = tmp_path / "r.md"
    sw.report(argparse.Namespace(report=str(f), md=str(out), stdout=False))
    assert "Flagged:" not in out.read_text()
