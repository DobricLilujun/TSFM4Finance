"""The registry is the contract: every model row must be complete, and every
adapter class must be importable *without* weights (so ``list``/``info``/``doctor``
work in a bare environment)."""
from __future__ import annotations

import importlib

import pytest

from tsfm_trl.errors import UnknownModelError
from tsfm_trl.model import ModelHub
from tsfm_trl.model.base import CAPS


def test_registry_has_the_zoo():
    rows = ModelHub.list()
    assert len(rows) >= 35, f"only {len(rows)} models registered"
    families = {s.family for s in rows}
    for fam in ("timesfm25", "chronos2", "chronos_bolt", "chronos_t5", "moirai", "moment",
                "toto2", "tirex", "timemoe", "sundial", "kronos", "ttm", "flowstate"):
        assert fam in families, f"family {fam} missing ({families})"


def test_every_row_is_documented():
    for s in ModelHub.list():
        assert s.key and s.family and s.env, s
        assert s.license, f"{s.key} has no license row"
        assert s.tier in ("A", "B", "C"), f"{s.key} tier {s.tier}"
        assert s.caps, f"{s.key} declares no capabilities"
        assert set(s.caps) <= CAPS, f"{s.key} uses unknown capability {set(s.caps) - CAPS}"
        assert s.trainable or "finetune" not in s.caps, \
            f"{s.key} advertises finetune but is not trainable"
        if s.weights and "/" in s.weights:
            assert s.weights[0].isalpha() or s.weights[0].isdigit(), s.weights


def test_loss_kinds_are_the_declared_vocabulary():
    bad = [s.key for s in ModelHub.list() if (s.loss or "none") not in
           ("quantile", "token", "vendor", "point", "none")]
    assert not bad, bad


def test_adapter_classes_import_without_weights():
    for s in ModelHub.list():
        mod, _, cls = s.adapter.partition(":")
        m = importlib.import_module(f"tsfm_trl.model.adapters.{mod}")
        assert hasattr(m, cls), f"{s.key}: tsfm_trl/model/adapters/{mod}.py has no {cls}"


def test_lookup_by_hf_id_and_path():
    assert ModelHub.lookup("google/timesfm-2.5-200m-pytorch").key == "timesfm-2.5-200m"
    assert ModelHub.lookup("/models/Salesforce/moirai-1.1-R-small").key == "moirai-1.1-R-small"
    with pytest.raises(UnknownModelError) as e:
        ModelHub.load("timesfm-4")
    assert "timesfm-2.5" in str(e.value) or "timesfm" in str(e.value)


def test_filters():
    assert all(s.tier == "A" for s in ModelHub.list(tier="A"))
    assert all(s.family in ("chronos2", "chronos_bolt", "chronos_t5")
               for s in ModelHub.list(family="chronos*"))
    # "classification" means *model-based* classification (embeddings or a native head);
    # every model can still be probed with statistical features.
    clf = ModelHub.list(task="classification")
    assert len(clf) >= 3, clf
    assert all(s.can("embed") or s.can("clf_head") for s in clf)
