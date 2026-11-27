"""Data loading, windowing and the label discipline. The property that matters most
here is that a fine-tune can never be handed a window containing the evaluation
span — that is what the ``split_end`` provenance is for."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tsfm_trl.data.classify import direction_dataset, extreme_dataset, volreg_dataset
from tsfm_trl.data.fixtures import tiny_finance, tiny_series, tiny_windows
from tsfm_trl.data.series import SeriesDataset, train_windows, windows


def test_series_from_array_and_csv(tmp_path):
    ds = SeriesDataset.load(np.arange(200.0))
    assert len(ds) == 200 and ds.splits["train"][1] == 140
    csv = tmp_path / "px.csv"
    df = pd.DataFrame({"timestamp": pd.date_range("2020-01-01", periods=120, freq="B"),
                       "close": np.cumsum(np.random.RandomState(0).normal(size=120)) + 100})
    df.to_csv(csv, index=False)
    ds2 = SeriesDataset.load(csv)
    assert ds2.target == "close" and isinstance(ds2.index, pd.DatetimeIndex)
    assert ds2.split("test")[-1] == df["close"].iloc[-1]


def test_a_missing_path_names_the_path_not_the_type():
    """The old message blamed the type (`cannot read a series from str`) for a mistyped path,
    which is the most common way to call this wrong in a sweep; it cost two detours through
    the CLI and an adapter before checking whether the file existed."""
    from tsfm_trl.errors import DataError
    with pytest.raises(DataError) as e:
        SeriesDataset.load("reports/pass3_trux/csv/definitely_not_there.csv")
    msg = str(e.value)
    assert "definitely_not_there.csv" in msg, msg
    assert "from str" not in msg, msg


def test_windows_are_ordered_and_disjoint_from_truth():
    y = np.arange(300.0)
    w = windows(y, horizon=10, ctx=50, stride=10)
    assert w and all(len(c) == 50 and len(t) == 10 for c, t in w)
    assert np.array_equal(w[0][1], y[50:60]) and np.array_equal(w[1][1], y[60:70])
    assert w[-1][1][-1] == y[-1]


def test_max_windows_sample_is_reproducible_and_time_ordered():
    y = np.arange(1000.0)
    a = windows(y, 10, 100, stride=1, max_windows=12, seed=7)
    b = windows(y, 10, 100, stride=1, max_windows=12, seed=7)
    assert [w[1][0] for w in a] == [w[1][0] for w in b]
    assert [w[1][0] for w in a] == sorted(w[1][0] for w in a)
    c = windows(y, 10, 100, stride=1, max_windows=12, seed=8)
    assert [w[1][0] for w in a] != [w[1][0] for w in c]


def test_train_windows_never_cross_the_split_end():
    """The windows a fine-tune sees must be reproducible from the *train slice only*."""
    ds = tiny_series(600)
    end = ds.splits["train"][1]
    w, info = train_windows(ds, horizon=21, ctx=256, max_windows=8)
    ref = windows(ds.values[:end], horizon=21, ctx=256, max_windows=8, seed=0)
    assert [tuple(np.round(x[1], 4)) for x in w] == [tuple(np.round(x[1], 4)) for x in ref]
    assert info["split_end_index"] == end and info["n_windows"] == len(w)
    assert w[-1][1][-1] <= ds.values[end - 1] + 1e-9 or np.isclose(w[-1][1][-1], ds.values[end - 1])
    # a window built from the *full* series reaches the test span (stride=1)
    full = windows(ds.values, horizon=21, ctx=256, stride=1)
    assert np.isclose(full[-1][1][-1], ds.values[-1])


def test_windows_refuse_to_run_on_too_little_data():
    ds = tiny_series(100)
    with pytest.raises(Exception) as e:
        train_windows(ds, horizon=64, ctx=256)
    assert "no train windows" in str(e.value)


def test_direction_labels_are_forward_looking():
    ds = tiny_finance(600)
    cds = direction_dataset(ds, horizon=5, ctx=100)
    assert cds.n_classes == 3                       # down / flat / up
    d = cds.describe()
    assert 0.0 < d["base_rate"] < 1.0 and d["train"] > 0 and d["test"] > 0
    # all three regimes occur in a regime-switching fixture
    cnt = np.bincount(cds.y, minlength=3)
    assert cnt.min() > 0, cnt


def test_extreme_and_volreg_labels_are_balanced_enough():
    ds = tiny_finance(900)
    ex = extreme_dataset(ds, horizon=5, ctx=100, k=1.5)
    assert ex.n_classes >= 2 and min(np.bincount(ex.y, minlength=ex.n_classes)) > 0
    vr = volreg_dataset(ds, horizon=21, ctx=100, n_bins=3)
    assert vr.n_classes == 3


def test_fixture_panel_has_ohlcv_for_kline_models():
    ds = tiny_finance()
    assert ds.panel is not None
    for c in ("open", "high", "low", "volume"):
        assert c in ds.panel.columns
    win = ds.panel_window(len(ds) - 60, len(ds) - 20)
    assert len(win) == 40


def test_tiny_windows_helper():
    w = tiny_windows(n=4, ctx=64, horizon=16)
    assert 0 < len(w) <= 4 and all(len(c) == 64 and len(t) == 16 for c, t in w)


def test_dir5_labels_have_five_levels_and_match_the_3_level_windows():
    """The ``*_dir5`` task is the same windows with ±2·band tails split out, so it must
    have 5 non-empty classes and exactly the same window count as the 3-level task.

    (On a random-walk fixture the *strong* classes dominate rather than the flat one:
    a 5-step move is ~sqrt(5)× the 1-step scale, so it clears 2·0.25·sd often. Nothing
    here may assume a class distribution.)
    """
    ds = tiny_finance(900)
    c5 = direction_dataset(ds, horizon=5, ctx=100, levels=5)
    c3 = direction_dataset(ds, horizon=5, ctx=100, levels=3)
    assert c5.n_classes == 5 and c3.n_classes == 3
    cnt = np.bincount(c5.y, minlength=5)
    assert (cnt > 0).all(), cnt
    assert c5.describe()["base_rate"] < 0.6
    assert len(c5.X) == len(c3.X)
    with pytest.raises(Exception):
        direction_dataset(ds, horizon=5, ctx=100, levels=4)


def test_classification_dataset_limit_keeps_the_split_and_the_container():
    """The matrix caps classification windows so every family is measured at the same
    scale — and the trainers index X with boolean masks, so the cap must keep X an
    object array and the train/test split proportional."""
    import numpy as np
    from tsfm_trl.data.classify import direction_dataset
    from tsfm_trl.data.fixtures import tiny_finance
    cds = direction_dataset(tiny_finance(900), horizon=5, ctx=60)
    n0 = cds.describe()["n_windows"]
    sub = cds.limit(64)
    assert len(sub.X) <= 64 < n0, (len(sub.X), n0)
    assert isinstance(sub.X, np.ndarray), type(sub.X)
    tr0 = float((sub.split == 0).mean())
    assert 0.4 < tr0 < 0.9, sub.describe()          # splits stay proportional
    assert sub.describe()["subsampled_from"] == n0
    assert cds.limit(10_000) is cds                   # a cap above the size is a no-op
