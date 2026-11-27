"""`--device cpu` has to mean the model runs on CPU, not that the row says so.

The trux re-run of the 12-series overlap asked for CPU and TimesFM-2.5 died with
`CUDA out of memory: GPU 0 has ... 14.94 MiB is free`: the vendor loader
(`TimesFM_2p5_200M_torch.from_pretrained`) has no device argument and takes the GPU when
torch can see one. The lane's meta said `device: cpu` the whole time. A benchmark row that
names a device it did not use is worse than a row that fails.
"""
from __future__ import annotations

import pytest

from tsfm_trl.errors import UnavailableError
from tsfm_trl.model.adapters.timesfm import TimesFM25, _TimesFMBase2


class _Params:
    def __init__(self, dev):
        self.dev = dev

    def parameters(self):
        return [object()] if self.dev == "empty" else [_Tensor(self.dev)]


class _Dev:
    """Enough of a torch.device for the adapter: it reads `.type`."""

    def __init__(self, kind):
        self.type = kind

    def __repr__(self):
        return self.type


class _Tensor:
    def __init__(self, dev):
        self.device = _Dev(dev)


class _Model:
    def __init__(self, dev, movable=True):
        self.dev, self.movable = dev, movable

    def parameters(self):
        return iter([_Tensor(self.dev)])     # torch hands back a generator, not a list

    def to(self, dev):
        if not self.movable:
            raise RuntimeError("vendor loader keeps a CUDA handle")
        self.dev = dev


class _Wrapper:
    def __init__(self, model):
        self.model = model


def _stub(dev_requested, dev_actual, movable=True):
    obj = TimesFM25.__new__(TimesFM25)
    obj.device = dev_requested
    obj._wrapper = _Wrapper(_Model(dev_actual, movable))
    return obj


def test_weights_loaded_on_the_gpu_are_moved_when_cpu_was_asked():
    obj = _stub("cpu", "cuda")
    _TimesFMBase2._pin_device(obj)
    assert obj._wrapper.model.dev == "cpu"


def test_a_gpu_request_is_left_alone():
    obj = _stub("cuda", "cuda")
    _TimesFMBase2._pin_device(obj)
    assert obj._wrapper.model.dev == "cuda"


def test_a_model_that_refuses_to_move_is_a_failure_not_a_mislabelled_row():
    obj = _stub("cpu", "cuda", movable=False)
    with pytest.raises(UnavailableError) as e:
        _TimesFMBase2._pin_device(obj)
    assert "refusing to record a row" in str(e.value)


def test_a_model_that_moves_to_the_wrong_place_still_fails():
    class _Stuck(_Model):
        def to(self, dev):            # accepts the call, changes nothing
            self.dev = "meta"
    obj = TimesFM25.__new__(TimesFM25)
    obj.device = "cpu"
    obj._wrapper = _Wrapper(_Stuck("cuda"))
    with pytest.raises(UnavailableError) as e:
        _TimesFMBase2._pin_device(obj)
    assert "still on" in str(e.value)
