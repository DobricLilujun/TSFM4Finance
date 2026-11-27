"""tsfm-trl — TRL for time-series foundation models.

One API over every open TSFM: load it, forecast with it, fine-tune it (forecasting
and classification), and post-train it against a reward.

    from tsfm_trl import ModelHub, ForecastTrainer, TrainingConfig
    from tsfm_trl.data import SeriesDataset

    m  = ModelHub.load("timesfm-2.5", device="auto")
    pr = m.predict([context], horizon=21)          # point + 0.1..0.9 quantiles
    cfg = TrainingConfig(ctx=512, horizon=21, budget="fs16")
    ft = ForecastTrainer(m, cfg).fit({"windows": w, "info": info})

Every capability question ("can this model be fine-tuned? does it emit quantiles?")
is answered by the registry + adapters with a *reason*, never a silent skip.
"""


def _usable_hf_modules_cache() -> None:
    """Point HF's dynamic-module cache somewhere writable, if the default is not.

    ``trust_remote_code`` models (Chronos-Bolt variants, Sundial, VisionTS, Time-MoE,
    Granite-TTM heads) write python into ``~/.cache/huggingface/modules``. On a box where
    somebody once ran a pipeline as root, that directory is root-owned and *every* such
    model fails with ``PermissionError`` — which a compatibility matrix would file as a
    model failure. Redirecting the cache is the whole fix, so it happens at import,
    with the reason printed once on stderr (never silently).
    """
    import os as _os
    import sys as _sys
    if _os.environ.get("HF_MODULES_CACHE"):
        return
    default = _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "modules")
    if not _os.path.isdir(default) or _os.access(default, _os.W_OK):
        return
    alt = _os.path.join(_os.environ.get("TSFMTRL_HOME") or _os.getcwd(),
                        ".cache", "hf_modules")
    _os.makedirs(alt, exist_ok=True)
    _os.environ["HF_MODULES_CACHE"] = alt
    print(f"tsfm-trl: {default} is not writable (root-owned?); "
          f"using HF_MODULES_CACHE={alt}", file=_sys.stderr)


_usable_hf_modules_cache()

from .errors import CapabilityError, TSFMTRLError, UnknownModelError, UnavailableError
from .model import ModelHub
from .model.base import LEVELS, Pred, TSFMModel
from .training.checkpoints import load_finetuned
from .training.classify import ClassificationTrainer
from .training.config import BUDGETS, TrainResult, TrainingConfig
from .training.forecast import ForecastTrainer
from .training.reward import RewardSpec, RewardTrainer

__version__ = "0.1.0"
__all__ = ["ModelHub", "TSFMModel", "Pred", "LEVELS", "load_finetuned", "TrainingConfig",
           "TrainResult", "BUDGETS", "ForecastTrainer", "ClassificationTrainer", "RewardTrainer",
           "RewardSpec", "TSFMTRLError", "UnknownModelError", "CapabilityError",
           "UnavailableError", "__version__"]
