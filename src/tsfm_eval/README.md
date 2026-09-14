# tsfm_eval

Unified evaluation library for time-series foundation models on financial tasks.

This package provides:

- **Standard data interface**: `TSDataset`, `load_dataset()`, `list_datasets()`
- **Task types**: `forecast`, `classify`, `anomaly`
- **Metrics**: RMSE, MAE, MASE, sMAPE, directional accuracy, rank correlation,
  accuracy, F1, AUC-ROC, etc.
- **Model adapter**: `BaseModel` with a single `predict()` contract
- **Evaluation**: `evaluate(pred, truth_df, meta)` and `run_benchmark()`

## Quick start

```python
import tsfm_eval

# List all public datasets under ./assets/data
print(tsfm_eval.list_datasets(open_only=True))

# Load one dataset
ds = tsfm_eval.load_dataset("sp500_daily")
print(ds.meta)

# Iterate rolling windows
for context, truth in ds.iter_rolling("test"):
    ...

# Evaluate a prediction
from tsfm_eval.schemas import PredictOutput
metrics = tsfm_eval.evaluate(
    PredictOutput(point=[101.0, 102.0]),
    truth,
    ds.meta,
)
print(metrics)
```

## Configuration

By default `tsfm_eval` looks for datasets in `./assets/data`. Override with:

```python
from tsfm_eval.config import set_data_root
set_data_root("/path/to/assets/data")
```

Or via the environment variable `TSFM_EVAL_DATA_ROOT`.
