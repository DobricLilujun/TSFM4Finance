# Development Guide

## Directory conventions

- **All data lives under `assets/`**. Never write inputs/outputs to a top-level `data/` directory.
  - `assets/data/open/` — public datasets.
  - `assets/data/closed/` — proprietary placeholder datasets.
  - `assets/leaderboard/leaderboard.json` — public leaderboard state.
  - `assets/outputs/deploy/index.html` — GitHub Pages bundle.

- **`src/tsfm_eval/`** is the standalone evaluation package. It must not import anything from `tsfm4finance`.

- **`tsfm4finance/`** is the web/arena layer. It re-exports contracts from `tsfm_eval` for backward compatibility and depends on `tsfm_eval`.

- **`scripts/`** contains build/deploy/maintenance scripts. `tests/` contains pytest tests.

## Adding a new dataset

1. Add a builder entry in `tsfm4finance/core/datasets.py` (or a new builder under `tsfm_eval/data/builders/`).
2. Run `uv run python scripts/build_datasets.py` to generate parquet + `meta.json`.
3. Verify it appears with `uv run python -c "import tsfm_eval; print(tsfm_eval.list_datasets())"`.
4. Add a smoke test in `tests/test_tsfm_eval/test_data.py` if needed.

## Adding a new model

1. Implement a subclass of `tsfm_eval.models.BaseModel` in `tsfm4finance/core/models/`.
2. Register it in the model factory so `available_models()` includes it.
3. Re-run the leaderboard seeder or use `/api/submit` to score it.

## Running tests

```bash
uv run pytest tests/ -q
```

## CI/CD

- `.github/workflows/ci.yml` runs tests and builds the deploy bundle on every push/PR.
- `.github/workflows/deploy_pages.yml` deploys `assets/outputs/deploy` to GitHub Pages on every push to `main`.
- `.github/workflows/refresh_data.yml` rebuilds datasets, refreshes the leaderboard and redeploys daily at 06:00 UTC.

## Publishing `tsfm_eval`

`src/tsfm_eval/pyproject.toml` is a normal setuptools package. To publish:

```bash
cd src/tsfm_eval
python -m build
python -m twine upload dist/*
```

Or keep it as a path dependency inside this monorepo.
