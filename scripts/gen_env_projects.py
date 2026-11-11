#!/usr/bin/env python3
"""Generate one uv project per model family (``envs/<family>/pyproject.toml``).

Pins are the versions the current benchmark actually runs with (read from the
working venvs with ``uv pip list``), so a fresh ``uv sync`` reproduces a known-good
environment instead of resolving to something that breaks on a different day.
Re-run after installing/upgrading a family to freeze the new state:

    python scripts/gen_env_projects.py            # regenerate envs/*/pyproject.toml
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# family -> (venv that works today, python, [packages to pin], notes)
FAMILIES: dict[str, tuple[str, str, list[str], str]] = {
    "core":    (".venv2", "3.11", ["torch==2.11.0", "transformers==5.22.1", "timesfm==3.0.2",
               "chronos-forecasting==2.3.2", "granite-tsfm==0.3.12", "datasets==2.17.1",
               "numpy==2.3.5", "pandas==2.3.3", "pyarrow==26.0.1", "statsmodels==0.15.0",
               "pmdarima==2.1.1", "scikit-learn==1.7.2", "scipy==1.17.1", "pytest>=8"],
               "sweep driver + TimesFM 3.0 + Chronos + baselines"),
    "legacy":  (".venv", "3.11", ["torch==2.13.0", "transformers==5.15.0", "timesfm==2.0.2",
               "chronos-forecasting==2.3.1", "numpy==2.4.6", "pandas==3.0.5", "pyarrow==25.0.1",
               "statsmodels==0.14.6", "pmdarima==2.1.1", "scikit-learn==1.9.0", "scipy==1.17.1"],
               "TimesFM 2.5-200M: the env that produced the published §4.4 numbers"),
    "moirai":  (".venv-moirai", "3.11", ["torch==2.4.1", "triton==3.0.0", "uni2ts==2.0.0",
               "gluonts==0.14.4", "numpy==1.26.4", "pandas==2.1.4", "pyarrow==16.1.0",
               "datasets==2.17.1", "einops==0.7.0", "torchmetrics==1.9.0", "scipy==1.11.4",
               "huggingface-hub<1.0"], "uni2ts pins torch<2.5"),
    "moment":  (".venv-moment", "3.11", ["torch==2.3.1", "triton==2.3.1", "momentfm==0.1.4",
               "transformers==4.33.3", "numpy>=1.25,<2", "pandas>=2.1,<3", "pyarrow>=15",
               "datasets>=2.16,<3", "scipy>=1.11"],
               "momentfm hard-pins huggingface-hub==0.24.0, so this env must not pin hub "
               "or the numpy/pandas versions momentfm pulls (pinning hub==0.36.2 makes the "
               "environment unsatisfiable); hub must stay <1.0 for its remote code"),
    "ttm":     (".venv-ttm", "3.12", ["torch==2.11.0", "transformers==5.22.1",
               "granite-tsfm==0.3.12", "datasets==5.1.0", "numpy==2.5.4", "pandas==3.1.1",
               "pyarrow==26.0.1", "einops==0.8.2", "statsmodels==0.15.0"],
               "Granite TTM r2 + FlowState r1 (tsfm_public)"),
    "toto":    (".venv-toto", "3.12", ["torch==2.15.0", "toto-2==2.0.0", "gluonts==0.17.0",
               "einops==0.8.2", "numpy==2.5.4", "pandas==2.3.3", "pyarrow==26.0.1",
               "datasets==5.1.0", "torchmetrics==1.9.0"], "toto-2 requires python>=3.12"),
    "tirex":   (".venv-tirex", "3.11", ["torch==2.15.0", "tirex-ts==1.4.2", "datasets==5.1.0",
               "numpy==2.4.6", "pandas==3.1.1", "pyarrow==26.0.1"], "TiRex v1"),
    "tirex2":  (".venv-tirex2", "3.12", ["torch==2.7.1", "triton==3.3.1", "tirex-2==0.4.0",
               "flashrnn==1.0.8", "mlstm-kernels==2.0.6", "ninja", "einops==0.8.2",
               "numpy==2.3.5", "pandas==2.3.3", "pyarrow==26.0.1", "datasets==5.1.0"],
               "IMPORTANT: mlstm_kernels/flashrnn use tl.make_block_ptr, removed in Triton "
               "3.9 - keep triton<3.6 (local torch 2.15/triton 3.9 cannot run them)"),
    "kronos":  (".venv-kronos", "3.11", ["torch==2.5.1", "triton==3.1.0", "einops==0.8.2",
               "numpy==2.3.5", "pandas==3.1.1", "pyarrow==26.0.1", "datasets==5.1.0"],
               "Kronos K-line model; also needs third_party/Kronos for its trainer"),
    "timemoe": (".venv-timemoe", "3.11", ["torch==2.5.1", "triton==3.1.0",
               "transformers==4.44.2", "huggingface-hub==0.36.2", "datasets==5.0.1",
               "numpy==2.4.6", "pandas==3.1.1", "pyarrow==26.0.1"],
               "Time-MoE remote code needs transformers==4.44 (fails on 4.46+/5.x)"),
}


COMMON = ["pydantic>=2.7", "pyyaml>=6", "requests>=2.31", "tqdm>=4.66", "pytest>=8"]


def installed(venv: str) -> dict[str, str]:
    out = subprocess.run(["uv", "pip", "list", "--python", f"{venv}/bin/python"],
                         capture_output=True, text=True).stdout
    d = {}
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2:
            d[parts[0].lower().replace("-", "_")] = parts[1]
    return d


def main() -> None:
    for fam, (venv, py, pins, note) in FAMILIES.items():
        # the watson cluster ships 3.12/3.13 and forbids uv python downloads
        # (python-downloads = "never"); 3.13 is excluded because the older families'
        # torch/triton pins have no cp313 wheels, so UV_PYTHON=3.12 picks 3.12.
        lo = py.split(".")[1]
        d = REPO / "envs" / fam
        d.mkdir(parents=True, exist_ok=True)
        deps = "\n".join(f'    "{p}",' for p in pins + COMMON)
        (d / "pyproject.toml").write_text(f'''[project]
name = "tsfm4finance-{fam}"
version = "0.1.0"
description = "Arena benchmark environment for the {fam} model family. {note}."
requires-python = ">=3.{lo},<3.13"
dependencies = [
{deps}
]

[tool.uv]
# library-style env: uv must not try to build/install this project itself
package = false
''')
        print(f"envs/{fam}/pyproject.toml  ({len(pins)} pins, python {py})")
    print("\ncreate them with:  for p in envs/*/; do uv sync --project $p; done")
    print("config.yaml envs then point at envs/<fam>/.venv/bin/python")


if __name__ == "__main__":
    main()
