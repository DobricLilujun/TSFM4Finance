"""Bake a deploy-ready single-file index.html.

Inlines live leaderboard + dataset + model data AND the university logos (as
base64) into the frontend template. No external JS/CSS — GitHub Pages serves
one self-contained file.
"""
import base64
import json
from pathlib import Path

import tsfm_eval
from tsfm4finance.paths import LEADERBOARD
from tsfm4finance.core.models import available_models

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SRC = ROOT / "tsfm4finance" / "web" / "index.html"
LOGOS = ROOT / "tsfm4finance" / "web" / "static" / "logos"
LB = LEADERBOARD / "leaderboard.json"
OUT = ROOT / "assets" / "outputs" / "deploy" / "index.html"
LEGACY_OUT = HERE / ".deploy" / "index.html"


def _public_leaderboard(records: list[dict]) -> list[dict]:
    """Drop private/proprietary rows from the public board."""
    return [
        r for r in records
        if r.get("mode") != "private"
        and "placeholder" not in str(r.get("model", "")).lower()
    ]


def _dataset_bundle() -> list[dict]:
    """Return public dataset metadata for the frontend."""
    return [m.model_dump() for m in tsfm_eval.list_datasets(open_only=True)]


def _model_bundle() -> list[dict]:
    """Return built-in model metadata for the frontend."""
    out = []
    for name, info in available_models().items():
        entry = {"name": name}
        entry.update(info) if isinstance(info, dict) else entry.update({"info": info})
        out.append(entry)
    return out


def _inline_logos(html: str) -> str:
    for name in ("princeton", "luxembourg", "northwestern"):
        p = LOGOS / f"{name}.png"
        if not p.exists():
            p = LOGOS / f"{name}.svg"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            ext = "svg+xml" if p.suffix == ".svg" else "png"
            uri = f"data:image/{ext};base64,{b64}"
            html = html.replace(f"static/logos/{name}.png", uri)
            html = html.replace(f"static/logos/{name}.svg", uri)
            print(f"inlined {name}: {len(b64)} b64 chars")
        else:
            print(f"WARN: {name} logo not found")
    return html


def main():
    src = SRC.read_text(encoding="utf-8")
    lb = json.load(open(LB))
    lb = _public_leaderboard(lb)
    ds = _dataset_bundle()
    md = _model_bundle()

    payload = json.dumps({"leaderboard": lb, "datasets": ds, "models": md},
                         ensure_ascii=False)
    out = _inline_logos(src.replace("__BUNDLE__", payload))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding="utf-8")

    # Keep the legacy path working so existing scripts / docs still find output.
    LEGACY_OUT.parent.mkdir(parents=True, exist_ok=True)
    LEGACY_OUT.write_text(out, encoding="utf-8")

    print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")
    print(f"wrote {LEGACY_OUT} ({LEGACY_OUT.stat().st_size//1024} KB)")
    print(f"  leaderboard {len(lb)} | datasets {len(ds)} | models {len(md)}")


if __name__ == "__main__":
    main()
