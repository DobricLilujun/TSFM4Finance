"""Bake a deploy-ready single-file index.html: inlines live leaderboard +
dataset + model data AND the university logos (as base64) into the frontend
template. No external JS/CSS — GitHub Pages serves one self-contained file.
"""
import base64
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "frontend" / "index.html"
LOGOS = HERE / "frontend" / "static" / "logos"
LB = HERE / "backend" / "leaderboard.json"
OUT = HERE / ".deploy" / "index.html"


def main():
    src = SRC.read_text(encoding="utf-8")
    lb = json.load(open(LB))
    ds = json.load(open(Path("/tmp/ds.json")))
    md = json.load(open(Path("/tmp/md.json")))

    payload = json.dumps({"leaderboard": lb, "datasets": ds, "models": md},
                         ensure_ascii=False)
    out = src.replace("__BUNDLE__", payload)

    # Inline the university logos as base64 data URIs so the single-file deploy
    # has no external asset dependencies.
    for name in ("princeton", "luxembourg", "northeastern"):
        p = LOGOS / f"{name}.png"
        if not p.exists():
            p = LOGOS / f"{name}.svg"
        if p.exists():
            b64 = base64.b64encode(p.read_bytes()).decode()
            ext = "svg+xml" if p.suffix == ".svg" else "png"
            uri = f"data:image/{ext};base64,{b64}"
            out = out.replace(f"static/logos/{name}.png", uri)
            out = out.replace(f"static/logos/{name}.svg", uri)
            print(f"inlined {name}: {len(b64)} b64 chars")
        else:
            print(f"WARN: {name} logo not found")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size//1024} KB)")
    print(f"  leaderboard {len(lb)} | datasets {len(ds)} | models {len(md)}")


if __name__ == "__main__":
    main()