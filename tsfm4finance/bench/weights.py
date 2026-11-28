"""Model-weight resolution: Hugging Face id **or** a local directory.

Nothing in the benchmark ever *requires* network access to Hugging Face.  Every
model is addressed by a string in ``bench/config.yaml`` (``weights:``), and this
module decides what that string means:

  1. a path (``/abs``, ``./rel``, ``~/home``) -> used as-is;
  2. a directory under ``$TSFM_WEIGHTS_ROOT`` (default ``<repo>/models``), tried as
     ``<root>/<org>/<repo>``, ``<root>/<org>__<repo>`` and ``<root>/<repo>`` -> that
     directory is used (this is the ``git clone`` / ``hf download`` layout);
  3. otherwise the string is passed through as a Hub id (online or the local Hub
     cache).

``TSFM_OFFLINE=1`` (or ``HF_HUB_OFFLINE=1``) additionally sets the offline flags so
libraries never try to reach the Hub - which is what you want on an air-gapped
GPU box where the weights were copied in beforehand.

CLI:
  python -m tsfm4finance.bench.weights --list-clone   # git clone / hf download lines
  python -m tsfm4finance.bench.weights --verify       # what is present locally
  python -m tsfm4finance.bench.weights --resolve amazon/chronos-2
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# Extra non-Hub sources that must exist locally (git repos we vendor ourselves).
VENDORED = {
    "kronos": "https://github.com/NeoQuasar/Kronos",       # -> third_party/Kronos
}

_required_offline = False


def weights_root() -> Path:
    return Path(os.environ.get("TSFM_WEIGHTS_ROOT") or (REPO / "models")).expanduser()


def set_offline(flag: bool = True) -> None:
    """Force libraries to stay off the network (they still use the local cache)."""
    global _required_offline
    _required_offline = bool(flag)
    if flag:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def is_offline() -> bool:
    return _required_offline or os.environ.get("TSFM_OFFLINE", "") in ("1", "true", "yes") \
        or os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true")


def looks_like_path(name: str) -> bool:
    return name.startswith(("/", "./", "../", "~/")) or Path(name).exists()


def local_candidates(name: str) -> list[Path]:
    root = weights_root()
    tail = name.rstrip("/").split("/")
    return [root / name, root / "__".join(tail), root / tail[-1]]


def resolve(name: str | None) -> str | None:
    """Return the path/id that should be handed to ``from_pretrained``."""
    if not name:
        return name
    name = str(Path(name).expanduser())
    if looks_like_path(name):
        p = Path(name).expanduser()
        if p.exists():
            return str(p)
    # Prefer the offline HF cache when it holds the repo: passing the *Hub id* keeps
    # strict loaders (uni2ts, tsfm_public, tirex) happy, because huggingface_hub
    # resolves the id locally under HF_HUB_OFFLINE.  Build that mirror with
    #   python -m tsfm4finance.bench.weights --link-into-cache .hf-cache
    # and export HF_HOME=$PWD/.hf-cache.
    if is_offline() and _in_hf_cache(name):
        set_offline(True)
        return name
    for cand in local_candidates(name):
        if cand.exists() and _looks_like_model_dir(cand):
            return str(cand)
    if cand_dir_exists_but_incomplete(name):
        # fail loudly: an offline box with a half-cloned repo otherwise produces a
        # confusing "model not found on the Hub" error
        raise FileNotFoundError(
            f"{name}: found a partial directory under {weights_root()} but it is missing "
            f"config/weights - re-clone it (see docs/08_offline_models.md)")
    if is_offline():
        set_offline(True)
    return name


def _hub_dirs() -> list[Path]:
    out = []
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        v = os.environ.get(var)
        if v:
            out += [Path(v) / "hub", Path(v)]
    out.append(Path(os.path.expanduser("~/.cache/huggingface/hub")))
    return [d for d in out if d.exists()]


def _in_hf_cache(name: str) -> bool:
    if "/" not in name or looks_like_path(name):
        return False
    needle = "models--" + name.replace("/", "--")
    return any(d.is_dir() and (d / needle).is_dir() for d in _hub_dirs())


def cand_dir_exists_but_incomplete(name: str) -> bool:
    root = weights_root()
    tail = name.rstrip("/").split("/")
    for cand in (root / name, root / "__".join(tail), root / tail[-1]):
        if cand.is_dir() and not _looks_like_model_dir(cand) and any(cand.iterdir()):
            return True
    return False


def _looks_like_model_dir(p: Path) -> bool:
    """A usable checkpoint directory: config and/or weights, at any depth
    (some repos keep weights in sub-folders, e.g. TiRex)."""
    if not p.is_dir():
        return False
    top = {f.name for f in p.iterdir()}
    if "config.json" in top or any(n.endswith((".safetensors", ".bin", ".pt", ".ckpt")) for n in top):
        return True
    for f in p.rglob("*"):
        if f.is_file() and f.suffix in (".safetensors", ".bin", ".pt", ".ckpt", ".msgpack"):
            return True
    return False


def local_kwargs(name: str | None) -> dict:
    """Kwargs for ``from_pretrained``: offline-safe, harmless for local paths."""
    return {"local_files_only": True} if is_offline() and not (name and looks_like_path(name)) \
        else {}


# Set the offline flags as early as possible: some libraries read them at import.
if os.environ.get("TSFM_OFFLINE", "") in ("1", "true", "yes"):
    set_offline(True)


# ------------------------------------------------------------------- inventory
def model_inventory() -> list[dict]:
    from .config import load_grid
    out = []
    g = load_grid()
    for key, m in sorted(g.models().items()):
        out.append({"key": key, "family": m.family, "env": m.env, "tier": m.tier,
                    "weights": m.weights or "(none: statistical baseline)"})
    return out


def git_urls(weights: str) -> list[str]:
    """Clone URLs for a Hub id (huggingface.co serves git+LFS)."""
    if "/" not in weights or looks_like_path(weights):
        return []
    org, repo = weights.split("/", 1)
    return [f"https://huggingface.co/{org}/{repo}"]


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list-clone", action="store_true",
                    help="print git clone / hf download commands for every registered model")
    ap.add_argument("--verify", action="store_true", help="check what is present locally")
    ap.add_argument("--resolve", default="", help="show how one id resolves right now")
    ap.add_argument("--root", default="", help="weights root (also $TSFM_WEIGHTS_ROOT)")
    ap.add_argument("--offline", action="store_true", help="export-style hint: force offline")
    ap.add_argument("--link-into-cache", default="",
                    help="build an HF-cache-shaped tree of symlinks from $TSFM_WEIGHTS_ROOT "
                         "(needed by libraries that insist on Hub APIs, e.g. TiRex); "
                         "value = target dir, e.g. .hf-cache")
    ap.add_argument("--from-cache", action="store_true",
                    help="materialise the local Hugging Face cache into $TSFM_WEIGHTS_ROOT "
                         "(run this on the machine that HAS network, then rsync models/ over)")
    a = ap.parse_args()

    if a.root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.root
    if a.offline:
        set_offline(True)

    if a.resolve:
        print(f"{a.resolve} -> {resolve(a.resolve)}")
        return

    inv = model_inventory()
    if a.list_clone:
        root = weights_root()
        print(f"# weights root: {root}   (export TSFM_WEIGHTS_ROOT to change it)")
        print(f"# then run: export TSFM_WEIGHTS_ROOT=$PWD/{root.name} TSFM_OFFLINE=1\n")
        print("git lfs install")
        for m in inv:
            w = m["weights"]
            for url in git_urls(w):
                tgt = w if "/" in w else w
                print(f"git clone {url} {root}/{w}")
        for fam, url in VENDORED.items():
            print(f"git clone {url} {REPO}/third_party/{fam.title()}")
        print("\n# or, for Hub repos, without git-lfs:")
        for m in inv:
            if "/" in m["weights"] and not looks_like_path(m["weights"]):
                print(f"huggingface-cli download {m['weights']} "
                      f"--local-dir {weights_root()}/{m['weights']}")
        return

    if a.link_into_cache:
        import hashlib
        target = Path(a.link_into_cache).expanduser()
        if not target.is_absolute():
            target = REPO / target
        hub = target / "hub"
        root = weights_root()
        n = 0
        for cfg in sorted(root.rglob("config.json")) + sorted(root.rglob("*.ckpt")):
            d = cfg.parent
            rel = d.relative_to(root)
            if rel.parts[0].startswith("models--"):
                continue
            repo_id = "/".join(rel.parts)
            marker = hub / f"models--{repo_id.replace('/', '--')}"
            snap = hashlib.sha1(str(rel).encode()).hexdigest()
            (marker / "snapshots" / snap).mkdir(parents=True, exist_ok=True)
            (marker / "refs").mkdir(parents=True, exist_ok=True)
            (marker / "refs" / "main").write_text(snap)
            dest = marker / "snapshots" / snap
            for f in d.rglob("*"):
                if f.is_file():
                    link = dest / f.relative_to(d)
                    link.parent.mkdir(parents=True, exist_ok=True)
                    if link.is_symlink() or link.exists():
                        link.unlink()
                    link.symlink_to(f.resolve())
            n += 1
            print(f"  linked {repo_id}")
        print(f"\n{len(list(hub.glob('models--*')))} repos linked into {target}")
        print(f"export HF_HOME={target} TSFM_OFFLINE=1")
        return

    if a.from_cache:
        import shutil
        hub = Path(os.path.expanduser("~/.cache/huggingface/hub"))
        root = weights_root()
        if not hub.exists():
            print(f"no HF cache at {hub}"); sys.exit(1)
        n = 0
        for d in hub.glob("models--*"):
            rel = d.name.replace("models--", "").replace("--", "/")
            snaps = sorted((d / "snapshots").glob("*")) if (d / "snapshots").exists() else []
            if not snaps:
                continue
            tgt = root / rel
            if (tgt / "config.json").exists() or any(tgt.glob("*.safetensors")):
                print(f"  exists: {rel}")
                continue
            tgt.mkdir(parents=True, exist_ok=True)
            for f in snaps[-1].rglob("*"):
                if f.is_file():
                    dst = tgt / f.relative_to(snaps[-1])
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.copy2(f, dst)
                    except OSError:
                        shutil.copyfile(f, dst)
            n += 1
            print(f"  wrote {rel} -> {tgt}")
        print(f"{n} models materialised under {root}")
        return

    if a.verify:
        root = weights_root()
        print(f"weights root: {root}   offline: {is_offline()}")
        ok = miss = 0
        for m in inv:
            w = m["weights"]
            if "/" not in w or w.startswith("("):
                print(f"  {m['key']:<22} baseline (no weights)")
                continue
            found = next((c for c in local_candidates(w) if _looks_like_model_dir(c)), None)
            if found:
                ok += 1
                print(f"  {m['key']:<22} LOCAL  {found}")
            elif Path(os.path.expanduser("~/.cache/huggingface/hub")).exists() and \
                    any((Path(os.path.expanduser("~/.cache/huggingface/hub"))).glob(f"models--{w.replace('/', '--')}*")):
                print(f"  {m['key']:<22} CACHED in ~/.cache/huggingface (works offline)")
                ok += 1
            else:
                miss += 1
                print(f"  {m['key']:<22} MISSING -> git clone https://huggingface.co/{w} "
                      f"{root}/{w}")
        print(f"\n{ok} available, {miss} missing")
        sys.exit(1 if miss else 0)

    print(__doc__)
    print("registered models:")
    for m in inv:
        print(f"  {m['key']:<22} tier={m['tier']} env={m['env']:<7} {m['weights']}")


if __name__ == "__main__":
    main()
