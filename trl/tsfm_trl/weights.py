"""Where a model's weights come from: a Hugging Face id, a local directory, or
the offline Hub cache — decided in exactly one place.

A model string can be

1. a path (``/abs``, ``./rel``, ``~/x``) → used as-is;
2. anything else → looked up under ``$TSFM_WEIGHTS_ROOT`` (default ``<repo>/models``)
   as ``<root>/<org>/<repo>``, ``<root>/<org>__<repo>`` or ``<root>/<repo>`` (the
   ``git clone`` / ``hf download`` layout of an air-gapped GPU box);
3. otherwise passed through as a Hub id (online, or the local Hub cache when
   ``TSFM_OFFLINE=1`` / ``HF_HUB_OFFLINE=1``).

The offline branch prefers to pass the *Hub id* rather than a directory, because
several strict loaders (uni2ts, tsfm_public, tirex) only accept ids — pair it with
``python -m tsfm_trl.weights --link-into-cache .hf-cache`` and ``HF_HOME=...``.

CLI::

    python -m tsfm_trl.weights --verify          # what is present locally
    python -m tsfm_trl.weights --list-clone      # the git/hf download commands
    python -m tsfm_trl.weights --resolve amazon/chronos-2
"""
from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_offline = False


def weights_root() -> Path:
    return Path(os.environ.get("TSFM_WEIGHTS_ROOT") or (REPO / "models")).expanduser()


def set_offline(flag: bool = True) -> None:
    global _offline
    _offline = bool(flag)
    if flag:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


def is_offline() -> bool:
    return _offline or os.environ.get("TSFM_OFFLINE", "") in ("1", "true", "yes") \
        or os.environ.get("HF_HUB_OFFLINE", "") in ("1", "true")


def looks_like_path(name: str) -> bool:
    return name.startswith(("/", "./", "../", "~/")) or Path(name).exists()


def local_candidates(name: str) -> list[Path]:
    root = weights_root()
    tail = name.rstrip("/").split("/")
    return [root / name, root / "__".join(tail), root / tail[-1]]


def _looks_like_model_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    top = {f.name for f in p.iterdir()}
    if "config.json" in top or any(n.endswith((".safetensors", ".bin", ".pt", ".ckpt")) for n in top):
        return True
    return any(f.is_file() and f.suffix in (".safetensors", ".bin", ".pt", ".ckpt", ".msgpack")
               for f in p.rglob("*"))


def _hub_dirs() -> list[Path]:
    out: list[Path] = []
    for var in ("HF_HOME", "HUGGINGFACE_HUB_CACHE", "XDG_CACHE_HOME"):
        v = os.environ.get(var)
        if v:
            out += [Path(v) / "hub", Path(v)]
    out.append(Path(os.path.expanduser("~/.cache/huggingface/hub")))
    return [d for d in out if d.exists()]


def in_hf_cache(name: str) -> bool:
    if "/" not in name or looks_like_path(name):
        return False
    needle = "models--" + name.replace("/", "--")
    return any(d.is_dir() and (d / needle).is_dir() for d in _hub_dirs())


def present(name: str) -> bool:
    """Cheap answer to "can this model be loaded here at all?" for the matrix."""
    if not name or looks_like_path(name):
        return bool(name) and Path(str(name)).expanduser().exists()
    if in_hf_cache(name):
        return True
    return any(_looks_like_model_dir(c) for c in local_candidates(name))


def resolve(name: str | None) -> str | None:
    if not name:
        return name
    name = str(Path(str(name)).expanduser())
    if looks_like_path(name):
        p = Path(name).expanduser()
        if p.exists():
            return str(p)
    if is_offline() and in_hf_cache(name):
        set_offline(True)
        return name
    for cand in local_candidates(name):
        if _looks_like_model_dir(cand):
            return str(cand)
    # a half-copied directory is worse than a missing one: it produces "not found
    # on the Hub" errors that send you debugging the network
    for cand in local_candidates(name):
        if cand.is_dir() and any(cand.iterdir()):
            raise FileNotFoundError(
                f"{name}: found {cand} but it has no config/weights — re-download it "
                f"(see docs/01_install.md)")
    if is_offline():
        set_offline(True)
    return name


def local_kwargs(name: str | None) -> dict:
    """Kwargs for ``from_pretrained``: offline-safe and harmless for local paths."""
    return ({"local_files_only": True}
            if is_offline() and not (name and looks_like_path(name)) else {})


if os.environ.get("TSFM_OFFLINE", "") in ("1", "true", "yes"):
    set_offline(True)


# ------------------------------------------------------------------------ CLI
def main() -> None:                                   # pragma: no cover
    import argparse
    from .model.registry import Registry
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check the registry against local files")
    ap.add_argument("--list-clone", action="store_true", help="print download commands")
    ap.add_argument("--resolve", default="", help="resolve one id right now")
    ap.add_argument("--root", default="", help="weights root (also $TSFM_WEIGHTS_ROOT)")
    a = ap.parse_args()
    if a.root:
        os.environ["TSFM_WEIGHTS_ROOT"] = a.root
    if a.resolve:
        print(f"{a.resolve} -> {resolve(a.resolve)}")
        return
    reg = Registry()
    root = weights_root()
    if a.list_clone:
        print(f"# weights root: {root}   (export TSFM_WEIGHTS_ROOT to change it)")
        print("git lfs install")
        for spec in reg.models():
            if spec.weights and not looks_like_path(spec.weights):
                print(f"git clone https://huggingface.co/{spec.weights} {root}/{spec.weights}")
        return
    n_ok = n_miss = 0
    print(f"weights root: {root}   offline: {is_offline()}")
    for spec in reg.models():
        if not spec.weights:
            print(f"  {spec.key:<24} (no weights: {spec.family})")
            continue
        ok = present(spec.weights)
        n_ok += ok
        n_miss += not ok
        print(f"  {spec.key:<24} {'LOCAL/CACHED' if ok else 'MISSING':<13} {spec.weights}")
    print(f"\n{n_ok} available, {n_miss} missing")


if __name__ == "__main__":                             # pragma: no cover
    main()
