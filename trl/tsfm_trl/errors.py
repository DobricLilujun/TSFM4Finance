"""Error types. Every failure in this library carries a *machine-readable* reason,
because the whole point of ``tsfm-trl doctor`` is to produce a table of what works
and what does not — a bare traceback is not a result."""
from __future__ import annotations


class TSFMTRLError(Exception):
    """Base class."""


class UnknownModelError(TSFMTRLError):
    def __init__(self, key: str, known):
        hint = ""
        if isinstance(known, (list, tuple)) and known:
            hint = f"  known models: {', '.join(list(known)[:12])}{' ...' if len(known) > 12 else ''}"
        super().__init__(f"unknown model {key!r}.{hint}")
        self.key = key


class CapabilityError(NotImplementedError):
    """The model does not support the requested capability (embed, clf_head, ...)."""

    def __init__(self, key: str, cap: str, why: str = ""):
        super().__init__(f"{key}: no '{cap}' support" + (f" ({why})" if why else ""))
        self.key, self.cap, self.why = key, cap, why


class UnavailableError(RuntimeError):
    """The model cannot be loaded here: missing package, missing weights, or an
    incompatible toolchain. ``reason`` is short and fit for a report cell."""

    def __init__(self, key: str, reason: str, install: str = ""):
        super().__init__(f"{key}: {reason}" + (f"  [install: {install}]" if install else ""))
        self.key, self.reason, self.install = key, reason, install


class DataError(TSFMTRLError):
    """Bad or insufficient data for the requested operation (e.g. no train
    windows fit inside the train split)."""
