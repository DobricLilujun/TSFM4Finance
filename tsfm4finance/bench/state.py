"""Resumable cell store for the multi-model benchmark.

Design contract
---------------
* One row per *cell* = the smallest unit of work that can crash on its own:
  ``(stage, series, model, adaptation, horizon, span, seed)``.
* A cell is written by ``begin()`` -> ``finish()`` / ``fail()`` / ``skip()``.
  Every write is its own transaction, so ``kill -9`` at any moment leaves the
  store consistent and every completed cell still counted.
* The heavy results blob lives in a side file per cell
  (``reports/cells/<cell_id>.json``) written atomically (tmp + rename) and
  referenced from the row.  Nothing rewrites a whole-run JSON per row.
* ``load_results()`` (the legacy shape used by report/paper generators) is a
  *derived export*, so the store can be replayed after an interrupt.

Concurrency: WAL mode + busy timeout, plus shard ownership (``--shard i/N``) so
several machines can work on disjoint cells of the same grid file.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1

_SCHEMA = f"""
PRAGMA user_version = {SCHEMA_VERSION};
CREATE TABLE IF NOT EXISTS cells (
    cell_id     TEXT PRIMARY KEY,
    stage       TEXT NOT NULL,
    series      TEXT NOT NULL,
    model       TEXT NOT NULL,
    adaptation  TEXT NOT NULL,
    horizon     INTEGER,
    span        TEXT,
    seed        INTEGER,
    status      TEXT NOT NULL,              -- pending running done failed skipped
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    reason      TEXT,                       -- skip reason / degradation note
    seconds     REAL,
    vram_mb     REAL,
    profile     TEXT,                       -- device/amp/ctx/batch actually used
    config_hash TEXT,
    payload     TEXT,                       -- small scalar summary (json)
    artifact    TEXT,                       -- path to full per-cell json
    updated_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS cells_stage  ON cells (stage);
CREATE INDEX IF NOT EXISTS cells_status ON cells (status);
CREATE INDEX IF NOT EXISTS cells_model  ON cells (model);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at REAL NOT NULL);
"""

TERMINAL = ("done", "failed", "skipped")


def cell_id(spec: dict) -> str:
    """Stable id for a cell spec. Keep the key order explicit: ids are the
    resumability contract, they must not depend on dict insertion order."""
    key = "|".join(
        f"{k}={spec.get(k)}"
        for k in ("stage", "series", "model", "adaptation", "horizon", "span", "seed")
    )
    return hashlib.sha1(key.encode()).hexdigest()[:16]


class CellStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.cells_dir = self.path.parent / "cells"
        self.cells_dir.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(str(self.path), timeout=60, check_same_thread=False)
        self.con.execute("PRAGMA journal_mode=WAL")
        self.con.execute("PRAGMA synchronous=FULL")
        self.con.execute("PRAGMA busy_timeout=60000")
        self.con.executescript(_SCHEMA)
        self.con.commit()

    # ------------------------------------------------------------------ helpers
    def _touch(self, cid: str, **kw):
        sets = ", ".join(f"{k}=?" for k in kw)
        vals = list(kw.values()) + [time.time(), cid]
        self.con.execute(f"UPDATE cells SET {sets}, updated_at=? WHERE cell_id=?", vals)
        self.con.commit()

    def put_meta(self, key: str, value) -> None:
        self.con.execute(
            "INSERT INTO meta(key,value,updated_at) VALUES(?,?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
            (key, json.dumps(value), time.time()),
        )
        self.con.commit()

    def get_meta(self, key: str, default=None):
        r = self.con.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return json.loads(r[0]) if r else default

    # ------------------------------------------------------------------ registry
    def plan(self, specs: list[dict], *, force: bool = False) -> tuple[list[dict], int, int]:
        """Insert planned cells. Returns (todo_specs, n_done, n_other_terminal)."""
        now = time.time()
        todo, ndone, nother = [], 0, 0
        for spec in specs:
            cid = cell_id(spec)
            self.con.execute(
                "INSERT INTO cells(cell_id,stage,series,model,adaptation,horizon,span,seed,status,"
                "config_hash,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?) "
                "ON CONFLICT(cell_id) DO NOTHING",
                (
                    cid, spec["stage"], str(spec.get("series")), spec["model"],
                    str(spec.get("adaptation")), spec.get("horizon"), spec.get("span"),
                    spec.get("seed"), "pending", spec.get("config_hash"), now,
                ),
            )
            row = self.con.execute(
                "SELECT status FROM cells WHERE cell_id=?", (cid,)
            ).fetchone()
            st = row[0]
            if force and st in TERMINAL:
                self.con.execute(
                    "UPDATE cells SET status='pending', error=NULL, reason=NULL, attempts=0, "
                    "updated_at=? WHERE cell_id=?", (now, cid),
                )
                st = "pending"
            if st in TERMINAL:
                ndone += st == "done"
                nother += st != "done"
            else:
                todo.append({**spec, "cell_id": cid})
        self.con.commit()
        return todo, ndone, nother

    def claim(self, cid: str) -> bool:
        """Atomically move pending -> running. False if another worker owns it."""
        cur = self.con.execute(
            "UPDATE cells SET status='running', attempts=attempts+1, updated_at=? "
            "WHERE cell_id=? AND status IN ('pending','running')",
            (time.time(), cid),
        )
        self.con.commit()
        return cur.rowcount == 1

    # ------------------------------------------------------------------ results
    def finish(self, cid: str, payload: dict, *, profile: str | None = None,
               seconds: float | None = None, vram_mb: float | None = None) -> str:
        art = self.cells_dir / f"{cid}.json"
        tmp = art.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, default=str))
        os.replace(tmp, art)
        small = {k: v for k, v in payload.items() if isinstance(v, (int, float, str, bool))}
        self._touch(cid, status="done", error=None, payload=json.dumps(small, default=str),
                    artifact=str(art), profile=profile, seconds=seconds, vram_mb=vram_mb)
        return str(art)

    def fail(self, cid: str, error: str, *, profile: str | None = None,
             seconds: float | None = None) -> None:
        self._touch(cid, status="failed", error=str(error)[:2000], profile=profile,
                    seconds=seconds)

    def skip(self, cid: str, reason: str) -> None:
        self._touch(cid, status="skipped", reason=str(reason)[:1000])

    def note(self, cid: str, reason: str) -> None:
        self._touch(cid, reason=str(reason)[:1000])

    def payload(self, cid: str) -> dict | None:
        row = self.con.execute(
            "SELECT artifact FROM cells WHERE cell_id=?", (cid,)
        ).fetchone()
        if not row or not row[0] or not Path(row[0]).exists():
            return None
        try:
            return json.loads(Path(row[0]).read_text())
        except Exception:
            return None

    # ------------------------------------------------------------------ queries
    def rows(self, stage: str | None = None, status: str | None = None) -> list[dict]:
        q, a = "SELECT * FROM cells", []
        cond = []
        if stage:
            cond.append("stage=?"); a.append(stage)
        if status:
            cond.append("status=?"); a.append(status)
        if cond:
            q += " WHERE " + " AND ".join(cond)
        q += " ORDER BY stage, series, model, horizon"
        cur = self.con.execute(q, a)
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]

    def counts(self) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for stage, status, n in self.con.execute(
            "SELECT stage,status,COUNT(*) FROM cells GROUP BY stage,status"
        ):
            out.setdefault(stage, {})[status] = n
        return out

    def model_counts(self, stage: str) -> dict[str, dict[str, int]]:
        out: dict[str, dict[str, int]] = {}
        for model, status, n in self.con.execute(
            "SELECT model,status,COUNT(*) FROM cells WHERE stage=? GROUP BY model,status",
            (stage,),
        ):
            out.setdefault(model, {})[status] = n
        return out

    def reset_running(self) -> int:
        """A process died mid-cell: make those cells claimable again."""
        cur = self.con.execute(
            "UPDATE cells SET status='pending', updated_at=? WHERE status='running'",
            (time.time(),),
        )
        self.con.commit()
        return cur.rowcount

    def retry_failed(self, stages: tuple[str, ...] | None = None) -> int:
        q = "UPDATE cells SET status='pending', updated_at=? WHERE status='failed'"
        a: list = [time.time()]
        if stages:
            q += " AND stage IN (" + ",".join("?" * len(stages)) + ")"
            a += list(stages)
        cur = self.con.execute(q, a)
        self.con.commit()
        return cur.rowcount

    # ------------------------------------------------------------------ export
    def export(self) -> dict[str, list[dict]]:
        """Legacy results shape: {stage: [payload, ...]} for done cells."""
        out: dict[str, list[dict]] = {}
        for r in self.rows(status="done"):
            p = self.payload(r["cell_id"])
            if p is not None:
                out.setdefault(r["stage"], []).append(p)
        return out


def human_progress(store: CellStore) -> str:
    """ASCII progress matrix: stage x status, plus per-model breakdown."""
    lines: list[str] = []
    for stage, c in sorted(store.counts().items()):
        done, run = c.get("done", 0), c.get("running", 0)
        fail, skip = c.get("failed", 0), c.get("skipped", 0)
        pend = c.get("pending", 0)
        tot = sum(c.values())
        bar = f"[{'#' * done}{'>' * run}{'.x~'[:1] * (fail + skip)}{'.' * pend}]"
        lines.append(
            f"{stage:<10} {done:>4}/{tot:<5} done={done} running={run} "
            f"failed={fail} skipped={skip} pending={pend} {bar}"
        )
    return "\n".join(lines)
