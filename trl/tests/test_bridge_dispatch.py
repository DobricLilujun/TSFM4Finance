"""The bridge dispatcher must never hand the worker a job number that already has a log.

Why this is worth a test: the Spark/GPU7 worker's job-selection rule is "first inbox/*.sh
with no outbox/<name>.log". A reused number therefore does not fail - the job sits in the
inbox forever while a healthy worker reports a fresh heartbeat. The old numbering counted
*.done.sh files, which misses jobs still in the inbox and jobs whose file was already
reaped, and it produced exactly that silent stall."""
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

HERE = Path(__file__).resolve().parent.parent
SCRIPT = HERE / "probes" / "trux" / "bridge" / "trux_dispatch.sh"


def _bridge(tmp_path):
    b = tmp_path / "bridge"
    for d in ("inbox", "inbox_done", "outbox"):
        (b / d).mkdir(parents=True)
    return b


def _dispatch(b, tmp_path):
    job = tmp_path / "job.sh"
    import datetime as _dt
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    job.write_text("#!/usr/bin/env bash\n"
                   f"# GENERATED {stamp} by gpu7_generate.sh from fixture.sh\n"
                   "echo hi\n")
    r = subprocess.run(["bash", str(SCRIPT), str(job)],
                       env={"BRIDGE": str(b), "PATH": "/usr/bin:/bin"},
                       capture_output=True, text=True, timeout=30)
    return r


def test_the_number_continues_past_finished_and_in_flight_jobs(tmp_path):
    b = _bridge(tmp_path)
    (b / "inbox_done" / "job_001.sh.done.sh").write_text("# done\n")
    (b / "outbox" / "job_001.sh.log").write_text("ran\n")
    (b / "inbox" / "job_002.sh").write_text("# still queued\n")
    r = _dispatch(b, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "queued job_003.sh" in r.stdout, r.stdout
    assert (b / "inbox" / "job_003.sh").is_file()


def test_a_reaped_job_still_blocks_its_number(tmp_path):
    """Log exists, job file already reaped elsewhere: counting done files would say 1, and
    the worker would ignore the new job_001 forever."""
    b = _bridge(tmp_path)
    (b / "outbox" / "job_001.sh.log").write_text("ran\n")
    r = _dispatch(b, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "queued job_002.sh" in r.stdout, r.stdout


def test_an_empty_bridge_starts_at_one(tmp_path):
    b = _bridge(tmp_path)
    r = _dispatch(b, tmp_path)
    assert r.returncode == 0, r.stderr
    assert "queued job_001.sh" in r.stdout, r.stdout


def test_a_flag_before_the_job_path_is_not_mistaken_for_the_job(tmp_path):
    """`trux_dispatch.sh --hold job.sh` used to read `--hold` as the job path, then fail the
    GENERATED-stamp check on it: the error named a flag, and the real job was never queued."""
    dispatch = REPO / "probes" / "trux" / "bridge" / "trux_dispatch.sh"
    bridge = tmp_path / "spark_bridge"
    for d in ("inbox", "inbox_done", "outbox"):
        (bridge / d).mkdir(parents=True)
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    job = tmp_path / "job.sh"
    job.write_text(f"#!/usr/bin/env bash\n# GENERATED {stamp} by gpu7_generate.sh from t.sh\necho hi\n")
    env = {**__import__("os").environ, "BRIDGE": str(bridge)}
    p = subprocess.run(["bash", str(dispatch), "--hold", str(job)],
                       capture_output=True, text=True, timeout=30, env=env)
    assert p.returncode == 0, p.stderr[-300:]
    assert list((bridge / "hold").glob("*.sh"))
    bad = subprocess.run(["bash", str(dispatch), "--nope", str(job)],
                         capture_output=True, text=True, timeout=30, env=env)
    assert bad.returncode == 2 and "unknown flag" in bad.stderr
