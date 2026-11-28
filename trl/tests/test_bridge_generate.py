"""The bridge generator has to inline *everything*, and prove it.

Two bridge jobs died the same way for different reasons: job_012 because a guard inside a generated
file compared against the placeholder the generator substitutes (so the check was `[ x != x ]` and
passed on an ungenerated file), job_014 because TRUX/TRUX_SSH/SRC were read from a bridge.env that
exists on trux and nowhere on the worker. Both are the same rule - one file crosses the bridge - and
both should fail in a test rather than on a far machine at 03:00.
"""
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
GEN = REPO / "probes" / "trux" / "bridge" / "gpu7_generate.sh"

TEMPLATE = """#!/usr/bin/env bash
set -euo pipefail
MARKER='@@'
TRUX=${TRUX:-"@@TRUX@@"}
SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}
case " $SERIES_LIST " in *"$MARKER"*) echo "FATAL: marker left"; exit 6 ;; esac
echo "TRUX=$TRUX"
echo "SERIES=$SERIES_LIST"
"""


def _bridge(tmp_path, **vals):
    env = tmp_path / "bridge.env"
    env.write_text("".join(f'{k}="{v}"\n' for k, v in vals.items()))
    return env


def _run(tmp_path, tpl, series_file):
    # the script reads bridge.env from its own directory, so the test ships a copy
    script = tmp_path / "gpu7_generate.sh"
    script.write_text(GEN.read_text())
    return subprocess.run(["bash", str(script), str(tpl), str(series_file),
                           str(tmp_path / "job.sh")], capture_output=True, text=True)


def test_generated_job_carries_the_series_and_the_endpoints(tmp_path):
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE)
    ser = tmp_path / "series.txt"
    ser.write_text("std_equity_1d_10107\nstd_fx_1d_eurjpy\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX="trux.example", TRUX_SSH="ssh -p 22 -i /k",
            GPU_SSH="ssh -o BatchMode=yes", SRC="/data/standard")
    r = _run(tmp_path, tpl, ser)
    assert r.returncode == 0, r.stderr
    job = (tmp_path / "job.sh").read_text()
    assert "@@" not in job.replace("MARKER='@@'", ""), job
    assert "std_equity_1d_10107 std_fx_1d_eurjpy" in job
    assert 'TRUX=${TRUX:-"trux.example"}' in job
    # and the guard inside the generated file still refuses a file whose default was restored by
    # hand: it looks for the marker, not for one particular value, which is why it survives
    # whatever the substitution produced
    import re
    bad = re.sub(r'SERIES_LIST=\$\{SERIES_LIST:-"[^"]*"\}',
                 'SERIES_LIST=${SERIES_LIST:-"@@SERIES@@"}', job)
    (tmp_path / "job2.sh").write_text(bad)
    r2 = subprocess.run(["bash", str(tmp_path / "job2.sh")], capture_output=True, text=True)
    assert r2.returncode == 6 and "marker" in r2.stdout, r2


def test_a_missing_endpoint_is_refused_before_dispatch(tmp_path):
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE)
    ser = tmp_path / "series.txt"
    ser.write_text("s1\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX_SSH="ssh", SRC="/d")   # TRUX absent
    r = _run(tmp_path, tpl, ser)
    assert r.returncode == 6 and "TRUX" in r.stderr, r.stderr


def test_a_token_the_generator_does_not_know_fails_the_run(tmp_path):
    """An unknown placeholder must not be shipped as a literal."""
    tpl = tmp_path / "tpl.sh"
    tpl.write_text(TEMPLATE.replace('TRUX=${TRUX:-"@@TRUX@@"}',
                                    'OUT=${OUT:-"@@OUT_DIR@@/lane.jsonl"}'))
    ser = tmp_path / "series.txt"
    ser.write_text("s1\n")
    _bridge(tmp_path, BRIDGE="/b", TRUX="t", TRUX_SSH="ssh", GPU_SSH="ssh", SRC="/d")
    r = _run(tmp_path, tpl, ser)
    assert r.returncode != 0 and "OUT_DIR" in r.stderr, r.stderr


def test_the_shipped_templates_generate_clean(tmp_path):
    """Every gpu7_*.sh template must survive the generator with no token left, including GPU_SSH,
    which is the credential the *worker* uses to reach trux (not the one trux uses to reach it)."""
    bridge = tmp_path / "bridge.env"
    bridge.write_text('BRIDGE="/b"\nTRUX="user@trux.example"\n'
                      'TRUX_SSH="ssh -i /on/trux/id_bridge"\n'
                      'GPU_SSH="ssh -o BatchMode=yes"\nSRC="/data/standard"\n')
    script = tmp_path / "gpu7_generate.sh"
    script.write_text(GEN.read_text())
    ser = tmp_path / "series.txt"
    ser.write_text("s1\ns2\n")
    for tpl in sorted(t for t in (REPO / "probes" / "trux" / "bridge").glob("gpu7_*.sh")
                      if t.name != "gpu7_generate.sh"):  # the generator's own comment carries a
                                                         # example token, by design
        r = subprocess.run(["bash", str(script), str(tpl), str(ser), str(tmp_path / "job.sh")],
                           capture_output=True, text=True)
        assert r.returncode == 0, f"{tpl.name}: {r.stderr}"
        job = (tmp_path / "job.sh").read_text()
        import re
        assert not re.findall(r"@@[A-Z_]+@@", job), f"{tpl.name}: token left in the generated job"
        if "@@GPU_SSH@@" in tpl.read_text():
            # the newer jobs pull the corpus with the worker's own credentials; that value must
            # arrive filled, because no bridge.env exists on the worker
            assert 'ssh -o BatchMode=yes' in job, f"{tpl.name}: worker-side ssh not inlined"


def test_dispatch_refuses_a_file_that_was_never_generated(tmp_path):
    """The generator aborting and the old output surviving in /tmp is how a superseded template got
    dispatched once (job_021). Dispatch reads the stamp the generator writes."""
    disp = REPO / "probes" / "trux" / "bridge" / "trux_dispatch.sh"
    bridge = tmp_path / "bridge"
    (bridge / "inbox").mkdir(parents=True)
    (bridge / "outbox").mkdir()
    env = tmp_path / "disp.sh"
    env.write_text(f'BRIDGE="{bridge}"\n')
    stale = tmp_path / "job_old.sh"
    stale.write_text("#!/usr/bin/env bash\necho hi\n")
    r = subprocess.run(["bash", str(disp), str(stale)], capture_output=True, text=True,
                       env={**os.environ, "BRIDGE": str(bridge)})
    assert r.returncode == 7 and "no GENERATED stamp" in r.stderr, r.stderr


def test_dispatch_refuses_a_stale_generated_file(tmp_path):
    disp = REPO / "probes" / "trux" / "bridge" / "trux_dispatch.sh"
    bridge = tmp_path / "bridge"
    (bridge / "inbox").mkdir(parents=True)
    (bridge / "outbox").mkdir()
    old = tmp_path / "job_old.sh"
    old.write_text("#!/usr/bin/env bash\n"
                   "# GENERATED 2020-01-01T00:00:00Z by gpu7_generate.sh from x.sh\n"
                   "echo hi\n")
    r = subprocess.run(["bash", str(disp), str(old)], capture_output=True, text=True,
                       env={**os.environ, "BRIDGE": str(bridge)})
    assert r.returncode == 7 and "generated" in r.stderr, r.stderr


def test_the_python_inside_the_templates_compiles(tmp_path):
    """bash -n cannot see an embedded heredoc. One template shipped an f-string that nested double
    quotes inside a double-quoted f-string: valid on 3.12, a SyntaxError on the 3.11 interpreter the
    worker actually runs, and it only exploded in the summary after a two-hour sweep had finished."""
    import ast
    import re
    for tpl in sorted(t for t in (REPO / "probes" / "trux" / "bridge").glob("gpu7_*.sh")
                      if t.name != "gpu7_generate.sh"):
        text = tpl.read_text()
        # any quoted heredoc, not just one named PY: a template may name its blocks
        blocks = re.findall(r"<<'([A-Z_]+)'\n(.*?)\n\1\n", text, flags=re.S)
        for i, (marker, block) in enumerate(blocks):
            try:
                ast.parse(block)
            except SyntaxError as e:
                raise AssertionError(f"{tpl.name} python block {i} ({marker}): {e}") from e


def _pick_device_fn(text: str) -> str:
    """Lift the device chooser out of a template so it can be run for real."""
    import re
    m = re.search(r"^(free_mib\(\) \{.*?^pick_device\(\) \{.*?^\}$\n?)", text, flags=re.S | re.M)
    assert m, "no free_mib/pick_device pair found"
    return m.group(1)


def _run_chooser(tmp_path, free_mib_output: str, floor: str = "4000", wait: str = "0",
                 retry: str = "1") -> tuple[str, str]:
    tpl = (REPO / "probes" / "trux" / "bridge" / "gpu7_060_windows.sh").read_text()
    stub = tmp_path / "envs" / "core" / "bin"
    stub.mkdir(parents=True, exist_ok=True)
    (stub / "python").write_text(f"#!{__import__('sys').executable}\nprint({free_mib_output!r})\n")
    (stub / "python").chmod(0o755)
    script = tmp_path / "chooser.sh"
    script.write_text(f"ENVROOT={tmp_path}/envs GPU_FLOOR_MIB={floor} GPU_WAIT_SECS={wait} "
                      f"GPU_RETRY_SECS={retry}\n" + _pick_device_fn(tpl) + "\npick_device\n")
    p = __import__("subprocess").run(["bash", str(script)], capture_output=True, text=True, timeout=30)
    return p.stdout.strip(), p.stderr


def test_the_device_chooser_returns_only_a_device_name(tmp_path):
    """`DEV=$(pick_device)` captures stdout, so a progress line printed by the chooser becomes
    the device string and the lane runs with --device '  waiting for the GPU: 1305 MiB free...'.
    The function's stdout must be exactly cuda or cpu; its narration belongs on stderr."""
    out, _ = _run_chooser(tmp_path, "99999")
    assert out == "cuda"
    out, err = _run_chooser(tmp_path, "0", floor="4000", wait="2", retry="1")
    assert out == "cpu", f"the chooser leaked progress into its return value: {out!r}"
    assert "waiting for the GPU" in err                 # narration belongs on stderr
    out, _ = _run_chooser(tmp_path, "4300", floor="4000")
    assert out == "cuda"   # the floor is a wait, not a verdict


def test_a_queued_bridge_job_can_be_held_back_before_the_worker_claims_it(tmp_path):
    """job_023 was dispatched, then the memory audit said 1.1 GiB free: a CPU chronos lane would
    have taken most of a day. Until the Spark worker moves a file into inbox_done it is still
    trux's to take back, so --hold exists to stage a generated job without releasing it."""
    dispatch = REPO / "probes" / "trux" / "bridge" / "trux_dispatch.sh"
    bridge = tmp_path / "spark_bridge"
    for d in ("inbox", "inbox_done", "outbox"):
        (bridge / d).mkdir(parents=True)
    job = tmp_path / "job.sh"
    import datetime
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    job.write_text(f"#!/usr/bin/env bash\n# GENERATED {stamp} by gpu7_generate.sh from t.sh\necho hi\n")
    p = __import__("subprocess").run(
        ["bash", str(dispatch), "--hold", str(job)],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "BRIDGE": str(bridge)})
    assert p.returncode == 0, p.stderr[-400:]
    held = list((bridge / "hold").glob("*.sh"))
    assert held, "nothing staged in hold/"
    assert not list((bridge / "inbox").glob("job_*.sh")), "the job reached the inbox anyway"
    # releasing it is the same command without --hold, on the held file
    p2 = __import__("subprocess").run(
        ["bash", str(dispatch), str(held[0])],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "BRIDGE": str(bridge)})
    assert p2.returncode == 0, p2.stderr[-400:]
    assert list((bridge / "inbox").glob("job_*.sh")), "the held job never reached the inbox"


def test_the_bridge_example_assigns_values_the_shell_will_not_execute(tmp_path):
    """`TRUX_SSH=ssh -i $HOME/.ssh/key -o ...` is not a value with spaces, it is an assignment of
    "ssh" followed by the shell trying to run `-i /home/.../key` as a command. On trux that
    truncated the endpoint, the generator died under set -e, and its stale /tmp output got
    dispatched. So every multi-word value in the example must be quoted."""
    ex = (REPO / "probes" / "trux" / "bridge" / "bridge.env.example").read_text()
    bad = []
    for ln in ex.splitlines():
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        name, _, val = ln.partition("=")
        if not name.replace("_", "").isalpha():
            continue
        if (" " in val.strip() or "$" in val) and not val.strip().startswith(('"', "'")):
            bad.append(name)
    assert not bad, f"unquoted multi-word values: {bad} - source would run their tail as a command"
    # and the file must actually source cleanly in a clean environment
    p = __import__("subprocess").run(
        ["bash", "-c", f"set -u; . {REPO / 'probes' / 'trux' / 'bridge' / 'bridge.env.example'}; "
                       'echo BRIDGE_SSH=[$BRIDGE_SSH]'],
        capture_output=True, text=True, timeout=30)
    assert p.returncode == 0, p.stderr[-300:]
    assert "command not found" not in p.stderr
    assert p.stdout.strip().startswith("BRIDGE_SSH=[ssh ")
