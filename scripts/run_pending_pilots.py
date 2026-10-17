"""
Orchestrator: runs several scenario scripts back-to-back against a single
already-running CARLA server, so one "kick it off and walk away" command
covers more ground than starting each script by hand between classes.

Usage:
    1. Start CARLA first, as always:
       cd CARLA\\WindowsNoEditor
       CarlaUE4.exe -quality-level=Low -windowed -ResX=800 -ResY=600
    2. Then, in a second terminal:
       py -3.7 scripts\\run_pending_pilots.py

Each scenario script runs as its own subprocess — NOT imported and called
in-process — so it gets a clean CARLA client connection and goes through its
own synchronous-mode-enable / finally-restore-async cleanup exactly as if you
had run it by hand. This matters: if two scripts shared one Python process,
a crash in the first could leave synchronous mode stuck for the second, or
leak the vehicle/collision-sensor state between runs. Subprocesses avoid all
of that at the cost of nothing (CARLA client setup is cheap).

The full console output of every run is captured to logs/<script>_<time>.log
— not just errors, the entire run — while still streaming live to your
terminal so you can watch if you're at the machine. This is the same
motivation as the error-log fix added to the scenario scripts on 2026-07-09
(that's how the reason 26/400 Scenario C episodes failed got lost): terminal
scrollback is not a reliable place to keep information you'll need later.

If one script fails or crashes, the orchestrator records it and moves on to
the next one rather than stopping the whole batch — same philosophy as the
per-episode try/except inside each scenario script, one level up.

Edit PENDING_SCRIPTS below to control what runs and in what order. Nothing
here changes any scenario script's own PILOT_MODE / NUM_EPISODES config —
this just runs each one exactly as currently configured, back to back.
"""
import subprocess
import sys
import os
import datetime

PROJECT_ROOT = r'C:\Users\jayad\Desktop\RPI_AINN'
SCRIPTS_DIR  = os.path.join(PROJECT_ROOT, 'scripts')
LOGS_DIR     = os.path.join(PROJECT_ROOT, 'logs')

# Edit this list to control what runs, and in what order.
# As of 2026-07-09: Scenario D's pilot passed (160/160 safe, clean) and is
# done. Scenario B's pilot ran clean data (13/13 pairs is_junction_t=True)
# but with very low yield (13/80 possible) — added diagnostic counters
# (nav_timeouts vs yield_verify_fails) before touching any constants. Re-run
# just B to get that breakdown.
PENDING_SCRIPTS = [
    'scenario_b_intersection.py',  # PILOT_MODE=True -> 20 episodes, now with diagnostic counters
]

PYTHON_LAUNCHER = ['py', '-3.7']   # matches project convention (py -3.7 ...)


def run_script(script_name):
    """
    Run one scenario script as a subprocess, streaming its output live to
    this console while also writing the full transcript to a log file.
    Returns (success: bool, log_path: str, elapsed_seconds: float).
    """
    script_path = os.path.join(SCRIPTS_DIR, script_name)
    timestamp   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_name    = f"{script_name.replace('.py', '')}_{timestamp}.log"
    log_path    = os.path.join(LOGS_DIR, log_name)

    print(f"\n{'=' * 70}")
    print(f"STARTING: {script_name}")
    print(f"Full log: {log_path}")
    print(f"{'=' * 70}\n", flush=True)

    start = datetime.datetime.now()

    with open(log_path, 'w') as logfile:
        logfile.write(f"=== {script_name} started {start.isoformat()} ===\n")
        logfile.flush()

        process = subprocess.Popen(
            PYTHON_LAUNCHER + [script_path],
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )
        # Tee: print each line live AND persist it to the log file.
        for line in process.stdout:
            print(line, end='', flush=True)
            logfile.write(line)
        process.wait()

        elapsed = (datetime.datetime.now() - start).total_seconds()
        success = (process.returncode == 0)
        logfile.write(f"\n=== {script_name} finished, "
                       f"returncode={process.returncode}, "
                       f"elapsed={elapsed:.1f}s ===\n")

    return success, log_path, elapsed


def main():
    os.makedirs(LOGS_DIR, exist_ok=True)

    print(f"Orchestrator starting. {len(PENDING_SCRIPTS)} script(s) queued:")
    for s in PENDING_SCRIPTS:
        print(f"  - {s}")
    print("\nMake sure CARLA is already running before this proceeds.\n")

    results = []
    for script_name in PENDING_SCRIPTS:
        success, log_path, elapsed = run_script(script_name)
        results.append((script_name, success, log_path, elapsed))
        status = "OK" if success else "FAILED (non-zero exit code)"
        print(f"\n>>> {script_name}: {status} ({elapsed:.1f}s) — "
              f"log at {log_path}\n", flush=True)

    print(f"\n{'=' * 70}")
    print("ORCHESTRATOR SUMMARY")
    print(f"{'=' * 70}")
    for script_name, success, log_path, elapsed in results:
        status = "OK" if success else "FAILED"
        print(f"  [{status:6s}] {script_name:35s} {elapsed:7.1f}s  {log_path}")
    n_failed = sum(1 for _, s, _, _ in results if not s)
    if n_failed:
        print(f"\n{n_failed}/{len(results)} script(s) exited with a non-zero "
              f"code — check their logs above before trusting the output CSVs.")
    else:
        print(f"\nAll {len(results)} script(s) completed (exit code 0). "
              f"That means the PROCESS ran clean — still check each CSV's "
              f"own sanity numbers (crash rate, is_junction_t, etc.) before "
              f"trusting the data itself.")


if __name__ == '__main__':
    main()
