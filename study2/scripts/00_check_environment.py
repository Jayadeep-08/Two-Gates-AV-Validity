"""STEP 0 - check that this machine can run the project, and that it computes the
same numbers as the machine the code was developed on.

Run this once on any new machine, and again after changing Python or packages.
It writes nothing except code/results/env_check_<hostname>.json.

Three kinds of check:
  1. REQUIRED packages: numpy, scipy, pandas. The project will not run without them.
  2. OPTIONAL accelerators: torch/CUDA. Only reported. Nothing in the project uses the
     GPU yet, and it will not until a CPU-vs-GPU equivalence test has passed (see
     scripts/README.md).
  3. NUMERICAL REPRODUCIBILITY: runs the IDM integrator on a fixed input and compares
     against reference values computed in the development environment. A mismatch
     means results from this machine are not comparable with the diary's results,
     and must be investigated before any real run.

Usage (from AINN_ReWORK/scripts):   python 00_check_environment.py
"""

import json
import os
import platform
import socket
import sys
import time

from _common import DERIVED, EXID_DATA, HIGHD_DATA, RESULTS

# Reference values: code/src/idm.py, designed trace, theta* = (30, 1.2, 2, 1.5, 2),
# computed on the development machine (Python 3.11, numpy 2.4). Rounded to 1e-6.
REF = {"sum_gap": 21374.462317, "sum_vF": 11487.574488, "sum_vL": 11423.341447}
TOL = 1e-4


def main():
    report = {"host": socket.gethostname(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
              "python": sys.version.split()[0], "platform": platform.platform(),
              "cpu_count": os.cpu_count(), "required": {}, "optional": {}, "problems": []}

    for name in ("numpy", "scipy", "pandas"):
        try:
            mod = __import__(name)
            report["required"][name] = mod.__version__
        except ImportError:
            report["required"][name] = None
            report["problems"].append(f"missing required package: {name}")

    # Optional checks must never be able to crash the required ones. The first
    # version caught only ImportError; on the laptop torch is installed but fails to
    # load a DLL (OSError, WinError 1114), which crashed the whole script.
    try:
        import torch
        cuda = bool(torch.cuda.is_available())
        report["optional"]["torch"] = torch.__version__
        report["optional"]["cuda_available"] = cuda
        if cuda:
            report["optional"]["gpu"] = torch.cuda.get_device_name(0)
            report["optional"]["cuda_version"] = torch.version.cuda
    except ImportError:
        report["optional"]["torch"] = None
    except Exception as e:                      # installed but broken - report, continue
        report["optional"]["torch"] = f"installed but fails to load: {type(e).__name__}: {e}"[:300]

    try:
        import psutil
        report["ram_gb"] = round(psutil.virtual_memory().total / 1e9, 1)
    except ImportError:
        report["ram_gb"] = "unknown (psutil not installed - fine)"

    # data presence
    def count(d, suffix):
        return len([f for f in os.listdir(d) if f.endswith(suffix)]) if os.path.isdir(d) else 0
    report["data"] = {
        "highD_tracks_files": count(HIGHD_DATA, "_tracks.csv"),
        "exiD_tracks_files": count(EXID_DATA, "_tracks.csv"),
        "derived_dirs": sorted(os.listdir(DERIVED)) if os.path.isdir(DERIVED) else []}
    if report["data"]["highD_tracks_files"] != 60:
        report["problems"].append("highD: expected 60 tracks files")
    if report["data"]["exiD_tracks_files"] != 93:
        report["problems"].append("exiD: expected 93 tracks files")

    # numerical reproducibility + speed
    if report["required"].get("numpy"):
        import numpy as np
        from idm import make_leader_trace, simulate_follower
        _, vl = make_leader_trace(0.04, 20.0, kind="informative")
        th = np.array([30.0, 1.2, 2.0, 1.5, 2.0])
        s, v, ok = simulate_follower(vl, 0.04, th, 30.0, 28.0)
        got = {"sum_gap": round(float(s.sum()), 6), "sum_vF": round(float(v.sum()), 6),
               "sum_vL": round(float(vl.sum()), 6)}
        diffs = {k: abs(got[k] - REF[k]) for k in REF}
        report["reproducibility"] = {"reference": REF, "this_machine": got,
                                     "abs_diff": diffs,
                                     "pass": all(d <= TOL for d in diffs.values())}
        if not report["reproducibility"]["pass"]:
            report["problems"].append("IDM integrator does NOT reproduce reference values")
        t0 = time.time()
        for _ in range(500):
            simulate_follower(vl, 0.04, th, 30.0, 28.0)
        report["idm_ms_per_500_step_trace"] = round((time.time() - t0) / 500 * 1000, 3)

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, f"env_check_{report['host']}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print("\nRESULT:", "OK - ready" if not report["problems"] else "PROBLEMS - see list above")
    print("written:", out)


if __name__ == "__main__":
    main()
