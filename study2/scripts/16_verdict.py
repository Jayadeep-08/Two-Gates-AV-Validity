"""STEP 7 - propagate the calibration region to a safety-verdict interval (Amendment K).

Stages (each checkpointed; rerun the same command to resume):
  0  load the Step 6 regions, freeze the scenario suite, and run the theta-blindness assertion
     (K.1: the verdict module must not be able to see the parameter vector)
  1  POINT   theta_hat, both ego policies, N_POINT seeds; W_MC by bootstrap over seeds
  2  EPISTEMIC  min and max of V over each region by differential evolution in ball coordinates
     (every evaluation is cached with its whole threshold curve, and a cached point counts for
     every region that contains it), then fresh-seed re-evaluation of each extremum
  3  ATTRIBUTION  one parameter at a time across its marginal interval, the rest at theta_hat
  4  ALEATORIC   per-driver headways drawn from the fitted distribution, N_ALEA blocks
  5  LITRANGE    the same search over an analyst-chosen literature box
  6  OUTSIDE (positive control) and SHUFFLE (negative control)
  7  CLASS       the delayed-IDM comparison
  8  PHI         verdict flips between the two ego policies across the region
  9  report  -> code/results/step7/{cache.jsonl, step7.json, step7.md}

Everything here is synthetic: no real trajectory is read, so all outputs are shareable.
Usage (from AINN_ReWORK/scripts):
  python 16_verdict.py                 # everything (resumable)
  python 16_verdict.py --quick         # a small smoke configuration
  python 16_verdict.py --summary       # rebuild the report from the saved results
"""

import os

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import hashlib
import inspect
import json
import math
import sys
import time
import zlib

import numpy as np
from scipy.optimize import differential_evolution
from scipy.stats import binomtest

from _common import RESULTS, SRC  # noqa: F401

import verdict as vd                                    # noqa: E402
from idm import PARAM_BOUNDS, PARAM_NAMES               # noqa: E402

OUT = os.path.join(RESULTS, "step7")
STEP6 = os.path.join(RESULTS, "step6", "step6.json")
GEN = os.path.join(RESULTS, "step5_generator", "generator.json")
PN = list(PARAM_NAMES)
LO = np.array([b[0] for b in PARAM_BOUNDS], float)   # one source of truth with the fitting code
HI = np.array([b[1] for b in PARAM_BOUNDS], float)
# Run sizes. K.5 fixed 120 instances, 100 seeds for POINT and the fresh re-evaluations, 50 for
# the search, and a search budget of about 600 evaluations per direction per start. DE with
# popsize 12 over 5 parameters carries a population of 60, so maxiter 10 gives 660 - the
# declared budget. (The build used maxiter 25, i.e. 1560 per start, which was over budget and
# four hours of avoidable compute; K.6 records the correction.)
# N_INST is PER FAMILY: vd.draw_instances takes a per-family count and returns three families,
# so 40 here is the pre-registered suite of 120 instances. Read it as a total and you triple
# the whole run for nothing.
N_INST, N_POINT, N_SEARCH, N_FINAL, N_ALEA = 40, 100, 50, 100, 40
DE_POPSIZE, DE_MAXITER, DE_SEED, DE_STARTS = 12, 10, 20261002, 3
# Which outcomes each region is searched on. The full threshold grid is searched for the
# primary region; the other two calibrations are searched on the primary threshold only, which
# is the one K.4 predicts over, so the three-region comparison is on an equal footing. Their
# widths at the other thresholds come from pooled points and are lower bounds (K.6).
SEARCH_KEYS = {"primary_robust": None}        # None = every outcome
SEARCH_KEYS_DEFAULT = ["c2.0"]
B_INST = 2000          # paired instance bootstrap for the policy contrast (stage 8)
TAU_BASE, TAU_AGGR, BG_LAG_S = 1.5, 1.0, 0.6
# K.2's LITRANGE arm: "the literature ranges of the Sustainability 2023 baseline". That paper
# reports IDM point values, not ranges, so a range had to be chosen. The box below is the span of
# the standard IDM defaults as given in Treiber & Kesting's textbook, which brackets the 2023
# paper's values; it is reported under that name, NOT as the 2023 paper's own interval (K.6).
LIT_BOX = {"v0": (30.0, 40.0), "T": (1.0, 2.0), "s0": (1.0, 3.0),
           "a": (0.8, 2.0), "b": (1.0, 3.0)}
B_BOOT_MC, N_PERM = 2000, 10000


def seed(label):
    """A label-derived seed for a numpy Generator (bootstraps, permutations). These are
    independent RNG objects, so two of them colliding would only repeat a resampling pattern,
    not couple two simulations. SIMULATION seeds do NOT come from here - see sim_base."""
    return zlib.crc32(label.encode("utf-8"))


# Every simulation stream in the script, in a fixed order. The index in this list is the
# stream's identity; its base is (index+1) * vd.K_BLOCK, and the kernel's offsets stay inside
# one block, so two streams cannot share a seed however many are added (K.10). Appending to
# this list is safe; reordering or removing changes every downstream number, so don't.
SIM_STREAMS = (["point/baseline", "point/aggressive", "search/crn", "eval/crn", "final/crn",
                "attr/crn", "alea/crn", "phi/crn"]
               + [f"phinull/crn/{i}" for i in range(16)]
               + ["T/default"] + [f"alea/T/{r}" for r in range(64)])
_SIM_INDEX = {name: i for i, name in enumerate(SIM_STREAMS)}


def sim_base(label):
    """The seed block for one simulation stream. Never a hash: a hash gives bases at arbitrary
    distances, and two of them can land exactly an integer combination of the kernel's offsets
    apart - which is what happened to alea/T/22 and alea/T/31 (20% of their draws identical)."""
    if label not in _SIM_INDEX:
        raise KeyError(f"simulation stream {label!r} is not registered in SIM_STREAMS; add it "
                       f"there (at the END) so it gets a block of its own")
    return (_SIM_INDEX[label] + 1) * vd.K_BLOCK


# ---------------------------------------------------------------------------- theta blindness
def assert_theta_blind():
    """K.1 / §6: the verdict-computation module receives trajectories and nothing else, and is
    checked on every run. The check reads the compiled code object (parameters, locals, globals
    and closure referenced by the function), not the source text, so prose cannot pass or fail
    it."""
    forbidden = {"theta", "th", "thetas", "params", "param", "par", "t_i", "sig_e", "sig_eta",
                 "v_0", "s_0", "idm", "_idm_acc", "simulate", "_run_one", "run_suite",
                 "t_med", "t_log_sd", "het_t", "bg_lag", "tau_ego"}
    bad = []
    checked = (vd._metrics, vd.metrics_from_traces, vd.run_suite, Suite.curve,
               Suite.per_instance)
    for fn in checked:
        f = getattr(fn, "py_func", fn)
        c = f.__code__
        names = set(c.co_varnames) | set(c.co_names) | set(c.co_freevars) | set(c.co_cellvars)
        if f.__name__ == "run_suite":
            # The kernel necessarily receives theta - it drives the background cars with it.
            # What the assertion buys here is the plumbing: the kernel must hand the raw
            # trajectories to the metric function directly, with no theta-dependent
            # post-processing in between.
            if "_metrics" not in set(c.co_names):
                bad.append("run_suite does not call _metrics directly")
            continue
        hit = {n for n in names if n.lower() in forbidden}
        if hit:
            bad.append(f"{f.__name__} references {sorted(hit)}")
        sig = list(inspect.signature(f).parameters)
        if any(n.lower() in forbidden for n in sig):
            bad.append(f"{f.__name__} takes {sig}")
        for g in names:                       # no module global that holds the parameter vector
            val = f.__globals__.get(g)
            if isinstance(val, np.ndarray) and val.shape == (5,):
                bad.append(f"{f.__name__} reads the 5-vector global {g!r}")
    if bad:
        sys.exit("STOP - theta-blindness assertion failed: " + "; ".join(bad))
    return {"checked": [getattr(f, "__name__", str(f)) for f in checked],
            "forbidden_tokens": sorted(forbidden),
            "limit": "a name blacklist plus a same-module 5-vector scan; it does not catch a "
                     "parameter vector reached through another module or a closure"}


def assert_seed_streams(n_inst):
    """No two seed streams may overlap - enumerated, not argued.

    Each run seeds two streams, `t_base + m*K_STRIDE + k` for the per-driver headways and
    `base + m*K_STRIDE + k` for the process noise, and every base is a distinct multiple of
    K_BLOCK (sim_base). Since the offset can never reach K_BLOCK, the blocks are disjoint and
    this check cannot fail. It stays in anyway, and enumerates every seed the run will use,
    because the version it replaced *looked* safe too: bases from CRC32s, offsets with coprime
    multipliers spread over 11 million values. Two of the forty heterogeneity bases landed
    exactly 7919*95 + 104729*4 apart, and 20% of their draws were the same numbers. The check
    caught it on the first real run; the geometry now makes the failure impossible (K.10).
    Streams that are MEANT to coincide (common random numbers across theta) share one base by
    construction and so are a single stream here."""
    streams = {}
    for lbl in SIM_STREAMS:
        if lbl.startswith("alea/T/") and int(lbl.rsplit("/", 1)[1]) >= N_ALEA:
            continue                      # heterogeneity blocks this configuration won't draw
        streams[lbl] = sim_base(lbl)
    kmax = max(N_POINT, N_SEARCH, N_FINAL)
    if kmax > vd.K_STRIDE or n_inst > vd.M_MAX:
        sys.exit(f"STOP - suite geometry ({n_inst} instances x {kmax} seeds) does not fit the "
                 f"seed block (M_MAX={vd.M_MAX}, K_STRIDE={vd.K_STRIDE})")
    owner, clash = {}, []
    for label, b in streams.items():
        for m in range(n_inst):
            for k in range(kmax):
                s = b + m * vd.K_STRIDE + k
                if s in owner and owner[s] != label:
                    clash.append(f"{owner[s]} and {label} share seed {s}")
                owner[s] = label
    if clash:
        sys.exit("STOP - seed streams overlap, so evaluations treated as independent are not: "
                 + "; ".join(sorted(set(clash))[:6]))
    return {"streams": sorted(streams), "n_streams": len(streams),
            "seeds_checked": len(owner), "n_instances": n_inst, "k_max": kmax,
            "block": vd.K_BLOCK, "stride": vd.K_STRIDE}


# ---------------------------------------------------------------------------- evaluation
class Suite:
    def __init__(self, theta_hat, n_inst, gen):
        self.inst = vd.draw_instances(n_inst)
        self.M = len(self.inst)
        (self.lead, self.off, self.X0, self.V0, self.fam, self.cut_step, self.cut_gap,
         self.cut_v) = vd.pack_instances(self.inst, theta_hat)
        self.phi = math.exp(-vd.DT / gen["best"]["tau"])
        self.sig_eta = gen["best"]["sig_eta"]
        self.omega = gen["best"]["omega"]
        self.lag_steps = int(round(BG_LAG_S / vd.DT))
        self.n_eval = 0

    def run(self, theta, n_seeds, base, tau=TAU_BASE, het=0, lag=0, T_log_sd=0.0, t_base=None):
        # The kernel lays each stream out as base + m*K_STRIDE + k, which stays inside one
        # K_BLOCK only while the suite fits the declared geometry (K.10).
        if n_seeds > vd.K_STRIDE or self.M > vd.M_MAX:
            raise ValueError(f"suite geometry exceeds the seed block: {self.M} instances x "
                             f"{n_seeds} seeds against M_MAX={vd.M_MAX}, K_STRIDE="
                             f"{vd.K_STRIDE}. Raise them in verdict.py together with K_BLOCK.")
        if t_base is None:
            t_base = sim_base("T/default")
        theta = np.ascontiguousarray(np.asarray(theta, float))
        n = self.M * n_seeds
        cr = np.empty(n); tt = np.empty(n); rs = np.empty(n); ot = np.empty(n)
        vd.run_suite(self.lead, self.off, self.X0, self.V0, theta, float(theta[1]),
                     float(T_log_sd), int(het), float(self.sig_eta), float(self.omega),
                     float(self.phi), float(tau), vd.DT, int(lag), int(n_seeds),
                     int(base), int(t_base),
                     self.cut_step, self.cut_gap, self.cut_v, cr, tt, rs, ot)
        self.n_eval += 1
        sh = (self.M, n_seeds)
        return cr.reshape(sh), tt.reshape(sh), rs.reshape(sh), ot.reshape(sh)

    def curve(self, cr, tt, ot=None):
        """The pre-registered outcome: the EGO-involved collision rate and the ego conflict rate
        at each threshold of the declared grid. A conflict is a close encounter or an
        ego-involved collision. Collisions elsewhere in the string are reported separately and
        never enter the outcome (S5 review of K)."""
        out = {"crash": float(cr.mean())}
        for t in vd.TTC_GRID:
            out[f"c{t}"] = float(np.mean((tt < t) | (cr > 0)))
        if ot is not None:
            out["other_crash"] = float(ot.mean())
        return out

    def per_instance(self, cr, tt, t):
        return np.mean((tt < t) | (cr > 0), axis=1)


def save(obj, path):
    """Atomic checkpoint: a truncated write must not destroy a multi-hour run."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=float)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


# ---------------------------------------------------------------------------- region geometry
class Region:
    """{theta : (theta-c)' S^+ (theta-c) <= c_fac*q}, parameterised by the unit ball so that
    every search point is feasible by construction."""

    def __init__(self, fit, tag):
        r = fit["regions"][tag]
        self.tag = tag
        self.center = np.array(fit["theta_hat"], float)
        self.cov = np.array(fit["cov"], float)
        self.q = r["c"] * fit["q_loo"]
        self.P = np.linalg.pinv(self.cov)
        w, Vm = np.linalg.eigh(self.cov)
        self.L = Vm @ np.diag(np.sqrt(np.clip(w, 0, None)))
        self.clipped = 0

    def theta(self, y):
        y = np.asarray(y, float)
        nrm = np.linalg.norm(y)
        u = y if nrm <= 1.0 else y / nrm
        th = self.center + math.sqrt(self.q) * (self.L @ u)
        cl = np.clip(th, LO, HI)
        if not np.allclose(cl, th):
            self.clipped += 1
        return cl

    def contains(self, th):
        d = np.asarray(th, float) - self.center
        return bool(d @ self.P @ d <= self.q * (1 + 1e-9))

    def surface(self, direction, scale=1.0):
        """A point at `scale` times the region radius along `direction`. Clipping to the
        parameter bounds can shorten it, so the ACHIEVED Mahalanobis radius is returned with it
        (S5 review finding 6) and the direction is re-scaled along the unclipped coordinates to
        get as close to the requested radius as the bounds allow."""
        d = np.asarray(direction, float) - self.center
        m = math.sqrt(max(d @ self.P @ d, 1e-300))
        raw = self.center + scale * math.sqrt(self.q) / m * d
        th = np.clip(raw, LO, HI)
        for _ in range(60):
            dd = th - self.center
            cur = math.sqrt(max(dd @ self.P @ dd, 1e-300)) / math.sqrt(self.q)
            # Which coordinates are still free to move - i.e. not pinned at a bound. This has
            # to be recomputed each pass: a coordinate pinned by the clip cannot contribute any
            # further radius, and the loop must stop when none are left.
            free = ~(np.isclose(th, LO) | np.isclose(th, HI))
            if cur >= scale - 1e-6 or not free.any():
                break
            th2 = th.copy()
            th2[free] = self.center[free] + (th[free] - self.center[free]) * 1.05
            th2 = np.clip(th2, LO, HI)
            if np.allclose(th2, th):
                break
            th = th2
        dd = th - self.center
        return th, float(math.sqrt(max(dd @ self.P @ dd, 1e-300)) / math.sqrt(self.q))


# ---------------------------------------------------------------------------- cache
def read_cache():
    """Every evaluation ever appended, de-duplicated.

    A stage that is interrupted and re-run appends its evaluations a second time, and a
    search can revisit a point it has already tried. Those rows carry the SAME curve (same
    theta, same seed base, same suite), so keeping both would not change any value - but it
    would inflate the point count behind the SHUFFLE regression and the diagnostic bands,
    i.e. claim more independent evidence than the run actually has. The key is everything
    that determines the curve, plus the label, so two genuinely different evaluations of the
    same theta are both kept."""
    fn = os.path.join(OUT, "cache.jsonl")
    rows, seen = [], {}
    if os.path.exists(fn):
        for line in open(fn, encoding="utf-8"):
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            k = (r.get("label"), r.get("region"), r.get("n_seeds"), r.get("tau"),
                 r.get("lag"), tuple(round(float(v), 12) for v in r.get("theta", ())))
            if k in seen:
                rows[seen[k]] = r                 # keep the most recent copy
            else:
                seen[k] = len(rows)
                rows.append(r)
    return rows


def append_cache(rec):
    fn = os.path.join(OUT, "cache.jsonl")
    lead = ""
    if os.path.exists(fn) and os.path.getsize(fn) > 0:
        with open(fn, "rb") as fh:
            fh.seek(-1, os.SEEK_END)
            lead = "" if fh.read(1) == b"\n" else "\n"
    with open(fn, "a", encoding="utf-8") as fh:
        fh.write(lead + json.dumps(rec, default=float) + "\n")


def band(cache, region, key, n_seeds):
    """Diagnostic only: min and max of `key` over every cached search point inside this region.
    The REPORTED interval comes from the fresh-seed re-evaluations (K.2); the difference between
    the two is the winner's curse of taking an extreme over many noisy evaluations."""
    pts = [(r["curve"][key], r["theta"]) for r in cache
           if r["n_seeds"] == n_seeds and r["lag"] == 0 and r["tau"] == TAU_BASE
           and r["label"] == "search" and region.contains(r["theta"])]
    if not pts:
        return None
    lo = min(pts, key=lambda z: z[0])
    hi = max(pts, key=lambda z: z[0])
    return {"min": lo[0], "argmin": lo[1], "max": hi[0], "argmax": hi[1], "n_points": len(pts)}


# ---------------------------------------------------------------------------- stages
def stage_point(S, th, out):
    if "point" in out:
        return out["point"]
    res = {}
    for name, tau in (("baseline", TAU_BASE), ("aggressive", TAU_AGGR)):
        cr, tt, rs, ot = S.run(th, N_POINT, sim_base(f"point/{name}"), tau=tau)
        c = S.curve(cr, tt, ot)
        rng = np.random.default_rng(seed(f"mc/{name}"))
        # Resample seed blocks within each instance. Keyed on the OUTCOME curve only:
        # other_crash is a diagnostic, not an outcome, and carries no W_MC.
        boots = {k: [] for k in S.curve(cr, tt)}
        for _ in range(B_BOOT_MC):
            idx = rng.integers(0, cr.shape[1], cr.shape)
            crb = np.take_along_axis(cr, idx, 1)
            ttb = np.take_along_axis(tt, idx, 1)
            b = S.curve(crb, ttb)
            for k in b:
                if k in boots:
                    boots[k].append(b[k])
        res[name] = {"curve": c, "rss_mean": float(rs.mean()),
                     "per_instance_c2.0": S.per_instance(cr, tt, 2.0).tolist(),
                     "W_MC": {k: float(np.quantile(v, 0.975) - np.quantile(v, 0.025))
                              for k, v in boots.items()},
                     "mc_se": {k: float(np.std(v, ddof=1)) for k, v in boots.items()},
                     "n_collisions": int(cr.sum()), "n_runs": int(cr.size)}
        print(f"[1] POINT {name}: {json.dumps({k: round(v, 4) for k, v in c.items()})}", flush=True)
    out["point"] = res
    return res


def evaluate(S, region_tag, theta, label, n_seeds=None, tau=TAU_BASE, lag=0, base=None,
             per_inst=False):
    """One suite evaluation. `base` defaults to a COMMON seed base so that evaluations meant to
    be compared with each other are contrasts at fixed noise rather than independent draws
    (K.2; S5 review findings 4 and 5)."""
    n_seeds = N_SEARCH if n_seeds is None else n_seeds      # resolved now, not at import
    cr, tt, _, ot = S.run(theta, n_seeds, sim_base("eval/crn") if base is None else base,
                          tau=tau, lag=lag)
    rec = {"theta": list(map(float, theta)), "curve": S.curve(cr, tt, ot), "n_seeds": n_seeds,
           "tau": tau, "lag": lag, "region": region_tag, "label": label}
    if per_inst:
        rec["per_instance_c2.0"] = S.per_instance(cr, tt, 2.0).tolist()
    append_cache(rec)
    return rec


def search(S, region, key, sign, cache_sink):
    """min (sign=+1) or max (sign=-1) of the outcome `key` over the region."""
    def f(y):
        th = region.theta(y)
        cr, tt, _, ot = S.run(th, N_SEARCH, sim_base("search/crn"))
        c = S.curve(cr, tt, ot)
        cache_sink.append({"theta": list(map(float, th)), "curve": c, "n_seeds": N_SEARCH,
                           "tau": TAU_BASE, "lag": 0, "region": region.tag, "label": "search"})
        return sign * c[key]
    best, nfev, msgs, starts = None, 0, [], []
    for st in range(DE_STARTS):               # K.5 fixes 3 starts; tol/atol 0 so maxiter binds
        r = differential_evolution(f, [(-1.0, 1.0)] * 5, popsize=DE_POPSIZE,
                                   maxiter=DE_MAXITER, tol=0.0, atol=0.0, polish=False,
                                   seed=DE_SEED + st, init="sobol" if st == 0 else "latinhypercube")
        nfev += int(r.nfev); msgs.append(str(r.message))
        # D7.4 convergence evidence: each start is an INDEPENDENT optimisation of the same
        # surface. Agreement between the starts is the only evidence available that the
        # reported extremum is the region's extremum and not one start's luck; disagreement
        # means the search budget is too small and the interval is understated.
        starts.append({"start": st, "value": sign * float(r.fun), "nfev": int(r.nfev),
                       "theta": list(map(float, region.theta(r.x)))})
        if best is None or r.fun < best[1]:
            best = (r.x.copy(), float(r.fun))
    return region.theta(best[0]), sign * best[1], nfev, msgs, starts


def main():
    global N_INST, N_POINT, N_SEARCH, N_FINAL, N_ALEA
    global DE_MAXITER, B_BOOT_MC, N_PERM, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="smallest configuration: exercises every stage and the stop paths in "
                         "a couple of minutes. The outcome is too coarsely quantised at this "
                         "size for D7.1 to pass, so a STOP here is expected, not a result.")
    ap.add_argument("--mid", action="store_true",
                    help="the validation configuration: big enough for the controls to mean "
                         "something and to reach the report. Run this FIRST. Results go to a "
                         "separate directory and are never mixed with the real run.")
    ap.add_argument("--summary", action="store_true",
                    help="re-print the report from an existing results directory without "
                         "running or writing anything")
    a = ap.parse_args()
    if a.quick:
        N_INST, N_POINT, N_SEARCH, N_FINAL, N_ALEA = 4, 10, 6, 12, 4
        DE_MAXITER, B_BOOT_MC, N_PERM = 3, 50, 200
        OUT = OUT + "_quick"
    elif a.mid:
        N_INST, N_POINT, N_SEARCH, N_FINAL, N_ALEA = 10, 40, 20, 60, 10
        DE_MAXITER, B_BOOT_MC, N_PERM = 8, 500, 2000
        OUT = OUT + "_mid"
    os.makedirs(OUT, exist_ok=True)
    res_file = os.path.join(OUT, "step7.json")
    out = json.load(open(res_file)) if os.path.exists(res_file) else {}
    if a.summary:
        report(out, read_only=True)
        return 0

    assert_theta_blind()
    ss = assert_seed_streams(len(vd.draw_instances(N_INST)))
    print(f"[0] theta-blindness and seed-stream assertions passed "
          f"({ss['seeds_checked']} seeds over {ss['n_streams']} disjoint streams, "
          f"block {ss['block']}, stride {ss['stride']})", flush=True)
    fit = json.load(open(STEP6))["fits"]["loc1"]
    gen = json.load(open(GEN))
    th = np.array(fit["theta_hat"], float)
    S = Suite(th, N_INST, gen)
    regions = {t: Region(fit, t) for t in fit["regions"]}
    h = hashlib.md5()
    for fn in (os.path.join(SRC, "verdict.py"), os.path.abspath(__file__)):
        h.update(open(fn, "rb").read())
    h.update(json.dumps({"theta": fit["theta_hat"], "cov": fit["cov"], "q": fit["q_loo"],
                         "c": {t: r["c"] for t, r in fit["regions"].items()},
                         "sizes": [N_INST, N_POINT, N_SEARCH, N_FINAL, N_ALEA, DE_POPSIZE,
                                   DE_MAXITER, DE_STARTS],
                         "config": ("quick" if a.quick else ("mid" if a.mid else "run"))},
                        sort_keys=True, default=float).encode())
    fpf = os.path.join(OUT, "fingerprint.txt")
    if os.path.exists(fpf) and open(fpf).read().strip() != h.hexdigest():
        sys.exit("STOP - saved Step 7 results come from different code or a different region; "
                 "move code/results/step7 aside")
    open(fpf, "w").write(h.hexdigest() + "\n")
    t_all = time.time()

    stage_point(S, th, out); save(out, res_file)

    # 2 epistemic search
    if "epistemic" not in out:
        keys = ["crash"] + [f"c{t}" for t in vd.TTC_GRID]
        ep = out.get("epistemic_partial", {})
        for tag, reg in regions.items():
            if tag in ep:
                continue
            sink = []
            t0 = time.time()
            ex = {}
            ks = SEARCH_KEYS.get(tag, SEARCH_KEYS_DEFAULT)
            for key in (keys if ks is None else ks):
                for sign, nm in ((1, "min"), (-1, "max")):
                    thx, val, nfev, msgs, starts = search(S, reg, key, sign, sink)
                    sv = [s["value"] for s in starts]
                    ex[f"{key}/{nm}"] = {"theta": list(map(float, thx)), "search_value": val,
                                         "nfev": nfev, "messages": msgs, "starts": starts,
                                         "start_spread": max(sv) - min(sv)}
            ep[tag] = {"extrema": ex, "clipped_points": reg.clipped,
                       "searched_keys": (keys if ks is None else ks),
                       "seconds": time.time() - t0}
            for r in sink:
                append_cache(r)
            out["epistemic_partial"] = ep
            save(out, res_file)
            print(f"[2] {tag}: {len(ex)} searches, {time.time() - t0:.0f} s, "
                  f"{reg.clipped} clipped", flush=True)
        out["epistemic"] = ep
        save(out, res_file)

    cache = read_cache()

    # 2b fresh-seed re-evaluation of every extremum
    if "final" not in out:
        # theta_hat on the SAME fresh seed base and the same seed count as the extrema, so that
        # the reported interval, the OUTSIDE control and the point value are all one contrast at
        # fixed noise. The POINT stage uses its own base and its own seed count and stays a
        # separate, independently seeded estimate (S5 review: mixed seed counts in the envelope).
        rec0 = evaluate(S, "final_point", th, "final/theta_hat", n_seeds=N_FINAL,
                        base=sim_base("final/crn"), per_inst=True)
        out["final_point"] = {"theta": list(map(float, th)), "curve": rec0["curve"],
                              "per_instance_c2.0": rec0["per_instance_c2.0"],
                              "n_seeds": N_FINAL, "base": "final/crn"}
        fin = {}
        for tag, ep in out["epistemic"].items():
            fin[tag] = {}
            for name, e in ep["extrema"].items():
                rec = evaluate(S, tag, np.array(e["theta"]), f"final/{tag}/{name}",
                               n_seeds=N_FINAL, base=sim_base("final/crn"), per_inst=True)
                key = name.split("/")[0]
                # n_seeds and base are stored with every record so that report() can VERIFY
                # that each member of the envelope pool is the same kind of evaluation,
                # instead of relying on a comment that says so (S5 review).
                fin[tag][name] = {"theta": e["theta"], "search_value": e["search_value"],
                                  "fresh_value": rec["curve"][key],
                                  "gap": rec["curve"][key] - e["search_value"],
                                  "n_seeds": N_FINAL, "base": "final/crn",
                                  "curve": rec["curve"]}
        out["final"] = fin
        save(out, res_file)
        print("[2b] extrema re-evaluated with fresh seeds", flush=True)

    # 3 attribution
    if "attribution" not in out:
        att = {}
        reg = regions["primary_robust"]
        half = np.array(fit["regions"]["primary_robust"]["marginal_half_width"])
        for j, nm in enumerate(PN):
            vals = []
            for w in np.linspace(-1, 1, 11):
                t2 = th.copy(); t2[j] = float(np.clip(th[j] + w * half[j], LO[j], HI[j]))
                rec = evaluate(S, "attribution", t2, f"attr/{nm}/{w:+.2f}",
                               base=sim_base("attr/crn"))
                vals.append({"value": t2[j], "curve": rec["curve"]})
            att[nm] = vals
            rng_ = [v["curve"]["c2.0"] for v in vals]
            print(f"[3] {nm}: c2.0 from {min(rng_):.4f} to {max(rng_):.4f}", flush=True)
        out["attribution"] = att
        save(out, res_file)

    # 4 aleatoric
    if "aleatoric" not in out:
        sd_log = float(json.load(open(os.path.join(RESULTS, "step6", "fit_loc1.json")))["T_log_sd"])
        vals = []
        for r in range(N_ALEA):
            cr, tt, _, ot = S.run(th, N_POINT, sim_base("alea/crn"), het=1, T_log_sd=sd_log,
                                  t_base=sim_base(f"alea/T/{r}"))
            vals.append(S.curve(cr, tt, ot))
        out["aleatoric"] = {"T_log_sd": sd_log, "blocks": vals,
                            "note": "the process noise is held fixed across blocks, so this "
                                    "range is heterogeneity only; Ĝ is approximated by a "
                                    "lognormal around theta_hat's T, dropping J.3's dependence "
                                    "on the initial headway",
                            "W_alea": {k: float(np.quantile([v[k] for v in vals], 0.975)
                                                - np.quantile([v[k] for v in vals], 0.025))
                                       for k in vals[0]}}
        save(out, res_file)
        print(f"[4] ALEATORIC over {N_ALEA} blocks: W_alea(c2.0) = "
              f"{out['aleatoric']['W_alea']['c2.0']:.4f}", flush=True)

    # 5 literature box
    if "litrange" not in out:
        lo = np.array([LIT_BOX[n][0] for n in PN]); hi = np.array([LIT_BOX[n][1] for n in PN])
        sink = []

        def fbox(z, sign):
            t2 = lo + np.asarray(z) * (hi - lo)
            cr, tt, _, ot = S.run(t2, N_SEARCH, sim_base("search/crn"))
            c = S.curve(cr, tt, ot)
            sink.append({"theta": list(map(float, t2)), "curve": c, "n_seeds": N_SEARCH,
                         "tau": TAU_BASE, "lag": 0, "region": "litrange", "label": "lit"})
            return sign * c["c2.0"]
        lit = {}
        for sign, nm in ((1, "min"), (-1, "max")):
            # Same optimiser settings as the region search (3 starts, no early stop), because
            # the literature-box width is compared with the region width and the two bounds
            # must be computed with the same effort to be comparable (K.8).
            best = None
            for st in range(DE_STARTS):
                r = differential_evolution(fbox, [(0.0, 1.0)] * 5, args=(sign,),
                                           popsize=DE_POPSIZE, maxiter=DE_MAXITER, tol=0.0,
                                           atol=0.0, polish=False, seed=DE_SEED + st,
                                           init="sobol" if st == 0 else "latinhypercube")
                if best is None or r.fun < best.fun:
                    best = r
            t2 = lo + best.x * (hi - lo)
            rec = evaluate(S, "litrange", t2, f"final/lit/{nm}", n_seeds=N_FINAL,
                           base=sim_base("final/crn"))
            lit[nm] = {"theta": list(map(float, t2)), "curve": rec["curve"]}
        for r in sink:
            append_cache(r)
        out["litrange"] = {"box": LIT_BOX, "source": "span of the standard Treiber-Kesting IDM "
                           "defaults, which brackets the Sustainability 2023 baseline's point "
                           "values; not that paper's own interval", **lit}
        save(out, res_file)
        print(f"[5] LITRANGE c2.0 {lit['min']['curve']['c2.0']:.4f} - "
              f"{lit['max']['curve']['c2.0']:.4f}", flush=True)

    # 6 controls
    if "controls" not in out:
        reg = regions["primary_robust"]
        # Both sides at N_FINAL seeds on the same base; the threshold is the Monte Carlo width
        # rescaled from the POINT stage's seed count to N_FINAL (the bootstrap resamples seed
        # blocks, so the width scales as 1/sqrt(n_seeds)).
        v0 = out["final_point"]["curve"]["c2.0"]
        wmc_point = out["point"]["baseline"]["W_MC"]["c2.0"]
        wmc = wmc_point * math.sqrt(N_POINT / N_FINAL)
        # OUTSIDE over ELEVEN pre-declared directions, not one (K.6, after the single-direction
        # version failed on the harness for a reason that had nothing to do with sensitivity).
        # The five principal axes of the region, both signs, plus the direction of the
        # within-region c2.0 argmax. Along the argmax direction the push to 3x radius drops v0
        # while raising b, and the two effects cancel, so that one direction measures the
        # cancellation rather than the instrument. Every direction's change is recorded; the
        # control statistic is the largest.
        w_eig, V_eig = np.linalg.eigh(reg.cov)
        dirs = []
        for j in np.argsort(w_eig)[::-1]:
            for sgn in (1.0, -1.0):
                dirs.append((f"axis{int(j)}{'+' if sgn > 0 else '-'}",
                             reg.center + sgn * V_eig[:, j]))
        dirs.append(("argmax_c2.0",
                     np.array(out["final"]["primary_robust"]["c2.0/max"]["theta"], float)))
        probes, rejected = [], []
        for nm, d in dirs:
            th3, achieved = reg.surface(d, scale=3.0)
            r3 = evaluate(S, "outside", th3, f"final/outside/{nm}", n_seeds=N_FINAL,
                          base=sim_base("final/crn"))
            p3 = {"direction": nm, "theta": list(map(float, th3)),
                  "achieved_radius": achieved, "v": r3["curve"]["c2.0"],
                  "abs_change": abs(r3["curve"]["c2.0"] - v0), "curve": r3["curve"]}
            # A probe the parameter bounds pulled back to the region itself is not OUTSIDE it,
            # whatever it does to the outcome, so it cannot carry the control (K.8).
            (probes if achieved >= 1.5 else rejected).append(p3)
        if not probes:
            save(out, res_file)
            sys.exit("STOP - every OUTSIDE direction was pulled back inside the region by the "
                     "parameter bounds, so D7.1 cannot be evaluated. Reported as such; no "
                     "verdict interval is claimed.")
        bestp = max(probes, key=lambda z: z["abs_change"])
        rec, achieved, d50 = {"curve": bestp["curve"]}, bestp["achieved_radius"], \
            bestp["abs_change"]
        th3 = np.array(bestp["theta"], float)
        # The threshold cannot be below the finest a rate over M x N_FINAL runs can resolve: a
        # W_MC that bootstraps to zero would otherwise let any nonzero change pass.
        floor_out = 1.0 / max(S.M * N_FINAL, 1)
        thr_out = max(wmc, floor_out)
        n_pass_dir = int(sum(p["abs_change"] > thr_out for p in probes))
        cache2 = [r for r in read_cache() if r["label"] == "search" and r["n_seeds"] == N_SEARCH]
        X = np.array([r["theta"] for r in cache2]); Y = np.array([r["curve"]["c2.0"] for r in cache2])
        sst = float(np.sum((Y - Y.mean()) ** 2))
        if len(X) < 10 or sst <= 1e-12:
            shuf = {"n_points": len(X), "r2_observed": None, "r2_permuted_mean": None,
                    "r2_permuted_p95": None, "sensitivity_p_value": None, "pass": None,
                    "note": "the outcome does not vary across the cached points, so the "
                            "theta -> V relation cannot be tested"}
        else:
            sd = X.std(0)
            keep = sd > 1e-9 * np.maximum(np.abs(X.mean(0)), 1.0)
            Xc = np.c_[np.ones(len(X)), (X[:, keep] - X[:, keep].mean(0)) / sd[keep]]
            # The design is fixed and only y is permuted, so the pseudo-inverse is computed once
            # and each of the N_PERM replicates is two matrix-vector products. (Calling lstsq
            # inside the loop costs a full decomposition per replicate - tens of minutes at the
            # run size, for an identical answer.)
            Xpinv = np.linalg.pinv(Xc)
            r2 = lambda yy: 1.0 - float(np.sum((yy - Xc @ (Xpinv @ yy)) ** 2)) \
                / max(float(np.sum((yy - yy.mean()) ** 2)), 1e-300)
            obs = r2(Y)
            rng_ = np.random.default_rng(seed("shuffle"))
            perm = [r2(rng_.permutation(Y)) for _ in range(N_PERM)]
            pm = float(np.mean(perm))
            p95 = float(np.quantile(perm, 0.95))
            # D7.2 as amended in K.6: the NEGATIVE control is that permuting the theta labels
            # destroys the relation, so the quantity that must be ~0 is the PERMUTED R2.
            # "~0" has to be measured against the right scale, not against a fixed 0.05: for a
            # fixed design with p regressors and n points, E[R2] under permutation is exactly
            # p/(n-1). At the run size that is 2e-4, so a fixed 0.05 threshold passes whatever
            # the data say; at n = 60 it is 0.085, so the same threshold would FAIL a healthy
            # instrument. The rule is therefore stated relative to that expectation (K.8).
            nullr2 = keep.sum() / max(len(X) - 1, 1)
            shuf = {"n_points": len(X), "n_parameters_used": int(keep.sum()),
                    "r2_observed": obs, "r2_permuted_mean": pm, "r2_permuted_p95": p95,
                    "r2_null_expected": float(nullr2),
                    "sensitivity_p_value": float((1 + np.sum(np.array(perm) >= obs)) / (N_PERM + 1)),
                    "sensitivity_note": "the cached points are an optimiser's trajectory, not a "
                                        "random design, so the OBSERVED R2 and its p-value are "
                                        "descriptive; they are not a test of sensitivity (that "
                                        "is D7.1's job)",
                    "pass": bool(pm <= max(3.0 * nullr2, 1e-6)
                                 and p95 <= max(8.0 * nullr2, 1e-5))}
        out["controls"] = {
            "outside": {"theta": list(map(float, th3)), "curve": rec["curve"],
                        "requested_radius": 3.0, "achieved_radius": achieved,
                        "v_theta_hat": v0, "abs_change": d50, "W_MC": wmc,
                        "W_MC_point": wmc_point, "n_seeds": N_FINAL,
                        "threshold": thr_out, "resolution_floor": floor_out,
                        "direction": bestp["direction"], "n_directions": len(probes),
                        "n_directions_passing": n_pass_dir, "probes": probes,
                        "rejected_probes": rejected,
                        "abs_change_argmax_only": next(
                            (p["abs_change"] for p in probes + rejected
                             if p["direction"] == "argmax_c2.0"), None),
                        "pass": bool(d50 > thr_out)},
            "shuffle": shuf}
        save(out, res_file)
        print(f"[6] OUTSIDE best of {len(probes)} directions ({bestp['direction']}, radius "
              f"{achieved:.2f}x): |dV| {d50:.4f} vs threshold {thr_out:.4f} -> "
              f"{'pass' if out['controls']['outside']['pass'] else 'FAIL'}; SHUFFLE "
              + ("not evaluable (the outcome does not vary across the cached points)"
                 if shuf.get("pass") is None else
                 f"permuted R2 {shuf['r2_permuted_mean']:.5f} against a null expectation of "
                 f"{shuf['r2_null_expected']:.5f} -> "
                 f"{'pass' if shuf['pass'] else 'FAIL'}"), flush=True)

    c = out["controls"]
    if not c["outside"]["pass"]:
        save(out, res_file)
        report(out)
        sys.exit("STOP - D7.1: OUTSIDE did not move the outcome by more than the Monte Carlo "
                 "width, so the suite cannot detect parameter effects. Reported as insensitive; "
                 "no verdict interval is claimed.")
    if c["shuffle"].get("pass") is False:
        save(out, res_file)
        report(out)
        sys.exit("STOP - D7.2: permuting the theta labels did not destroy the theta -> V "
                 "relation, which indicates leakage.")

    # 7 delayed-IDM class arm
    if "class_delayed" not in out:
        reg = regions["primary_robust"]
        cls = {"point": evaluate(S, "class", th, "class/point", n_seeds=N_FINAL,
                                 lag=S.lag_steps, base=sim_base("final/crn"))["curve"]}
        for nm in ("min", "max"):
            t2 = np.array(out["final"]["primary_robust"][f"c2.0/{nm}"]["theta"])
            cls[nm] = evaluate(S, "class", t2, f"class/{nm}", n_seeds=N_FINAL,
                               lag=S.lag_steps, base=sim_base("final/crn"))["curve"]
        out["class_delayed"] = {"bg_lag_s": BG_LAG_S, **cls}
        save(out, res_file)
        print(f"[7] delayed IDM: crash {cls['point']['crash']:.4f}, c2.0 {cls['point']['c2.0']:.4f}",
              flush=True)

    # 8 verdict flips between the two ego policies, over every fresh extremum of the region
    if "phi" not in out:
        reg = regions["primary_robust"]
        pts = {"theta_hat": th}
        for nm, e in out["final"]["primary_robust"].items():
            if reg.contains(e["theta"]):
                pts[nm] = np.array(e["theta"], float)
        per = {}
        for nm, t2 in pts.items():
            d = {}
            for pol, tau in (("baseline", TAU_BASE), ("aggressive", TAU_AGGR)):
                cr, tt, _, _ = S.run(t2, N_FINAL, sim_base("phi/crn"), tau=tau)
                d[pol] = S.per_instance(cr, tt, 2.0)
            per[nm] = d
        # K.3: Phi is the fraction of instances whose safety ranking between the two policies
        # "flips somewhere in Theta-hat". An instance flips when BOTH orderings occur somewhere
        # among the evaluated points - not when a point disagrees with theta_hat. Referencing
        # theta_hat would miss an instance that is + at one extremum and - at another while
        # tied at theta_hat, and would make a tie at theta_hat disqualify the instance
        # entirely (K.8).
        # Null calibration (K.8). The flip statistic is an OR over many points, so it grows with
        # the number of points even when theta does nothing at all: a per-instance rate is a
        # proportion over N_FINAL seeds, and the sign of a small difference is partly noise. The
        # same statistic is therefore computed with theta held AT theta_hat and only the seed
        # base varying, over the same number of points. Phi is read against that floor.
        per0 = {}
        for i in range(len(pts)):
            d0 = {}
            for pol, tau in (("baseline", TAU_BASE), ("aggressive", TAU_AGGR)):
                cr, tt, _, _ = S.run(th, N_FINAL, sim_base(f"phinull/crn/{i}"), tau=tau)
                d0[pol] = S.per_instance(cr, tt, 2.0)
            per0[f"null{i}"] = d0
        np_, nn_ = np.zeros(S.M, dtype=bool), np.zeros(S.M, dtype=bool)
        for d0 in per0.values():
            s_ = np.sign(d0["baseline"] - d0["aggressive"])
            np_ |= s_ > 0
            nn_ |= s_ < 0
        k_null = int((np_ & nn_).sum())
        sgn = {nm: np.sign(per[nm]["baseline"] - per[nm]["aggressive"]) for nm in pts}
        any_pos = np.zeros(S.M, dtype=bool)
        any_neg = np.zeros(S.M, dtype=bool)
        for nm in pts:
            any_pos |= sgn[nm] > 0
            any_neg |= sgn[nm] < 0
        flips = any_pos & any_neg
        k = int(flips.sum())
        bt = binomtest(k, S.M)
        # Kept for continuity with the earlier definition and reported beside it.
        base_sign = sgn["theta_hat"]
        flips_vs_hat = np.zeros(S.M, dtype=bool)
        for nm in pts:
            if nm != "theta_hat":
                flips_vs_hat |= (sgn[nm] != base_sign) & (sgn[nm] != 0) & (base_sign != 0)
        n_tied = int(np.sum(base_sign == 0))
        n_elig = int(np.sum(any_pos | any_neg))     # an instance with a strict ordering anywhere
        bte = binomtest(k, n_elig) if n_elig > 0 else None
        # The AGGREGATE contrast behind the headline claim: at each theta, the difference in
        # the suite-mean conflict rate between the two ego policies, with a percentile
        # bootstrap over INSTANCES (the sampling unit). The two policies run on the same seed
        # base, so the difference is paired within each instance and the bootstrap resamples
        # instances, not noise. "Divergent verdicts" means the SIGN of this difference is not
        # settled inside the region; a sign that reverses with both CIs excluding zero is a
        # strictly stronger statement than a flip count.
        rng_d = np.random.default_rng(seed("phi/boot"))
        bidx = rng_d.integers(0, S.M, (B_INST, S.M))
        delta = {}
        for nm in pts:
            dd = per[nm]["baseline"] - per[nm]["aggressive"]
            bs = dd[bidx].mean(1)
            q = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
            delta[nm] = {"delta": float(dd.mean()), "ci": q,
                         "excludes_zero": bool(q[0] > 0 or q[1] < 0),
                         "sign": int(np.sign(dd.mean()))}
        s0 = delta["theta_hat"]
        reversed_ = [nm for nm, d in delta.items()
                     if nm != "theta_hat" and d["sign"] == -s0["sign"] and d["sign"] != 0]
        reversed_sig = [nm for nm in reversed_
                        if delta[nm]["excludes_zero"] and s0["excludes_zero"]]
        # The decision-level statement, which is weaker than a sign reversal and is what the
        # study is actually about: at how many theta inside the SAME confidence region would an
        # analyst call the policy comparison resolved? If it is resolved at some theta and not
        # at others, two analysts with equally defensible calibrations write different
        # conclusions from the same data, without either of them ever seeing the sign flip.
        resolved = sorted(nm for nm, d in delta.items() if d["excludes_zero"])
        unresolved = sorted(nm for nm, d in delta.items() if not d["excludes_zero"])
        out["phi"] = {"n_instances": S.M, "n_theta_points": len(pts), "n_flipped": k,
                      "phi": k / S.M, "ci": list(bt.proportion_ci(0.95)),
                      "n_flipped_null": k_null, "phi_null": k_null / S.M,
                      "ci_null": list(binomtest(k_null, S.M).proportion_ci(0.95)),
                      "null_note": "theta held at theta_hat, only the seed base varying, over "
                                   "the same number of points: the share of flips that seed "
                                   "noise alone produces",
                      "n_flipped_vs_theta_hat": int(flips_vs_hat.sum()),
                      "n_tied_at_theta_hat": n_tied, "n_eligible": n_elig,
                      "phi_eligible": (k / n_elig) if n_elig > 0 else None,
                      "ci_eligible": list(bte.proportion_ci(0.95)) if bte else None,
                      "denominator": "all instances (K.3); phi_eligible counts only instances "
                                     "with a strict ordering somewhere in the region",
                      "mean_rate_baseline": float(per["theta_hat"]["baseline"].mean()),
                      "mean_rate_aggressive": float(per["theta_hat"]["aggressive"].mean()),
                      "contrast": delta, "b_instance_boot": B_INST,
                      "sign_reversed_at": reversed_,
                      "sign_reversed_significantly_at": reversed_sig,
                      "n_resolved": len(resolved), "n_unresolved": len(unresolved),
                      "resolved_at": resolved, "unresolved_at": unresolved,
                      "resolved_at_theta_hat": bool(s0["excludes_zero"])}
        save(out, res_file)
        print(f"[8] PHI {k}/{S.M} instances flip their policy ranking across {len(pts)} points "
              f"in the region (seed-only null: {k_null}/{S.M}); aggregate contrast at theta_hat "
              f"{s0['delta']:+.4f} [{s0['ci'][0]:+.4f}, {s0['ci'][1]:+.4f}], sign reverses at "
              f"{len(reversed_)} point(s) ({len(reversed_sig)} with both CIs excluding zero)",
              flush=True)

    out["meta"] = {"n_instances": S.M, "n_point_seeds": N_POINT, "n_search_seeds": N_SEARCH,
                   "n_final_seeds": N_FINAL, "n_alea_blocks": N_ALEA,
                   "de": {"popsize": DE_POPSIZE, "maxiter": DE_MAXITER, "seed": DE_SEED},
                   "suite_evaluations": S.n_eval, "minutes": (time.time() - t_all) / 60,
                   "theta_hat": list(map(float, th)),
                   "config": ("quick" if a.quick else ("mid" if a.mid else "run"))}
    save(out, res_file)
    report(out)
    return 0


def report(out, read_only=False):
    fit = json.load(open(STEP6))["fits"]["loc1"]
    regions = {t: Region(fit, t) for t in fit["regions"]}
    cache = read_cache()
    ns = out.get("meta", {}).get("n_search_seeds", N_SEARCH)
    keys = [f"c{t}" for t in vd.TTC_GRID] + ["crash"]
    # REPORTED interval: the fresh-seed re-evaluation of each searched extremum (K.2).
    # The cache band is a diagnostic and quantifies the winner's curse.
    rep, diag = {}, {}
    for tag in regions:
        rep[tag], diag[tag] = {}, {}
        fin = out.get("final", {}).get(tag, {})
        # The interval is the ENVELOPE over every fresh-seed evaluation at a point inside this
        # region - the extrema searched for all outcomes, plus theta_hat. Two separate searches
        # re-evaluated on fresh seeds can cross when the width is near the Monte Carlo noise, so
        # taking one as "min" and the other as "max" can invert; the envelope cannot.
        # pool EVERY fresh-seed evaluation that lies inside this region, whichever region's
        # search produced it. The three regions are CONCENTRIC with the same shape matrix and
        # radii proportional to sqrt(c), so plain is a subset of generator_only is a subset of
        # robust, exactly - the intervals are nested by construction and cannot invert.
        # Membership of the pool is checked, not assumed: the same seed count and the same seed
        # base, so no endpoint can win the extremum by being noisier than the others; and
        # containment is re-tested here because clipping a searched point to the parameter
        # bounds can push it off its own region's surface.
        nf = (out.get("meta") or {}).get("n_final_seeds", N_FINAL)
        pool, dropped = [], 0
        for g in out.get("final", {}).values():
            for e in g.values():
                if e.get("n_seeds", nf) != nf or e.get("base", "final/crn") != "final/crn":
                    dropped += 1
                    continue
                if regions[tag].contains(e["theta"]):
                    pool.append((e["curve"], e["theta"]))
        if out.get("final_point"):
            pool.append((out["final_point"]["curve"], out["final_point"]["theta"]))
        for k in keys:
            lo, hi = fin.get(f"{k}/min"), fin.get(f"{k}/max")
            if pool:
                vals = [(c[k], th) for c, th in pool if k in c]
                mn, mx = min(vals, key=lambda z: z[0]), max(vals, key=lambda z: z[0])
                # The winner's curse is the search-to-fresh gap for THIS key's own searched
                # extremum. The envelope endpoint can be a different point (another key's
                # extremum, or theta_hat), so whether the two coincide is recorded rather than
                # assumed (S5 review).
                own_lo = bool(lo and np.allclose(mn[1], lo["theta"]))
                own_hi = bool(hi and np.allclose(mx[1], hi["theta"]))
                rep[tag][k] = {"min": mn[0], "max": mx[0], "width": mx[0] - mn[0],
                               "argmin": mn[1], "argmax": mx[1], "n_fresh_points": len(vals),
                               "n_dropped_wrong_seeding": dropped,
                               "search_min": lo["search_value"] if lo else None,
                               "search_max": hi["search_value"] if hi else None,
                               "endpoint_is_own_extremum": (own_lo, own_hi),
                               "winners_curse": ((lo["search_value"] - lo["fresh_value"]) if lo else None,
                                                 (hi["search_value"] - hi["fresh_value"]) if hi else None)}
            b = band(cache, regions[tag], k, ns)
            if b:
                diag[tag][k] = b
    out["interval"], out["cache_band"] = rep, diag
    if "point" not in out:
        print("nothing to report yet: the POINT stage has not run in this results directory",
              flush=True)
        return
    pt = out["point"]["baseline"]
    # A report can be written from a PARTIAL results file - on a control failure (D7.1/D7.2) the
    # run stops and reports before `meta` exists, and `--summary` may be pointed at a run that
    # was interrupted. Fall back to the live configuration and say so, rather than crashing.
    mt = {"n_instances": "?", "n_point_seeds": N_POINT, "n_search_seeds": N_SEARCH,
          "n_final_seeds": N_FINAL, "n_alea_blocks": N_ALEA, "suite_evaluations": "?",
          "minutes": float("nan"), "theta_hat": list(map(float, fit["theta_hat"]))}
    mt.update(out.get("meta") or {})
    f4 = lambda v: ", ".join(f"{x:.3f}" for x in v)
    L = ["# Step 7 - the safety-verdict interval (Amendment K)", ""]
    if "meta" not in out:
        L += ["> **PARTIAL RUN.** This results file has no `meta` block, so the run did not "
              "reach the end - either a pre-registered control stopped it or it was "
              "interrupted. Seed counts below are the ones configured now, not necessarily "
              "the ones that produced every number.", ""]
    L += [f"Suite: {mt['n_instances']} instances (3 families x {N_INST}); "
          f"{mt['suite_evaluations']} suite evaluations; {mt['minutes']:.0f} min. "
          f"Search {mt['n_search_seeds']} seeds, re-evaluation {mt['n_final_seeds']}, "
          f"POINT {mt['n_point_seeds']}.",
          f"theta_hat = ({f4(mt['theta_hat'])})", "",
          "## The verdict interval across the declared threshold grid (robust region)", "",
          "| outcome | V(theta_hat) | W_MC | epistemic interval | W_epi | R_epi/MC |",
          "|---|---|---|---|---|---|"]
    # W_MC is estimated at the POINT stage's seed count; the interval is built from N_FINAL
    # evaluations. The bootstrap resamples seed blocks, so the width scales as 1/sqrt(n_seeds)
    # and the ratio is taken at the interval's own seed count.
    nfin, npt = mt["n_final_seeds"], mt["n_point_seeds"]
    sc = math.sqrt(npt / nfin)
    v_at = (out.get("final_point") or {}).get("curve") or pt["curve"]
    for k in keys:
        r = rep.get("primary_robust", {}).get(k)
        if not r:
            continue
        # Noise floor: a rate over M x n runs cannot be resolved more finely than 1/(M n), so a
        # W_MC at or below that is not an estimate of anything and the ratio built on it would
        # be an artefact of the grid. Reported as n/a rather than as a large number.
        wmc = pt["W_MC"][k] * sc
        # The rescaled width describes an N_FINAL-seed estimate, so the resolution floor is
        # 1/(M x n_final), not 1/(M x n_point).
        floor = (1.0 / max(mt["n_instances"] * nfin, 1)
                 if isinstance(mt["n_instances"], int) else 0.0)
        ratio = "n/a" if wmc <= max(0.0, floor) else f"{r['width'] / wmc:.1f}"
        L.append(f"| {k} | {v_at.get(k, float('nan')):.4f} | {wmc:.4f} | "
                 f"{r['min']:.4f} - {r['max']:.4f} | {r['width']:.4f} | {ratio} |")
    L += ["", f"V(theta_hat) and the interval are all {nfin}-seed evaluations on one common "
          f"seed base, so the point value lies inside the interval by construction. W_MC is the "
          f"POINT stage's 95% Monte Carlo width rescaled from {npt} to {nfin} seeds "
          f"(x{sc:.3f})."]
    wc = [abs(z) for z in (rep.get("primary_robust", {}).get("c2.0", {}).get("winners_curse")
                           or ()) if z is not None]
    own = rep.get("primary_robust", {}).get("c2.0", {}).get("endpoint_is_own_extremum")
    L += ["", "Each endpoint is the envelope over fresh-seed re-evaluations of the searched "
          "extrema inside the region (K.2), so it is a conservative lower bound on the true "
          "interval. For c2.0 the searched-then-re-evaluated gap is "
          f"{f4(wc) if wc else 'n/a'} - the winner's curse of taking an extreme over many "
          "noisy evaluations. That gap belongs to c2.0's own searched extrema; the printed "
          f"endpoints are {'both' if own and all(own) else ('partly' if own and any(own) else 'not')} "
          "those same points (the envelope can be set by another outcome's extremum or by "
          "theta_hat), so the two numbers are reported separately rather than subtracted.", "",
          ]
    # D7.4, operationalised in K.6: the three DE starts are independent optimisations of the
    # same surface. If they disagree by an amount comparable with the interval, the search -
    # not the region - is setting the width, and the interval must be read as a lower bound.
    ep4 = ((out.get("epistemic") or out.get("epistemic_partial") or {})
           .get("primary_robust", {}) or {}).get("extrema", {})
    sp = {k: e["start_spread"] for k, e in ep4.items() if "start_spread" in e}
    if sp:
        # Each extremum's spread is judged against ITS OWN outcome's width, not against c2.0's
        # (S5 review): a spread in the crash rate compared with a conflict-rate width is a
        # comparison of two different scales.
        L += ["", "## D7.4 - did the search converge?", "",
              "| extremum | spread across the 3 independent starts | that outcome's W_epi | "
              "ratio | evaluations |", "|---|---|---|---|---|"]
        ratios = {}
        for k in sorted(sp):
            kk = k.split("/")[0]
            wk = rep.get("primary_robust", {}).get(kk, {}).get("width")
            rr = (sp[k] / wk) if (wk and wk > 0) else None
            if rr is not None:
                ratios[k] = rr
            L.append(f"| {k} | {sp[k]:.4f} | "
                     f"{('%.4f' % wk) if wk is not None else 'n/a'} | "
                     f"{('%.2f' % rr) if rr is not None else 'n/a'} | {ep4[k]['nfev']} |")
        if not ratios:
            L += ["", "No outcome has a positive epistemic width, so there is nothing to compare "
                  "the start spread with and D7.4 is not evaluable at this configuration."]
        else:
            wk_name, worst = max(ratios.items(), key=lambda z: z[1])
            L += ["", f"Worst ratio {worst:.2f} (at {wk_name})" +
                  (". Every start agrees to well inside the width it is being compared with, so "
                   "the reported extrema are the region's, not one start's." if worst < 0.25 else
                   ". **At least one spread is NOT small relative to its own width**: those "
                   "bounds are reported as lower bounds set partly by the search budget, and "
                   "the honest reading is that the true region extremum lies at or beyond the "
                   "endpoint.")]
    L += ["", "## The same interval under the three calibrations (c2.0)", "",
          "| region | c | interval | width | vs the plain region |", "|---|---|---|---|---|"]
    wp = rep.get("plain", {}).get("c2.0", {}).get("width")
    for tag in ("primary_robust", "generator_only", "plain"):
        r = rep.get(tag, {}).get("c2.0")
        if not r:
            continue
        rel = "-" if not wp else f"{r['width'] / wp:.2f}x"
        L.append(f"| {tag} | {fit['regions'][tag]['c']:.3f} | {r['min']:.4f} - {r['max']:.4f} "
                 f"| {r['width']:.4f} | {rel} |")
    skd = {t: (((out.get("epistemic") or out.get("epistemic_partial") or {}).get(t) or {})
               .get("searched_keys")) for t in regions}
    L += ["", "Every region is searched on c2.0 with the same budget, so this comparison is on "
          "an equal footing. Searched outcomes per region: "
          + "; ".join(f"{t}: {len(v) if v else '?'}" for t, v in skd.items())
          + ". At the other thresholds the non-primary regions' widths come from pooled "
          "evaluations rather than their own searches and are lower bounds (K.6)."]
    if "attribution" in out:
        L += ["", "## What drives the width (one parameter at a time, c2.0)", "",
              "| parameter | range | span |", "|---|---|---|"]
        for nm, vals in out["attribution"].items():
            v = [x["curve"]["c2.0"] for x in vals]
            L.append(f"| {nm} | {min(v):.4f} - {max(v):.4f} | {max(v) - min(v):.4f} |")
    if "aleatoric" in out:
        wa = out["aleatoric"]["W_alea"]["c2.0"]
        we = rep.get("primary_robust", {}).get("c2.0", {}).get("width")
        L += ["", f"- **W_alea** (c2.0, {mt['n_alea_blocks']} heterogeneity blocks): "
              f"{wa:.4f}" + ("" if not we or wa <= 0 else f"; **R_epi/alea = {we / wa:.1f}**")]
    if "litrange" in out:
        lo, hi = out["litrange"]["min"]["curve"]["c2.0"], out["litrange"]["max"]["curve"]["c2.0"]
        L.append(f"- **LITRANGE** ({out['litrange']['source']}): c2.0 {lo:.4f} - {hi:.4f} "
                 f"(width {hi - lo:.4f})")
    if "controls" in out:
        c = out["controls"]
        sh = c["shuffle"]
        L += ["", "## Controls", "",
              f"- **OUTSIDE** (positive, best of "
              f"{c['outside'].get('n_directions', 1)} pre-declared directions - "
              f"{c['outside'].get('direction', 'argmax_c2.0')}, achieved radius "
              f"{c['outside'].get('achieved_radius', float('nan')):.2f}x of the requested 3x; "
              f"the parameter bounds can shorten it): changes c2.0 by "
              f"{c['outside']['abs_change']:.4f} against a threshold of "
              f"{c['outside'].get('threshold', c['outside']['W_MC']):.4f} "
              + ("(W_MC)" if c['outside']['W_MC'] >= c['outside'].get('resolution_floor', 0)
                 else "(the 1/(M x n) resolution floor; W_MC bootstrapped to "
                      f"{c['outside']['W_MC']:.4f})") + " -> "
              f"**{'pass' if c['outside']['pass'] else 'FAIL - D7.1 says Step 7 stops and the suite is reported as insensitive'}**",
              f"- **SHUFFLE** (negative): " + (sh.get("note") or
               f"permuted R2 = {sh['r2_permuted_mean']:.5f} (p95 "
               f"{sh['r2_permuted_p95']:.5f}) against the no-association expectation "
               f"p/(n-1) = {sh.get('r2_null_expected', float('nan')):.5f} over "
               f"{sh['n_points']} points and {sh['n_parameters_used']} parameters -> "
               "**" + ("pass" if sh["pass"] else "FAIL - permuting the labels did NOT destroy "
                       "the relation, which indicates leakage") + "**. "
               f"The observed fit is R2 = {sh['r2_observed']:.3f} (one-sided p "
               f"{sh['sensitivity_p_value']:.4f}), which is **descriptive only**: the cached "
               f"points are an optimiser's trajectory, not a random design, so this R2 is not "
               f"a sensitivity test - that is D7.1's job.")]
        if c["outside"].get("probes"):
            L += ["", "| OUTSIDE direction | achieved radius | c2.0 | &#124;dV&#124; |",
                  "|---|---|---|---|"]
            for p in sorted(c["outside"]["probes"], key=lambda z: -z["abs_change"]):
                L.append(f"| {p['direction']} | {p['achieved_radius']:.2f}x | {p['v']:.4f} "
                         f"| {p['abs_change']:.4f} |")
            L += ["", f"{c['outside'].get('n_directions_passing', '?')} of "
                  f"{c['outside'].get('n_directions', '?')} directions exceed the threshold on "
                  f"their own, so the control does not rest on a single lucky line. The "
                  f"threshold is {c['outside'].get('threshold', float('nan')):.4f} = "
                  f"max(W_MC, 1/(M x n) = "
                  f"{c['outside'].get('resolution_floor', float('nan')):.4f}). "
                  + (f"{len(c['outside'].get('rejected_probes') or [])} direction(s) were "
                     f"excluded because the parameter bounds pulled them back to under 1.5x "
                     f"the region radius. " if c["outside"].get("rejected_probes") else "")
                  + (f"Along the within-region c2.0 argmax alone the change is "
                     f"{c['outside']['abs_change_argmax_only']:.4f}; K.7 records why that "
                     f"single direction is uninformative here (a 3x push along it lowers s0 "
                     f"toward more conflict while lowering v0 and pinning b at its bound, and "
                     f"the effects cancel)."
                     if c["outside"].get("abs_change_argmax_only") is not None else "")]
    if "class_delayed" in out:
        cd = out["class_delayed"]
        L += ["", f"- **CLASS (delayed IDM, {cd['bg_lag_s']} s)**: ego-involved collision rate "
              f"{cd['point']['crash']:.4f} against {v_at['crash']:.4f} for plain IDM; "
              f"c2.0 {cd['point']['c2.0']:.4f} against {v_at['c2.0']:.4f} "
              f"(both at {nfin} seeds on the common base)"]
    if "phi" in out:
        p = out["phi"]
        L += ["", f"- **Verdict flips**: {p['n_flipped']} of {p['n_instances']} instances "
              f"(Phi = {p['phi']:.3f}, 95% CI {p['ci'][0]:.3f}-{p['ci'][1]:.3f}) change which ego "
              f"policy is safer somewhere inside the region, across "
              f"{p.get('n_theta_points', '?')} evaluated points"
              + (f". **Seed-only null: {p['n_flipped_null']} of {p['n_instances']} "
                 f"(Phi_null = {p['phi_null']:.3f}, 95% CI {p['ci_null'][0]:.3f}-"
                 f"{p['ci_null'][1]:.3f})** - the same statistic with theta held at theta_hat "
                 f"and only the seed base varying over the same number of points. Phi is read "
                 f"against that floor, not against zero"
                 if "phi_null" in p else "")
              + (f". {p['n_tied_at_theta_hat']} instances tie at theta_hat, which under the "
                 f"corrected definition (K.8) no longer disqualifies them; "
                 f"{p['n_eligible']} have a strict ordering somewhere in the region, and among "
                 f"those the rate is "
                 f"{p['phi_eligible']:.3f} (95% CI {p['ci_eligible'][0]:.3f}-"
                 f"{p['ci_eligible'][1]:.3f}). The headline Phi keeps K.3's denominator (all "
                 f"instances)" if p.get("n_eligible") else "")]
        if "contrast" in p:
            c0 = p["contrast"]["theta_hat"]
            L += ["", "| theta point | baseline - aggressive (c2.0) | 95% CI over instances | "
                  "sign |", "|---|---|---|---|"]
            for nm, d in p["contrast"].items():
                L.append(f"| {nm} | {d['delta']:+.4f} | {d['ci'][0]:+.4f} - {d['ci'][1]:+.4f} "
                         f"| {'+' if d['sign'] > 0 else ('-' if d['sign'] < 0 else '0')}"
                         f"{'' if d['excludes_zero'] else ' (CI spans 0)'} |")
            L += ["", f"At theta_hat the baseline policy's conflict rate is "
                  f"{c0['delta']:+.4f} against the aggressive one"
                  + ("" if c0["excludes_zero"] else " (CI spans zero, so the ordering is not "
                     "resolved even at the point estimate)")
                  + f". The sign reverses at {len(p['sign_reversed_at'])} of "
                  f"{len(p['contrast']) - 1} other points in the region, "
                  f"{len(p['sign_reversed_significantly_at'])} of them with both CIs excluding "
                  f"zero. A reversal with both CIs excluding zero is the strong form of the "
                  f"divergent-verdict claim; a reversal with CIs that span zero says only that "
                  f"the ordering is unresolved."]
            if "n_resolved" in p:
                L += ["", f"**The decision-level reading.** At {p['n_resolved']} of "
                      f"{p['n_resolved'] + p['n_unresolved']} evaluated theta inside the region "
                      f"the policy comparison is RESOLVED (the instance-bootstrap interval "
                      f"excludes zero); at the other {p['n_unresolved']} it is not. At "
                      f"theta_hat itself it is "
                      f"**{'resolved' if p['resolved_at_theta_hat'] else 'NOT resolved'}**."
                      + ((" So two analysts whose calibrations are both inside the same "
                          "confidence region would write different conclusions from the same "
                          "data - one that the policies differ, one that the evidence does not "
                          "separate them - without the sign ever reversing. That is the "
                          "divergent verdict this study is about, stated as a decision rather "
                          "than as a number.")
                         if p["n_resolved"] and p["n_unresolved"] else
                         (" The comparison has the same status everywhere in the region, so at "
                          "this level the verdict does NOT diverge: the width of the interval "
                          "is real, but it does not change what an analyst would conclude "
                          "about these two policies."))]
    n_coll = pt["n_collisions"]
    L += ["", f"- Ego-involved collisions in POINT: {n_coll} of {pt['n_runs']} runs"
          + ("" if n_coll >= 100 else "  -> **under 100, so the collision-rate results are "
             "reported as underpowered, not as a null (D7.3)**")]
    if "other_crash" in pt["curve"]:
        L += [f"- Collisions elsewhere in the string (background pairs, NOT in the outcome): "
              f"rate {pt['curve']['other_crash']:.4f}"
              + (f"; delayed-IDM arm {out['class_delayed']['point']['other_crash']:.4f}"
                 if "class_delayed" in out
                 and "other_crash" in out["class_delayed"]["point"] else "")]
    if read_only:
        # --summary may be pointed at a run that is still going in another process. Writing the
        # results file back would overwrite whatever that process has committed since this one
        # read it, so a read-only report touches neither the results file nor step7.md.
        open(os.path.join(OUT, "step7_summary.md"), "w", encoding="utf-8").write(
            "\n".join(L) + "\n")
    else:
        save(out, os.path.join(OUT, "step7.json"))
        open(os.path.join(OUT, "step7.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


if __name__ == "__main__":
    sys.exit(main())
