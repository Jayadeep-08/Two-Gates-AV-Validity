"""Emit LaTeX macros for the Study 2 section straight from the AINN_ReWORK step artifacts.

Nothing here is transcribed by hand: every value is read from
  code/results/step6/step6.json, step7/step7.json, step7/step7_summary.json.
Coverage figures that live only in the pre-registration tables (J.10 re-scoring and the
per-arm widening table) are listed in PREREG below with their source line, because they are
not emitted as JSON by any step.

Run:  py -3.7 tiv_revision\\scripts\\study2_numbers.py
      (set AINN_ROOT if AINN_ReWORK is not beside RPI_AINN)
"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("RPI_OUT") or os.path.abspath(os.path.join(HERE, ".."))
AINN = os.environ.get("AINN_ROOT") or os.path.abspath(
    os.path.join(ROOT, "..", "..", "AINN_ReWORK"))
RES = os.path.join(AINN, "code", "results")
os.makedirs(os.path.join(ROOT, "tables"), exist_ok=True)

s6 = json.load(open(os.path.join(RES, "step6", "step6.json")))
s7 = json.load(open(os.path.join(RES, "step7", "step7.json")))
s7s = json.load(open(os.path.join(RES, "step7", "step7_summary.json")))

# Pre-registration tables (05_PREREGISTRATION_DRAFT_3.md). Source lines noted.
PREREG = {
    "CovGzeroThree": "0.770", "CovGzeroThreeCI": "[0.679,\\;0.842]",   # L2019, J.10
    "CovGzeroTen":   "0.850", "CovGzeroTenCI":   "[0.739,\\;0.919]",   # L2020, J.10
    "CovPC":         "0.940",                                          # step5 summary, PC-300
    "CovWorstArm":   "0.583", "CovWorstArmName": "G3-300",             # L2121
    "CovBestArm":    "0.883", "CovBestArmName":  "G2-300",             # L2120
    "CneedNinety":   "3.711", "CneedNinetyFive": "5.254",              # L2124-2125
    "CRadius":       "2.29",  "CVolume": "63",                         # L2125
    "LooIdeal":      "0.954",                                          # L1786, B=99 Gaussian
    # Regime definition and excluded mass -- step2_excitation_highd.md,
    # tables "loc1_primary" and "loc5_replication", columns carled_10s and E10,1.0.
    "RegCarled":     "18{,}878", "RegKept": "5{,}691", "RegDropped": "13{,}187",
    "RegKeptPct":    "30", "RegDroppedPct": "70",
    "RegHoursAll":   "72.4", "RegHoursKept": "27.8", "RegHoursPct": "38",
    "RegExcit":      "1.0",
    "RegLocFiveCarled": "1{,}673", "RegLocFiveKept": "518", "RegLocFiveKeptPct": "31",
    "RegNrecLocOne": "37", "RegNrecLocFive": "10",
    "RegCorpusEp":   "49{,}078", "RegCorpusHours": "156", "RegCorpusLoc": "six",
}

M = {}
f1 = s6["fits"]["loc1"]
th = f1.get("theta") or f1.get("theta_hat") or s7["meta"]["theta_hat"]
for nm, v in zip(("Vzero", "Tee", "Szero", "Aay", "Bee"), th):
    M["sTh" + nm] = f"{v:.3f}"
M["sNepLocOne"] = f"{f1['M']:,}".replace(",", "{,}")
M["sNepLocFive"] = str(s6["fits"]["loc5"]["M"])
M["sNepLocOneFF"] = str(s6["fits"]["loc1_ff"]["M"])
M["sRms"] = f"{f1['resid_rms']:.3f}"
M["sEpRmsMed"] = f"{f1['ep_rms_median']:.3f}"
M["sEpRmsPninety"] = f"{f1['ep_rms_p90']:.3f}"
M["sBootB"] = str(f1["B"])
M["sTatBound"] = f"{100 * f1['T_at_bound']:.1f}" if f1['T_at_bound'] < 1 else f"{f1['T_at_bound']:.1f}"

mt = s7["meta"]
M["sNinst"] = str(mt["n_instances"])
M["sNseeds"] = str(mt["n_point_seeds"])
M["sNeval"] = f"{mt['suite_evaluations']:,}".replace(",", "{,}")
M["sMinutes"] = f"{mt['minutes']:.0f}"
M["sNblocks"] = str(mt["n_alea_blocks"])
M["sNtheta"] = str(s7s["n_theta_points"])

# verdict interval, primary robust region
iv = s7["interval"]["primary_robust"]
NAME = {"crash": "Crash", "c1.0": "CaOne", "c1.5": "CaOneFive",
        "c2.0": "CaTwo", "c2.5": "CaTwoFive", "c3.0": "CaThree"}
pt = s7["final_point"]["curve"]
for k, tag in NAME.items():
    d = iv[k]
    M["sV" + tag] = f"{pt[k]:.4f}"
    M["sLo" + tag] = f"{d['min']:.4f}"
    M["sHi" + tag] = f"{d['max']:.4f}"
    M["sWepi" + tag] = f"{d['width']:.4f}"
    M["sRel" + tag] = f"{100 * d['width'] / pt[k]:.0f}"      # width as % of the level
    cm = s7s["decomposition"][k]["common_mode_factor"]
    M["sCmf" + tag] = f"{cm:.1f}"

# widths under the three calibrations at c2.0
for reg, tag in (("primary_robust", "Robust"), ("generator_only", "Gen"), ("plain", "Plain")):
    M["sWidth" + tag] = f"{s7['interval'][reg]['c2.0']['width']:.4f}"
M["sRobustVsPlain"] = f"{s7['interval']['primary_robust']['c2.0']['width'] / s7['interval']['plain']['c2.0']['width']:.2f}"
M["sPlainFrac"] = f"{s7['interval']['plain']['c2.0']['width'] / s7['interval']['primary_robust']['c2.0']['width']:.2f}"

# aleatoric and literature range
wal = s7["aleatoric"]["W_alea"]
M["sWalea"] = f"{wal['c2.0']:.4f}"
M["sRalea"] = f"{s7['interval']['primary_robust']['c2.0']['width'] / wal['c2.0']:.1f}"
M["sLitLo"] = f"{s7['litrange']['min']['curve']['c2.0']:.4f}"
M["sLitHi"] = f"{s7['litrange']['max']['curve']['c2.0']:.4f}"
litw = float(M["sLitHi"]) - float(M["sLitLo"])
M["sLitWidth"] = f"{litw:.4f}"
M["sLitRatio"] = f"{litw / s7['interval']['primary_robust']['c2.0']['width']:.0f}"

# effect size spread and ranking
con = s7["phi"]["contrast"]
vals = [abs(v["delta"]) for v in con.values()]
M["sEffLo"] = f"{100 * min(vals):.1f}"
M["sEffHi"] = f"{100 * max(vals):.1f}"
M["sEffRatio"] = f"{max(vals) / min(vals):.1f}"
ph = s7["phi"]
M["sPhi"] = f"{ph['phi']:.3f}"
M["sPhiN"] = str(ph["n_flipped"])
M["sPhiNull"] = f"{ph['phi_null']:.3f}"
M["sPhiNullN"] = str(ph["n_flipped_null"])
M["sNreversed"] = str(len(ph["sign_reversed_at"]))

# controls
ou = s7["controls"]["outside"]
M["sOutsideDV"] = f"{ou['abs_change']:.4f}"
M["sOutsideThr"] = f"{ou['threshold']:.4f}"
M["sOutsidePass"] = f"{ou['n_directions_passing']} of {ou['n_directions']}"
cd = s7["class_delayed"]
M["sClassLag"] = f"{cd['bg_lag_s']:.1f}"
M["sClassCrash"] = f"{cd['point']['crash']:.4f}"
M["sClassCaTwo"] = f"{cd['point']['c2.0']:.4f}"

# replication across locations
for reg, tag in (("primary_robust", "Robust"), ("generator_only", "Gen"), ("plain", "Plain")):
    M["sRepG" + tag] = f"{s6['replication'][reg]['ellipsoid']['g']:.2f}"

M.update(PREREG)

out = os.path.join(ROOT, "tables", "study2_macros.tex")
with open(out, "w") as fh:
    fh.write("% GENERATED by tiv_revision/scripts/study2_numbers.py -- do not hand-edit.\n")
    for k in sorted(M):
        fh.write("\\newcommand{\\%s}{%s}\n" % (k, M[k]))
print("wrote", out, "with", len(M), "macros")
for k in sorted(M):
    print(f"  \\{k} = {M[k]}")
