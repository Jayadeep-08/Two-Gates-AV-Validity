r"""Post-build sanity checks for the manuscript.

Three checks, run against the build log and the sources:
  1. undefined references and citations   -- catches a float or bibitem deleted by an edit
  2. orphaned labels                      -- catches a \ref deleted while its \label survives;
                                             LaTeX never warns about this
  3. overfull boxes above a threshold     -- catches content too wide for the column

Run after every structural edit, not only before submission:
    py -3.7 tiv_revision\\scripts\\check_build.py [logfile]
Exit code: 0 all clear, 1 a check failed, 2 the build needs another pdflatex pass.
"""
import glob, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "combined.log")
OVERFULL_PT = 5.0
fails = 0

if not os.path.exists(LOG):
    sys.exit("no log at %s -- build first" % LOG)
log = open(LOG, encoding="utf-8", errors="replace").read()

rerun = ("Rerun to get cross-references right" in log
         or "Label(s) may have changed" in log
         or "Rerun to get citations correct" in log)
if rerun:
    print("BUILD INCOMPLETE -- LaTeX asked to be run again.")
    print("Every cross-reference reads as undefined until the .aux file is written,")
    print("so the list below would be noise. Run pdflatex once more, then re-check:")
    print("    pdflatex combined && pdflatex combined")
    print("    py -3.7 scripts\\check_build.py combined.log")
    sys.exit(2)

print("1. undefined references and citations")
und = sorted(set(re.findall(r"(?:Reference|Citation) `([^']+)' on page \d+ undefined", log)))
if und:
    fails += 1
    for u in und: print("   MISSING  %s" % u)
else:
    print("   none")

print("2. orphaned labels (defined, never cited)")
src = ""
for f in glob.glob(os.path.join(ROOT, "sections", "*.tex")) + \
         glob.glob(os.path.join(ROOT, "tables", "*.tex")) + \
         [os.path.join(ROOT, "combined.tex")]:
    src += open(f, encoding="utf-8", errors="replace").read()
labels = set(re.findall(r"\\label\{([^}]+)\}", src))
refs = set(re.findall(r"\\(?:ref|eqref|autoref|pageref)\{([^}]+)\}", src))
orphans = sorted(l for l in labels - refs if not l.startswith("sec:"))
if orphans:
    fails += 1
    for o in orphans: print("   ORPHAN   %s  (defined but never cited)" % o)
else:
    print("   none")
unsec = sorted(l for l in labels - refs if l.startswith("sec:"))
if unsec:
    print("   (uncited section labels, usually fine: %s)" % ", ".join(unsec))

print("3. overfull boxes above %.0f pt" % OVERFULL_PT)
big = sorted((float(x) for x in re.findall(r"Overfull \\hbox \(([0-9.]+)pt", log)
              if float(x) > OVERFULL_PT), reverse=True)
if big:
    fails += 1
    for b in big[:10]: print("   OVERFULL %.1f pt" % b)
else:
    print("   none")

print("\n%s" % ("FAILED -- %d check(s)" % fails if fails else "all checks pass"))
sys.exit(1 if fails else 0)
