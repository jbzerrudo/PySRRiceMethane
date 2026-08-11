"""
7_form_recurrence.py — count which equation FORMS recur across independent seeds
==============================================================================
WHAT IT REPLACES

Rule A picks the seed whose single best candidate scored highest on the 80/20
hold-out. That is a maximum over 12 seeds x 4 candidates = 48 evaluations, so
the winning number is an order statistic, not a performance estimate. Measured
on the Mase run of 2 August 2026: the 12-seed mid-complexity distribution is
+0.4162 +/- 0.2195 and Rule A returned +0.7888, which is 1.70 sd above the mean.
The expected maximum of 12 draws from any symmetric noise is 1.63 sd. Stage 8
then gave +0.579 for the same form.

It also makes one irreproducible run load-bearing. PySR is not bit-reproducible
unless deterministic=True AND parallelism="serial" (pysr/sr.py line 1865), so
"seed 47" cannot be regenerated.

This script implements the selection rule used in the tropical-cyclone paper
instead: run N seeds, count how often each FORM recurs across their Pareto
fronts, and carry forward the forms that recur. A count over independent seeds
degrades gracefully under non-determinism, and it never touches the hold-out, so
it does not contaminate the stage-8 CV the way Rule A does.

HOW "SAME FORM" IS DEFINED

Every numeric constant is replaced by one placeholder C, then the expression is
canonicalised through sympy. So c1*V0*exp(c2*V0*t) from two different seeds with
different fitted constants collapse to one entry, V0*exp(C*V0*t).

VALIDATION, 3 August 2026

Run on the 12 native 6-hourly fronts from the mountains paper, this reproduces
every count that paper reports, without being told any of them:

  complexity 8   exponential present                      8 of 12   (paper: 8 of 12)
                 V0*exp(C*V0*t),        loss  95.754      2 of 12   (paper: 2, loss 95.8)
                 (V0+t)*exp(C*t),       loss 113.158      6 of 12   (paper: 6, loss 113.2)
  complexity 9   C*V0*(V0*t + 1),       loss  97.089      7 of 12   (paper: 7, loss 97.1)
                 V0 - t/(C + C/V0),     loss  96.153      2 of 12   (paper: "algebraically
                                                                     different", loss 96.2)

KNOWN LIMITATION, read this before quoting a count

A single placeholder cannot tell where a constant sits. Two members of the same
algebraic family that park their constants differently get separate entries. At
complexity 7 this split the hyperbolic decay into V0/(C*(t+1)) at 4 of 12 and
C*V0/(C+t) at 2 of 12, which a person reading the fronts would group as one form
at 6 of 12. So the counts are a LOWER BOUND on recurrence. Read the printed form
strings before quoting a number; merging two lines by eye is legitimate and is
what was done by hand for the mountains paper.

USAGE

    python 7_form_recurrence.py <folder>                 # auto-detects layout
    python 7_form_recurrence.py <folder> --min-runs 6    # promotion threshold

Handles both layouts:
    mountains   run_*_seed*/pareto_front.csv
    Paper 1     seed_*/pareto_equations_*.csv

Declare the threshold BEFORE looking at the output. 6 of 12 is the natural
choice: a form a majority of independent seeds find.

Author: Jef Zerrudo / Claude.  Requires pandas, sympy.
==============================================================================
"""

import argparse
import glob
import os
import re
import warnings

import pandas as pd
import sympy as sp

warnings.filterwarnings("ignore")

C = sp.Symbol("C")
SYMPY_LOCALS = {"abs": sp.Abs, "Abs": sp.Abs}


def find_fronts(root):
    """Return [(seed, csv_path), ...] for either layout."""
    hits = []
    for pat, rx in ((os.path.join(root, "**", "pareto_front.csv"), r"seed(\d+)"),
                    (os.path.join(root, "**", "pareto_equations_*.csv"), r"seed_?(\d+)")):
        for p in glob.glob(pat, recursive=True):
            m = re.search(rx, p)
            if m:
                hits.append((int(m.group(1)), p))
    # one front per seed; if a seed has several, take the newest
    best = {}
    for seed, p in hits:
        if seed not in best or os.path.getmtime(p) > os.path.getmtime(best[seed]):
            best[seed] = p
    return sorted(best.items())


def signature(expr_str):
    """(canonical signature, readable form) with every constant mapped to C."""
    e = sp.sympify(expr_str, locals=SYMPY_LOCALS)
    e = e.xreplace({f: C for f in e.atoms(sp.Float)})
    e = e.xreplace({r: C for r in e.atoms(sp.Rational)
                    if r not in (sp.S.One, sp.S.NegativeOne, sp.S.Zero)})
    try:
        e = sp.simplify(e)
    except Exception:
        pass
    return sp.srepr(e), str(e)


def load(root):
    fronts = find_fronts(root)
    if not fronts:
        raise SystemExit(
            f"\n  [STOP] no pareto_front.csv or pareto_equations_*.csv under\n"
            f"         {root}\n")
    rows, failed = [], 0
    for seed, path in fronts:
        d = pd.read_csv(path)
        col = ("sympy_format" if "sympy_format" in d.columns else
               "equation_named" if "equation_named" in d.columns else "equation")
        for _, x in d.iterrows():
            src = x[col] if isinstance(x[col], str) else x["equation"]
            try:
                sig, form = signature(src)
            except Exception:
                failed += 1
                continue
            rows.append(dict(seed=seed, complexity=int(x["complexity"]),
                             loss=float(x["loss"]), sig=sig, form=form,
                             equation=x["equation"]))
    return pd.DataFrame(rows), len(fronts), failed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--min-runs", type=int, default=6,
                    help="promotion threshold, declare it before looking")
    ap.add_argument("--top", type=int, default=3,
                    help="forms to list per complexity")
    a = ap.parse_args()

    f, n_runs, failed = load(a.root)
    print(f"  {n_runs} seeds, {len(f):,} equations, {f.sig.nunique()} distinct forms")
    if failed:
        print(f"  [WARN] {failed} equations could not be parsed and were skipped")
    print(f"  promotion threshold: a form present in >= {a.min_runs} of {n_runs} seeds\n")

    print(f"  {'cplx':>4s} {'runs':>6s} {'best loss':>11s}   form")
    print(f"  {'-'*4} {'-'*6} {'-'*11}   {'-'*56}")
    for cx in sorted(f.complexity.unique()):
        g = f[f.complexity == cx]
        agg = (g.groupby("sig")
                .agg(runs=("seed", "nunique"), loss=("loss", "min"),
                     form=("form", "first"))
                .sort_values(["runs", "loss"], ascending=[False, True]))
        for i, (_, r) in enumerate(agg.iterrows()):
            if i >= a.top:
                break
            mark = " <=" if r.runs >= a.min_runs else "   "
            print(f"  {cx:4d} {r.runs:3d}/{n_runs:<2d} {r.loss:11.4f}{mark} {r.form}")

    promoted = (f.groupby("sig")
                 .agg(runs=("seed", "nunique"), complexity=("complexity", "min"),
                      loss=("loss", "min"), form=("form", "first"))
                 .query("runs >= @a.min_runs")
                 .sort_values(["runs", "loss"], ascending=[False, True]))

    print(f"\n  {'='*74}\n  PROMOTED: {len(promoted)} form(s) present in >= "
          f"{a.min_runs} of {n_runs} seeds\n  {'='*74}")
    if promoted.empty:
        print("\n  Nothing recurs at this threshold. That is itself a result: the seeds\n"
              "  are not agreeing on structure, so no single equation from this run can\n"
              "  be defended as reproducible. Shrink the search space (drop operators\n"
              "  that duplicate each other's role, lower MAXSIZE) and run again before\n"
              "  promoting anything.\n")
    else:
        for _, r in promoted.iterrows():
            print(f"\n  {r.runs}/{n_runs} seeds   complexity {r.complexity}   "
                  f"best loss {r.loss:.4f}")
            print(f"    {r.form}")

    print("\n  Read the form strings before quoting a count. A single placeholder\n"
          "  cannot tell where a constant sits, so one algebraic family can appear\n"
          "  on two lines. Counts are a lower bound. Merging by eye is legitimate.\n")


if __name__ == "__main__":
    main()
