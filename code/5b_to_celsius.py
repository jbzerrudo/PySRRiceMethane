"""
5b_to_celsius.py — express Tv_K in degrees Celsius before PySR
==============================================================================
WHERE IT SITS

Between box (5) and box (6). It touches ONLY the PySR input file. Nothing in
boxes (0) to (5) has to be re-run, because subtracting a constant is a location
shift and every one of those steps is shift-invariant:

  box (0)  the dependency check centres and scales before the SVD
  box (1)  PIMP and Boruta are random-forest based; a shift is monotone, so the
           split partition and the permutation importances are identical
  box (2)  selects column names only
  box (3)  the VIF is CENTRED because VIF_ADD_CONSTANT = True, so the intercept
           of the auxiliary regression absorbs the shift exactly and the VIF is
           unchanged. It was the ABSENCE of that constant that let the 273 K
           offset destroy Tv_K in the first place.
  box (4)  as box (1)
  box (5)  as box (2)

Box (6) is the only step that sees the difference, because PySR fits raw
constants instead of a centred design matrix.

WHY

Mase, seed 47, 2 August 2026: exp( appears ZERO times in all 28 equations of the
Pareto front, complexity 1 through 35. It was not simplified away, it never
entered.

Tv_K has mean/sd = 48.56, so exp(b * Tv_K) needs b near 2.6e-19 to stay finite.
PySR initialises constants near O(1) and cannot reach that magnitude, so the
exponential is never competitive. The search bought the temperature signal with
tanh(0.027 * AUC_wet - Tv_K) instead, which is 95.85% saturated: a late-season
switch on accumulated ponding, not a temperature response.

July's Mase run carried bare VPD, mean/sd 1.12, and produced 16 exp occurrences.
The difference is scaling, not physics.

WHY 273.15 AND NOT THE SAMPLE MEAN

  * It is a physical constant, not a quantity estimated from the data, so there
    is no leakage into the CV folds and nothing to declare beyond one sentence.
  * It is identical at all four arms, so the pooled equation stays directly
    comparable with the three site equations.
  * The equation then reads in degrees Celsius, and a fitted exp(b * T_C) gives
    Q10 = exp(10 * b) straight off the coefficient.

Centring on the sample mean would do none of these three things.

WHAT IT DOES NOT DO

It does not guarantee an exponential appears. It removes the numerical barrier
that made one unreachable. PySR may still prefer tanh. That is now a result
about the data rather than an artefact of the units.

USAGE

    python 5b_to_celsius.py JPN_retvars_pass2.csv

Writes <name>_C.csv beside the original and leaves the original untouched.
Point INPUT_FILE in 6_PySR.py (line ~103) at the _C file.

The scan runs on every arm and prints mean/sd for all predictors, so if another
arm retained a second large-offset variable you will see it before PySR does.

Author: Jef Zerrudo / Claude.  Requires numpy, pandas.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

SRC_COL = "Tv_K"
DST_COL = "Tv_C"
ZERO_C = 273.15

# Plausible virtual air temperature in kelvin. Outside this the column is not
# what the script thinks it is, or the file has already been converted.
K_RANGE = (250.0, 330.0)

# Above this, exp(b * x) needs a constant PySR's optimiser cannot reach from an
# O(1) initialisation. Tv_K measured 48.56 at Mase; VPD, which did produce
# exponentials in July, measured 1.12.
RATIO_WARN = 10.0

MISSING_FLAGS = [-9999, -999900, -99999]

NOT_PREDICTORS = {"site", "w", "Date", "Deltime", "time",
                  "F_CH4_F", "F_CH4_F_orig"}


def ratio(v):
    """mean / sd, the quantity that decides whether exp(b * x) is reachable."""
    v = v.to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size < 2:
        return np.nan
    sd = float(v.std(ddof=1))
    return np.inf if sd == 0.0 else abs(float(v.mean())) / sd


def scan(d, title):
    print(f"\n  {title}")
    print(f"  {'predictor':<22s}{'mean':>12s}{'sd':>12s}{'|mean|/sd':>12s}")
    rows = []
    for c in d.columns:
        if c in NOT_PREDICTORS:
            continue
        v = pd.to_numeric(d[c], errors="coerce")
        if not np.isfinite(v.to_numpy(float)).any():
            continue
        rows.append((c, float(v.mean()), float(v.std(ddof=1)), ratio(v)))
    rows.sort(key=lambda r: (-r[3] if np.isfinite(r[3]) else -1e18))
    for c, m, s, r in rows:
        flag = "  <-- exp unreachable" if np.isfinite(r) and r > RATIO_WARN else ""
        print(f"  {c:<22s}{m:>12.4g}{s:>12.4g}{r:>12.2f}{flag}")
    return rows


def main():
    if len(sys.argv) < 2:
        raise SystemExit("\n  usage: python 5b_to_celsius.py <pass-2 retvars csv>\n")
    src = sys.argv[1]
    if not os.path.isfile(src):
        raise SystemExit(f"\n  [STOP] not found: {src}\n")
    root, ext = os.path.splitext(src)
    dst = sys.argv[2] if len(sys.argv) > 2 else f"{root}_C{ext}"

    d = pd.read_csv(src, low_memory=False)
    print(f"  read  {src}")
    print(f"        {d.shape[0]:,} rows x {d.shape[1]} columns")

    if DST_COL in d.columns:
        raise SystemExit(
            f"\n  [STOP] '{DST_COL}' is already present. This file has been\n"
            f"         converted. Converting twice would subtract 546.30 K.\n")
    if SRC_COL not in d.columns:
        raise SystemExit(
            f"\n  [STOP] no '{SRC_COL}' column in this file.\n"
            f"         Either the arm did not retain it through box (5), or the\n"
            f"         path points at the wrong file. Columns present:\n"
            f"         {[c for c in d.columns if c not in NOT_PREDICTORS]}\n")

    raw = d[SRC_COL]
    v = pd.to_numeric(raw, errors="coerce")

    # A flag value shifted by 273.15 becomes an ordinary-looking number and is
    # then unrecoverable, so stop rather than guess what was intended.
    hits = {f: int((v == f).sum()) for f in MISSING_FLAGS}
    hits = {f: n for f, n in hits.items() if n}
    if hits:
        raise SystemExit(
            f"\n  [STOP] {SRC_COL} still contains missing flags: {hits}\n"
            f"         Shifting them by {ZERO_C} would disguise them as real\n"
            f"         temperatures. Clean them to NaN first, then re-run.\n")

    med = float(v.median())
    if not (K_RANGE[0] <= med <= K_RANGE[1]):
        raise SystemExit(
            f"\n  [STOP] median {SRC_COL} is {med:.2f}, outside {K_RANGE} K.\n"
            f"         That column is not virtual temperature in kelvin, or the\n"
            f"         file is already in Celsius. Nothing written.\n")

    n_nan = int(v.isna().sum())
    before = ratio(v)
    print(f"\n  {SRC_COL}: min {v.min():.2f}  median {med:.2f}  max {v.max():.2f} K"
          f"   ({n_nan:,} NaN)")

    scan(d, "BEFORE, sorted by |mean|/sd")

    # Convert in place so column ORDER is preserved, then rename.
    d[SRC_COL] = v - ZERO_C
    d = d.rename(columns={SRC_COL: DST_COL})
    after = ratio(d[DST_COL])

    print(f"\n  {DST_COL}: min {d[DST_COL].min():.2f}  median "
          f"{d[DST_COL].median():.2f}  max {d[DST_COL].max():.2f} degC")
    print(f"  |mean|/sd  {before:.2f}  ->  {after:.2f}"
          f"   (VPD was 1.12 in July, when 16 exp terms appeared)")
    if after > RATIO_WARN:
        print(f"  [WARN] still above {RATIO_WARN:.0f}. The shift did not do enough.")

    others = [(c, r) for c, _, _, r in scan(d, "AFTER, sorted by |mean|/sd")
              if np.isfinite(r) and r > RATIO_WARN]
    if others:
        print(f"\n  [WARN] {len(others)} other predictor(s) still have a large "
              f"offset relative to their spread:")
        for c, r in others:
            print(f"         {c:<22s} |mean|/sd = {r:.2f}")
        print("         exp() of these is unreachable for the same reason. Decide\n"
              "         per variable whether a declared shift is defensible.")

    d.to_csv(dst, index=False)
    print(f"\n  wrote {dst}")
    print(f"        {d.shape[0]:,} rows x {d.shape[1]} columns")
    print(f"\n  Point INPUT_FILE in 6_PySR.py at this file. Equations will report")
    print(f"  {DST_COL} in degC, and a fitted exp(b * {DST_COL}) gives Q10 = exp(10*b).")
    print(f"\n  Declare in the methods: virtual temperature was expressed in degrees")
    print(f"  Celsius (Tv_K - 273.15) before symbolic regression. It is a location")
    print(f"  shift by a physical constant, so it changes no VIF, no importance and")
    print(f"  no fold assignment; it only brings the exponential within the reach of")
    print(f"  the constant optimiser.")


if __name__ == "__main__":
    main()
