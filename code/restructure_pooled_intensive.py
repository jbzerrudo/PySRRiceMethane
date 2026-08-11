#!/usr/bin/env python3
"""
restructure_pooled_intensive.py
==============================================================================
Restructure the pooled 3-site CSV so a COMMON equation is findable.

THE PROBLEM
    AUC is EXTENSIVE: cm x h accumulated since the record began. Its magnitude is
    set by how long and how wet the record is, not by the process. The three
    sites barely share any of its range, so an equation fitted where AUC is
    +8000 is asked to predict where AUC is -5000, a value it has never seen. It
    cannot transfer, whatever its functional form.

THE FIX
    Divide by elapsed time. AUC/Deltime is MEAN PONDING DEPTH in cm: it stops
    growing with season length, so it is the same kind of number at every site.
    All new columns are obtainable from ORYZA, which simulates depth, plus a
    clock. No fitted constant and no site-specific scaling is introduced.

TWO DIAGNOSTICS ARE PRINTED, AND THEY MEASURE DIFFERENT THINGS
    3-site overlap : the fraction of each site's rows lying inside the interval
                     all three sites share. Low overlap means predicting a
                     held-out site requires extrapolation. This is what breaks
                     transfer.
    eta^2          : the fraction of the variable's total variance that lies
                     BETWEEN sites. High eta^2 means the variable largely
                     encodes which site a row came from, so a pooled equation
                     can use it as a site label instead of as a driver. This is
                     the trap named in B.1.8.

    Measured on your data:

        variable       eta^2   3-site overlap (JP-MSE / PH-IR / SK-CRK)
        AUC            0.595     3% / 12% /   2%
        AUC_wet        0.373    14% / 100% /  4%
        AUC_dry        0.608     0% /  5% / 100%
        hbar           0.714    58% /  2% /  42%
        hbar_wet       0.553    80% /  4% /  55%
        hbar*es        0.657    79% /  8% /  98%
        hbar_wet*es    0.266    87% / 41% / 100%   <-- best on both measures
        fwet           0.732    DISJOINT
        hbar_dry       0.792    DISJOINT

    fwet and hbar_dry have NO value shared by all three sites, so they are
    dropped automatically. SK-CRK is flooded 100% of the time and PH-IR 15%,
    which leaves those two variables with no common ground at all.

WHAT THIS SCRIPT DOES
    1. adds hbar, hbar_wet, hbar_dry, fwet and the two es interactions
    2. drops any new column whose 3-site interval is empty
    3. drops AUC, AUC_wet, AUC_dry   (set DROP_EXTENSIVE = False to keep them)
    4. writes the new CSV and prints the diagnostic table

    Use the output as the GAM-RF union input for the POOLED arm ONLY. Per-site
    runs keep AUC: within one site it is a fine predictor and the transfer
    problem does not arise.

USAGE
    python restructure_pooled_intensive.py [input.csv] [output.csv]

Author: Jef Zerrudo / Claude.  Requires numpy, pandas.
==============================================================================
"""

import sys
import numpy as np
import pandas as pd

# ── CONFIG ───────────────────────────────────────────────────────────────────
IN_CSV  = "POOL_3sites_growingseason_VPDfix_REPAIRED.csv"
OUT_CSV = "POOL_3sites_INTENSIVE.csv"

SITE, TIME, DELTIME, TARGET = "site", "Date", "Deltime", "F_CH4_F"
EXTENSIVE      = ["AUC", "AUC_wet", "AUC_dry"]
DROP_EXTENSIVE = True
DROP_DISJOINT  = True     # drop new columns with no value shared by all sites
MIN_DELTIME    = 24.0     # a mean over less than a day is noise, not a mean


def parse_dates_per_site(d, time_col, site_col):
    """Parse dates one site at a time, inferring day-first or month-first from that
    site's own rows rather than element by element.

    A site whose first field ever exceeds 12 must be day-first; a site whose second
    field ever exceeds 12 must be month-first. Deciding per site is what makes the
    39 per cent of rows with both fields <= 12 recoverable: on their own they are
    ambiguous, but their site is not.
    """
    s = d[time_col].astype(str)
    f = s.str.extract(r"^(\d+)[/-](\d+)[/-](\d+)")
    out = pd.Series(pd.NaT, index=d.index, dtype="datetime64[ns]")
    for site, g in d.groupby(site_col):
        i = g.index
        a = pd.to_numeric(f.loc[i, 0], errors="coerce")
        b = pd.to_numeric(f.loc[i, 1], errors="coerce")
        if a.max() > 12 and b.max() > 12:
            raise SystemExit(f"  [STOP] site {site}: both date fields exceed 12, "
                             f"the convention cannot be inferred. Fix the source file.")
        dayfirst = bool(a.max() > 12) if a.max() > 12 or b.max() > 12 else True
        out.loc[i] = pd.to_datetime(s.loc[i], dayfirst=dayfirst, errors="coerce")
        print(f"  [DATE] {site}: parsed as {'DD/MM' if dayfirst else 'MM/DD'}, "
              f"{out.loc[i].min():%Y-%m-%d} to {out.loc[i].max():%Y-%m-%d}, "
              f"{out.loc[i].dt.floor('D').nunique()} distinct days")
    if out.isna().any():
        print(f"  [WARN] {int(out.isna().sum())} dates failed to parse")
    return out


def eta_squared(d, col):
    """Fraction of total variance lying between sites. Near 1 = a site label."""
    v = pd.to_numeric(d[col], errors="coerce")
    ok = v.notna()
    if ok.sum() < 10:
        return np.nan
    grand = v[ok].mean()
    sst = float(((v[ok] - grand) ** 2).sum())
    ssb = float(sum(len(g) * (g.mean() - grand) ** 2
                    for _, g in v[ok].groupby(d.loc[ok, SITE])))
    return ssb / sst if sst > 0 else np.nan


def overlap(d, col):
    """(lo, hi, {site: fraction inside}) or None if the sites share nothing."""
    v = pd.to_numeric(d[col], errors="coerce")
    ok = v.notna()
    g = v[ok].groupby(d.loc[ok, SITE])
    lo, hi = g.min().max(), g.max().min()
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        return None
    return lo, hi, {s: float(((x >= lo) & (x <= hi)).mean()) for s, x in g}


def report(d, cols, header):
    print(f"\n{header}")
    print(f"  {'variable':14s}{'eta^2':>7s}   3-site overlap")
    for c in cols:
        if c not in d.columns:
            continue
        o = overlap(d, c)
        txt = ("DISJOINT" if o is None else
               "  ".join(f"{s} {100*f:3.0f}%" for s, f in sorted(o[2].items())))
        print(f"  {c:14s}{eta_squared(d, c):7.3f}   {txt}")


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else IN_CSV
    dst = sys.argv[2] if len(sys.argv) > 2 else OUT_CSV

    d = pd.read_csv(src, low_memory=False)
    print(f"read {src}   {d.shape[0]} rows x {d.shape[1]} columns")
    report(d, EXTENSIVE, "BEFORE, the extensive water variables:")

    # sort by site then time so the running fraction below is genuinely running.
    #
    # BUG FIX, 30 July 2026. This line used format="mixed", which is WRONG for this
    # file. The pooled CSV mixes two date conventions: JP-MSE rows are MM/DD/YYYY
    # and PH-IR and SK-CRK rows are DD/MM/YYYY. Mixed inference resolves each
    # element independently, so every date whose day is <= 12 has day and month
    # swapped. That is 39 per cent of the pooled rows (2,752 / 1,556 / 3,037).
    # Parse per site with that site's own convention instead.
    d["__t__"] = parse_dates_per_site(d, TIME, SITE)
    d = d.sort_values([SITE, "__t__"], kind="mergesort").reset_index(drop=True)

    dt = pd.to_numeric(d[DELTIME], errors="coerce")
    usable = dt >= MIN_DELTIME

    for s_col, new in [("AUC", "hbar"), ("AUC_wet", "hbar_wet"), ("AUC_dry", "hbar_dry")]:
        if s_col in d.columns:
            d[new] = (pd.to_numeric(d[s_col], errors="coerce") / dt).where(usable)

    if "depth" in d.columns:
        wet = (pd.to_numeric(d["depth"], errors="coerce") > 0).astype(float)
        d["fwet"] = wet.groupby(d[SITE]).cumsum() / (wet.groupby(d[SITE]).cumcount() + 1.0)

    if "es" in d.columns:
        es = pd.to_numeric(d["es"], errors="coerce")
        for c in ["hbar", "hbar_wet"]:
            if c in d.columns:
                d[f"{c}*es"] = d[c] * es

    added = [c for c in ["hbar", "hbar_wet", "hbar_dry", "fwet", "hbar*es", "hbar_wet*es"]
             if c in d.columns]
    report(d, added, "AFTER, the intensive replacements:")

    disjoint = [c for c in added if overlap(d, c) is None] if DROP_DISJOINT else []
    keep = [c for c in added if c not in disjoint]
    d = d.drop(columns=disjoint + ["__t__"])

    dropped_ext = []
    if DROP_EXTENSIVE:
        dropped_ext = [c for c in EXTENSIVE if c in d.columns]
        d = d.drop(columns=dropped_ext)

    d.to_csv(dst, index=False)

    lost = int(pd.to_numeric(d[keep[0]], errors="coerce").isna().sum()) if keep else 0
    print(f"\nkept new columns  : {keep}")
    print(f"dropped, disjoint : {disjoint if disjoint else 'none'}")
    print(f"dropped, extensive: {dropped_ext if dropped_ext else 'none'}")
    print(f"rows with NaN in the new columns (Deltime < {MIN_DELTIME:.0f} h): {lost}")
    print(f"target unchanged  : sum({TARGET}) = "
          f"{pd.to_numeric(d[TARGET], errors='coerce').sum():.3f}")
    print(f"\nwrote {dst}   {d.shape[0]} rows x {d.shape[1]} columns")
    print("\nPOOLED arm only. Per-site runs keep AUC.")


if __name__ == "__main__":
    main()
