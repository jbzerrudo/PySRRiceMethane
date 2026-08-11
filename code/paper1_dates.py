"""
paper1_dates.py — one date parser for the whole Paper 1 pipeline
==============================================================================
WHY THIS EXISTS

Your three site files do not share a date convention:

    JPN-MSE_cm.csv                        MM/DD/YYYY   (field 1 runs 4 to 9)
    KORCRK_2018_..._growingseason.csv     DD/MM/YYYY   (field 1 runs 1 to 31)
    PHLIR_2016_cleaned.csv                DD/MM/YYYY   (field 1 runs 1 to 31)

The pooled CSV concatenates all three, so it carries BOTH conventions in one
column. 7,345 of its 19,026 rows have both fields at 12 or under and are
therefore ambiguous when read on their own.

`parse_dates_robust()` in PySR7v2.py, NNRF_diagnostics.py and the stage-8 script
picks ONE dayfirst flag for the whole column using a monotonicity score. Verified
30 July 2026:

    PER-SITE FILES   correct at all three sites. The monotonicity criterion picks
                     dayfirst=False for Mase (100.00% against 99.87%) and
                     dayfirst=True for Cheorwon and IRRI (100.00% against 99.87
                     and 99.86). Nothing in the per-site runs is affected.

    POOLED FILE      WRONG for JP-MSE. One flag cannot serve two conventions, so
                     it chooses dayfirst=True, which is right for PH-IR and
                     SK-CRK and wrong for JP-MSE. Mase rows are reported as
                     2012-01-05 to 2012-12-08 instead of 2012-04-16 to
                     2012-09-10. The day COUNT survives, 148 either way, because
                     the swap is a bijection, but the day GROUPING does not:
                     rows from one real day are scattered across several, and
                     rows from different days are merged.

Anything that groups by day on the pooled file is therefore misgrouping one
third of its rows. That includes day-grouped CV folds and the LOSO day blocks.

THE FIX

Decide the convention per site rather than per file. Each site is internally
consistent, so within a site the ambiguity is resolvable even where an individual
timestamp is not.

USAGE

    from paper1_dates import parse_dates

    df["__t__"] = parse_dates(df)                       # uses "Date" and "site"
    df["__t__"] = parse_dates(df, time_col="Date")      # no site column: global
    df["DAY"]   = df["__t__"].dt.normalize()            # correct day grouping

Drop-in replacement for parse_dates_robust: with no `site` column the behaviour
is identical to the existing function, so per-site scripts are unaffected.

Author: Jef Zerrudo / Claude.  Requires numpy, pandas.
==============================================================================
"""

import re
import warnings

import numpy as np
import pandas as pd

__all__ = ["parse_dates", "infer_convention", "audit_dates"]

_SPLIT = re.compile(r"^\s*(\d+)[/-](\d+)[/-](\d+)")


def _mono_score(parsed):
    """Fraction of consecutive gaps that do not go backwards. The correct
    convention on a chronologically ordered file scores 1.0."""
    pv = parsed.dropna()
    d = pv.diff().dropna()
    if len(d) == 0:
        return 0.0
    return float((d >= pd.Timedelta(0)).sum()) / len(d)


def infer_convention(series):
    """Return (dayfirst, reason).

    Field ranges decide it outright when one field exceeds 12, because only a day
    can. That is exact and needs no heuristic. Otherwise fall back to the
    monotonicity score, which is what the existing pipeline uses.
    """
    s = series.astype(str).str.strip()
    f = s.str.extract(_SPLIT)
    a = pd.to_numeric(f[0], errors="coerce")
    b = pd.to_numeric(f[1], errors="coerce")

    if a.notna().any() and b.notna().any():
        a_big, b_big = bool(a.max() > 12), bool(b.max() > 12)
        if a_big and not b_big:
            return True, f"field 1 reaches {int(a.max())}, so it is the day"
        if b_big and not a_big:
            return False, f"field 2 reaches {int(b.max())}, so it is the day"
        if a_big and b_big:
            raise ValueError("both date fields exceed 12; the column mixes "
                             "conventions within itself and cannot be resolved")

    scores = {}
    for flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores[flag] = _mono_score(
                pd.to_datetime(s, errors="coerce", dayfirst=flag, format="mixed"))
    best = max(scores, key=lambda k: scores[k])
    return best, (f"monotonicity {scores[best]:.2%} against "
                  f"{scores[not best]:.2%}, no field above 12")


def parse_dates(df, time_col="Date", site_col="site", verbose=True):
    """Parse `time_col` to datetime, one convention per site.

    With no `site_col` in the frame, resolves the whole column at once, which
    reproduces the existing parse_dates_robust behaviour.
    """
    s = df[time_col].astype(str).str.strip()
    out = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns]")

    groups = ([(None, df.index)] if site_col not in df.columns
              else [(k, g.index) for k, g in df.groupby(site_col, sort=True)])

    for site, idx in groups:
        dayfirst, reason = infer_convention(s.loc[idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out.loc[idx] = pd.to_datetime(s.loc[idx], errors="coerce",
                                          dayfirst=dayfirst, format="mixed")
        if verbose:
            v = out.loc[idx]
            name = "(whole column)" if site is None else str(site)
            print(f"  [DATE] {name:<10s} {'DD/MM' if dayfirst else 'MM/DD'}  "
                  f"{v.min():%Y-%m-%d} to {v.max():%Y-%m-%d}  "
                  f"{v.dt.normalize().nunique()} days  "
                  f"[{reason}]")

    bad = int(out.isna().sum())
    if bad:
        print(f"  [WARN] {bad} timestamps failed to parse and are NaT")
    return out


def audit_dates(df, time_col="Date", site_col="site"):
    """Print what each site's raw fields look like without parsing anything.
    Use this before trusting any file you have not seen before."""
    s = df[time_col].astype(str).str.strip()
    f = s.str.extract(_SPLIT)
    a = pd.to_numeric(f[0], errors="coerce")
    b = pd.to_numeric(f[1], errors="coerce")
    groups = ([(None, df.index)] if site_col not in df.columns
              else [(k, g.index) for k, g in df.groupby(site_col, sort=True)])
    print(f"  {'site':<12s}{'field 1':>12s}{'field 2':>12s}   convention   ambiguous rows")
    for site, idx in groups:
        amb = int(((a[idx] <= 12) & (b[idx] <= 12)).sum())
        try:
            dayfirst, _ = infer_convention(s.loc[idx])
            conv = "DD/MM" if dayfirst else "MM/DD"
        except ValueError:
            conv = "UNRESOLVABLE"
        print(f"  {str(site):<12s}{f'{int(a[idx].min())}-{int(a[idx].max())}':>12s}"
              f"{f'{int(b[idx].min())}-{int(b[idx].max())}':>12s}   {conv:<12s} "
              f"{amb:,} of {len(idx):,}")
