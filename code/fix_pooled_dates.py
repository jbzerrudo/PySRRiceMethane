"""
fix_pooled_dates.py — rewrite a pooled CSV's Date column to ISO format
==============================================================================
WHY

The pooled file carries two date conventions in one column: JP-MSE rows are
MM/DD/YYYY and PH-IR and SK-CRK rows are DD/MM/YYYY. 7,345 of 19,026 rows have
both fields at 12 or under, so they are ambiguous read on their own.

6_PySR.py parses dates in two places, each picking ONE dayfirst flag for the
whole column:

    line ~1214   the day-grouped 80/20 hold-out that selects the Rule A seed
    line ~548    the parser written into the generated stage-8 CV script

One flag cannot serve two conventions, so on the pooled arm one third of the
rows land on the wrong day. That corrupts the day grouping in both the seed
selection and the 5-fold CV, which are exactly the numbers you report.

THE FIX

Rewrite the Date column as ISO 8601, YYYY-MM-DD HH:MM, and sort by site then
time so the file is strictly chronological within each site.

Verified chain, tested on your data:
  * pandas reads ISO correctly with dayfirst=False.
  * With dayfirst=True it still swaps ambiguous days, a pandas quirk of
    format="mixed", so ISO alone is not enough.
  * BUT parse_dates_robust picks by monotonicity, and on the ISO file it scores
    dayfirst=False at 100.00% against dayfirst=True at 99.87%, so it chooses
    correctly and all three site windows come out right.

AFTER RUNNING, CHECK THIS in the PySR console. It must say dayfirst=False:

    dayfirst=False  parsed=19026/19026  monotonic=100.00%   <-- chosen

If it ever chooses dayfirst=True on this file, stop; the day grouping is wrong.

The three per-site files do NOT need this. Each is internally consistent and
your existing parser resolves all three correctly, verified.

USAGE

    python fix_pooled_dates.py POOLED_retvars_pass2.csv

Writes <name>_ISO.csv beside the original and leaves the original untouched.
Point 6_PySR.py at the _ISO file for the pooled arm.

Author: Jef Zerrudo / Claude.  Requires numpy, pandas, paper1_dates.py
==============================================================================
"""

import os
import sys

import pandas as pd

from paper1_dates import parse_dates, audit_dates

TIME_COL = "Date"
SITE_COL = "site"


def main():
    if len(sys.argv) < 2:
        raise SystemExit("\n  usage: python fix_pooled_dates.py <pooled csv>\n")
    src = sys.argv[1]
    if not os.path.isfile(src):
        raise SystemExit(f"\n  [STOP] not found: {src}\n")
    root, ext = os.path.splitext(src)
    dst = sys.argv[2] if len(sys.argv) > 2 else f"{root}_ISO{ext}"

    d = pd.read_csv(src, low_memory=False, dtype={TIME_COL: str})
    print(f"  read {src}")
    print(f"       {d.shape[0]:,} rows x {d.shape[1]} columns\n")

    if SITE_COL not in d.columns:
        raise SystemExit(
            f"\n  [STOP] no '{SITE_COL}' column. This script is for the POOLED file.\n"
            "         The three per-site files do not need it.\n")

    audit_dates(d, TIME_COL, SITE_COL)
    print()
    t = parse_dates(d, TIME_COL, SITE_COL)

    if t.isna().any():
        raise SystemExit(f"\n  [STOP] {int(t.isna().sum())} dates failed to parse.\n")

    # Chronological order within each site is the check that the parse is right
    print()
    ok = True
    for s, g in t.groupby(d[SITE_COL]):
        mono = float((g.diff().dropna() >= pd.Timedelta(0)).mean())
        flag = "OK" if mono > 0.999 else "*** OUT OF ORDER ***"
        print(f"  {s:8s} chronological within the file: {mono:.2%}   {flag}")
        ok &= mono > 0.999
    if not ok:
        print("\n  [WARN] a site is not in chronological order. The parse may still be")
        print("         right if the source file was never sorted, but check it.")

    d[TIME_COL] = t
    d = d.sort_values([SITE_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    d[TIME_COL] = d[TIME_COL].dt.strftime("%Y-%m-%d %H:%M")
    d.to_csv(dst, index=False)

    print(f"\n  wrote {dst}")
    print(f"        Date is now ISO 8601, so dayfirst no longer applies.")
    print(f"        Point 6_PySR.py INPUT_FILE at this file for the POOLED arm.")
    print(f"\n  CHECK in the PySR console that the parser reports:")
    print(f"        dayfirst=False  parsed={len(d)}/{len(d)}  monotonic=100.00%   <-- chosen")


if __name__ == "__main__":
    main()
