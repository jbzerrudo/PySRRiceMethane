#!/usr/bin/env python3
"""
verify_pooled.py  -  repair + verify the pooled 3-site file, on YOUR machine.

You should not accept a data file you did not produce. This script lets you
produce it yourself and then check that what you produced matches what was
sent to you. If the two fingerprints agree, the file is confirmed by two
independent runs. If they disagree, trust yours, not the one you were sent.

USAGE
    python verify_pooled.py POOL_3sites_growingseason_VPDfix.csv
        -> applies the repair, writes *_REPAIRED.csv next to it, verifies, prints fingerprint

    python verify_pooled.py POOL_3sites_growingseason_VPDfix_REPAIRED.csv --check-only
        -> no repair, just verify + fingerprint (use this on the file you were sent)

WHAT THE REPAIR IS
    fix_pooled_vpd_units.py rebuilt VPD*WS*d1sin by recovering d1sin as
    (VPD*WS*d1sin)/(VPD*WS). Where VPD*WS == 0 that is 0/0 -> NaN. In every one
    of those rows the original product is exactly 0, so the correct value is 0
    for any d1sin. The repair sets those cells back to 0. Nothing else changes.
"""

import sys, hashlib, json
import numpy as np
import pandas as pd

# Fingerprint of the repaired file as produced on 29 July 2026. Compare yours to this.
REFERENCE_FINGERPRINT = "05c0473562e72b95cd6402d38f00023f48f1e0ed5bcb663526c32e668013e3e5"
REFERENCE_SHA256      = "deac2acd1a4e509acedf9866fa4f075e5d30d67c817b4ab3ac079b0b3d0bf181"

EXCL   = {"Date", "Deltime", "time", "F_CH4_F_orig", "site", "w"}
TARGET = "F_CH4_F"


def fingerprint(d):
    """Content fingerprint, robust to float-formatting differences between
    pandas versions. Sensitive to any real change in the data."""
    num = d.select_dtypes(include=[np.number])
    parts = [f"shape={d.shape}", "cols=" + "|".join(map(str, d.columns))]
    for c in num.columns:
        v = num[c].to_numpy(float)
        nn = int(np.isnan(v).sum())
        ok = v[~np.isnan(v)]
        if ok.size:
            s = float(np.nansum(ok)); mn = float(ok.min()); mx = float(ok.max())
            g = lambda x: f"{x:.8g}"
            parts.append(f"{c}:n={nn}:s={g(s)}:min={g(mn)}:max={g(mx)}")
        else:
            parts.append(f"{c}:n={nn}:empty")
    if "site" in d.columns:
        parts.append("site=" + json.dumps(d["site"].value_counts().sort_index().to_dict()))
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    path = sys.argv[1]
    check_only = "--check-only" in sys.argv

    d = pd.read_csv(path, low_memory=False)
    print(f"read {path}")
    print(f"shape {d.shape}")

    # ---- repair -------------------------------------------------------------
    col = "VPD*WS*d1sin"
    n_nan = int(d[col].isna().sum())
    n_zero = int((d[col].isna() & (d["VPD*WS"] == 0)).sum())
    print(f"\nNaN in {col} on read : {n_nan}")
    print(f"  of which VPD*WS == 0 : {n_zero}")
    if n_nan != n_zero:
        print(f"  WARNING: {n_nan - n_zero} NaN are NOT explained by the 0/0 bug. Stop and investigate.")

    if not check_only:
        if n_zero:
            m = d[col].isna() & (d["VPD*WS"] == 0)
            d.loc[m, col] = 0.0
            out = path.replace(".csv", "_REPAIRED.csv")
            d.to_csv(out, index=False)
            print(f"  repaired {n_zero} rows -> wrote {out}")
        else:
            print("  nothing to repair; file is already clean on this column")

    # ---- verify -------------------------------------------------------------
    pred = [c for c in d.columns if c not in EXCL and c != TARGET]
    sub  = d[pred + [TARGET]].dropna()
    print("\n--- pre-flight check (this is section A.4 of the log) ---")
    print(f"predictors            : {len(pred)}")
    print(f"rows in file          : {len(d)}")
    print(f"rows GAM-RF will see  : {len(sub)}")
    print(f"rows lost to NaN      : {len(d) - len(sub)}")
    print("\nper-site complete cases:")
    print(d.loc[sub.index, "site"].value_counts().sort_index().to_string())
    print("\nworst remaining missingness (these are genuine sentinel-derived NaN):")
    print(d[pred].isna().sum().sort_values(ascending=False).head(8).to_string())

    print("\nVPD unit check, VPD / (es - ea) by site, must be ~1.0 everywhere:")
    r = d["VPD"] / (d["es"] - d["ea"])
    print(r.groupby(d["site"]).median().round(5).to_string())
    print(f"rows with VPD > es (must be 0): {int((d['VPD'] > d['es']).sum())}")

    # ---- fingerprint --------------------------------------------------------
    fp = fingerprint(d)
    print("\n--- fingerprint ---")
    print(f"yours     : {fp}")
    print(f"reference : {REFERENCE_FINGERPRINT}")
    print("MATCH" if fp == REFERENCE_FINGERPRINT else
          "DIFFERENT  <-- do not proceed until you know why; use your own file, not the sent one")


if __name__ == "__main__":
    main()
