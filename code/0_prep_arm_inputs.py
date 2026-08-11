"""
0_prep_arm_inputs.py — box (0) preparation for all four arms
==============================================================================
Run this BEFORE the union. It does the two things that must be right and that
are easy to get wrong by hand:

  1. Divides VPD by 10 at Mase and Cheorwon. Those two files carry VPD in hPa
     while es and ea are in kPa; IRRI and the pooled file are already kPa.
     Measured ratio of VPD to (es - ea): 10.003 at Mase, 9.995 at Cheorwon,
     1.000 at IRRI.

  2. Deletes the box (0) columns for that arm:
       Mase, IRRI    hdry, DelTsa, ea, AUC, h*DelTsa
       Cheorwon      hdry, DelTsa, ea, AUC_dry, hwet, AUC, h*DelTsa
       Pooled        hdry, DelTsa, ea, h*DelTsa

     Every one of these removes a MEASURED exact identity:
       depth       = hwet - hdry            residual 0.000e+00 at all sites
       DelTsa      = Tsoil - Tair           residual ~5e-15 at all sites
       VPD         = es - ea                exact once VPD is in kPa
       AUC         = AUC_wet - AUC_dry      residual 2.1e-11 JPN, 8.2e-12 PHL
       h*DelTsa    = h*Ts - h*Ta            residual 1e-4 JPN/PHL, 5e-14 KOR

     Cheorwon takes more because hdry and AUC_dry are identically zero over all
     8,365 of its rows, which additionally makes depth = hwet and AUC = AUC_wet
     exact there and nowhere else.

It writes a NEW file and never overwrites the original, then verifies its own
work: it reports what was dropped, the VPD ratio after conversion, and whether
any exact linear dependency survives.

USAGE
    python 0_prep_arm_inputs.py                 # all four arms
    python 0_prep_arm_inputs.py JPN KOR         # named arms only

Edit ARMS below so the paths point at your files.

Author: Jef Zerrudo / Claude.  Requires numpy, pandas.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd

MISSING_FLAGS = [-9999, -999900, -99999]

# The columns the collinearity checker never puts in the design. The dependency
# check below must ignore them too, or it reports alarms for pairs that never
# reach the VIF loop: F_CH4_F against its unit-converted twin F_CH4_F_orig, and
# time against dayhr.
EXCLUDE_HEADERS = ["site", "w", "Date", "Deltime", "time", "F_CH4_F_orig", "F_CH4_F"]

# ── EDIT THESE PATHS ────────────────────────────────────────────────────────
BASE = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV"

ARMS = {
    "JPN": dict(
        src=os.path.join(BASE, r"JPN\Data-Metadata\JPN-MSE_cm.csv"),
        dst=os.path.join(BASE, r"JPN\Data-Metadata\JPN-MSE_cm_box0.csv"),
        vpd_to_kpa=True,
        drop=["hdry", "DelTsa", "ea", "AUC", "h*DelTsa"],
    ),
    "KOR": dict(
        src=os.path.join(BASE, r"KOR\Data_Metadata\Papale_hampel_cleaned\KORCRK_2018_papale_hampel_growingseason.csv"),
        dst=os.path.join(BASE, r"KOR\Data_Metadata\Papale_hampel_cleaned\KORCRK_2018_papale_hampel_growingseason_box0.csv"),
        vpd_to_kpa=True,
        drop=["hdry", "DelTsa", "ea", "AUC_dry", "hwet", "AUC", "h*DelTsa"],
    ),
    "PHL": dict(
        src=os.path.join(BASE, r"PHL\Data_Metadata\PHLIR_2016_cleaned.csv"),
        dst=os.path.join(BASE, r"PHL\Data_Metadata\PHLIR_2016_cleaned_box0.csv"),
        vpd_to_kpa=False,
        drop=["hdry", "DelTsa", "ea", "AUC", "h*DelTsa"],
    ),
    "POOLED": dict(
        src=os.path.join(BASE, r"POOLED\Data-Metadata\POOL_3sites_INTENSIVE.csv"),
        dst=os.path.join(BASE, r"POOLED\Data-Metadata\POOL_3sites_INTENSIVE_box0.csv"),
        vpd_to_kpa=False,
        drop=["hdry", "DelTsa", "ea", "h*DelTsa", "fwet"],
    ),
}
# ────────────────────────────────────────────────────────────────────────────


def num(d, c):
    if c not in d.columns:
        return None
    return pd.to_numeric(d[c], errors="coerce").replace(MISSING_FLAGS, np.nan)


def exact_dependencies(d, tol=1e-8):
    """Exact linear dependencies among the predictors that will reach the VIF
    loop, by SVD null space. Columns in EXCLUDE_HEADERS are removed first.

    Text columns such as Date and site become all-NaN under to_numeric and are
    still float dtype, so they survive select_dtypes. Dropping rows before
    removing them empties the frame and makes every later check vacuous. Remove
    all-NaN columns FIRST.
    """
    d = d.drop(columns=[c for c in EXCLUDE_HEADERS if c in d.columns])
    X = d.apply(lambda c: pd.to_numeric(c, errors="coerce"))
    X = X.select_dtypes(include=[np.number]).replace(MISSING_FLAGS, np.nan)
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.loc[:, X.notna().any()]          # drop columns that are entirely NaN
    X = X.dropna()
    if len(X) < 2:
        return [], []
    cols = [c for c in X.columns if float(np.nanstd(X[c].to_numpy(float))) > 0]
    consts = [c for c in X.columns if c not in cols]
    if len(cols) < 2:
        return [], consts
    A = X[cols].to_numpy(float)
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    out = []
    for k, val in enumerate(sv):
        if sv[0] > 0 and val / sv[0] < tol:
            v = Vt[k]
            idx = np.argsort(-np.abs(v))[:4]
            out.append([(cols[i], float(v[i])) for i in idx if abs(v[i]) > 1e-3])
    return out, consts


def prep(name, cfg):
    print(f"\n{'='*74}\n  {name}\n{'='*74}")
    if not os.path.isfile(cfg["src"]):
        print(f"  [SKIP] not found: {cfg['src']}")
        return False

    d = pd.read_csv(cfg["src"], low_memory=False)
    print(f"  read  {cfg['src']}")
    print(f"        {d.shape[0]:,} rows x {d.shape[1]} columns")

    # 1. VPD units
    es, ea, vpd = num(d, "es"), num(d, "ea"), num(d, "VPD")
    if es is not None and ea is not None and vpd is not None:
        ratio = (vpd / (es - ea)).replace([np.inf, -np.inf], np.nan).median()
        print(f"  VPD / (es - ea) before : {ratio:.4f}")
        if cfg["vpd_to_kpa"]:
            if not (8.0 < ratio < 12.0):
                print(f"  [STOP] vpd_to_kpa is True but the ratio is {ratio:.4f}, "
                      f"not near 10. Refusing to divide.")
                return False
            d["VPD"] = pd.to_numeric(d["VPD"], errors="coerce") / 10.0
            after = (num(d, "VPD") / (es - ea)).replace([np.inf, -np.inf], np.nan).median()
            print(f"  VPD divided by 10, ratio now : {after:.4f}")
        elif not (0.9 < ratio < 1.1):
            print(f"  [WARN] vpd_to_kpa is False but the ratio is {ratio:.4f}. Check it.")

    # 2. box (0) deletions
    present = [c for c in cfg["drop"] if c in d.columns]
    absent = [c for c in cfg["drop"] if c not in d.columns]
    d = d.drop(columns=present)
    print(f"  dropped ({len(present)}) : {present}")
    if absent:
        print(f"  listed but not present : {absent}")

    os.makedirs(os.path.dirname(cfg["dst"]), exist_ok=True)
    d.to_csv(cfg["dst"], index=False)
    print(f"  wrote {cfg['dst']}")
    print(f"        {d.shape[0]:,} rows x {d.shape[1]} columns")

    # 3. verify
    tgt = num(d, "F_CH4_F")
    if tgt is not None:
        print(f"  target unchanged : sum(F_CH4_F) = {tgt.sum():.3f}")
    deps, consts = exact_dependencies(d)
    n_pred = len([c for c in d.columns if c not in EXCLUDE_HEADERS])
    print(f"  predictors reaching the VIF loop : {n_pred}")
    if consts:
        print(f"  [WARN] zero-variance columns remain : {consts}")
    if deps:
        print(f"  [WARN] {len(deps)} exact linear dependency(ies) REMAIN:")
        for terms in deps:
            print("         " + "  ".join(f"{c}({w:+.3f})" for c, w in terms))
        print("         Add one member of each to this arm's drop list.")
    else:
        print("  [OK]   no exact linear dependency remains")
    return True


if __name__ == "__main__":
    wanted = [a.upper() for a in sys.argv[1:]] or list(ARMS)
    bad = [a for a in wanted if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s): {bad}. Choose from {list(ARMS)}")
    ok = [prep(a, ARMS[a]) for a in wanted]
    print(f"\n{'='*74}")
    print(f"  {sum(ok)} of {len(wanted)} arms written. Point each config's "
          f"INPUT_FILE at the _box0 file.")
    print(f"{'='*74}\n")
