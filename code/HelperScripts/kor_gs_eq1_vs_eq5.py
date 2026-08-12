#!/usr/bin/env python3
"""
SK-CRK growing-season robustness check: how does the FULL-RECORD temperature
form (Eq. 1) perform when restricted to the growing season, versus the operative
in-season form (Eq. 5)?

This answers the likely reviewer probe "what if you fit the 2018 full-season
equation to Apr-Sep only?" two ways, both under the SAME day-grouped 5-fold CV
as PySR7v2.py Stage 8 (RandomState(42) day shuffle, block folds, per-fold
curve_fit warm-started, |CoV| flagged at 0.5):

  Eq. 1 REFIT  : refit Eq. 1's coefficients on the growing season, score by CV.
                 -> "is the temperature form genuinely worse in-season, or just
                     mis-tuned?"  (the honest comparison)
  Eq. 1 NO-REFIT: apply Eq. 1's published full-record coefficients to the
                 growing season, no refit, whole-window R2.
                 -> "do the full-record coefficients transfer?"  (weaker)
  Eq. 5        : the operative in-season form, refit by CV (sanity: ~0.344).

Equations (manuscript):
  Eq. 1 (eq:kor):    F = gamma*(h*VPD) + beta*(alpha*(SR*v) + dayhr)*exp(asinh_Ta)
                         + exp(delta*exp(asinh_Ta))
  Eq. 5 (eq:kor-gs): F = k*(AUC_wet + b/(a + g/sqrt(AUC_wet + c)))*(SR + E)
                         [the manuscript (AUC_wet+c)^f with f = -0.5, i.e. 1/sqrt]

UNITS: the published coefficients assume depth in cm (Table 2): h*VPD in cm*kPa,
AUC_wet in cm*h. This script auto-detects each by magnitude and rescales to cm if
your file is in metres, printing what it did. h*VPD enters Eq. 1 linearly so its
unit only shifts the warm start (the refit R2 is unit-invariant); AUC_wet enters
Eq. 5 non-linearly, so it is genuinely normalised to cm*h to match the warm start.

Run:  python kor_gs_eq1_vs_eq5.py
  or: python kor_gs_eq1_vs_eq5.py path/to/KOR-CRK_2018_updated.csv
"""

import sys, os, argparse, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ============================ CONFIG (edit me) ===============================
# Path to the FULL-COLUMN growing-season KOR CSV (must carry Eq. 1's inputs).
INPUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated.csv"

# Restrict to the growing season (9 Apr - 30 Sep 2018). Leave True; if the file
# is already the in-season subset this is a no-op, if it is the full Apr-Dec
# record this clips it. The script prints the resulting row count either way.
GROWING_SEASON_ONLY = True
GS_START, GS_END = "2018-04-09", "2018-09-30"

# Units. "auto" detects by magnitude and normalises to cm; override if you know.
H_VPD_UNITS   = "auto"   # "auto" | "cm" | "m"   (cm*kPa vs m*kPa)  -- Eq. 1
AUC_WET_UNITS = "auto"   # "auto" | "cm" | "m"   (cm*h  vs m*h)     -- Eq. 5

# Column names (defaults match the RUN2 retvars files). Fix if your header differs;
# the script prints the header and tells you which name is missing.
COLS = dict(hVPD="h*VPD", SRv="SR*v", dayhr="dayhr", asinh_Ta="asinh_Ta",
            AUC_wet="AUC_wet", SR="SR")
TARGET_COL, TIME_COL = "F_CH4_F", "Date"

# Published warm starts (cm convention).
# Eq. 1 full-record coefficients (Table 6, gamma already per-cm = -3.99e-2):
EQ1_WARM = dict(alpha=4.8e-3, beta=-5.3e-3, gamma=-3.99e-2, delta=5.4e-2)
# Eq. 5 growing-season coefficients (Table 6; f fixed at -0.5 == sqrt):
EQ5_WARM = dict(a=0.07465542, b=1.0970329, c=2.0335147,
                k=1.597917e-6, E=747.46045, g=-7.643319)
# =============================================================================

MISSING_FLAGS = [-9999, -999900, -99999]
RANDOM_STATE, N_FOLDS = 42, 5


# ------------------------------- models --------------------------------------
def eq1(X, alpha, beta, gamma, delta):
    hVPD, SRv, dayhr, asinh_Ta = X
    eu = np.exp(asinh_Ta)
    return gamma * hVPD + beta * (alpha * SRv + dayhr) * eu + np.exp(delta * eu)

def eq5(X, a, b, c, k, E, g):           # f = -0.5  ->  1/sqrt(AUC_wet + c)
    AUC_wet, SR = X
    return k * (AUC_wet + b / (a + g / np.sqrt(AUC_wet + c))) * (SR + E)

EQ1_NAMES = ["alpha", "beta", "gamma", "delta"]
EQ5_NAMES = ["a", "b", "c", "k", "E", "g"]
EQ1_P0    = [EQ1_WARM[n] for n in EQ1_NAMES]
EQ5_P0    = [EQ5_WARM[n] for n in EQ5_NAMES]


# ------------------------------- helpers -------------------------------------
def parse_dates_robust(series):
    s = series.astype(str).str.strip()
    best, best_key = None, (-1.0, -1)
    for dayfirst in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=dayfirst, format="mixed")
        diffs = parsed.dropna().diff().dropna()
        mono = float((diffs >= pd.Timedelta(0)).sum()) / max(1, len(diffs))
        key = (mono, int(parsed.notna().sum()))
        if key > best_key:
            best, best_key = parsed, key
    return best

def r2_rmse_mae(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    if m.sum() < 10:
        return np.nan, np.nan, np.nan, int(m.sum())
    y, yhat = y[m], yhat[m]
    ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1 - np.sum((y - yhat) ** 2) / ss) if ss > 0 else np.nan
    return r2, float(np.sqrt(np.mean((y - yhat) ** 2))), float(np.mean(np.abs(y - yhat))), len(y)

def normalise_to_cm(df, col, mode, kind):
    """kind: 'hVPD' (cm*kPa ~ tens; m*kPa < ~1) or 'AUC' (cm*h ~ thousands; m*h < ~few hundred)."""
    s = pd.to_numeric(df[col], errors="coerce")
    mx = float(s.abs().max())
    thresh = 2.0 if kind == "hVPD" else 500.0
    if mode == "m":
        is_m = True
    elif mode == "cm":
        is_m = False
    else:                               # auto
        is_m = mx < thresh
    unit_now = "m" if is_m else "cm"
    factor = 100.0 if is_m else 1.0
    tag = "x100 -> cm" if is_m else "kept as cm"
    print(f"  {col:9s} max|{mx:9.3g}|  detected {unit_now}  ({tag})")
    return s * factor


# ------------------------- day-grouped 5-fold CV -----------------------------
def cv(df, fn, p0, names, packer):
    days = df["DAY"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(days)
    fold_size = len(days) // N_FOLDS
    fr2, fp = [], []
    for k in range(N_FOLDS):
        test_days = days[k * fold_size:(k + 1) * fold_size] if k < N_FOLDS - 1 else days[k * fold_size:]
        tr = df[df["DAY"].isin(np.setdiff1d(days, test_days))]
        te = df[df["DAY"].isin(test_days)]
        try:
            popt, _ = curve_fit(fn, packer(tr), tr[TARGET_COL].values, p0=p0, maxfev=40000)
        except Exception:
            popt = np.full(len(p0), np.nan)
        r2, *_ = r2_rmse_mae(te[TARGET_COL].values, fn(packer(te), *popt))
        fr2.append(r2); fp.append(popt)
    fr2, fp = np.array(fr2), np.vstack(fp)
    r = fr2[np.isfinite(fr2)]
    mean = float(np.mean(r)) if len(r) else np.nan
    sd = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    covs = {}
    for j, nm in enumerate(names):
        v = fp[:, j][np.isfinite(fp[:, j])]
        covs[nm] = float(np.std(v, ddof=1) / abs(v.mean())) if (len(v) > 1 and abs(v.mean()) > 1e-12) else np.nan
    maxcov = max([c for c in covs.values() if np.isfinite(c)], default=np.nan)
    return fr2, fp, mean, sd, covs, maxcov


# ----------------------------------- main ------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="?", default=INPUT_CSV)
    csv_path = ap.parse_args().csv

    df = pd.read_csv(csv_path)
    print(f"Loaded {csv_path}\n  {len(df)} rows, {df.shape[1]} columns")
    print("  columns: " + ", ".join(map(str, df.columns)) + "\n")

    for c in df.select_dtypes(include=["object"]).columns:
        if c == TIME_COL:
            continue
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() > 0:
            df[c] = conv
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace(MISSING_FLAGS, np.nan)

    df[TIME_COL] = parse_dates_robust(df[TIME_COL])
    df = df.dropna(subset=[TIME_COL]).copy()
    df["DAY"] = df[TIME_COL].dt.date

    if GROWING_SEASON_ONLY:
        n0 = len(df)
        df = df[(df[TIME_COL] >= pd.Timestamp(GS_START)) &
                (df[TIME_COL] < pd.Timestamp(GS_END) + pd.Timedelta(days=1))].copy()
        print(f"Growing-season filter [{GS_START} .. {GS_END}]: {n0} -> {len(df)} rows")
    print(f"Date range: {df[TIME_COL].min():%Y-%m-%d} .. {df[TIME_COL].max():%Y-%m-%d}   "
          f"days={df['DAY'].nunique()}   (paper in-season n=8365)\n")

    have_eq1 = all(COLS[k] in df.columns for k in ("hVPD", "SRv", "dayhr", "asinh_Ta"))
    have_eq5 = all(COLS[k] in df.columns for k in ("AUC_wet", "SR"))
    if not have_eq1:
        miss = [COLS[k] for k in ("hVPD", "SRv", "dayhr", "asinh_Ta") if COLS[k] not in df.columns]
        print(f"[skip Eq. 1] missing columns: {miss}  (this file cannot feed the temperature form)\n")
    if not have_eq5:
        miss = [COLS[k] for k in ("AUC_wet", "SR") if COLS[k] not in df.columns]
        print(f"[skip Eq. 5] missing columns: {miss}\n")
    if not (have_eq1 or have_eq5):
        sys.exit("Neither equation's columns are present. Fix COLS at the top.")

    print("Unit detection (normalised to cm convention):")
    if have_eq1:
        df["_hVPD"] = normalise_to_cm(df, COLS["hVPD"], H_VPD_UNITS, "hVPD")
    if have_eq5:
        df["_AUCwet"] = normalise_to_cm(df, COLS["AUC_wet"], AUC_WET_UNITS, "AUC")
    print()

    drop = [TARGET_COL]
    if have_eq1: drop += ["_hVPD", COLS["SRv"], COLS["dayhr"], COLS["asinh_Ta"]]
    if have_eq5: drop += ["_AUCwet", COLS["SR"]]
    df = df.dropna(subset=drop).copy()
    y = df[TARGET_COL].values

    results = {}

    if have_eq1:
        packer1 = lambda d: (d["_hVPD"].values, d[COLS["SRv"]].values,
                             d[COLS["dayhr"]].values, d[COLS["asinh_Ta"]].values)
        # full-data refit
        try:
            popt, _ = curve_fit(eq1, packer1(df), y, p0=EQ1_P0, maxfev=40000)
            fr2, _, _, _ = r2_rmse_mae(y, eq1(packer1(df), *popt))
        except Exception as e:
            popt, fr2 = np.full(len(EQ1_P0), np.nan), np.nan
        # CV refit
        f, p, m, sd, cov, mx = cv(df, eq1, EQ1_P0, EQ1_NAMES, packer1)
        # no-refit whole window (published full-record coeffs)
        nr_r2, *_ = r2_rmse_mae(y, eq1(packer1(df), *EQ1_P0))
        results["eq1_refit"] = (m, sd, mx, f, cov)
        results["eq1_norefit"] = nr_r2
        print("=" * 70)
        print("Eq. 1 (temperature, asinh(Ta) double-exponential) on the growing season")
        print("=" * 70)
        print(f"  full-data refit R2          = {fr2:.3f}")
        print(f"  per-fold CV R2              = {', '.join(f'{x:.3f}' for x in f)}")
        print(f"  day-grouped CV R2          = {m:.3f} +/- {sd:.3f}   max|CoV| = {mx:.2f}")
        print(f"  NO-REFIT (full-record coef) = {nr_r2:.3f}   whole-window\n")

    if have_eq5:
        packer5 = lambda d: (d["_AUCwet"].values, d[COLS["SR"]].values)
        try:
            popt, _ = curve_fit(eq5, packer5(df), y, p0=EQ5_P0, maxfev=40000)
            fr2, *_ = r2_rmse_mae(y, eq5(packer5(df), *popt))
        except Exception:
            fr2 = np.nan
        recon_r2, *_ = r2_rmse_mae(y, eq5(packer5(df), *EQ5_P0))
        f, p, m, sd, cov, mx = cv(df, eq5, EQ5_P0, EQ5_NAMES, packer5)
        results["eq5_refit"] = (m, sd, mx)
        print("=" * 70)
        print("Eq. 5 (AUC_wet, operative in-season form) -- sanity check vs paper")
        print("=" * 70)
        print(f"  whole-window reconstruction = {recon_r2:.3f}   (paper 0.343)")
        print(f"  per-fold CV R2              = {', '.join(f'{x:.3f}' for x in f)}")
        print(f"  day-grouped CV R2          = {m:.3f} +/- {sd:.3f}   max|CoV| = {mx:.2f}   "
              f"(paper 0.344 +/- 0.101)\n")

    if have_eq1 and have_eq5:
        e1m, e1sd, e1mx, _, _ = results["eq1_refit"]
        e5m, e5sd, e5mx = results["eq5_refit"]
        print("=" * 70)
        print("VERDICT  (growing season, day-grouped 5-fold CV)")
        print("=" * 70)
        print(f"  Eq. 5  AUC_wet      CV R2 = {e5m:.3f} +/- {e5sd:.3f}")
        print(f"  Eq. 1  temperature  CV R2 = {e1m:.3f} +/- {e1sd:.3f}  (refit in-season)")
        print(f"         temperature  R2    = {results['eq1_norefit']:.3f}        (no refit, full-record coef)")
        gap = e5m - e1m
        if np.isfinite(gap):
            verdict = ("underperforms" if gap > 0 else "matches/exceeds")
            print(f"\n  Eq. 1 {verdict} Eq. 5 in-season by {gap:+.3f} CV R2.")
            if gap > 0:
                print("  Consistent with sec:window: the temperature form's full-record")
                print("  strength is the cold drained shoulder, which is gone in-season.")
    print()


if __name__ == "__main__":
    main()
