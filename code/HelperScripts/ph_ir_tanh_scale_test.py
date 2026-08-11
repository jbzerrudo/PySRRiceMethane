"""
PH-IR Eq. 4 — tanh interior-scale identifiability diagnostic
============================================================
One-off test for Laurent's point: should the tanh terms in the PH-IR
equation carry an interior scale, i.e. tanh(c1 * h*VPD) * tanh(c2 * SR*HODsin)
instead of bare tanh(h*VPD) * tanh(SR*HODsin)?

This script refits Eq. 4 in BOTH forms under the SAME day-grouped 5-fold CV
as PySR7v2.py's auto-generated Stage 8 (RandomState(42) day shuffle, block
folds, per-fold curve_fit warm-started, CoV = std/|mean| flagged at 0.5),
so the numbers are directly comparable to Table 5.

Eq. 4 (paper):
    F = alpha + ( gamma*(SR*Ts) + beta
                  + (eps + tanh(h*VPD)) * tanh(SR*HODsin) ) * exp(delta*AUC)

Decision rule:
  * If BOTH c1 and c2 have across-fold |CoV| < 0.5, they are identified ->
    report 1/c1, 1/c2 as characteristic scales (with units), as you did AUC0.
  * Otherwise they are not separately recoverable on these data -> the
    §4.5(vii) limitation stands as written.

This does NOT modify PySR7v2.py and touches only PH-IR. No PySR search,
no figure regeneration.

Run:  python ph_ir_tanh_scale_test.py
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ─── CONFIG — EDIT THESE FOUR THINGS ──────────────────────────────────────
# 1) Path to the PH-IR feature-engineered CSV (the same file Eq. 4 was fit on).
INPUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\vif5\postgamrf2_run\PHL-IR_2016_retdvars2.csv"

# 2) Map each Eq.-4 term to its EXACT column name in your CSV.
#    These are best-guess defaults; the script prints the CSV header and
#    aborts with a clear message if any name is wrong, so just correct it.
COLS = {
    "AUC":       "AUC",          # cumulative ponded-water exposure (cm*h)
    "SR_Ts":     "SR*Ts",        # SR * Ts        (W m^-2 * degC)
    "h_VPD":     "h*VPD",        # h  * VPD       (cm * kPa)
    "SR_HODsin": "SR*HODsin",    # SR * HODsin    (W m^-2)
}

# 3) Target and time columns (match PySR7v2.py defaults; change only if needed).
TARGET_COL = "F_CH4_F"
TIME_COL   = "Date"

# 4) Published full-data Eq.-4 coefficients (Table 6) used as warm starts.
#    alpha, beta, gamma, delta, eps
WARM_BASE = dict(alpha=0.96, beta=1.69, gamma=2.4e-4, delta=3.2e-3, eps=-1.54)
#    interior scales: start in tanh's active band (arguments are large/saturated)
WARM_C1 = 1e-1   # scale on h*VPD  (cm*kPa range ~ tens)
WARM_C2 = 1e-2   # scale on SR*HODsin (W m^-2 range ~ hundreds)

# 5) Where to write CSV outputs (default: a folder next to this script).
#OUTPUT_DIR = os.path.join(
#    os.path.dirname(os.path.abspath(__file__)), "tanh_scale_out")
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\vif5\postgamrf2_run\tanh_scale_out"
# ──────────────────────────────────────────────────────────────────────────

MISSING_FLAGS = [-9999, -999900, -99999]
N_FOLDS = 5
RANDOM_STATE = 42

# Pack order MUST match the unpack order inside the model functions.
MODEL_COLS = ["AUC", "SR_Ts", "h_VPD", "SR_HODsin"]


# ─── models ───────────────────────────────────────────────────────────────
def eq4_base(X, alpha, beta, gamma, delta, eps):
    auc, sr_ts, h_vpd, sr_hodsin = X
    inner = gamma * sr_ts + beta + (eps + np.tanh(h_vpd)) * np.tanh(sr_hodsin)
    return alpha + inner * np.exp(delta * auc)


def eq4_scaled(X, alpha, beta, gamma, delta, eps, c1, c2):
    auc, sr_ts, h_vpd, sr_hodsin = X
    inner = gamma * sr_ts + beta + (eps + np.tanh(c1 * h_vpd)) * np.tanh(c2 * sr_hodsin)
    return alpha + inner * np.exp(delta * auc)


BASE_P0   = [WARM_BASE[k] for k in ("alpha", "beta", "gamma", "delta", "eps")]
SCALED_P0 = BASE_P0 + [WARM_C1, WARM_C2]
BASE_NAMES   = ["alpha", "beta", "gamma", "delta", "eps"]
SCALED_NAMES = BASE_NAMES + ["c1", "c2"]


# ─── helpers (copied from PySR7v2.py to match behaviour) ──────────────────
def parse_dates_robust(series):
    s = series.astype(str).str.strip()
    cands = []
    for df_flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=df_flag, format="mixed")
        pv = parsed.dropna()
        diffs = pv.diff().dropna()
        mono = float((diffs >= pd.Timedelta(0)).sum()) / max(1, len(diffs))
        cands.append({"parsed": parsed, "valid": int(parsed.notna().sum()), "mono": mono})
    best = max(cands, key=lambda c: (c["mono"], c["valid"]))
    return best["parsed"]


def compute_metrics(y_true, y_pred):
    finite = np.isfinite(y_pred)
    if finite.sum() < 10:
        return np.nan, np.nan, np.nan
    res = y_true[finite] - y_pred[finite]
    ss_tot = float(np.sum((y_true[finite] - y_true[finite].mean()) ** 2))
    r2 = float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae = float(np.mean(np.abs(res)))
    return r2, rmse, mae


def pack_X(frame):
    return tuple(frame[COLS[c]].values for c in MODEL_COLS)


# ─── load + validate ──────────────────────────────────────────────────────
def load():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {INPUT_CSV}\n  {len(df)} rows, {df.shape[1]} columns\n")
    print("CSV columns:")
    print("  " + ", ".join(map(str, df.columns)) + "\n")

    needed = [COLS[c] for c in MODEL_COLS] + [TARGET_COL, TIME_COL]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        print("[ABORT] These configured names are NOT columns in the CSV:")
        for m in missing:
            print(f"    {m!r}")
        print("\nFix the COLS / TARGET_COL / TIME_COL entries at the top to "
              "match the header printed above, then rerun.")
        sys.exit(1)

    for col in df.select_dtypes(include=["object"]).columns:
        if col == TIME_COL:
            continue
        conv = pd.to_numeric(df[col], errors="coerce")
        if conv.notna().sum() > 0:
            df[col] = conv
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].replace(MISSING_FLAGS, np.nan)

    df[TIME_COL] = parse_dates_robust(df[TIME_COL])
    df = df.dropna(subset=[TIME_COL]).copy()
    df["DAY"] = df[TIME_COL].dt.date

    model_cols = [COLS[c] for c in MODEL_COLS]
    df = df.dropna(subset=model_cols + [TARGET_COL]).copy()
    print(f"Complete cases (Eq.-4 columns + target): {len(df)}  "
          f"Days: {df['DAY'].nunique()}")
    print("  (paper PH-IR n = 3620; if this differs, your complete-case set "
          "or column choice differs from the paper run)\n")
    return df


# ─── full-data fit ────────────────────────────────────────────────────────
def full_fit(df):
    X = pack_X(df)
    y = df[TARGET_COL].values
    print("=" * 70)
    print("FULL-DATA FIT")
    print("=" * 70)
    out = {}
    for tag, fn, p0, names in [
        ("base",   eq4_base,   BASE_P0,   BASE_NAMES),
        ("scaled", eq4_scaled, SCALED_P0, SCALED_NAMES),
    ]:
        try:
            popt, _ = curve_fit(fn, X, y, p0=p0, maxfev=40000)
        except Exception as e:
            print(f"  [{tag}] curve_fit FAILED: {e}")
            out[tag] = None
            continue
        r2, rmse, mae = compute_metrics(y, fn(X, *popt))
        out[tag] = dict(popt=popt, r2=r2, rmse=rmse, mae=mae, names=names)
        print(f"  [{tag:6s}] R2={r2:.4f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
        print("            " + "  ".join(f"{n}={v:.4g}" for n, v in zip(names, popt)))
    if out.get("scaled"):
        c1, c2 = out["scaled"]["popt"][5], out["scaled"]["popt"][6]
        print(f"\n  Full-data interior scales:")
        print(f"    c1 = {c1:.4g}  ->  1/c1 = {abs(1/c1):.4g} cm*kPa "
              f"(characteristic scale of h*VPD)")
        print(f"    c2 = {c2:.4g}  ->  1/c2 = {abs(1/c2):.4g} W m^-2 "
              f"(characteristic scale of SR*HODsin)")
    print()
    return out


# ─── day-grouped CV (replicates PySR7v2 Stage 8 exactly) ──────────────────
def cv(df, fn, p0, names):
    days = df["DAY"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(days)
    fold_size = len(days) // N_FOLDS

    fold_r2, fold_params = [], []
    for k in range(N_FOLDS):
        if k < N_FOLDS - 1:
            test_days = days[k * fold_size:(k + 1) * fold_size]
        else:
            test_days = days[k * fold_size:]
        train_days = np.setdiff1d(days, test_days)
        tr = df[df["DAY"].isin(train_days)]
        te = df[df["DAY"].isin(test_days)]
        try:
            popt, _ = curve_fit(fn, pack_X(tr), tr[TARGET_COL].values,
                                p0=p0, maxfev=40000)
        except Exception:
            popt = np.full(len(p0), np.nan)
        r2, _, _ = compute_metrics(te[TARGET_COL].values, fn(pack_X(te), *popt))
        fold_r2.append(r2)
        fold_params.append(popt)
    return np.array(fold_r2), np.vstack(fold_params)


def report_cv(tag, fold_r2, fold_params, names):
    r = fold_r2[np.isfinite(fold_r2)]
    mean = float(np.mean(r)) if len(r) else np.nan
    sd1 = float(np.std(r, ddof=1)) if len(r) > 1 else np.nan
    sd0 = float(np.std(r, ddof=0)) if len(r) else np.nan
    print(f"  [{tag}] per-fold R2: " + ", ".join(f"{x:.3f}" for x in fold_r2))
    print(f"         CV R2 = {mean:.3f} +/- {sd1:.3f} (sample sd) "
          f"| {sd0:.3f} (pop sd)")
    cov = {}
    for j, nm in enumerate(names):
        v = fold_params[:, j]
        v = v[np.isfinite(v)]
        if len(v) > 1 and abs(v.mean()) > 1e-12:
            c = float(np.std(v, ddof=1) / abs(v.mean()))
        else:
            c = np.nan
        cov[nm] = (float(v.mean()) if len(v) else np.nan, c)
    return mean, sd1, cov


def _stability_rows(form, fold_params, warm, names):
    rows = []
    for j, nm in enumerate(names):
        v = fold_params[:, j]
        v = v[np.isfinite(v)]
        if len(v):
            mean = float(v.mean())
            std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
            cov = (std / abs(mean)) if (np.isfinite(std) and abs(mean) > 1e-12) else float("nan")
            vmin, vmax = float(v.min()), float(v.max())
        else:
            mean = std = cov = vmin = vmax = float("nan")
        rows.append(dict(
            form=form, param_idx=j, param_name=nm,
            warm_start=float(warm[j]), fold_mean=mean, fold_std=std,
            fold_cov_abs=cov, fold_min=vmin, fold_max=vmax,
            unstable_flag=int(np.isfinite(cov) and cov > 0.5)))
    return rows


def _fold_rows(form, fold_r2, fold_params, names):
    rows = []
    for k in range(len(fold_r2)):
        rec = dict(form=form, fold=k + 1, r2=float(fold_r2[k]))
        for j, nm in enumerate(names):
            rec[nm] = float(fold_params[k, j])
        rows.append(rec)
    return rows


def main():
    df = load()
    full = full_fit(df)

    print("=" * 70)
    print("DAY-GROUPED 5-FOLD CV  (RandomState(42) day shuffle, block folds)")
    print("=" * 70)
    b_r2, b_par = cv(df, eq4_base, BASE_P0, BASE_NAMES)
    s_r2, s_par = cv(df, eq4_scaled, SCALED_P0, SCALED_NAMES)
    b_mean, _, _ = report_cv("base  ", b_r2, b_par, BASE_NAMES)
    print()
    s_mean, _, s_cov = report_cv("scaled", s_r2, s_par, SCALED_NAMES)

    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)
    print(f"  Base (paper Eq.4) CV R2   = {b_mean:.3f}   "
          f"(Table 5 reports 0.476 +/- 0.345)")
    print(f"  Scaled (c1,c2)    CV R2   = {s_mean:.3f}")
    print(f"  delta CV R2 (scaled-base) = {s_mean - b_mean:+.3f}   "
          f"(gain from the two extra params)")
    print()
    for nm, unit in [("c1", "cm*kPa"), ("c2", "W m^-2")]:
        mean, c = s_cov[nm]
        verdict = "IDENTIFIED (|CoV|<0.5)" if (np.isfinite(c) and c < 0.5) \
            else "NOT identified (|CoV|>=0.5)"
        scale = abs(1 / mean) if (np.isfinite(mean) and abs(mean) > 1e-12) else np.nan
        print(f"  {nm}: fold mean={mean:.4g}  |CoV|={c:.3f}  -> {verdict}")
        if np.isfinite(scale):
            print(f"      1/{nm} = {scale:.4g} {unit}")
    both_ok = all(np.isfinite(s_cov[n][1]) and s_cov[n][1] < 0.5 for n in ("c1", "c2"))
    print()
    if both_ok:
        print("  => Both interior scales are identifiable. You may report")
        print("     1/c1 and 1/c2 as characteristic scales (like AUC0) and")
        print("     relax the §4.5(vii) limitation accordingly.")
    else:
        print("  => At least one interior scale is not separately recoverable")
        print("     on these data (saturated-tanh regime). The §4.5(vii)")
        print("     limitation stands as written: the scale exists but is")
        print("     not identifiable in the present unconstrained search.")

    # ── write CSV outputs ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fold_df = pd.DataFrame(
        _fold_rows("base", b_r2, b_par, BASE_NAMES)
        + _fold_rows("scaled", s_r2, s_par, SCALED_NAMES))
    fold_df.to_csv(os.path.join(OUTPUT_DIR, "tanh_scale_cv_fold_metrics.csv"),
                   index=False)

    stab_df = pd.DataFrame(
        _stability_rows("base", b_par, BASE_P0, BASE_NAMES)
        + _stability_rows("scaled", s_par, SCALED_P0, SCALED_NAMES))
    stab_df.to_csv(os.path.join(OUTPUT_DIR, "tanh_scale_coefficient_stability.csv"),
                   index=False)

    def _full(tag, key):
        d = full.get(tag)
        return float(d[key]) if d else float("nan")
    c1_mean, c1_cov = s_cov["c1"]
    c2_mean, c2_cov = s_cov["c2"]
    summary = dict(
        n_rows=int(len(df)), n_days=int(df["DAY"].nunique()),
        base_full_R2=_full("base", "r2"), scaled_full_R2=_full("scaled", "r2"),
        base_cv_R2=float(b_mean), scaled_cv_R2=float(s_mean),
        delta_cv_R2=float(s_mean - b_mean),
        c1_fold_mean=c1_mean, c1_cov_abs=c1_cov,
        c1_inv_scale_cmkPa=(abs(1 / c1_mean) if abs(c1_mean) > 1e-12 else float("nan")),
        c1_identified=int(np.isfinite(c1_cov) and c1_cov < 0.5),
        c2_fold_mean=c2_mean, c2_cov_abs=c2_cov,
        c2_inv_scale_Wm2=(abs(1 / c2_mean) if abs(c2_mean) > 1e-12 else float("nan")),
        c2_identified=int(np.isfinite(c2_cov) and c2_cov < 0.5))
    pd.DataFrame([summary]).to_csv(
        os.path.join(OUTPUT_DIR, "tanh_scale_summary.csv"), index=False)

    print(f"\n  [SAVED] 3 CSVs in {OUTPUT_DIR}")
    print("          tanh_scale_cv_fold_metrics.csv")
    print("          tanh_scale_coefficient_stability.csv")
    print("          tanh_scale_summary.csv")


if __name__ == "__main__":
    main()
