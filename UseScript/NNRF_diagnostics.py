"""
NNRF_diagnostics_fixed.py — Layer-3 (NN+RF) check for the PySR rice-CH4 equations.

Corrects two configuration bugs in the original run:
  (1) PREDICTORS was left at the placeholder ["AUC"], so RF/NN only ever saw AUC.
      -> Here PREDICTORS is AUTO-DERIVED from each site's retdvars2 CSV columns
         (every column except Date / target / known non-predictors). Nothing to
         forget to replace. The derived list + count is printed for you to check.
  (2) The residual was computed against PH's hardcoded exp(6.38e-3*AUC) for ALL
      sites. -> Here each site uses its OWN selected equation and OWN fitted
      coefficients, taken verbatim from that site's Stage-8 script. Predictors are
      pulled BY NAME (order-independent), so column order can't break anything.

Because PREDICTORS is now the full post-VIF set, the dropna() removes the SAME rows
as Stage 8, so the day-grouped folds (seed 42) match Stage 8 and the R2 values are
directly comparable — exactly what Layer 3 needs.

  CEILING  — RF+NN on drivers -> measured flux. Small gap to PySR's CV R2 = the
             form is near-optimal; large gap = signal left on the table.
  RESIDUAL — RF+NN on drivers -> (measured - PySR prediction). R2 ~ 0 = noise
             (equation certified sufficient); R2 > 0 = a systematic term is
             missing, and the importance ranking says which driver carries it.

RF/NN are diagnostics only — not deployed, not embedded in ORYZA.

Author: Jef Zerrudo / Claude
Requires: numpy, pandas, scikit-learn
"""

import os
import warnings
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor
from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")   # MLP convergence / sklearn chatter

# ─── Global settings (match Stage 8) ────────────────────────────────────────────
TARGET_COL    = "F_CH4_F"
TIME_COL      = "Date"
MISSING_FLAGS = [-9999, -999900, -99999]
N_FOLDS       = 5
RANDOM_STATE  = 42      # MUST match each Stage-8 fold seed (it is 42 in all three)

# Columns that are never predictors (auto-derive excludes these + TARGET + TIME).
NON_PREDICTORS = {TIME_COL, TARGET_COL, "Deltime", "time", "F_CH4_F_orig"}


# ─── Per-site equations (verbatim from each Stage-8 script) ─────────────────────
# Predictors pulled BY NAME from the cleaned dataframe, so order is irrelevant.
# The selected (reported) form per site:
#   KOR -> complex_ruleA   (CV R2 0.423 ± 0.099, all coeffs stable)
#   JPN -> simple_ruleA    (auto-best; CV R2 0.426 ± 0.172, stable. Complex is more
#                           accurate but coeff-unstable, so the reported form is this.)
#   PHL -> complex_ruleA   (CV R2 0.476 ± 0.345, all coeffs stable)
# To test a different form, swap the function + params in SITES below.

def eq_kor_simple(df, a, b, c, d, e, f, g):
    AUC_wet = df["AUC_wet"].to_numpy(float)
    SR      = df["SR"].to_numpy(float)
    return d * (AUC_wet + b / (a + g * (AUC_wet + c) ** f)) * (SR + e)

def eq_jpn_simple(df, a):
    AUC = df["AUC"].to_numpy(float)
    return AUC * a

def eq_phl_complex(df, a, b, c, d, e):
    AUC       = df["AUC"].to_numpy(float)
    SR_Ts     = df["SR*Ts"].to_numpy(float)
    h_VPD     = df["h*VPD"].to_numpy(float)
    SR_HODsin = df["SR*HODsin"].to_numpy(float)
    return a + (SR_Ts * c + b + (e + np.tanh(h_VPD)) * np.tanh(SR_HODsin)) * np.exp(AUC * d)


# ─── Per-site configuration ─────────────────────────────────────────────────────
# Set the csv / outdir paths to your machine. Coefficients are the fitted (warm-start)
# constants from each Stage-8 script; do not change unless you reselect the form.
SITES = {
    "KOR": dict(
        csv=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv",
        outdir=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\NNRF\KOR\nnrf_kor_newrun",
        equation=eq_kor_simple,
        params=[7.465542e-02, 1.097033e+00, 2.033515e+00, 1.597917e-06, 7.474605e+02, -5.000000e-01, -7.643319e+00],
        form="simple_ruleA (auto-best)",
        pysr_cv_r2=0.344,
    ),
    #"JPN": dict(
    #    csv=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\JPN\JPN-MSE_2012_retainedvars2.csv",
    #    outdir=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\NNRF\JPN\nnrf_newrun",
    #    equation=eq_jpn_simple,
    #    params=[8.401471e-02],
    #    form="simple_ruleA (auto-best)",
    #    pysr_cv_r2=0.426,
    # ),
    # "PHL": dict(
    #    csv=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\vif5\postgamrf2_run\PHL-IR_2016_retdvars2.csv",
    #    outdir=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\NNRF\PHL\nnrf_phl_newrun",
    #    equation=eq_phl_complex,
    #    params=[9.636868e-01, 1.686024e+00, 2.388674e-04, 3.167974e-03, -1.537868e+00],
    #    form="complex_ruleA",
    #    pysr_cv_r2=0.476,
    # ),
}


# ─── Models (unchanged) ─────────────────────────────────────────────────────────
def make_rf():
    return RandomForestRegressor(
        n_estimators=400, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE)

def make_nn():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", max_iter=2000, early_stopping=True,
                             n_iter_no_change=20, random_state=RANDOM_STATE)),
    ])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


# ─── Helpers (verbatim from PySR7v2 / Stage 8 so preprocessing matches) ─────────
def parse_dates_robust(series, verbose=True):
    s = series.astype(str).str.strip()
    n = len(s)
    cands = []
    for df_flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=df_flag, format="mixed")
        valid = int(parsed.notna().sum())
        pv = parsed.dropna()
        diffs = pv.diff().dropna()
        mono = float((diffs >= pd.Timedelta(0)).sum()) / max(1, len(diffs))
        cands.append({"dayfirst": df_flag, "parsed": parsed, "valid": valid, "mono": mono})
    best = max(cands, key=lambda c: (c["mono"], c["valid"]))
    if verbose:
        for c in cands:
            flag = "  <-- chosen" if c is best else ""
            print(f"    dayfirst={str(c['dayfirst']):<5} parsed={c['valid']}/{n} "
                  f"monotonic={c['mono']:.2%}{flag}")
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


def make_day_folds(day_norm, n_folds, seed):
    """Day-grouped folds, identical construction to Stage 8."""
    days = pd.DatetimeIndex(day_norm.drop_duplicates()).to_numpy().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(days)
    fs = len(days) // n_folds
    folds = []
    for k in range(n_folds):
        test_days = days[k * fs:] if k == n_folds - 1 else days[k * fs:(k + 1) * fs]
        folds.append(day_norm.isin(test_days).values)
    return folds


def cv_eval(make_model, X, y, folds):
    r2s, rmses, maes = [], [], []
    for te in folds:
        tr = ~te
        m = make_model()
        m.fit(X[tr], y[tr])
        r2, rmse, mae = compute_metrics(y[te], m.predict(X[te]))
        r2s.append(r2); rmses.append(rmse); maes.append(mae)
    return np.array(r2s), np.array(rmses), np.array(maes)


def fmt(arr):
    a = arr[np.isfinite(arr)]
    return f"{a.mean():+.3f} ± {a.std():.3f}" if len(a) else "  n/a"


# ─── Run one site ───────────────────────────────────────────────────────────────
def run_site(site, cfg):
    print("=" * 78)
    print(f"SITE: {site}   form: {cfg['form']}")
    print(f"  Input : {cfg['csv']}")
    print("=" * 78)
    os.makedirs(cfg["outdir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(cfg["csv"])
    for col in df.columns:
        if col != TIME_COL:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.replace(MISSING_FLAGS, np.nan)

    # (1) AUTO-DERIVE predictors from the CSV — the full post-VIF set PySR used.
    predictors = [c for c in df.columns if c not in NON_PREDICTORS]
    print(f"\n  Predictors auto-derived ({len(predictors)}): {predictors}")
    for need in (TARGET_COL, TIME_COL):
        if need not in df.columns:
            raise KeyError(f"{site}: column '{need}' not found")

    print("\n  Parsing dates:")
    dates = parse_dates_robust(df[TIME_COL])

    keep = predictors + [TARGET_COL]
    sub = df[keep].copy()
    sub["__date__"] = dates
    sub = sub.dropna(subset=keep + ["__date__"]).reset_index(drop=True)
    day_norm = sub["__date__"].dt.normalize()
    print(f"  Complete cases: {len(sub):,} half-hours over {day_norm.nunique():,} days")

    X = sub[predictors].to_numpy(float)
    y = sub[TARGET_COL].to_numpy(float)
    folds = make_day_folds(day_norm, N_FOLDS, RANDOM_STATE)

    rows = []

    # (CEILING) drivers -> measured flux
    print("\n  (1) CEILING — drivers -> measured flux")
    rf_r2, _, _ = cv_eval(make_rf, X, y, folds)
    nn_r2, _, _ = cv_eval(make_nn, X, y, folds)
    print(f"      RF  CV R2 = {fmt(rf_r2)}")
    print(f"      NN  CV R2 = {fmt(nn_r2)}")
    rows += [{"task": "ceiling", "model": "RF", "cv_r2_mean": float(np.nanmean(rf_r2)),
              "cv_r2_std": float(np.nanstd(rf_r2))},
             {"task": "ceiling", "model": "NN", "cv_r2_mean": float(np.nanmean(nn_r2)),
              "cv_r2_std": float(np.nanstd(nn_r2))}]
    best_ceiling = np.nanmax([np.nanmean(rf_r2), np.nanmean(nn_r2)])
    gap = best_ceiling - cfg["pysr_cv_r2"]
    print(f"      PySR CV R2 (form) = {cfg['pysr_cv_r2']:+.3f}   "
          f"ceiling gap = {gap:+.3f}  -> "
          f"{'form near-optimal' if gap < 0.05 else 'signal left on the table'}")

    # (RESIDUAL) drivers -> (measured - PySR prediction)
    print("\n  (2) RESIDUAL — drivers -> (measured - PySR prediction)")
    pysr_pred = cfg["equation"](sub, *cfg["params"])
    pysr_pred = np.asarray(pysr_pred, dtype=float)
    resid = y - pysr_pred
    finite = np.isfinite(resid)
    eq_r2 = compute_metrics(y, pysr_pred)[0]
    print(f"      PySR fit on all data: R2 = {eq_r2:+.3f}  "
          f"(residual std = {np.nanstd(resid[finite]):.3f})")
    rf_r2r, _, _ = cv_eval(make_rf, X, resid, folds)
    nn_r2r, _, _ = cv_eval(make_nn, X, resid, folds)
    print(f"      RF  on residual: CV R2 = {fmt(rf_r2r)}")
    print(f"      NN  on residual: CV R2 = {fmt(nn_r2r)}")
    rows += [{"task": "residual", "model": "RF", "cv_r2_mean": float(np.nanmean(rf_r2r)),
              "cv_r2_std": float(np.nanstd(rf_r2r))},
             {"task": "residual", "model": "NN", "cv_r2_mean": float(np.nanmean(nn_r2r)),
              "cv_r2_std": float(np.nanstd(nn_r2r))}]
    best_resid = np.nanmax([np.nanmean(rf_r2r), np.nanmean(nn_r2r)])
    verdict = ("residuals ~ NOISE -> equation CERTIFIED sufficient" if best_resid < 0.05
               else "minor residual structure -> optional correction term" if best_resid < 0.20
               else "SUBSTANTIAL residual structure -> consider a PySR-on-residual term")
    print(f"      VERDICT: best residual CV R2 = {best_resid:+.3f}  ->  {verdict}")

    # Which driver carries the leftover signal (impurity + permutation importance).
    rf_imp = make_rf()
    rf_imp.fit(X[finite], resid[finite])
    imp = rf_imp.feature_importances_
    perm = permutation_importance(rf_imp, X[finite], resid[finite],
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    order = np.argsort(perm.importances_mean)[::-1]
    print("\n      Residual importance (which driver the form under-uses):")
    print(f"        {'predictor':<14} {'impurity':>9} {'perm_mean':>10} {'perm_std':>9}")
    imp_rows = []
    for i in order:
        print(f"        {predictors[i]:<14} {imp[i]:>9.3f} "
              f"{perm.importances_mean[i]:>10.3f} {perm.importances_std[i]:>9.3f}")
        imp_rows.append({"predictor": predictors[i],
                         "impurity_importance": float(imp[i]),
                         "perm_importance_mean": float(perm.importances_mean[i]),
                         "perm_importance_std": float(perm.importances_std[i])})

    p_diag = os.path.join(cfg["outdir"], f"nnrf_diagnostics_{ts}.csv")
    p_imp = os.path.join(cfg["outdir"], f"residual_importance_{ts}.csv")
    pd.DataFrame(rows).to_csv(p_diag, index=False)
    pd.DataFrame(imp_rows).to_csv(p_imp, index=False)
    print(f"\n  [SAVED] {p_diag}")
    print(f"  [SAVED] {p_imp}\n")


def main():
    print("NN+RF Layer-3 diagnostic (corrected) — day-grouped 5-fold CV, seed",
          RANDOM_STATE)
    for site, cfg in SITES.items():
        run_site(site, cfg)


if __name__ == "__main__":
    main()