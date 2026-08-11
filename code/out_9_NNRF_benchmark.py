"""
9_NNRF_benchmark.py — RF and NN benchmark for the four PySR equations
==============================================================================
REPLACES 7_NNRF_diagnostics.py, which encoded the June equations. Every arm's
equation changed between 2 and 10 August 2026, JPN and PHL were commented out,
and its KOR form used `AUC`, which no longer exists.

WHAT THIS ANSWERS

Wassmann's charge is that the equations are "self-serving mathematical
exercises", which in practice means: you never showed they beat something
simpler. The mountains paper rebuts that with five benchmarks under an
identical CV protocol. Paper 1 has had nothing. This is that section.

Two tests per arm, both on the SAME day-grouped folds stage 8 used (seed 42):

  CEILING   RF and NN on the drivers -> measured flux. This is how much any
            model can extract from these predictors. It does NOT depend on
            which equation you picked, so it runs even if you have not chosen.
            Small gap to the PySR CV R2 means the equation is near-optimal.
            Large gap means signal is being left on the table.

  RESIDUAL  RF and NN on the drivers -> (measured - equation prediction).
            R2 near 0 means what the equation misses is noise, so the form is
            sufficient. R2 well above 0 means a systematic term is missing, and
            the permutation importance says which driver carries it.

The equation is evaluated at its PUBLISHED coefficients, not refitted. The
residual of the reported equation is the thing of interest.

RF and NN are diagnostics and benchmarks. Neither is deployed and neither goes
into ORYZA.

EQUATIONS, all verbatim from the arm's own report

  JPN  complex_ruleA        cplx 22  CV +0.542  run_20260805_122821 seed 49
  KOR  best_accuracy_ruleA  cplx 40  CV +0.654  run_20260805_170113 seed 45
  PHL  auto-best = recurring cplx  8  CV +0.465  run_20260806_104044 seed 53   (v2)
  POOL best_accuracy_ruleA  cplx 40  CV +0.596  run_20260810_111532 seed 45

USAGE
    python 9_NNRF_benchmark.py                # all four arms
    python 9_NNRF_benchmark.py JPN POOLED     # named arms only

Author: Jef Zerrudo / Claude.  Requires numpy, pandas, scikit-learn.
==============================================================================
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

TARGET_COL = "F_CH4_F"
TIME_COL = "Date"
MISSING_FLAGS = [-9999, -999900, -99999]
N_FOLDS = 5
RANDOM_STATE = 42          # must match stage 8's fold seed

# Not predictors. Mirrors EXCLUDE_HEADERS and DROP_PREDICTORS in 6_PySR.py so
# RF and NN see exactly the design PySR searched. `w` and `site` matter on the
# pooled arm: `w` is a perfect site label (0.900724 / 1.751934 / 0.758159).
NON_PREDICTORS = {TIME_COL, TARGET_COL, "Deltime", "time", "F_CH4_F_orig",
                  "site", "w", "WD"}

BASE = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV"
OUT = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\NNRF"


# ── the four equations, at their published coefficients ─────────────────────
def eq_jpn(d):
    """Mase, complex_ruleA, complexity 22, CV +0.542.
    CH4 = exp(z) - (u*VPD / VPD*WS) / (3.0412269 - z),  z = 2.1130118e-7*AUC_wet*Tv_C^2
    """
    z = 2.1130118e-7 * d["AUC_wet"].to_numpy(float) * d["Tv_C"].to_numpy(float) ** 2
    return np.exp(z) - (d["u*VPD"].to_numpy(float) / d["VPD*WS"].to_numpy(float)) \
        / (3.0412269 - z)


def eq_kor(d):
    """Cheorwon, best_accuracy_ruleA, complexity 40, CV +0.654."""
    AUC = d["AUC_wet"].to_numpy(float)
    SR = d["SR"].to_numpy(float)
    SRv = d["SR*v"].to_numpy(float)
    SRu = d["SR*u"].to_numpy(float)
    SRH = d["SR*HODsin"].to_numpy(float)
    Tv = d["Tv_C"].to_numpy(float)
    dayhr = d["dayhr"].to_numpy(float)
    hinv = d["h_inv"].to_numpy(float)
    inner = ((dayhr * np.log(Tv)) + (-9.83549 / hinv)) \
        + ((SRu * -0.019528149)
           - (((AUC - SRu) + (SRH * -7.484453)) / (AUC - (hinv * 6.249428))))
    return (((AUC - SRv) + SR) * 6.3612692e-6) * (np.exp(np.sqrt(Tv)) - inner) \
        + 0.4964769


def eq_phl(d):
    """IRRI, complexity 8, CV +0.465. OPERATIVE since 11 Aug 2026.

    v2: REPLACES the complexity-24 form. Stage 8's coefficient_stability.csv
    flags the c24 form's sqrt(SR*Ts) exponent coefficient at |CoV| = 0.949
    (unstable_flag = 1), so the declared screen rejects it and the paper now
    reports the complexity-8 auto-best, which is also the 6/12 recurring form.
    Verbatim from seed 53:
      CH4 = -1676.962 / (sqrt(SR*Ts) + (-216.37347 - AUC_dry))
    """
    SRTs = d["SR*Ts"].to_numpy(float)
    AUCd = d["AUC_dry"].to_numpy(float)
    return -1676.962 / (np.sqrt(SRTs) + (-216.37347 - AUCd))


def eq_pooled(d):
    """Pooled, best_accuracy_ruleA, complexity 40, CV +0.596."""
    hb = d["hbar_wet"].to_numpy(float)
    hi = d["h_inv"].to_numpy(float)
    Tv = d["Tv_C"].to_numpy(float)
    SRH = d["SR*HODsin"].to_numpy(float)
    SRv = d["SR*v"].to_numpy(float)
    hv = d["h*v"].to_numpy(float)
    hs = d["h*sinTOD"].to_numpy(float)
    VW = d["VPD*WS"].to_numpy(float)
    inner = (((SRH + (SRv / 1.6380328)) + (((hv - hs) - (hb * Tv)) * Tv))
             * -1.7110912e-7) * (hb - 5.6869154)
    return (hb - 0.40055007) * ((((hi * 0.00034019744)
                                  - (Tv * (Tv * inner))) * hb)
                                + np.exp(VW * -39.338554))


ARMS = {
    "JPN": dict(
        csv=os.path.join(BASE, r"JPN\Data-Metadata\JPN_retvars_pass2_C.csv"),
        outdir=os.path.join(OUT, "JPN"), equation=eq_jpn,
        form="complex_ruleA cplx 22", pysr_cv_r2=0.542),
    "KOR": dict(
        csv=os.path.join(BASE, r"KOR\Data_Metadata\Papale_hampel_cleaned\KOR_retvars_pass2_C.csv"),
        outdir=os.path.join(OUT, "KOR"), equation=eq_kor,
        form="best_accuracy_ruleA cplx 40", pysr_cv_r2=0.654),
    "PHL": dict(
        csv=os.path.join(BASE, r"PHL\Data_Metadata\PHL_retvars_pass2_C.csv"),
        outdir=os.path.join(OUT, "PHL"), equation=eq_phl,
        form="auto-best/recurring cplx 8", pysr_cv_r2=0.465),
    "POOLED": dict(
        csv=os.path.join(BASE, r"POOLED\Data-Metadata\POOLED_retvars_pass2_ISO_C.csv"),
        outdir=os.path.join(OUT, "POOLED"), equation=eq_pooled,
        form="best_accuracy_ruleA cplx 40", pysr_cv_r2=0.596),
}


def make_rf():
    return RandomForestRegressor(n_estimators=400, min_samples_leaf=5,
                                 n_jobs=-1, random_state=RANDOM_STATE)


def make_nn():
    pipe = Pipeline([
        ("scale", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), activation="relu",
                             solver="adam", max_iter=2000, early_stopping=True,
                             n_iter_no_change=20, random_state=RANDOM_STATE)),
    ])
    return TransformedTargetRegressor(regressor=pipe, transformer=StandardScaler())


def parse_dates_robust(series, verbose=True):
    s = series.astype(str).str.strip()
    n = len(s)
    cands = []
    for flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=flag, format="mixed")
        pv = parsed.dropna()
        diffs = pv.diff().dropna()
        cands.append({"dayfirst": flag, "parsed": parsed,
                      "valid": int(parsed.notna().sum()),
                      "mono": float((diffs >= pd.Timedelta(0)).sum()) / max(1, len(diffs))})
    best = max(cands, key=lambda c: (c["mono"], c["valid"]))
    if verbose:
        for c in cands:
            tag = "  <-- chosen" if c is best else ""
            print(f"    dayfirst={str(c['dayfirst']):<5} parsed={c['valid']}/{n} "
                  f"monotonic={c['mono']:.2%}{tag}")
    return best["parsed"]


def metrics(y, p):
    m = np.isfinite(p)
    if m.sum() < 10:
        return np.nan
    ss = float(np.sum((y[m] - y[m].mean()) ** 2))
    return float(1 - np.sum((y[m] - p[m]) ** 2) / ss) if ss > 0 else np.nan


def day_folds(day, n_folds, seed):
    days = pd.DatetimeIndex(day.drop_duplicates()).to_numpy().copy()
    rng = np.random.RandomState(seed)
    rng.shuffle(days)
    fs = len(days) // n_folds
    return [day.isin(days[k * fs:] if k == n_folds - 1
                     else days[k * fs:(k + 1) * fs]).values
            for k in range(n_folds)]


def cv(make_model, X, y, folds):
    out = []
    for te in folds:
        tr = ~te
        m = make_model()
        m.fit(X[tr], y[tr])
        out.append(metrics(y[te], m.predict(X[te])))
    return np.array(out, dtype=float)


def fmt(a):
    a = a[np.isfinite(a)]
    return f"{a.mean():+.3f} ± {a.std():.3f}" if len(a) else "   n/a"


def run_arm(arm, cfg):
    print("=" * 78)
    print(f"  {arm}   form: {cfg['form']}   PySR CV R2 = {cfg['pysr_cv_r2']:+.3f}")
    print("=" * 78)
    if not os.path.isfile(cfg["csv"]):
        print(f"  [SKIP] not found: {cfg['csv']}\n")
        return
    os.makedirs(cfg["outdir"], exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    df = pd.read_csv(cfg["csv"], low_memory=False)
    for c in df.columns:
        if c not in (TIME_COL, "site"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.replace(MISSING_FLAGS, np.nan)

    predictors = [c for c in df.columns if c not in NON_PREDICTORS]
    print(f"\n  Predictors ({len(predictors)}): {predictors}")
    print("\n  Parsing dates:")
    dates = parse_dates_robust(df[TIME_COL])

    sub = df[predictors + [TARGET_COL]].copy()
    sub["__d__"] = dates
    sub = sub.dropna().reset_index(drop=True)
    day = sub["__d__"].dt.normalize()
    print(f"  Complete cases: {len(sub):,} half-hours over {day.nunique():,} days")

    X = sub[predictors].to_numpy(float)
    y = sub[TARGET_COL].to_numpy(float)
    folds = day_folds(day, N_FOLDS, RANDOM_STATE)
    rows = []

    print("\n  (1) CEILING — drivers -> measured flux")
    rf = cv(make_rf, X, y, folds)
    nn = cv(make_nn, X, y, folds)
    print(f"      RF   CV R2 = {fmt(rf)}")
    print(f"      NN   CV R2 = {fmt(nn)}")
    print(f"      PySR CV R2 = {cfg['pysr_cv_r2']:+.3f}")
    ceiling = float(np.nanmax([np.nanmean(rf), np.nanmean(nn)]))
    gap = ceiling - cfg["pysr_cv_r2"]
    verdict = ("equation is at the ceiling" if gap < 0.05 else
               "modest headroom" if gap < 0.15 else
               "SIGNAL LEFT ON THE TABLE")
    print(f"      ceiling {ceiling:+.3f}, gap {gap:+.3f}  ->  {verdict}")
    rows += [{"task": "ceiling", "model": "RF", "r2_mean": float(np.nanmean(rf)),
              "r2_std": float(np.nanstd(rf))},
             {"task": "ceiling", "model": "NN", "r2_mean": float(np.nanmean(nn)),
              "r2_std": float(np.nanstd(nn))},
             {"task": "ceiling", "model": "PySR", "r2_mean": cfg["pysr_cv_r2"],
              "r2_std": np.nan}]

    print("\n  (2) RESIDUAL — drivers -> (measured - equation)")
    pred = np.asarray(cfg["equation"](sub), dtype=float)
    ok = np.isfinite(pred)
    print(f"      equation on all rows: R2 = {metrics(y, pred):+.3f}   "
          f"({int((~ok).sum())} non-finite)")
    resid = y - pred
    resid[~ok] = np.nan
    keep = np.isfinite(resid)
    Xr, rr = X[keep], resid[keep]
    fr = [f[keep] for f in folds]
    rf_r = cv(make_rf, Xr, rr, fr)
    nn_r = cv(make_nn, Xr, rr, fr)
    print(f"      RF on residual: CV R2 = {fmt(rf_r)}")
    print(f"      NN on residual: CV R2 = {fmt(nn_r)}")
    best = float(np.nanmax([np.nanmean(rf_r), np.nanmean(nn_r)]))
    v2 = ("residual is NOISE -> form certified sufficient" if best < 0.05 else
          "minor residual structure -> optional correction term" if best < 0.20 else
          "SUBSTANTIAL residual structure -> a term is missing")
    print(f"      best residual CV R2 = {best:+.3f}  ->  {v2}")
    rows += [{"task": "residual", "model": "RF", "r2_mean": float(np.nanmean(rf_r)),
              "r2_std": float(np.nanstd(rf_r))},
             {"task": "residual", "model": "NN", "r2_mean": float(np.nanmean(nn_r)),
              "r2_std": float(np.nanstd(nn_r))}]

    m = make_rf()
    m.fit(Xr, rr)
    perm = permutation_importance(m, Xr, rr, n_repeats=10,
                                  random_state=RANDOM_STATE, n_jobs=-1)
    order = np.argsort(perm.importances_mean)[::-1]
    print("\n      Which driver the form under-uses:")
    print(f"        {'predictor':<16}{'impurity':>10}{'perm_mean':>11}{'perm_std':>10}")
    imp = []
    for i in order:
        print(f"        {predictors[i]:<16}{m.feature_importances_[i]:>10.3f}"
              f"{perm.importances_mean[i]:>11.3f}{perm.importances_std[i]:>10.3f}")
        imp.append({"predictor": predictors[i],
                    "impurity": float(m.feature_importances_[i]),
                    "perm_mean": float(perm.importances_mean[i]),
                    "perm_std": float(perm.importances_std[i])})

    p1 = os.path.join(cfg["outdir"], f"nnrf_benchmark_{arm}_{ts}.csv")
    p2 = os.path.join(cfg["outdir"], f"residual_importance_{arm}_{ts}.csv")
    pd.DataFrame(rows).to_csv(p1, index=False)
    pd.DataFrame(imp).to_csv(p2, index=False)
    print(f"\n  [SAVED] {p1}\n  [SAVED] {p2}\n")


def main():
    wanted = [a.upper() for a in sys.argv[1:]] or list(ARMS)
    bad = [a for a in wanted if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(ARMS)}")
    print(f"RF + NN benchmark — day-grouped {N_FOLDS}-fold CV, seed {RANDOM_STATE}\n")
    for a in wanted:
        run_arm(a, ARMS[a])


if __name__ == "__main__":
    main()
