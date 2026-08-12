#!/usr/bin/env python3
"""
ruleA_declared_screen.py  (v2)
==============================
Run Rule A exactly as Section 2.9 of the manuscript declares it:

    "From each of the twelve seeds we promote the four labelled candidates,
     discard any that fail the coefficient-stability screen, and report the
     highest day-grouped hold-out R^2 among those that remain."

v2 replaces the v1 reimplementation, whose selftest failed for two reasons:
  1. the label regex dropped every candidate whose label contains
     parentheses ("Auto-best (highest score)", "Best accuracy (most
     complex)"), so half the candidates never entered the audit;
  2. v1 assigned one parameter per printed literal WITHOUT sympy's automatic
     sign-folding/distribution, so shared coefficients stayed tied and the
     non-identifiability the deposited screen detects (Mase knee CoV 1.75,
     IRRI c24 CoV 0.95) could not appear.

v2 therefore copies the stage-8 conventions VERBATIM from code/6_PySR.py
(v6.4) instead of approximating them:
  * complete cases over the run's full predictor list, missing flags
    [-9999, -999900, -99999], robust date parsing (auto dayfirst);
  * folds: unique DAYs in order of appearance, shuffled once with
    RandomState(42), five contiguous chunks, last chunk takes the remainder;
  * parameterisation: _safe_sympify(equation, predictors), then every
    sympy Number atom except {0, 1, -1} becomes one free parameter with the
    printed value as warm start (sympy's evaluation folds signs into the
    Float atoms, which is what unties shared coefficients into +/- pairs,
    and turns sqrt into a fitted 0.5 exponent);
  * per-fold refit: curve_fit, maxfev=40000, no weights; on failure the
    warm start is recorded for that fold;
  * scoring: non-finite predictions dropped, >=10 finite required;
  * CoV: std(ddof=1)/|mean| over the finite fold values, NaN if
    |mean| <= 1e-12; unstable if any finite CoV > 0.5;
  * fold-R^2 spread: std(ddof=1), matching cv_summary.txt.

Validate before trusting: --selftest reruns each arm's deposited winning
seed (read from seed_selection_summary_*.txt) and compares against the
deposited stage8_auto_*/cv_summary.txt and coefficient_stability.csv,
printing PASS/FAIL per slot. Run the full sweep only after all selftest
rows pass.

Usage:
    python ruleA_declared_screen.py --repo "C:\\Users\\zerru001\\git\\PySRRiceMethane" --selftest
    python ruleA_declared_screen.py --repo "C:\\Users\\zerru001\\git\\PySRRiceMethane"

Requires: numpy, pandas, scipy, sympy.
"""

import argparse
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import sympy as sp
from scipy.optimize import curve_fit

# ── stage-8 constants (code/6_PySR.py v6.4) ────────────────────────────────
TARGET_COL    = "F_CH4_F"
TIME_COL      = "Date"
MISSING_FLAGS = [-9999, -999900, -99999]
N_FOLDS       = 5
RANDOM_STATE  = 42
MAXFEV        = 40000
UNSTABLE_COV  = 0.5

ARMS = {
    "JPN":    dict(run="results/pysr_runs/JPN/Metadata/rerun_05Aug2026/run_20260805_122821",
                   data="data/JPN_retvars_pass2_C.csv"),
    "KOR":    dict(run="results/pysr_runs/KOR/Metadata/rerun_05Aug2026/run_20260805_170113",
                   data="data/KOR_retvars_pass2_C.csv"),
    "PHL":    dict(run="results/pysr_runs/PHL/Metadata/rerun_06Aug2026/run_20260806_104044",
                   data="data/PHL/Data_Metadata/PHL_retvars_pass2_C.csv"),
    "POOLED": dict(run="results/pysr_runs/POOLED/Metadata/rerun_10Aug2026/run_20260810_111532",
                   data="data/POOLED/Data-Metadata/POOLED_retvars_pass2_ISO_C.csv"),
}

# Deposited slot names for the four labels (stage-8 _slot_prefix convention).
SLOT_OF_LABEL = {
    "Auto-best (highest score)":    "simple_ruleA",
    "Knee of Pareto front":         "knee_ruleA",
    "Mid-complexity candidate":     "complex_ruleA",
    "Best accuracy (most complex)": "best_accuracy_ruleA",
}

# ── helpers copied verbatim from code/6_PySR.py (v6.4) ─────────────────────
_SAFE_TRANSLATE = [
    ("*", "_STAR_"), ("/", "_SLASH_"), (".", "_DOT_"),
    ("-", "_DASH_"), ("+", "_PLUS_"), (" ", "_SP_"),
]

def _is_unsafe_predictor(name):
    return any(tok in name for tok in [c for c, _ in _SAFE_TRANSLATE])

def _sanitize_name(name):
    out = name
    for bad, good in _SAFE_TRANSLATE:
        out = out.replace(bad, good)
    return out

def _build_translation(predictors):
    safe = [_sanitize_name(p) for p in predictors]
    pairs_unsafe = [(p, s) for p, s in zip(predictors, safe)
                    if _is_unsafe_predictor(p)]
    pairs_unsafe.sort(key=lambda kv: -len(kv[0]))
    reverse = list(zip(safe, predictors))
    reverse.sort(key=lambda kv: -len(kv[0]))
    return safe, pairs_unsafe, reverse

def _apply_subs(s, pairs):
    out = s
    for a, b in pairs:
        out = out.replace(a, b)
    return out

def _safe_sympify(eq_str, predictors):
    try:
        safe_preds, eq_subs, _ = _build_translation(predictors)
        eq_safe = _apply_subs(eq_str, eq_subs)
        local = {sp_name: sp.Symbol(sp_name) for sp_name in safe_preds}
        for i, sp_name in enumerate(safe_preds):
            local[f"x{i}"] = sp.Symbol(sp_name)
        return sp.sympify(eq_safe, locals=local)
    except Exception:
        return None

def parse_dates_robust(series, verbose=False):
    s = series.astype(str).str.strip()
    cands = []
    for df_flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=df_flag,
                                    format="mixed")
        valid = int(parsed.notna().sum())
        pv = parsed.dropna()
        mono = (float((pv.diff().dropna() >= pd.Timedelta(0)).sum())
                / max(1, len(pv.diff().dropna())))
        cands.append({"dayfirst": df_flag, "parsed": parsed,
                      "valid": valid, "mono": mono})
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

# ── report parsing ─────────────────────────────────────────────────────────
BLOCK_RE = re.compile(r"^\s*--- Equation #\d+ \((?P<label>.+)\) ---\s*$")

def read_predictors(report_text):
    m = re.search(r"^Predictors:\s*(.+)$", report_text, re.M)
    if not m:
        raise RuntimeError("no 'Predictors:' line in equation report")
    return [p.strip() for p in m.group(1).split(",")]

def read_candidates(report_text):
    """The four labelled candidates: label, complexity, hold-out R2, equation.
    Line-based, so labels containing parentheses are kept whole (v1 bug)."""
    lines = report_text.splitlines()
    out, i = [], 0
    while i < len(lines):
        m = BLOCK_RE.match(lines[i])
        if not m:
            i += 1
            continue
        fields, eq = {}, None
        j = i + 1
        while j < len(lines) and not BLOCK_RE.match(lines[j]) \
                and not lines[j].startswith("==="):
            line = lines[j]
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip()] = val.strip()
                if key.strip() == "Equation":
                    k = j + 1
                    while k < len(lines) and not lines[k].strip():
                        k += 1
                    if k < len(lines) and lines[k].strip().startswith("CH4 ="):
                        eq = lines[k].strip()[len("CH4 ="):].strip()
                    break
            j += 1
        r2_key = next((k for k in fields if k.rstrip("\u00b2") == "R"), None)
        if eq and r2_key and "Complexity" in fields:
            out.append(dict(label=m["label"].strip(),
                            complexity=int(fields["Complexity"]),
                            holdout_r2=float(fields[r2_key]),
                            eq=eq))
        i = j
    return out

# ── model building (stage-8 parameterisation) ──────────────────────────────
def build_model(eq_str, predictors):
    expr = _safe_sympify(eq_str, predictors)
    if expr is None:
        raise ValueError("sympify failed")
    consts = [n for n in expr.atoms(sp.Number)
              if n not in (sp.S.Zero, sp.S.One, sp.S.NegativeOne)]
    param_syms = [sp.Symbol(f"p{i}") for i in range(len(consts))]
    free = expr.xreplace(dict(zip(consts, param_syms)))
    p0 = np.array([float(c) for c in consts])

    _, _, reverse = _build_translation(predictors)
    back = dict(reverse)
    var_syms = sorted(free.free_symbols - set(param_syms), key=lambda s: s.name)
    cols = [back.get(s.name, s.name) for s in var_syms]
    f = sp.lambdify(var_syms + param_syms, free, modules="numpy")

    def model(X, *p):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = f(*X, *p)
        y = np.asarray(y, dtype=float)
        if y.ndim == 0:
            y = np.full(len(X[0]), float(y))
        return y
    return model, p0, cols

# ── data + folds (stage-8 main() conventions) ──────────────────────────────
def load_arm_frame(repo, cfg, predictors):
    df = pd.read_csv(os.path.join(repo, cfg["data"]))
    for col in df.select_dtypes(include=["object"]).columns:
        if col == TIME_COL:
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(MISSING_FLAGS, np.nan)
    df[TIME_COL] = parse_dates_robust(df[TIME_COL])
    df = df.dropna(subset=[TIME_COL])
    df["DAY"] = df[TIME_COL].dt.date
    df = df.dropna(subset=predictors + [TARGET_COL])
    days = df["DAY"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(days)
    fold_size = len(days) // N_FOLDS
    folds = []
    for k in range(N_FOLDS):
        test_days = days[k * fold_size:] if k == N_FOLDS - 1 \
            else days[k * fold_size:(k + 1) * fold_size]
        folds.append(set(test_days))
    return df, folds

def screen_candidate(df, folds, eq_str, predictors):
    model, p0, cols = build_model(eq_str, predictors)
    y_all = df[TARGET_COL].to_numpy(float)
    X_all = tuple(df[c].to_numpy(float) for c in cols)
    day = df["DAY"].to_numpy()
    r2s, params = [], []
    for test_days in folds:
        te = np.array([d in test_days for d in day])
        tr = ~te
        Xtr = tuple(x[tr] for x in X_all)
        Xte = tuple(x[te] for x in X_all)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                popt, _ = curve_fit(model, Xtr, y_all[tr], p0=p0,
                                    maxfev=MAXFEV)
        except Exception:
            popt = p0.copy()
        params.append(np.asarray(popt, float))
        r2, _, _ = compute_metrics(y_all[te], model(Xte, *popt))
        r2s.append(r2)
    r2s = np.array(r2s, float)
    fp = np.stack(params, axis=0)
    covs = []
    for j in range(fp.shape[1]):
        v = fp[:, j][np.isfinite(fp[:, j])]
        if len(v) == 0:
            covs.append(np.nan)
            continue
        mean = float(v.mean())
        std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
        covs.append((std / abs(mean)) if (np.isfinite(std) and
                                          abs(mean) > 1e-12) else np.nan)
    covs = np.array(covs, float)
    max_cov = np.nanmax(covs) if len(covs) and np.any(np.isfinite(covs)) else 0.0
    unstable = bool(np.any(np.isfinite(covs) & (covs > UNSTABLE_COV)))
    return (float(np.nanmean(r2s)), float(np.nanstd(r2s, ddof=1)),
            float(max_cov), unstable, len(p0))

# ── deposited references (for --selftest) ──────────────────────────────────
def deposited_references(repo, cfg):
    run = os.path.join(repo, cfg["run"])
    sel = glob.glob(os.path.join(run, "seed_selection_summary_*.txt"))
    m = re.search(r"Selected seed for Stage 8:\s*(\d+)",
                  open(sel[0], encoding="utf-8").read())
    seed = int(m.group(1))
    stage8 = sorted(glob.glob(os.path.join(run, "stage8_auto_*")))[-1]
    cv_ref = {}
    for line in open(os.path.join(stage8, "cv_summary.txt"), encoding="utf-8"):
        m = re.match(r"\s*(\S+_ruleA)\s+mean R\S*\s+=\s+([-+0-9.]+)\s+\S+\s+([0-9.]+)",
                     line)
        if m and "nf" not in line:
            cv_ref[m.group(1)] = (float(m.group(2)), float(m.group(3)))
    cs = pd.read_csv(os.path.join(stage8, "coefficient_stability.csv"))
    cov_ref = (cs[cs.slot.str.endswith("_ruleA")]
               .groupby("slot")["fold_cov_abs"].max().to_dict())
    return seed, cv_ref, cov_ref

# ── modes ──────────────────────────────────────────────────────────────────
def run_selftest(repo, arm, cfg):
    seed, cv_ref, cov_ref = deposited_references(repo, cfg)
    rp = glob.glob(os.path.join(repo, cfg["run"], f"seed_{seed}",
                                "equation_report_*.txt"))[0]
    text = open(rp, encoding="utf-8").read()
    predictors = read_predictors(text)
    df, folds = load_arm_frame(repo, cfg, predictors)
    print(f"\n== {arm} selftest, deposited winning seed {seed}, "
          f"complete cases {len(df)} ==")
    print(f"{'slot':<22}{'CV here':>14}{'CV deposited':>16}"
          f"{'CoV here':>10}{'CoV dep.':>10}  verdict")
    all_ok = True
    for c in read_candidates(text):
        slot = SLOT_OF_LABEL.get(c["label"])
        if slot is None:
            continue
        mean, sd, max_cov, unstable, _ = screen_candidate(
            df, folds, c["eq"], predictors)
        ref_mean, ref_sd = cv_ref.get(slot, (np.nan, np.nan))
        ref_cov = cov_ref.get(slot, np.nan)
        ok = (abs(mean - ref_mean) <= 0.02
              and (unstable == (ref_cov > UNSTABLE_COV)))
        all_ok &= ok
        print(f"{slot:<22}{mean:+.3f} \u00b1 {sd:.3f}   "
              f"{ref_mean:+.3f} \u00b1 {ref_sd:.3f}   "
              f"{max_cov:8.2f}{ref_cov:10.2f}  "
              f"{'PASS' if ok else 'FAIL'}")
    print(f"{arm}: {'ALL PASS' if all_ok else 'FAILED, do not run the sweep'}")
    return all_ok

def run_sweep(repo, arm, cfg, out_prefix):
    rps = sorted(glob.glob(os.path.join(repo, cfg["run"], "seed_*",
                                        "equation_report_*.txt")))
    predictors = read_predictors(open(rps[0], encoding="utf-8").read())
    df, folds = load_arm_frame(repo, cfg, predictors)
    rows = []
    for rp in rps:
        seed = int(re.search(r"seed_(\d+)", rp).group(1))
        for c in read_candidates(open(rp, encoding="utf-8").read()):
            try:
                mean, sd, max_cov, unstable, n_par = screen_candidate(
                    df, folds, c["eq"], predictors)
                rows.append({**c, "arm": arm, "seed": seed, "n_params": n_par,
                             "cv_r2_mean": mean, "cv_r2_sd": sd,
                             "max_cov": max_cov, "stable": not unstable,
                             "error": ""})
            except Exception as e:
                rows.append({**c, "arm": arm, "seed": seed, "n_params": np.nan,
                             "cv_r2_mean": np.nan, "cv_r2_sd": np.nan,
                             "max_cov": np.nan, "stable": False,
                             "error": f"{type(e).__name__}: {e}"})
    tab = pd.DataFrame(rows).sort_values("holdout_r2", ascending=False)
    path = f"{out_prefix}_{arm}.csv"
    tab.to_csv(path, index=False)
    n_bad = int((tab["error"] != "").sum())
    print(f"\n== {arm}: {len(tab)} candidates screened "
          f"({n_bad} parse/fit errors, see CSV) ==")
    print(tab.head(8)[["seed", "label", "complexity", "holdout_r2",
                       "cv_r2_mean", "max_cov", "stable"]].to_string(index=False))
    adm = tab[tab["stable"]]
    if len(adm):
        w = adm.iloc[0]
        print(f"{arm}: declared Rule A selects seed {w['seed']}, {w['label']} "
              f"(c={w['complexity']}), hold-out R2 = {w['holdout_r2']:.3f}, "
              f"CV = {w['cv_r2_mean']:.3f} \u00b1 {w['cv_r2_sd']:.3f}, "
              f"max |CoV| = {w['max_cov']:.2f}")
    else:
        print(f"{arm}: NO admissible candidate; inspect {path}")
    print(f"full audit table -> {path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True,
                    help="path to the PySRRiceMethane clone (contains data/, results/)")
    ap.add_argument("--arm", choices=list(ARMS), action="append")
    ap.add_argument("--out", default="ruleA_declared_screen")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    arms = args.arm or list(ARMS)
    if args.selftest:
        results = [run_selftest(args.repo, a, ARMS[a]) for a in arms]
        print("\nSELFTEST:", "ALL ARMS PASS, the sweep can be trusted"
              if all(results) else "at least one FAIL, do not run the sweep")
    else:
        for a in arms:
            run_sweep(args.repo, a, ARMS[a], args.out)

if __name__ == "__main__":
    main()
