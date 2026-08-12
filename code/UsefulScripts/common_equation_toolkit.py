#!/usr/bin/env python3
"""
common_equation_toolkit.py
==============================================================================
Find a CH4 equation that actually TRANSFERS between sites, instead of one that
merely fits the pooled cloud.

THE DIAGNOSIS THIS IMPLEMENTS
    AUC is an EXTENSIVE quantity: cm x h accumulated since the record began. Its
    magnitude is set by how long and how wet the record is, not by the process.
    Measured on the three-site pool:

        JP-MSE / PH-IR / SK-CRK, fraction of each site's rows inside the
        three-site common interval (as printed by stage 1, with the 24 h guard):

            AUC   [0.000, 409.7] cm h   3.2% / 12.1% /  2.3%
            hbar  [2.114,   3.15] cm   58.0% /  1.8% / 42.2%
            fwet  [0.686,  0.891]      77.0% /  2.2% / 62.6%

        PH-IR stays low because it is drained 85% of the time; no change of
        variable fixes that, and it is the reason strict LOSO is capped.

    An equation in absolute AUC can never transfer, because the held-out site sits
    outside the training support by construction. Converting to an INTENSIVE
    quantity (mean ponding depth, cm) fixes the support problem:

        es*exp(c*AUC*es)            strict LOSO  -0.15 / +0.16 / +0.29   worst -0.15
        es*exp(c*(AUC/Deltime)*es)  strict LOSO  +0.28 / +0.21 / +0.27   worst +0.21

WHAT THIS SCRIPT DOES
    STAGE 1  Derive intensive water features and write an augmented CSV.
    STAGE 2  Harvest every Pareto equation from PySR equation_report_*.txt files.
    STAGE 3  Auto-parameterise each equation (one free parameter per DISTINCT
             numeric literal, the stage-8 convention) and score it by
             STRICT LOSO: fit on two sites, predict the third, no refit.
    STAGE 4  RANK BY WORST-SITE STRICT LOSO, not by pooled R2. This is the step
             the pipeline never had. Pooled R2 and transferability are close to
             uncorrelated in your data; ranking on the former picks equations
             that fail on the latter.

    It also prints the PySR call with site-balanced weights, which removes the
    other half of the problem: SK-CRK supplies 44% of pooled rows, so an
    unweighted search is 44% optimised for Cheorwon.

USAGE
    python common_equation_toolkit.py                 # stages 1-4 on REPORT_GLOB
    python common_equation_toolkit.py --features-only # stage 1 only

Author: Jef Zerrudo / Claude.  Requires numpy, pandas, scipy.
==============================================================================
"""

import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

warnings.filterwarnings("ignore")
np.seterr(all="ignore")

# ── CONFIG ───────────────────────────────────────────────────────────────────
IN_CSV      = r"POOL_3sites_growingseason_VPDfix_REPAIRED.csv"
OUT_CSV     = r"POOL_3sites_INTENSIVE.csv"
REPORT_GLOB = r"equation_report_*.txt"       # PySR reports to re-rank
TARGET      = "F_CH4_F"
SITE        = "site"
TIME        = "Date"
DELTIME     = "Deltime"

MIN_DELTIME   = 24.0   # burn-in guard: mean depth is noisy in the first day
MAX_COMPLEXITY = 35
MAX_PARAMS     = 6     # equations with more free constants are not identifiable here
TOP_K_STRUCT   = 15    # run the slower structure-only pass on this many leaders


# ── STAGE 1: intensive water features ────────────────────────────────────────
def add_intensive_features(d):
    """Extensive -> intensive. Each new column is a rate or a mean, so its scale
    does not grow with season length and it is comparable between sites.
    Every one is obtainable from ORYZA (which simulates depth) plus a clock."""
    dt = pd.to_numeric(d[DELTIME], errors="coerce")
    ok = dt >= MIN_DELTIME

    def per_hour(col, name):
        if col in d.columns:
            v = pd.to_numeric(d[col], errors="coerce") / dt
            d[name] = v.where(ok)

    per_hour("AUC",     "hbar")        # mean net ponding depth since record start, cm
    per_hour("AUC_wet", "hbar_wet")    # mean wet-side depth, cm
    per_hour("AUC_dry", "hbar_dry")    # mean dry-side depth, cm

    # Fraction of elapsed time the field has been flooded (dimensionless, 0-1).
    if "depth" in d.columns:
        wet = (pd.to_numeric(d["depth"], errors="coerce") > 0).astype(float)
        d["fwet"] = wet.groupby(d[SITE]).cumsum() / (
            wet.groupby(d[SITE]).cumcount() + 1.0)

    # Interactions with the thermal driver, matching the recurrent pooled form.
    if "es" in d.columns:
        es = pd.to_numeric(d["es"], errors="coerce")
        for c in ["hbar", "hbar_wet"]:
            if c in d.columns:
                d[f"{c}*es"] = d[c] * es
    return d


def stage1():
    d = pd.read_csv(IN_CSV, low_memory=False)
    n0 = len(d)
    d = add_intensive_features(d)
    d.to_csv(OUT_CSV, index=False)
    new = [c for c in ["hbar", "hbar_wet", "hbar_dry", "fwet", "hbar*es", "hbar_wet*es"]
           if c in d.columns]
    print(f"[stage 1] {IN_CSV} -> {OUT_CSV}   rows {n0}   added: {new}")
    print("\n  support overlap (fraction of each site's rows inside the 3-site common interval)")
    for col in ["AUC", "hbar", "fwet"]:
        if col not in d.columns:
            continue
        g = d.dropna(subset=[col]).groupby(SITE)[col]
        lo, hi = g.min().max(), g.max().min()
        frac = {s: float(((v >= lo) & (v <= hi)).mean()) for s, v in
                d.dropna(subset=[col]).groupby(SITE)[col]}
        pct = "  ".join(f"{s} {100*f:5.1f}%" for s, f in sorted(frac.items()))
        print(f"    {col:6s} [{lo:9.3f},{hi:9.3f}]   {pct}")
    return d


# ── STAGE 2: harvest Pareto equations from PySR reports ──────────────────────
EQ_RE  = re.compile(r"Equation #(\d+)\s+\[complexity = (\d+)\].*?CH4 = (.+?)(?=\n\s*\n|\n  Equation|\Z)",
                    re.S)
PRED_RE = re.compile(r"^Predictors:\s*(.+)$", re.M)


def harvest(paths):
    out = []
    for p in paths:
        txt = open(p, encoding="utf-8", errors="ignore").read()
        m = PRED_RE.search(txt)
        preds = [x.strip() for x in m.group(1).split(",")] if m else []
        for num, cx, eq in EQ_RE.findall(txt):
            out.append(dict(source=os.path.basename(os.path.dirname(p)) or os.path.basename(p),
                            eq_no=int(num), complexity=int(cx),
                            equation=" ".join(eq.split()), predictors=preds))
    return pd.DataFrame(out)


# ── STAGE 3: auto-parameterise a printed equation ────────────────────────────
NUM_RE = re.compile(r"(?<![A-Za-z_])(\d+\.\d+(?:[eE][-+]?\d+)?|\d+(?:[eE][-+]?\d+)|\d+\.)")
SAFE = {"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "tanh": np.tanh, "np": np}


def _tok(i):
    s, i = "", i + 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(97 + r) + s
    return f"zz{s}zz"


def parameterise(eq, feature_names):
    """Return (func(X,*p), p0, used_features) using the stage-8 convention:
    one free parameter per DISTINCT numeric literal; repeats tie to one parameter."""
    expr = eq
    order = sorted(feature_names, key=len, reverse=True)
    tokmap = {}
    for k, name in enumerate(order):
        # Identifier-boundary match, so short names like 'q' or 'es' are not
        # substituted inside 'sqrt' or 'exp' or inside a longer feature name.
        pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
        if pat.search(expr):
            t = _tok(k)
            expr = pat.sub(t, expr)
            tokmap[t] = name
    lits = []
    def sub(m):
        v = float(m.group(1))
        for j, u in enumerate(lits):
            if u == v:
                return f"p[{j}]"
        lits.append(v)
        return f"p[{len(lits)-1}]"
    expr = NUM_RE.sub(sub, expr)
    used = [tokmap[t] for t in tokmap if t in expr]
    if not used or not lits or len(lits) > MAX_PARAMS:
        return None
    idx = {n: i for i, n in enumerate(used)}
    for t, name in tokmap.items():
        if t in expr:
            expr = expr.replace(t, f"X[{idx[name]}]")
    code = compile(expr, "<eq>", "eval")
    def f(X, *p):
        return eval(code, {"__builtins__": {}}, {**SAFE, "X": X, "p": list(p)})
    return f, lits, used


# ── scoring ──────────────────────────────────────────────────────────────────
def r2(y, p):
    p = np.asarray(p, float)
    if p.ndim == 0:
        p = np.full_like(y, float(p))
    m = np.isfinite(p) & np.isfinite(y)
    if m.sum() < 20:
        return np.nan
    ss = np.sum((y[m] - y[m].mean()) ** 2)
    return 1 - np.sum((y[m] - p[m]) ** 2) / ss if ss > 0 else np.nan


def pack(g, used):
    return tuple(g[c].to_numpy(float) for c in used)


def strict_loso(f, p0, used, d, sites):
    out = {}
    for s in sites:
        tr, te = d[d[SITE] != s], d[d[SITE] == s]
        n = tr.groupby(SITE)[TARGET].transform("size").to_numpy(float)
        w = (len(tr) / 2.0) / n                      # site-balanced training
        try:
            pp, _ = curve_fit(f, pack(tr, used), tr[TARGET].to_numpy(float),
                              p0=p0, sigma=1 / np.sqrt(w), maxfev=60000)
            out[s] = r2(te[TARGET].to_numpy(float), f(pack(te, used), *pp))
        except Exception:
            out[s] = np.nan
    return out


def structure_only(f, p0, used, d, sites, n_folds=5, seed=42):
    out = {}
    for s in sites:
        g = d[d[SITE] == s]
        days = pd.DatetimeIndex(g["__day__"].drop_duplicates()).to_numpy().copy()
        rng = np.random.RandomState(seed); rng.shuffle(days)
        fs = max(1, len(days) // n_folds)
        sc = []
        for k in range(n_folds):
            te = g["__day__"].isin(days[k*fs:] if k == n_folds-1 else days[k*fs:(k+1)*fs]).values
            tr = ~te
            try:
                pp, _ = curve_fit(f, pack(g[tr], used), g[tr][TARGET].to_numpy(float),
                                  p0=p0, maxfev=60000)
                sc.append(r2(g[te][TARGET].to_numpy(float), f(pack(g[te], used), *pp)))
            except Exception:
                sc.append(np.nan)
        out[s] = float(np.nanmean(sc))
    return out


# ── PySR configuration to use next time ──────────────────────────────────────
PYSR_SNIPPET = '''
# ── Run PySR with SITE-BALANCED weights on the INTENSIVE feature set ─────────
# Without weights the search is ~44% optimised for SK-CRK, which supplies 44%
# of pooled rows. weights= is native PySR and needs no Julia.
import numpy as np
from pysr import PySRRegressor

counts = df["site"].value_counts()
w = (len(df) / len(counts)) / df["site"].map(counts).to_numpy(float)

model = PySRRegressor(
    niterations=2000, maxsize=35, populations=20,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["exp", "log", "sqrt", "tanh"],
    nested_constraints={"exp": {"exp": 1, "log": 0},
                        "log": {"log": 0, "exp": 1},
                        "tanh": {"tanh": 0}},
    constraints={"exp": 8, "log": 8, "tanh": 8, "sqrt": 10},
    parsimony=0.0032, batching=True, random_state=SEED,
)
model.fit(X, y, weights=w)          # <- the one-line change that matters
'''


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    d = stage1()
    if "--features-only" in sys.argv:
        print("\n[PySR call to use next time]"); print(PYSR_SNIPPET); return

    paths = sorted(glob.glob(REPORT_GLOB)) + sorted(glob.glob(os.path.join("**", REPORT_GLOB),
                                                             recursive=True))
    paths = sorted(set(paths))
    if not paths:
        print(f"\n[stage 2] no files matched {REPORT_GLOB}. Point REPORT_GLOB at your "
              f"equation_report_*.txt files."); print(PYSR_SNIPPET); return

    eqs = harvest(paths)
    eqs = eqs[eqs.complexity <= MAX_COMPLEXITY].drop_duplicates("equation").reset_index(drop=True)
    print(f"\n[stage 2] harvested {len(eqs)} distinct Pareto equations from {len(paths)} report(s)")

    d["__day__"] = pd.to_datetime(d[TIME], errors="coerce", format="mixed").dt.normalize()
    sites = sorted(d[SITE].dropna().unique())
    feats = [c for c in d.columns if c not in {SITE, TIME, TARGET, "__day__", "w", "F_CH4_F_orig"}]
    for c in feats + [TARGET]:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    print(f"[stage 3] strict LOSO on each equation ({len(sites)} fits each) ...")
    rows = []
    for i, r in eqs.iterrows():
        got = parameterise(r.equation, feats)
        if got is None:
            continue
        f, p0, used = got
        sub = d.dropna(subset=used + [TARGET, "__day__"])
        if len(sub) < 500 or sub[SITE].nunique() < len(sites):
            continue
        sl = strict_loso(f, p0, used, sub, sites)
        vals = [sl[s] for s in sites]
        rows.append(dict(source=r.source, eq_no=r.eq_no, complexity=r.complexity,
                         n_params=len(p0), drivers="|".join(used),
                         **{f"strict_{s}": sl[s] for s in sites},
                         worst_strict=np.nanmin(vals) if np.isfinite(vals).any() else np.nan,
                         equation=r.equation))
    res = pd.DataFrame(rows).sort_values("worst_strict", ascending=False).reset_index(drop=True)

    print(f"[stage 4] structure-only pass on the top {TOP_K_STRUCT} ...")
    for i in res.head(TOP_K_STRUCT).index:
        got = parameterise(res.at[i, "equation"], feats)
        if got is None:
            continue
        f, p0, used = got
        sub = d.dropna(subset=used + [TARGET, "__day__"])
        so = structure_only(f, p0, used, sub, sites)
        for s in sites:
            res.at[i, f"struct_{s}"] = so[s]

    res.to_csv("common_equation_ranking.csv", index=False)
    show = ["complexity", "n_params", "worst_strict"] + [f"strict_{s}" for s in sites] + ["equation"]
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
    print("\n" + "=" * 110)
    print("RANKED BY WORST-SITE STRICT LOSO  (fit on two sites, predict the third, no refit)")
    print("=" * 110)
    print(res.head(20)[show].to_string(index=False, float_format=lambda v: f"{v:7.3f}"))
    print("\nsaved -> common_equation_ranking.csv")
    print("\nRead the ranking, not the pooled R2. An equation at the top of this table")
    print("predicts a site it has never seen. One at the top of a pooled-R2 table does not.")
    print(PYSR_SNIPPET)


if __name__ == "__main__":
    main()
