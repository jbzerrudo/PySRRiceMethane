#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sobol_sensitivity.py
Variance-based (Sobol / given-data) sensitivity analysis of the recovered
per-site CH4 flux equations (Paper 1: PH-IR, JP-MSE, SK-CRK).

WHAT IT DOES
  For each site it (1) rebuilds the recovered symbolic equation FORM, (2) refits
  its coefficients to that site's own CSV by least squares -- this makes the model
  self-consistent with the file's units, sidestepping the m/cm inconsistencies
  across datasets -- (3) reports the reconstruction R^2 so you can confirm the fit
  matches the paper's in-sample skill, and (4) computes how the equation's output
  variance apportions among its driver inputs, two complementary ways:

    * given-data first-order index Si  (CORRELATION-INCLUSIVE): uses the real,
      correlated predictor values as observed. Answers "in this data, how much of
      the model's output variance is explained by knowing driver i".
    * independent Sobol Si / STi (STRUCTURAL): samples each input independently
      over its observed range (Saltelli/Jansen estimators). Answers "structurally,
      ignoring correlation, how sensitive is the equation to input i, alone (Si)
      and including interactions (STi)".

  Comparing the two shows where correlation matters (the SK-CRK story).

REQUIREMENTS:  numpy, pandas, scipy, matplotlib   (all standard; no SALib/shap)
USAGE:  edit INPUT_DIR / FILES below if needed, then:  python sobol_sensitivity.py
"""

import os, glob, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
warnings.filterwarnings("ignore")

# =====================================================================
# USER SETTINGS
# =====================================================================
INPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\finaldatasets"
OUTPUT_DIR = os.path.join(INPUT_DIR, "sobol_out")      # results written here

# CSV filenames (as delivered). If your files are named differently, either fix
# these strings or rely on the glob fallback patterns below.
FILES = {
    "PH-IR":        "PHL-IR_2016_retdvars2.csv",
    "JP-MSE":       "JPN-MSE_2012_retainedvars2.csv",
    "SK-CRK full":  "KOR-CRK_2018.0_retdvars2.csv",
    "SK-CRK g.s.":  "KOR-CRK_2018_updated_retvars2.csv",
}
GLOB_FALLBACK = {                     # used if the exact filename is not found
    "PH-IR":       ["*PHL*IR*.csv", "*PHL*.csv"],
    "JP-MSE":      ["*JPN*MSE*.csv", "*JP*MSE*.csv"],
    "SK-CRK full": ["*KOR*all*.csv", "*KOR*2018.0*.csv"],
    "SK-CRK g.s.": ["*KOR*grow*.csv", "*KOR*updated*.csv"],
}

# Unit harmonization to cm.  Only PH-IR and SK-CRK growing-season are fully cm.
# JP-MSE: ALL depth-derived variables (incl. AUC) are still in m -> x100.
# SK-CRK full: raw depth was converted, but its depth-DERIVED terms stayed in m -> x100.
# (x100 = m -> cm; AUC is depth-derived, so AUC scales too.)
UNIT_TO_CM = {
    "PH-IR":       [],                       # already cm-consistent
    "JP-MSE":      ["AUC", "h*Pr", "h*v"],    # depth-derived, in m
    "SK-CRK full": ["h*VPD", "h*Pr"],         # depth-derived, in m
    "SK-CRK g.s.": [],                        # already cm-consistent
}

TARGET   = "F_CH4_F"
N_SAMPLE = 2 ** 15          # base sample size for the Saltelli/Sobol sampler
N_BINS   = 20               # bins for the given-data first-order estimator
CLIP_PCT = (1, 99)          # sample independent Sobol over this percentile range
SEED     = 42

# =====================================================================
# EQUATION FORMS  (functional form fixed; coefficients are refit per file)
# Each entry: input columns, a label for the "primary" driver, the model f(X, *p),
# initial coefficient guesses (paper Table 6, only used to seed the refit), and a
# reference in-sample R^2 from the paper for your sanity check.
# =====================================================================

def f_phir(X, a, b, g, d, eps):
    # PH-IR Eq.(4) reduced:  a + (g*(SR*Ts) + b + eps*tanh(SR*HODsin)) * exp(d*AUC)
    AUC, SRTs, SRHOD = X
    return a + (g * SRTs + b + eps * np.tanh(SRHOD)) * np.exp(d * AUC)

def f_jpmse(X, g, d):
    # JP-MSE Eq.(3) bilinear:  AUC * (g + d*(SR*VPD))
    AUC, SRVPD = X
    return AUC * (g + d * SRVPD)

def f_korfull(X, a, b, g, d):
    # SK-CRK full Eq.(1):  g*(h*VPD) + b*(a*(SR*v)+dayhr)*E + exp(d*E),  E = exp(asinh_Ta)
    hVPD, SRv, dayhr, E = X
    return g * hVPD + b * (a * SRv + dayhr) * E + np.exp(d * E)

def f_korgs(X, a, b, c, k, E, f, g):
    # SK-CRK g.s. Eq.(5):  k*(AUC_wet + b/(a + g*(AUC_wet+c)^f)) * (SR + E)
    AW, SR = X
    return k * (AW + b / (a + g * np.power(np.maximum(AW + c, 1e-9), f))) * (SR + E)

SITES = {
    "PH-IR": dict(
        file="PH-IR", func=f_phir,
        inputs=["AUC", "SR*Ts", "SR*HODsin"],
        driver_labels=["AUC (exp kernel)", "SR*Ts", "SR*HODsin"],
        transform={"E_of": None},
        p0=[0.87, 1.67, 2.3e-4, 3.3e-3, -1.72], ref_r2=0.567,
    ),
    "JP-MSE": dict(
        file="JP-MSE", func=f_jpmse,
        inputs=["AUC", "SR*VPD"],
        driver_labels=["AUC", "SR*VPD"],
        transform={"E_of": None},
        p0=[6.5e-4, 5.6e-8], ref_r2=0.409,   # Table 6 (cm) coeffs; valid after AUC x100
    ),
    "SK-CRK full": dict(
        file="SK-CRK full", func=f_korfull,
        inputs=["h*VPD", "SR*v", "dayhr", "asinh_Ta"],
        driver_labels=["h*VPD", "SR*v", "dayhr", "temperature (asinh_Ta)"],
        transform={"E_of": "asinh_Ta"},   # this column is fed to the model as exp(col)
        p0=[4.8e-3, -5.3e-3, -3.99e-2, 5.4e-2], ref_r2=0.449,
    ),
    "SK-CRK g.s.": dict(
        file="SK-CRK g.s.", func=f_korgs,
        inputs=["AUC_wet", "SR"],
        driver_labels=["AUC_wet", "SR"],
        transform={"E_of": None},
        p0=[0.075, 1.10, 2.03, 1.6e-6, 747.0, -0.50, -7.64], ref_r2=0.343,
    ),
}

# =====================================================================
# HELPERS
# =====================================================================
def resolve_path(site):
    fn = FILES[site]
    p = os.path.join(INPUT_DIR, fn)
    if os.path.exists(p):
        return p
    for pat in GLOB_FALLBACK.get(site, []):
        hits = glob.glob(os.path.join(INPUT_DIR, pat))
        if hits:
            print(f"   [glob] {site}: using {os.path.basename(hits[0])}")
            return hits[0]
    raise FileNotFoundError(f"No CSV for {site} in {INPUT_DIR} (looked for {fn})")

def num(df, c):
    return pd.to_numeric(df[c], errors="coerce").values

def r2_score(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    ss_res = np.sum((y[m] - yhat[m]) ** 2)
    ss_tot = np.sum((y[m] - np.mean(y[m])) ** 2)
    return 1.0 - ss_res / ss_tot

def build_design(df, cfg):
    """Return the model-input matrix (list of arrays, in equation order) and target,
    applying the exp() transform to the temperature column where the form needs it."""
    E_of = cfg["transform"]["E_of"]
    to_cm = set(UNIT_TO_CM.get(cfg["file"], []))
    cols = []
    for c in cfg["inputs"]:
        v = num(df, c)
        if c in to_cm:                   # harmonize m -> cm for depth-derived terms
            v = v * 100.0
        if c == E_of:                    # model uses exp(asinh_Ta)
            v = np.exp(v)
        cols.append(v)
    y = num(df, TARGET)
    return cols, y

def given_data_first_order(xi, yhat, n_bins=N_BINS):
    """First-order index from real (correlated) samples: Var(E[yhat|xi]) / Var(yhat),
    estimated by ranking xi into equal-count bins. Correlation-inclusive."""
    m = np.isfinite(xi) & np.isfinite(yhat)
    xi, y = xi[m], yhat[m]
    order = np.argsort(xi)
    y = y[order]
    edges = np.linspace(0, len(y), n_bins + 1).astype(int)
    means = [y[edges[k]:edges[k + 1]].mean() for k in range(n_bins) if edges[k + 1] > edges[k]]
    means = np.array(means)
    counts = np.array([edges[k + 1] - edges[k] for k in range(n_bins) if edges[k + 1] > edges[k]])
    grand = np.average(means, weights=counts)
    between = np.average((means - grand) ** 2, weights=counts)
    return max(0.0, between / np.var(y))

def sobol_saltelli(func_vec, ranges, n=N_SAMPLE, seed=SEED):
    """Independent-input Sobol S1 (Saltelli) and ST (Jansen), inputs ~ Uniform(range).
    func_vec takes a list of 1-D arrays (equation input order) and returns yhat."""
    rng = np.random.default_rng(seed)
    d = len(ranges)
    def sample():
        return [rng.uniform(lo, hi, n) for (lo, hi) in ranges]
    A = sample(); B = sample()
    yA = func_vec(A); yB = func_vec(B)
    good = np.isfinite(yA) & np.isfinite(yB)
    varY = np.var(np.concatenate([yA[good], yB[good]]))
    S1, ST = [], []
    for i in range(d):
        AB = list(A); AB[i] = B[i]
        yAB = func_vec(AB)
        g = good & np.isfinite(yAB)
        S1.append(np.mean(yB[g] * (yAB[g] - yA[g])) / varY)          # Saltelli 2010
        ST.append(0.5 * np.mean((yA[g] - yAB[g]) ** 2) / varY)        # Jansen 1999
    return np.array(S1), np.array(ST)

# =====================================================================
# MAIN
# =====================================================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.ravel()

    for ax, (site, cfg) in zip(axes, SITES.items()):
        print("=" * 66); print(site)
        df = pd.read_csv(resolve_path(site))
        cols, y = build_design(df, cfg)

        # ---- refit the equation form to this file's columns ----
        stack = np.vstack(cols + [y])
        keep = np.all(np.isfinite(stack), axis=0)
        Xk = [c[keep] for c in cols]; yk = y[keep]
        try:
            popt, _ = curve_fit(cfg["func"], tuple(Xk), yk, p0=cfg["p0"], maxfev=60000)
        except Exception as e:
            print("   refit failed, using paper coefficients:", str(e)[:70])
            popt = np.array(cfg["p0"])
        yhat = cfg["func"](tuple(Xk), *popt)
        r2 = r2_score(yk, yhat)
        print(f"   n={keep.sum()}  reconstruction R2={r2:.3f}  (paper in-sample ~{cfg['ref_r2']})")

        # ---- sensitivity: given-data first-order (correlation-inclusive) ----
        gd = np.array([given_data_first_order(Xk[i], yhat) for i in range(len(Xk))])

        # ---- sensitivity: independent Sobol S1/ST (structural) ----
        ranges = [tuple(np.percentile(Xk[i], CLIP_PCT)) for i in range(len(Xk))]
        func_vec = lambda Xlist: cfg["func"](tuple(Xlist), *popt)
        S1, ST = sobol_saltelli(func_vec, ranges)

        for lab, a, b, c in zip(cfg["driver_labels"], gd, S1, ST):
            rows.append(dict(site=site, driver=lab, given_data_Si=round(a, 3),
                             sobol_Si=round(b, 3), sobol_STi=round(c, 3)))
            print(f"     {lab:26s}  given-data Si={a:5.2f}   Sobol Si={b:5.2f}  STi={c:5.2f}")

        # ---- plot: given-data first-order shares ----
        idx = np.arange(len(cfg["driver_labels"]))
        ax.barh(idx, gd, color="#FF9B00")
        ax.set_yticks(idx); ax.set_yticklabels(cfg["driver_labels"], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 1)
        ax.set_title(f"{site}   (fit R$^2$={r2:.2f})", fontsize=11)
        ax.set_xlabel("given-data first-order index")

    fig.suptitle("Variance-based driver sensitivity of the recovered CH$_4$ equations",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_png = os.path.join(OUTPUT_DIR, "sobol_sensitivity.png")
    fig.savefig(out_png, dpi=200, bbox_inches="tight")

    res = pd.DataFrame(rows)
    out_csv = os.path.join(OUTPUT_DIR, "sobol_indices.csv")
    res.to_csv(out_csv, index=False)
    print("=" * 66)
    print("Saved:", out_csv)
    print("Saved:", out_png)
    print("\nInterpretation: a large given-data Si for AUC/AUC_wet at PH-IR, JP-MSE and")
    print("SK-CRK g.s. confirms water-exposure dominance; a large Si for temperature at")
    print("SK-CRK full (with AUC absent from that window's set) is the dual-window contrast.")

if __name__ == "__main__":
    main()
