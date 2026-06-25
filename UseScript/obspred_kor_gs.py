#!/usr/bin/env python3
"""
SK-CRK growing-season subset: observed vs predicted half-hourly CH4 for the
three Rule-A (seed 42) candidate equations -- auto-best (= Eq. 5), Pareto knee,
and most-accurate. Whole-record (in-sample) scatter; reads only the
growing-season retvars2 CSV. Full-data coefficients.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CSV    = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv"
OUTPNG = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\obspred_kor_gs_new.png"
TARGET = "F_CH4_F"
COLOR  = "#762a83"
LIMS   = (-90, 140)

# --- the three Rule-A candidates (seed 42), full-data coefficients ----------
def eq_auto_best(df):   # #15, C18  == Eq. 5
    return 1.597917e-6 * (df["AUC_wet"] + 1.0970329 / (0.07465542 - 7.643319 / np.sqrt(df["AUC_wet"] + 2.0335147))) * (df["SR"] + 747.46045)

def eq_knee(df):        # #17, C20
    return (df["AUC_wet"] + 0.14832489 / (0.07478618 - 7.6433325 / np.sqrt(df["AUC_wet"] + 1.5054631))) * ((df["SR"] + (674.06104 - df["SR*HODsin"])) * 1.5518393e-6)

def eq_best(df):        # #29, C35
    return ((((df["AUC_wet"] - df["SR"]) + (np.tanh(df["WD"]) / (0.07466244 - 7.643604 / np.sqrt(df["AUC_wet"] + 1.1335045))))
             * (df["SR"] + ((df["WD"] - df["SR*HODsin"]) + 678.24854))) * 1.6777989e-6) - (np.exp(np.tanh(df["SR*v"] - (df["h_inv"] + 3.7988737))) + 1.1335045)

CANDS = [("Auto-best (= Eq. 5)", 18, eq_auto_best, "passes CoV"),
         ("Knee of Pareto front", 20, eq_knee, "fails CoV"),
         ("Best accuracy (most complex)", 35, eq_best, "fails CoV")]

def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2   = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae  = np.mean(np.abs(y - yhat))
    return r2, rmse, mae, len(y)

df = pd.read_csv(CSV)
y  = df[TARGET].values

fig, axes = plt.subplots(1, 3, figsize=(12, 4.1))
for ax, (name, cx, fn, cov) in zip(axes, CANDS):
    p = fn(df).values
    r2, rmse, mae, n = metrics(y, p)
    print(f"{name:32s} C{cx:>2}  R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}")
    ax.scatter(y, p, s=3, alpha=0.18, color=COLOR, edgecolors="none")
    ax.plot(LIMS, LIMS, "k--", lw=1, alpha=0.8)
    ax.set_xlim(LIMS); ax.set_ylim(LIMS); ax.set_aspect("equal", "box")
    ax.set_xlabel(r"Observed CH$_4$ (mg m$^{-2}$ h$^{-1}$)")
    if ax is axes[0]:
        ax.set_ylabel(r"Predicted CH$_4$")
    ax.set_title(f"{name}\nC{cx} — {cov}", fontsize=9)
    ax.text(0.04, 0.96, f"$R^2$={r2:.3f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
            transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

fig.suptitle("SK-CRK growing season — observed vs predicted (Rule-A seed 42 candidates)", fontsize=10)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=200, bbox_inches="tight")
print(f"saved -> {OUTPNG}")
