#!/usr/bin/env python3
"""
Figure B.11 -- observed vs. predicted half-hourly CH4 for the three labelled
SK-CRK growing-season candidates (Rule-A seed 42), evaluated over the in-season
subset (KOR-CRK_2018_updated_retvars2.csv, 9 Apr-30 Sep 2018, n=8,365) with each
candidate's full-data coefficients.

  #15 auto-best (Eq. 5)  -- PASSES the coefficient-stability screen
  #17 knee               -- fails (c CoV 1.08)
  #29 best-accuracy      -- fails (b CoV 3.35)

Annotated R2/RMSE/MAE are whole-subset (in-sample) values. Verified against the
known reconstruction: #15 -> R2=0.343, RMSE=8.15, MAE=4.98 (matches Fig. 6).
Predictions under-disperse: the predicted band is compressed relative to the
observed spread (the equations track central tendency, not the excursions).
Flux units: mg CH4 m^-2 h^-1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

KOR_GS_PATH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv"
OUTPNG      = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\obspred_kor_gs_new1.png"  # copy to figs/obspred_kor_gs.png

PREDS  = ["AUC_wet", "SR", "WD", "SR*v", "h_inv", "SR*HODsin"]
TARGET = "F_CH4_F"

# --- the three candidate forms (seed 42), full-data coefficients --------------
def predict_auto(df):   # #15 == Eq. 5
    a, b, c, d, e, g = 0.07465542, 1.0970329, 2.0335147, 1.597917e-6, 747.46045, -7.643319
    return d * (df["AUC_wet"] + b / (a + g / np.sqrt(df["AUC_wet"] + c))) * (df["SR"] + e)

def predict_knee(df):   # #17
    return (df["AUC_wet"]
            + (0.14832489 / (0.07478618 - (7.6433325 / np.sqrt(df["AUC_wet"] + 1.5054631))))) \
        * ((df["SR"] + (674.06104 - df["SR*HODsin"])) * 1.5518393e-6)

def predict_best(df):   # #29
    return ((((df["AUC_wet"] - df["SR"])
              + (np.tanh(df["WD"]) / (0.07466244 - (7.643604 / np.sqrt(df["AUC_wet"] + 1.1335045)))))
             * (df["SR"] + ((df["WD"] - df["SR*HODsin"]) + 678.24854)))
            * 1.6777989e-6) \
        - (np.exp(np.tanh(df["SR*v"] - (df["h_inv"] + 3.7988737))) + 1.1335045)

CANDS = [("#15  auto-best (Eq. 5)  \u2014 passes CoV", predict_auto, "#2c7fb8"),
         ("#17  knee  \u2014 fails CoV",                predict_knee, "#7a7a7a"),
         ("#29  best-accuracy  \u2014 fails CoV",        predict_best, "#7a7a7a")]

def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2   = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae  = np.mean(np.abs(y - yhat))
    return r2, rmse, mae, len(y)

# --- evaluate + plot ----------------------------------------------------------
df = pd.read_csv(KOR_GS_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
df = df.dropna(subset=["Date"] + PREDS + [TARGET])
obs = df[TARGET].to_numpy()

preds = {lbl: fn(df).to_numpy() for lbl, fn, _ in CANDS}
allv = np.concatenate([obs] + [p for p in preds.values()])
lim = [np.nanpercentile(allv, 0.2), np.nanpercentile(allv, 99.8)]   # clip extreme outliers for readability

fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, (lbl, fn, col) in zip(axes, CANDS):
    pred = preds[lbl]
    r2, rmse, mae, n = metrics(obs, pred)
    print(f"{lbl:42s} R2={r2:.3f}  RMSE={rmse:.2f}  MAE={mae:.2f}  n={n}")
    ax.plot(lim, lim, ls="--", color="0.5", lw=1, zorder=1)
    ax.scatter(obs, pred, s=5, alpha=0.18, color=col, edgecolors="none", zorder=2)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal", "box")
    ax.set_title(lbl, fontsize=10, fontweight="bold", loc="left")
    ax.set_xlabel(r"Observed $F_{\mathrm{CH_4}}$ (mg m$^{-2}$ h$^{-1}$)")
    ax.text(0.04, 0.96, f"$R^2$={r2:.3f}\nRMSE={rmse:.2f}\nMAE={mae:.2f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
axes[0].set_ylabel(r"Predicted $F_{\mathrm{CH_4}}$ (mg m$^{-2}$ h$^{-1}$)")
fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print(f"saved -> {OUTPNG}")
