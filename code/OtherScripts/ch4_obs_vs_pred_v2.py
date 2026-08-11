#!/usr/bin/env python3
"""
SK-CRK growing-season subset: observed vs predicted CH4 reconstruction.

Standalone -- reads ONLY the growing-season CSV, so it never touches the
full-record PH-IR / SK-CRK / JP-MSE files (and cannot raise the missing-column
KeyError that the three-panel script does when a full-record path is wrong).
Styling is identical to the panels in ch4_obs_vs_pred.py.

Equation eq:kor-gs (auto-best / Eq. 5):
    F = d*(AUC_wet + b/(a + g/sqrt(AUC_wet + c))) * (SR + e)
Full-data coefficients (PySR Pareto #15, seed 42); full-data rather than
CV-mean because the offset c is non-identifiable under CV (per-fold c spans
~4,400-16,300). Metrics printed are whole-series reconstruction scores, NOT
the held-out CV R2 (= 0.344 +/- 0.101) reported in the manuscript.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# --- the only input: the growing-season retvars2 CSV (8,365 complete rows) ---
KOR_GS_PATH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\postgamrf2_run\KOR-CRK_2018_updated_retvars2.csv"
OUTPNG      = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\ch4_obs_vs_pred_timeseries_korgs.png"

TARGET = "F_CH4_F"
ROLL   = "7D"          # trailing rolling-mean window; set to None to disable
COLOR  = "#762a83"     # same purple as the full-record SK-CRK panel


def predict_kor_gs(df):
    a, b, c, d, e, g = 0.07465542, 1.0970329, 2.0335147, 1.597917e-6, 747.46045, -7.643319
    return d * (df["AUC_wet"] + b / (a + g / np.sqrt(df["AUC_wet"] + c))) * (df["SR"] + e)


def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2   = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae  = np.mean(np.abs(y - yhat))
    return r2, rmse, mae, len(y)


df = pd.read_csv(KOR_GS_PATH)
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
df = df.dropna(subset=["Date", "AUC_wet", "SR", TARGET]).sort_values("Date").set_index("Date")

obs  = df[TARGET].values
pred = predict_kor_gs(df).values
r2, rmse, mae, n = metrics(obs, pred)
print(f"SK-CRK gs  n={n}  {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}  "
      f"R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")

fig, ax = plt.subplots(1, 1, figsize=(11, 3.4))
ax.plot(df.index, obs,  lw=0.6, color="0.55", alpha=0.5, label="Observed")
ax.plot(df.index, pred, lw=0.8, color=COLOR, alpha=0.8, label="Predicted")
if ROLL:
    ax.plot(df.index, df[TARGET].rolling(ROLL).mean(),
            lw=1.8, color="k", label=f"Observed ({ROLL} mean)")
    ax.plot(df.index, pd.Series(pred, index=df.index).rolling(ROLL).mean(),
            lw=1.8, color=COLOR, label=f"Predicted ({ROLL} mean)")
ax.set_title("SK-CRK (growing season, 9 Apr-30 Sep 2018)", loc="left", fontweight="bold")
ax.set_ylabel(r"CH$_4$ flux (mg m$^{-2}$ h$^{-1}$)")
ax.set_xlabel("Month")
ax.text(0.99, 0.95,
        f"$R^2$={r2:.3f}   RMSE={rmse:.2f}   MAE={mae:.2f}   (n={n})",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
ax.xaxis.set_major_formatter(DateFormatter("%b"))
ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)
fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print(f"saved -> {OUTPNG}")