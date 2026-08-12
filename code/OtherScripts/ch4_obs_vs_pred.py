#!/usr/bin/env python3
"""
Observed vs. predicted CH4 flux time series for the three rice-paddy sites,
using the reported symbolic-regression equations (Paper 1).

Each equation is evaluated on the *raw* second-GAM-RF feature matrix (the
'retdvars2' / postgamrf2 files) -- the actual PySR7v2 input, after the cascade
GAM-RF1 -> VIF<=5 (postcollin) -> GAM-RF2 (retdvars2). The pipeline applies NO
input standardization, so the published CV-mean coefficients evaluate directly
on these columns. Records are restricted to complete cases over each run's exact
predictor list (i.e. df.dropna(subset=predictors+[TARGET]) as in the run
scripts), reproducing the records PySR used: n = 12781 SK-CRK, 3620 PH-IR,
6760 JP-MSE.

METRICS CAVEAT: the R2 / RMSE / MAE printed here are WHOLE-SERIES RECONSTRUCTION
scores (fixed CV-mean coefficients applied to every record). They are NOT the
held-out cross-validation scores reported in the manuscript:
    SK-CRK  CV R2 = 0.423   PH-IR  CV R2 = 0.476   JP-MSE  CV R2 = 0.463
Whole-series scores sit close to but differ from the CV scores (in-sample
optimism; JP-MSE comes out slightly lower because AUC acts as a seasonal clock).
Flux units: mg CH4 m^-2 h^-1 (same scale as F_CH4_F in the input files).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------------------------------------------------------------------
# 1. PySR7v2 input files (postgamrf2 / retdvars2 = second GAM-RF output)
# ----------------------------------------------------------------------------
PATHS = {
    "PH-IR":  r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\vif5\postgamrf2_run\PHL-IR_2016_retdvars2.csv",
    "SK-CRK": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\postgamrf2_run\KOR-CRK_2018.0_retdvars2.csv",
    "JP-MSE": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\JPN\JPN-MSE_2012_retainedvars2.csv",
}

# Exact predictor list each run fed to PySR -- used ONLY to reproduce the run's
# complete-case record set. The equations below reference a subset of these.
PREDICTORS = {
    "PH-IR":  ['AUC', 'h*VPD', 'SR*Ts', 'SR*HODsin', 'h*sinTOD'],
    "SK-CRK": ['dayhr', 'SR', 'WD', 'h*Pr', 'h*VPD', 'SR*v', 'q*WS', 'asinh_Ta'],
    "JP-MSE": ['AUC', 'h*Pr', 'uzonal', 'buoy_TsTa', 'SRxWS', 'SR*VPD',
               'u*VPD', 'v*VPD', 'SR*HODsin', 'VPD*WS*d1sin', 'h*v'],
}

TARGET = "F_CH4_F"
ROLL   = "7D"      # rolling-mean trend overlay window; set ROLL = None to disable
OUTPNG = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\ch4_obs_vs_pred_timeseries.png"

# ----------------------------------------------------------------------------
# 2. Reported equations (Greek letters = manuscript coefficients).
#    Columns are accessed by their exact header names.
# ----------------------------------------------------------------------------
def predict_phl(df):
    # eq:phl   F = a + (g*(SR*Ts) + b + (e + tanh(h*VPD))*tanh(SR*HODsin)) * exp(d*AUC)
    alpha, beta, gamma, delta, eps = 0.96, 1.69, 2.4e-4, 3.2e-3, -1.54
    return (alpha
            + (gamma * df["SR*Ts"] + beta
               + (eps + np.tanh(df["h*VPD"])) * np.tanh(df["SR*HODsin"]))
            * np.exp(delta * df["AUC"]))

def predict_kor(df):
    # eq:kor   F = g*(h*VPD) + b*(a*(SR*v) + dayhr)*exp(asinh(Ta)) + exp(d*exp(asinh(Ta)))
    alpha, beta, gamma, delta = 4.8e-3, -5.3e-3, -3.99, 5.4e-2
    e_aT = np.exp(df["asinh_Ta"])          # 'asinh_Ta' column already holds asinh(Ta)
    return gamma * df["h*VPD"] + beta * (alpha * df["SR*v"] + df["dayhr"]) * e_aT \
           + np.exp(delta * e_aT)

def predict_jpn(df):
    # eq:jpn-bilinear   F = AUC*(g + d*(SR*VPD))
    gamma, delta = 6.5e-2, 5.6e-6
    return df["AUC"] * (gamma + delta * df["SR*VPD"])

SITES = {
    # dayfirst: PH-IR dates are dd/mm/yyyy; SK-CRK and JP-MSE are mm/dd/yy(yy).
    # (SK-CRK also mixes 2- and 4-digit years -> format="mixed" handles per element.)
    "PH-IR":  dict(fn=predict_phl, color="#1b7837", dayfirst=True),    # aerobic AWD
    "SK-CRK": dict(fn=predict_kor, color="#762a83", dayfirst=False),   # aggressive AWD
    "JP-MSE": dict(fn=predict_jpn, color="#2166ac", dayfirst=False),   # continuous flooding
}

# ----------------------------------------------------------------------------
# 3. Evaluate + plot
# ----------------------------------------------------------------------------
def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2   = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae  = np.mean(np.abs(y - yhat))
    return r2, rmse, mae, len(y)

fig, axes = plt.subplots(len(SITES), 1, figsize=(11, 10))

metrics_rows = []   # collect per-site metrics for CSV export

for ax, (site, cfg) in zip(axes, SITES.items()):
    df = pd.read_csv(PATHS[site])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=cfg["dayfirst"])
    df = df.dropna(subset=["Date"] + PREDICTORS[site] + [TARGET])   # == run's row filter
    df = df.sort_values("Date").set_index("Date")

    obs  = df[TARGET].values
    pred = cfg["fn"](df).values
    r2, rmse, mae, n = metrics(obs, pred)
    print(f"{site:7s} n={n:5d}  {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}  "
          f"R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    
    metrics_rows.append({"site": site, "n": n,
                         "start": f"{df.index.min():%Y-%m-%d}",
                         "end":   f"{df.index.max():%Y-%m-%d}",
                         "R2": round(r2, 3), "RMSE": round(rmse, 3),
                         "MAE": round(mae, 3)})
    
    ax.plot(df.index, obs,  lw=0.6, color="0.55", alpha=0.5, label="Observed")
    ax.plot(df.index, pred, lw=0.8, color=cfg["color"], alpha=0.8, label="Predicted")
    if ROLL:
        ax.plot(df.index, df[TARGET].rolling(ROLL).mean(),
                lw=1.8, color="k", label=f"Observed ({ROLL} mean)")
        ax.plot(df.index, pd.Series(pred, index=df.index).rolling(ROLL).mean(),
                lw=1.8, color=cfg["color"], label=f"Predicted ({ROLL} mean)")

    ax.set_title(site, loc="left", fontweight="bold")
    ax.set_ylabel(r"CH$_4$ flux (mg m$^{-2}$ h$^{-1}$)")
    ax.text(0.99, 0.95,
            f"$R^2$={r2:.3f}   RMSE={rmse:.2f}   MAE={mae:.2f}   (n={n})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)

axes[-1].set_xlabel("Month")
fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print(f"saved -> {OUTPNG}")

OUTCSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV_CHECK\ch4_obs_vs_pred_metrics.csv"
pd.DataFrame(metrics_rows).to_csv(OUTCSV, index=False)
print(f"saved -> {OUTCSV}")