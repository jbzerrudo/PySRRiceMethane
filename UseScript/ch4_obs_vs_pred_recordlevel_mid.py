#!/usr/bin/env python3
"""
Observed vs. predicted CH4 flux for the NCGG-10 record-level MID-COMPLEXITY
equations (Appendix A), evaluated over the full record at each site -- the
record-level analogue of the main-text reconstruction figure.

Equations (Appendix A, Mid tier), record-level fitted constants:
  JP-MSE (C19): CH4 = tanh(0.141) * exp(0.131*(Tv - tanh(u*VPD)))
                       / (1.341 - tanh(-0.00611/(h*Pr - 0.909)))
  PH-IR  (C19): CH4 = (0.414 + exp(0.00913*(sqrt(SR*Ts) + 0.323*AUC)))
                       / exp(tanh(0.00291*SR*HODsin - 0.840))
  SK-CRK (C20): CH4 = exp(0.106*(Tair - 0.000744*SR*v))
                       + (Tair + 3.80)*dayhr*(-0.00771) - (h*VPD)/0.277

Needs the ORIGINAL feature CSVs (all variables): the retdvars2 files do NOT
contain Tv (JP-MSE) or raw Tair (SK-CRK).

Printed R2/RMSE/MAE are WHOLE-RECORD reconstruction scores (in-sample), not the
record-level hold-out R2 in Appendix A (0.68/0.73/0.51). Flux: mg CH4 m^-2 h^-1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------------------------------------------------------------------
# 1. EDIT: original feature CSVs (all variables) + column-name map
# ----------------------------------------------------------------------------
PATHS = {
    "PH-IR":  r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\PHL-IR_2016.csv",
    "SK-CRK": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\2018\v5\KOR-CRK_2018_retainedvars_postgam2.csv",
    "JP-MSE": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\JPN\Run1\JPN-MSE_2012_retainedvars_postgam2.csv",
}

# Map each variable the equations use to its column name in YOUR CSV.
# Guesses follow the retdvars2 convention -- VERIFY against your headers,
# especially Tv and Tair, which are NOT in the retdvars2 files.
COLS = {
    "Tv":       "Tv_K",         # JP-MSE virtual temperature    <-- VERIFY
    "uVPD":     "u*VPD",
    "hPr":      "h*Pr",
    "SRTs":     "SR*Ts",
    "AUC":      "AUC",
    "SRHODsin": "SR*HODsin",
    "Tair":     "Tair",       # SK-CRK raw air temperature     <-- VERIFY
    "SRv":      "SR*v",
    "dayhr":    "dayhr",
    "hVPD":     "h*VPD",
}
TARGET = "F_CH4_F"     # CH4 flux column                       <-- VERIFY
DATE   = "Date"
ROLL   = "7D"          # rolling-mean overlay; None to disable
OUTPNG = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\ch4_obs_vs_pred_recordlevel_mid.png"
OUTCSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV_CHECK\ch4_obs_vs_pred_recordlevel_mid_metrics.csv"

# ----------------------------------------------------------------------------
# 2. Appendix A Mid-complexity equations
# ----------------------------------------------------------------------------
def predict_jpn(df):
    Tv, uVPD, hPr = df[COLS["Tv"]] - 273.15, df[COLS["uVPD"]], df[COLS["hPr"]]
    return np.tanh(0.141) * np.exp(0.131 * (Tv - np.tanh(uVPD))) \
           / (1.341 - np.tanh(-0.00611 / (hPr - 0.909)))

def predict_phl(df):
    SRTs, AUC, SRHODsin = df[COLS["SRTs"]], df[COLS["AUC"]], df[COLS["SRHODsin"]]
    return (0.414 + np.exp(0.00913 * (np.sqrt(SRTs) + 0.323 * AUC))) \
           / np.exp(np.tanh(0.00291 * SRHODsin - 0.840))

def predict_kor(df):
    Tair, SRv = df[COLS["Tair"]], df[COLS["SRv"]]
    dayhr, hVPD = df[COLS["dayhr"]], df[COLS["hVPD"]]
    return np.exp(0.106 * (Tair - 0.000744 * SRv)) \
           + (Tair + 3.80) * dayhr * (-0.00771) - hVPD / 0.277

SITES = {
    "PH-IR":  dict(fn=predict_phl, cols=["SRTs", "AUC", "SRHODsin"],     color="#1b7837", dayfirst=True),
    "SK-CRK": dict(fn=predict_kor, cols=["Tair", "SRv", "dayhr", "hVPD"], color="#762a83", dayfirst=False),
    "JP-MSE": dict(fn=predict_jpn, cols=["Tv", "uVPD", "hPr"],           color="#2166ac", dayfirst=False),
}

# ----------------------------------------------------------------------------
# 3. Evaluate + plot
# ----------------------------------------------------------------------------
def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2 = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, np.sqrt(np.mean((y - yhat) ** 2)), np.mean(np.abs(y - yhat)), len(y)

fig, axes = plt.subplots(len(SITES), 1, figsize=(11, 10))
rows = []

for ax, (site, cfg) in zip(axes, SITES.items()):
    need = [COLS[c] for c in cfg["cols"]]
    df = pd.read_csv(PATHS[site])
    missing = [c for c in need + [TARGET, DATE] if c not in df.columns]
    if missing:
        raise SystemExit(f"{site}: missing columns {missing}\n  available: {list(df.columns)}")

    df[DATE] = pd.to_datetime(df[DATE], format="mixed", dayfirst=cfg["dayfirst"])
    df = df.dropna(subset=[DATE] + need + [TARGET]).sort_values(DATE).set_index(DATE)

    obs  = df[TARGET].values
    pred = cfg["fn"](df).values
    r2, rmse, mae, n = metrics(obs, pred)
    print(f"{site:7s} n={n:5d}  {df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}  "
          f"R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    rows.append({"site": site, "n": n, "start": f"{df.index.min():%Y-%m-%d}",
                 "end": f"{df.index.max():%Y-%m-%d}", "R2": round(r2, 3),
                 "RMSE": round(rmse, 3), "MAE": round(mae, 3)})

    ax.plot(df.index, obs,  lw=0.6, color="0.55", alpha=0.5, label="Observed")
    ax.plot(df.index, pred, lw=0.8, color=cfg["color"], alpha=0.8, label="Predicted")
    if ROLL:
        ax.plot(df.index, df[TARGET].rolling(ROLL).mean(), lw=1.8, color="k",
                label=f"Observed ({ROLL} mean)")
        ax.plot(df.index, pd.Series(pred, index=df.index).rolling(ROLL).mean(),
                lw=1.8, color=cfg["color"], label=f"Predicted ({ROLL} mean)")

    ax.set_title(f"{site} - record-level Mid form", loc="left", fontweight="bold")
    ax.set_ylabel(r"CH$_4$ flux (mg m$^{-2}$ h$^{-1}$)")
    ax.text(0.99, 0.95, f"$R^2$={r2:.3f}   RMSE={rmse:.2f}   MAE={mae:.2f}   (n={n})",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.xaxis.set_major_formatter(DateFormatter("%b"))
    ax.legend(loc="upper left", fontsize=8, ncol=2, framealpha=0.85)

axes[-1].set_xlabel("Month")
fig.tight_layout()
fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
print(f"saved -> {OUTPNG}")
pd.DataFrame(rows).to_csv(OUTCSV, index=False)
print(f"saved -> {OUTCSV}")
