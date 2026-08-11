#!/usr/bin/env python3
"""
SK-CRK dual-window comparison (manuscript Fig. 6 replacement).

Two stacked panels, both SK-CRK, same site different analysis window:
  TOP    full record  (9 Apr-31 Dec 2018, n~12,781) -- Eq. 1, temperature
         double-exponential. The post-harvest drained fallow (1 Oct onward) is
         SHADED: this is the off-season the growing-season run excludes.
  BOTTOM growing season (9 Apr-30 Sep 2018, n~8,365) -- Eq. 5, AUC_wet form
         (PySR auto-best #15, seed 42), the operative KOR equation.

Purpose: show the dual-window contrast (Table 7 / sec:window). The full record
scores the higher R^2 (~0.449), but the lift is the cold drained shoulder a
temperature kernel predicts cheaply -- not better active-season skill. Removing
it (bottom panel) drops R^2 to ~0.343 and returns AUC_wet as the driver.

METRICS CAVEAT: R2 / RMSE / MAE are WHOLE-SERIES RECONSTRUCTION scores (fixed
published coefficients applied to every record), NOT held-out CV. They should
print close to: full record ~0.449, growing season ~0.343.

SET BEFORE RUNNING: KOR_FULL_PATH must point at the Apr-Dec file behind your
current Fig. 5 (this is the only file not produced by the in-season zip).
Flux units: mg CH4 m^-2 h^-1.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------------------------------------------------------------------
# Files (SK-CRK dates are mm/dd/yyyy -> dayfirst=False for both)
# ----------------------------------------------------------------------------
KOR_FULL_PATH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\postgamrf2_run\KOR-CRK_2018.0_retdvars2.csv"   # <-- SET THIS: the Apr-Dec file behind current Fig. 5
KOR_GS_PATH   = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv"  # in-season subset (from the zip)

OUTPNG = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE\ch4_skcrk_dualwindow.png"

TARGET = "F_CH4_F"
ROLL   = "7D"

# Columns each panel drops on to reproduce its run's record set. These are the
# columns the equation references; if your full-record n comes out != 12,781,
# your Apr-Dec run dropped over its full 8-predictor retained set -- add the
# rest (e.g. 'h*Pr','WD','q*WS','SR') to PREDS_FULL to match exactly.
PREDS_FULL = ["asinh_Ta", "h*VPD", "SR*v", "dayhr"]
PREDS_GS   = ["AUC_wet", "SR", "WD", "SR*v", "h_inv", "SR*HODsin"]

# ----------------------------------------------------------------------------
# Equations (Greek = manuscript coefficients; columns by exact header name)
# ----------------------------------------------------------------------------
def predict_kor_full(df):
    # eq:kor (Eq. 1)  F = gamma*(h*VPD) + beta*(alpha*(SR*v) + dayhr)*exp(asinh_Ta)
    #                     + exp(delta*exp(asinh_Ta))
    # NOTE: gamma = -3.99 is the published fit for h*VPD with depth in METRES. If
    # your full-record 'h*VPD' column is cm-based, that term needs *0.01 or the
    # panel won't reproduce R^2 ~ 0.449.
    alpha, beta, gamma, delta = 4.8e-3, -5.3e-3, -3.99, 5.4e-2
    eu = np.exp(df["asinh_Ta"])
    return gamma * df["h*VPD"] + beta * (alpha * df["SR*v"] + df["dayhr"]) * eu \
        + np.exp(delta * eu)

def predict_kor_gs(df):
    # eq:kor-gs (Eq. 5)  F = d*(AUC_wet + b/(a + g/sqrt(AUC_wet + c)))*(SR + e)
    # PySR auto-best #15, seed 42; full-data coefficients.
    a, b, c, d, e, g = 0.07465542, 1.0970329, 2.0335147, 1.597917e-6, 747.46045, -7.643319
    return d * (df["AUC_wet"] + b / (a + g / np.sqrt(df["AUC_wet"] + c))) * (df["SR"] + e)

# ----------------------------------------------------------------------------
# Evaluate + plot
# ----------------------------------------------------------------------------
def metrics(y, yhat):
    m = np.isfinite(y) & np.isfinite(yhat)
    y, yhat = y[m], yhat[m]
    r2   = 1.0 - np.sum((y - yhat) ** 2) / np.sum((y - y.mean()) ** 2)
    rmse = np.sqrt(np.mean((y - yhat) ** 2))
    mae  = np.mean(np.abs(y - yhat))
    return r2, rmse, mae, len(y)

PANELS = [
    dict(path=KOR_FULL_PATH, fn=predict_kor_full, preds=PREDS_FULL, shade=True,
         color="#762a83",
         title="SK-CRK \u2014 full record (9 Apr\u201331 Dec), Eq. 1 (temperature)"),
    dict(path=KOR_GS_PATH,   fn=predict_kor_gs,   preds=PREDS_GS,   shade=False,
         color="#762a83",
         title="SK-CRK \u2014 growing season (9 Apr\u201330 Sep), Eq. 5 (AUC$_{\\mathrm{wet}}$)"),
]

fig, axes = plt.subplots(2, 1, figsize=(11, 7.5))

for ax, p in zip(axes, PANELS):
    df = pd.read_csv(p["path"])
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=False)
    df = df.dropna(subset=["Date"] + p["preds"] + [TARGET]).sort_values("Date").set_index("Date")

    obs  = df[TARGET].values
    pred = p["fn"](df).values
    r2, rmse, mae, n = metrics(obs, pred)
    print(f"{df.index.min():%Y-%m-%d}..{df.index.max():%Y-%m-%d}  "
          f"R2={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}  n={n}")

    ax.plot(df.index, obs, lw=0.6, color="0.6", alpha=0.5, label="Observed")
    ax.plot(df.index, df[TARGET].rolling(ROLL).mean(),
            lw=1.8, color="k", label=f"Observed ({ROLL} mean)")
    ax.plot(df.index, pd.Series(pred, index=df.index).rolling(ROLL).mean(),
            lw=1.8, color=p["color"], label=f"Predicted ({ROLL} mean)")

    if p["shade"]:
        ax.axvspan(pd.Timestamp("2018-10-01"), df.index.max(),
                   color="#cdbf9a", alpha=0.30,
                   label="post-harvest drained fallow\n(excluded in growing-season run)")
        ax.axvline(pd.Timestamp("2018-10-01"), color="0.35", ls="--", lw=1)

    ax.set_title(p["title"], loc="left", fontweight="bold")
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
