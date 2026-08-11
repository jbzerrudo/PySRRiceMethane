#!/usr/bin/env python3
"""
Observed vs. predicted CH4 flux time series for the three rice-paddy sites.

PH-IR and JP-MSE use their FULL-RECORD forms and files (unchanged) -- both are
already active-season-only records, so there is no off-season to strip. The
SK-CRK panel uses the GROWING-SEASON subset run on
KOR-CRK_2018_updated_retvars2.csv (9 Apr-30 Sep 2018), putting all three panels
on a common active-season footing.

SK-CRK CANDIDATE SELECTOR (KOR_CANDIDATE below):
    "auto_best"     -> #15, complexity 18  == Eq. 5 in the manuscript  [DEFAULT]
    "knee"          -> #17, complexity 20
    "best_accuracy" -> #29, complexity 35
seed_42 (Rule-A winner). Stage-8 day-grouped CV / coefficient-stability:
    #15 auto_best   CV R2 = 0.344   max|CoV| = 0.49   STABLE  (the reported Eq. 5)
    #17 knee        CV R2 = 0.377   c CoV = 1.08       FAILS screen
    #29 best_accur. CV R2 = 0.415   b CoV = 3.35       FAILS screen
seed_42 has NO mid-complexity candidate (it did not generate for this seed).
=> By Stage 8, #15 is the only admissible form; #17/#29 are kept here for visual
   comparison only and should NOT replace Eq. 5.

Each equation is evaluated on the raw second-GAM-RF feature matrix (the
retvars2 / postgamrf2 file). The pipeline applies NO input standardization, so
the published coefficients evaluate directly on these columns. Records are
restricted to complete cases over each run's exact predictor list, reproducing
the records each run used: n = 8365 SK-CRK (growing season), 3620 PH-IR,
6760 JP-MSE.

METRICS CAVEAT: the R2 / RMSE / MAE printed here are WHOLE-SERIES RECONSTRUCTION
scores (fixed coefficients applied to every record). They are NOT the held-out
cross-validation scores reported in the manuscript:
    SK-CRK (growing season) CV R2 = 0.344   PH-IR CV R2 = 0.476   JP-MSE CV R2 = 0.463
Whole-series scores sit close to the CV scores (SK-CRK growing season ~0.343).
Flux units: mg CH4 m^-2 h^-1 (same scale as F_CH4_F in the input files).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter

# ----------------------------------------------------------------------------
# 0. SK-CRK candidate to plot: "auto_best" (Eq. 5) | "knee" | "best_accuracy"
# ----------------------------------------------------------------------------
KOR_CANDIDATE = "auto_best"   # <- change to "knee" or "best_accuracy" to compare

# ----------------------------------------------------------------------------
# 1. PySR input files (postgamrf2 / retvars2 = second GAM-RF output)
# ----------------------------------------------------------------------------
PATHS = {
    "PH-IR":  r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\vif5\postgamrf2_run\PHL-IR_2016_retdvars2.csv",
    "SK-CRK": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv",   # growing-season subset
    "JP-MSE": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\JPN\JPN-MSE_2012_retainedvars2.csv",
}

# Exact predictor list each run fed to PySR -- used ONLY to reproduce the run's
# complete-case record set. The equations below reference a subset of these.
PREDICTORS = {
    "PH-IR":  ['AUC', 'h*VPD', 'SR*Ts', 'SR*HODsin', 'h*sinTOD'],
    "SK-CRK": ['AUC_wet', 'SR', 'WD', 'SR*v', 'h_inv', 'SR*HODsin'],   # growing-season retvars2 columns
    "JP-MSE": ['AUC', 'h*Pr', 'uzonal', 'buoy_TsTa', 'SRxWS', 'SR*VPD',
               'u*VPD', 'v*VPD', 'SR*HODsin', 'VPD*WS*d1sin', 'h*v'],
}

TARGET = "F_CH4_F"
ROLL   = "7D"      # rolling-mean trend overlay window; set ROLL = None to disable
OUTPNG = (r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\000_RICEMETHANE"
          rf"\ch4_obs_vs_pred_timeseries_v2_{KOR_CANDIDATE}.png")

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

# --- SK-CRK growing-season candidates (seed 42) --------------------------------
def predict_kor_gs(df):
    # #15 auto-best, complexity 18 == eq:kor-gs (Eq. 5).  STABLE (max|CoV|=0.49).
    # F = d*(AUC_wet + b/(a + g/sqrt(AUC_wet + c)))*(SR + e)
    # Full-data coefficients (not CV-mean): the offset c is non-identifiable
    # under CV (per-fold c spans ~4,400-16,300), so a CV-mean c averages noise.
    a, b, c, d, e, g = 0.07465542, 1.0970329, 2.0335147, 1.597917e-6, 747.46045, -7.643319
    return d * (df["AUC_wet"] + b / (a + g / np.sqrt(df["AUC_wet"] + c))) * (df["SR"] + e)

def predict_kor_knee(df):
    # #17 knee, complexity 20.  FAILS coefficient-stability screen (c CoV 1.08).
    # Plotted for comparison only; NOT the reported Eq. 5.
    return (df["AUC_wet"]
            + (0.14832489 / (0.07478618 - (7.6433325 / np.sqrt(df["AUC_wet"] + 1.5054631))))) \
        * ((df["SR"] + (674.06104 - df["SR*HODsin"])) * 1.5518393e-6)

def predict_kor_best(df):
    # #29 best-accuracy, complexity 35.  FAILS screen (b CoV 3.35, c CoV 0.79).
    # Plotted for comparison only; NOT the reported Eq. 5.
    return ((((df["AUC_wet"] - df["SR"])
              + (np.tanh(df["WD"]) / (0.07466244 - (7.643604 / np.sqrt(df["AUC_wet"] + 1.1335045)))))
             * (df["SR"] + ((df["WD"] - df["SR*HODsin"]) + 678.24854)))
            * 1.6777989e-6) \
        - (np.exp(np.tanh(df["SR*v"] - (df["h_inv"] + 3.7988737))) + 1.1335045)

KOR_FNS = {
    "auto_best":     predict_kor_gs,
    "knee":          predict_kor_knee,
    "best_accuracy": predict_kor_best,
}
if KOR_CANDIDATE not in KOR_FNS:
    raise ValueError(f"KOR_CANDIDATE must be one of {list(KOR_FNS)}, got {KOR_CANDIDATE!r}")

def predict_jpn(df):
    # eq:jpn-bilinear   F = AUC*(g + d*(SR*VPD))
    gamma, delta = 6.5e-2, 5.6e-6
    return df["AUC"] * (gamma + delta * df["SR*VPD"])

SITES = {
    # dayfirst: PH-IR dates are dd/mm/yyyy; SK-CRK and JP-MSE are mm/dd/yy(yy).
    "PH-IR":  dict(fn=predict_phl,               color="#1b7837", dayfirst=True),    # aerobic AWD, full record
    "SK-CRK": dict(fn=KOR_FNS[KOR_CANDIDATE],    color="#762a83", dayfirst=False),   # growing-season subset (selected candidate)
    "JP-MSE": dict(fn=predict_jpn,               color="#2166ac", dayfirst=False),   # continuous flooding, full record
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

    title = site if site != "SK-CRK" else f"SK-CRK (growing season, {KOR_CANDIDATE})"
    ax.set_title(title, loc="left", fontweight="bold")
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

OUTCSV = (r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV_CHECK"
          rf"\ch4_obs_vs_pred_metrics_{KOR_CANDIDATE}.csv")
pd.DataFrame(metrics_rows).to_csv(OUTCSV, index=False)
print(f"saved -> {OUTCSV}")