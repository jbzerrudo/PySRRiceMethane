"""
11_figures.py (v1.7) -- the three data-dependent figures for Paper 1
==============================================================================
v1.7 (12 Aug 2026): earth/WUR palette replaces Okabe-Ito, validated for
    protan/deutan/tritan separation (OKLab, all pairs, white surface).
    KOR path is the Papale/Hampel-cleaned CSV. Regenerate ALL figures so the
    Cheorwon panels stop showing the uncleaned record: the old Figure 4
    annotated n=8,365 where the cleaned analysis set has 8,364 complete
    cases (tab:sites footnote b).

Regenerates, from the August 2026 runs and at the equations' PUBLISHED
coefficients (no refitting):

  A. ch4_obs_vs_pred_timeseries_v3.png     four panels, one per arm: observed
     and predicted half-hourly CH4 with 7-day rolling means, in-sample R2,
     RMSE, MAE annotated. Operative equations only.
  B. pareto_front_<arm>.png (x4)           accuracy-complexity Pareto front at
     the reported seed, the four labelled candidates marked, operative filled.
  C. obspred_<arm>.png (x4)                observed vs predicted scatter for
     the four labelled candidates of the reported seed.

REPORTED SEEDS (hold-out criterion): Mase 49, Cheorwon 45, IRRI 53, POOLED 45.

All equations below are verbatim from those seeds' equation reports.
The operative equations are: Mase c22, Cheorwon c40, IRRI c8 (post-screen),
POOLED c40.

USAGE
    python 11_figures.py            # everything
    python 11_figures.py A          # time series only
    python 11_figures.py B C        # pareto + scatters
    python 11_figures.py D          # DAT panels (pooled arm, one per site)
    python 11_figures.py A MASE     # one figure, one arm

Outputs land in  ...\RUN2\FIGS
Author: Jef Zerrudo / Claude.  Requires numpy, pandas, matplotlib.
==============================================================================
"""

import os
import sys
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

TARGET = "F_CH4_F"
TIME_COL = "Date"
MISSING_FLAGS = [-9999, -999900, -99999]

BASE = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV"
PYSR_ROOT = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\PYSR"
OUT = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FIGS"

# Earth/WUR palette (v1.7), colourblind-aware. Validated all-pairs in OKLab
# against protanopia/deuteranopia/tritanopia on a white surface; arm colours
# are never co-plotted in one panel and every panel is title-labelled, so
# identity never rests on colour alone. POOLED wears the WUR green family:
# the exact brand green #34B233 sits at DeltaE 3.2 from the turmeric under
# protanopia, so the validated darker step #279027 is used instead.
COLOR = {"MASE": "#0C7CA0",      # deep lake teal -- continuous flooding
         "CHEORWON": "#96421F",  # terracotta    -- intermittent flooding
         "IRRI": "#C99700",      # turmeric      -- AWD dry season
         "POOLED": "#279027"}    # WUR green     -- the pooled equation
GREY = "#9a9a9a"


# ── equations, verbatim ─────────────────────────────────────────────────────
def _c(d, name):
    return pd.to_numeric(d[name], errors="coerce").to_numpy(float)


# MASE seed 49 ---------------------------------------------------------------
def mase_auto(d):    # c3, report R2 0.286
    return _c(d, "AUC_wet") * 0.00072553806

def mase_knee(d):    # c8, report R2 0.476 (REJECTED by stability screen)
    return np.exp(_c(d, "Tv_C") ** 2 * _c(d, "AUC_wet") * 2.1631375e-7)

def mase_mid(d):     # c22, report R2 0.450  << OPERATIVE
    z = _c(d, "AUC_wet") * _c(d, "Tv_C") ** 2 * 2.1130118e-7
    return np.exp(z) - (_c(d, "u*VPD") / _c(d, "VPD*WS")) / (3.0412269 - z)

def mase_best(d):    # c39, report R2 0.346
    A, T = _c(d, "AUC_wet"), _c(d, "Tv_C")
    hPr, VW, uV = _c(d, "h*Pr"), _c(d, "VPD*WS"), _c(d, "u*VPD")
    inner = (T / (A + (hPr * 2.726118 - VW * VW * 0.7045468))) \
        * ((A - hPr) * -0.02332521 * T + A)
    return (6.4613116e-5 / (inner - 1.9055021 - A * 0.00032838323)) \
        * (A * (T - uV)) + 0.97030455


# CHEORWON seed 45 -----------------------------------------------------------
def kor_auto(d):     # c3, report R2 0.533
    return _c(d, "Tv_C") - 15.227551

def kor_knee(d):     # c4, report R2 0.625  (the recurring Q10 kernel)
    return np.exp(_c(d, "Tv_C") * 0.08502509)

def kor_mid(d):      # c22, report R2 0.700
    A, hi, T = _c(d, "AUC_wet"), _c(d, "h_inv"), _c(d, "Tv_C")
    return ((_c(d, "SR") + A - _c(d, "SR*v"))
            * (np.exp(np.sqrt(T)) - (np.sqrt(hi) - A / (A - hi * 6.251832)))) \
        * 6.378604e-6

def kor_best(d):     # c40, report R2 0.744  << OPERATIVE
    A, SR, SRv = _c(d, "AUC_wet"), _c(d, "SR"), _c(d, "SR*v")
    SRu, SRH, T = _c(d, "SR*u"), _c(d, "SR*HODsin"), _c(d, "Tv_C")
    dayhr, hi = _c(d, "dayhr"), _c(d, "h_inv")
    inner = (dayhr * np.log(T) + (-9.83549 / hi)) \
        + (SRu * -0.019528149 - ((A - SRu + SRH * -7.484453) / (A - hi * 6.249428)))
    return ((A - SRv + SR) * 6.3612692e-6) * (np.exp(np.sqrt(T)) - inner) + 0.4964769


# IRRI seed 53 ---------------------------------------------------------------
def phl_auto(d):     # c8, report R2 0.543  << OPERATIVE (post-screen)
    return -1676.962 / (np.sqrt(_c(d, "SR*Ts")) + (-216.37347 - _c(d, "AUC_dry")))

def phl_knee(d):     # c9, report R2 0.359
    return (_c(d, "Tv_C") - 24.959808) * (-613.9796 / (-213.10391 - _c(d, "AUC_dry")))

def phl_mid(d):      # c24, report R2 0.606 (REJECTED by stability screen)
    T, uz, SRH = _c(d, "Tv_C"), _c(d, "uzonal"), _c(d, "SR*HODsin")
    hwet, SRTs, Ad = _c(d, "hwet"), _c(d, "SR*Ts"), _c(d, "AUC_dry")
    return ((T + (((4.2946577 - uz) - SRH * 0.017328158) / np.exp(hwet) - 24.07842))
            * np.exp((np.sqrt(SRTs) - Ad) * 0.004744538)) + 0.8820041

def phl_best(d):     # c40, report R2 0.091
    T, uz, SRH = _c(d, "Tv_C"), _c(d, "uzonal"), _c(d, "SR*HODsin")
    hwet, hs, hA = _c(d, "hwet"), _c(d, "h*sinTOD"), _c(d, "h_ASINH_cm")
    SRTs, Ad = _c(d, "SR*Ts"), _c(d, "AUC_dry")
    br = (((Ad - SRH) * 0.017164316) - (uz - 2.9094393)) \
        / np.exp((hwet - hs) / (hA - Ad))
    return (uz * 0.15228048
            + (T + (br - 25.1438))
            * (np.exp((np.sqrt(Ad + SRTs) - Ad) * 0.005614984) - 0.009335757)) \
        + 1.768934


# POOLED seed 45 -------------------------------------------------------------
def pool_auto(d):    # c7, report R2 0.505
    return np.exp(np.sqrt(_c(d, "Tv_C"))) * (_c(d, "hbar_wet") * 0.013211074)

def pool_knee(d):    # c8, report R2 0.525
    T = _c(d, "Tv_C")
    return _c(d, "hbar_wet") * np.exp(T * 0.15075056) / T

def pool_mid(d):     # c21, report R2 0.585
    hb, SRv, VW = _c(d, "hbar_wet"), _c(d, "SR*v"), _c(d, "VPD*WS")
    hi, T = _c(d, "h_inv"), _c(d, "Tv_C")
    return (hb - SRv * 0.0003887753) \
        * ((np.exp(VW / -0.039896015) + hi * 0.0012095191)
           + np.exp(-3.8196046 + T * 0.1589963))

def pool_best(d):    # c40, report R2 0.639  << OPERATIVE
    hb, hi, T = _c(d, "hbar_wet"), _c(d, "h_inv"), _c(d, "Tv_C")
    SRH, SRv = _c(d, "SR*HODsin"), _c(d, "SR*v")
    hv, hs, VW = _c(d, "h*v"), _c(d, "h*sinTOD"), _c(d, "VPD*WS")
    inner = ((SRH + SRv / 1.6380328 + ((hv - hs) - hb * T) * T)
             * -1.7110912e-7) * (hb - 5.6869154)
    return (hb - 0.40055007) * (((hi * 0.00034019744 - T * T * inner) * hb)
                                + np.exp(VW * -39.338554))


ARMS = {
    "MASE": dict(
        csv=os.path.join(BASE, r"JPN\Data-Metadata\JPN_retvars_pass2_C.csv"),
        run="run_20260805_122821", seed=49, label="Mase",
        candidates=[("auto-best", 3, mase_auto), ("knee", 8, mase_knee),
                    ("mid (operative)", 22, mase_mid), ("most accurate", 39, mase_best)],
        operative=("mid (operative)", mase_mid, 22)),
    "CHEORWON": dict(
        csv=os.path.join(BASE, r"KOR\Data_Metadata\Papale_hampel_cleaned\KOR_retvars_pass2_C.csv"),
        run="run_20260805_170113", seed=45, label="Cheorwon",
        candidates=[("auto-best", 3, kor_auto), ("knee", 4, kor_knee),
                    ("mid", 22, kor_mid), ("most accurate (operative)", 40, kor_best)],
        operative=("most accurate (operative)", kor_best, 40)),
    "IRRI": dict(
        csv=os.path.join(BASE, r"PHL\Data_Metadata\PHL_retvars_pass2_C.csv"),
        run="run_20260806_104044", seed=53, label="IRRI",
        candidates=[("auto-best (operative)", 8, phl_auto), ("knee", 9, phl_knee),
                    ("mid (rejected)", 24, phl_mid), ("most accurate", 40, phl_best)],
        operative=("auto-best (operative)", phl_auto, 8)),
    "POOLED": dict(
        csv=os.path.join(BASE, r"POOLED\Data-Metadata\POOLED_retvars_pass2_ISO_C.csv"),
        run="run_20260810_111532", seed=45, label="Pooled",
        candidates=[("auto-best", 7, pool_auto), ("knee", 8, pool_knee),
                    ("mid", 21, pool_mid), ("most accurate (operative)", 40, pool_best)],
        operative=("most accurate (operative)", pool_best, 40)),
}


# ── helpers ─────────────────────────────────────────────────────────────────
def parse_dates(series):
    s = series.astype(str).str.strip()
    best, bestmono = None, -1
    for flag in (False, True):
        parsed = pd.to_datetime(s, errors="coerce", dayfirst=flag, format="mixed")
        diffs = parsed.dropna().diff().dropna()
        mono = float((diffs >= pd.Timedelta(0)).sum()) / max(1, len(diffs))
        if mono > bestmono:
            best, bestmono = parsed, mono
    return best


def load(cfg):
    d = pd.read_csv(cfg["csv"], low_memory=False)
    d = d.replace(MISSING_FLAGS, np.nan)
    d["__date__"] = parse_dates(d[TIME_COL])
    return d


def stats(y, p):
    m = np.isfinite(y) & np.isfinite(p)
    y, p = y[m], p[m]
    r2 = 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    return r2, float(np.sqrt(np.mean((y - p) ** 2))), float(np.mean(np.abs(y - p))), int(m.sum())


def find_run(runname):
    for root, dirs, _ in os.walk(PYSR_ROOT):
        if runname in dirs:
            return os.path.join(root, runname)
    return None


def _grid(t, *series):
    """Reindex onto a regular 30-min grid so plotted lines break across gaps."""
    df = pd.DataFrame({i: s for i, s in enumerate(series)})
    df.index = pd.DatetimeIndex(t)
    df = df[~df.index.duplicated()].sort_index()
    full = pd.date_range(df.index.min(), df.index.max(), freq="30min")
    df = df.reindex(full)
    return df.index, [df[i].to_numpy(float) for i in df.columns]


def _panel(ax, arm, label, t, y, p, clip_note=False, ylab=True, stats_in_title=True):
    r2, rmse, mae, n = stats(y, p)
    tt, (yy, pp) = _grid(t, y, p)
    roll = lambda v: pd.Series(v).rolling(336, min_periods=48, center=True).mean()
    ax.plot(tt, yy, color=GREY, lw=0.3, alpha=0.45)
    ax.plot(tt, pp, color=COLOR[arm], lw=0.3, alpha=0.45)
    ax.plot(tt, roll(yy), color="black", lw=1.6, label="observed, 7-day mean")
    ax.plot(tt, roll(pp), color=COLOR[arm], lw=1.6, label="predicted, 7-day mean")
    ymax = 1.05 * np.nanmax(yy)
    nclip = int(np.nansum(pp > ymax))
    if clip_note and nclip:
        ax.set_ylim(top=ymax)
        ax.annotate(f"{nclip} near-pole prediction(s) exceed the frame",
                    xy=(0.985, 0.72), xycoords="axes fraction", ha="right",
                    fontsize=7, color=COLOR[arm])
    if ylab:
        ax.set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$)", fontsize=8)
    if stats_in_title:
        ax.set_title(label + f"   in-sample R$^2$={r2:.2f}, RMSE={rmse:.2f}, "
                     f"MAE={mae:.2f}  (n={n:,})", fontsize=9, loc="left")
    else:
        ax.annotate(label + f"  (R$^2$={r2:.2f})", xy=(0.02, 0.92),
                    xycoords="axes fraction", fontsize=8.5, fontweight="semibold")
    ax.tick_params(labelsize=7)


# ── figure A: time series ───────────────────────────────────────────────────
def fig_timeseries(wanted, ts):
    arms = [a for a in ARMS if a in wanted]
    nrow = len(arms)
    fig = plt.figure(figsize=(11, 2.7 * nrow))
    gs = fig.add_gridspec(nrow, 3, hspace=0.55, wspace=0.16)
    for i, arm in enumerate(arms):
        cfg = ARMS[arm]
        if not os.path.isfile(cfg["csv"]):
            print(f"  [SKIP {arm}] {cfg['csv']} not found"); continue
        d = load(cfg)
        name, fn, cx = cfg["operative"]
        head = f"{cfg['label']}  --  complexity {cx}, seed {cfg['seed']}"
        if arm == "POOLED" and "site" in d.columns:
            # one segment per site-year; a stacked record has no continuous time axis
            order = [s for s in ["JP-MSE", "Mase", "PH-IR", "IRRI", "SK-CRK", "Cheorwon"]
                     if s in set(d["site"])] or sorted(set(d["site"]))
            y_all = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
            p_all = np.asarray(fn(d), dtype=float)
            r2, rmse, mae, n = stats(y_all, p_all)
            fig.text(0.085, gs[i, 0].get_position(fig).y1 + 0.012,
                     head + f"   in-sample R$^2$={r2:.2f}, RMSE={rmse:.2f}, "
                     f"MAE={mae:.2f}  (n={n:,}; one panel per site-year)",
                     fontsize=9, ha="left")
            display = {"JP-MSE": "Mase", "PH-IR": "IRRI", "SK-CRK": "Cheorwon"}
            for j, site in enumerate(order[:3]):
                ax = fig.add_subplot(gs[i, j])
                sub = d[d["site"] == site]
                _panel(ax, arm, display.get(site, site), sub["__date__"],
                       pd.to_numeric(sub[TARGET], errors="coerce").to_numpy(float),
                       np.asarray(fn(sub), dtype=float),
                       ylab=(j == 0), stats_in_title=False)
                if j == 2:
                    ax.legend(fontsize=6.5, loc="lower right", frameon=False)
        else:
            ax = fig.add_subplot(gs[i, :])
            y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
            p = np.asarray(fn(d), dtype=float)
            _panel(ax, arm, head, d["__date__"], y, p, clip_note=(arm == "MASE"))
            ax.legend(fontsize=7, loc="upper right", frameon=False)
    for out in (os.path.join(OUT, f"ch4_obs_vs_pred_timeseries_v3_{ts}.png"),
                os.path.join(OUT, "ch4_obs_vs_pred_timeseries_v3.png")):
        fig.savefig(out, dpi=300, bbox_inches="tight")
        print(f"  [SAVED] {out}")
    plt.close(fig)


# ── figure B: pareto fronts ─────────────────────────────────────────────────
def fig_pareto(wanted, ts):
    for arm in [a for a in ARMS if a in wanted]:
        cfg = ARMS[arm]
        rundir = find_run(cfg["run"])
        if rundir is None:
            print(f"  [SKIP {arm}] run folder {cfg['run']} not under {PYSR_ROOT}"); continue
        sd = os.path.join(rundir, f"seed_{cfg['seed']}")
        pcsv = [f for f in os.listdir(sd) if f.startswith("pareto_equations")]
        pf = pd.read_csv(os.path.join(sd, pcsv[0]))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        ax.plot(pf["complexity"], pf["loss"], "-o", color=GREY, ms=3.5, lw=1,
                mfc="white", zorder=2)
        marks = {c: lab for lab, c, _ in cfg["candidates"]}
        opname, _, opc = cfg["operative"]
        for c, lab in marks.items():
            row = pf[pf["complexity"] == c]
            if row.empty:
                continue
            operative = (c == opc)
            ax.plot(row["complexity"], row["loss"], "o",
                    color=COLOR[arm], ms=9 if operative else 7,
                    mfc=COLOR[arm] if operative else "white",
                    mew=1.6, zorder=3)
            ax.annotate(lab, (float(row["complexity"].iloc[0]), float(row["loss"].iloc[0])),
                        textcoords="offset points", xytext=(6, 6), fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel("complexity", fontsize=9)
        ax.set_ylabel("loss (MSE)", fontsize=9)
        ax.set_title(f"{cfg['label']}, seed {cfg['seed']}", fontsize=10, loc="left")
        ax.tick_params(labelsize=8)
        fig.tight_layout()
        for out in (os.path.join(OUT, f"pareto_front_{arm.lower()}_{ts}.png"),
                    os.path.join(OUT, f"pareto_front_{arm.lower()}.png")):
            fig.savefig(out, dpi=300)
            print(f"  [SAVED] {out}")
        plt.close(fig)


# ── figure C: obs vs pred scatters ──────────────────────────────────────────
def fig_obspred(wanted, ts):
    for arm in [a for a in ARMS if a in wanted]:
        cfg = ARMS[arm]
        if not os.path.isfile(cfg["csv"]):
            print(f"  [SKIP {arm}] {cfg['csv']} not found"); continue
        d = load(cfg)
        y = pd.to_numeric(d[TARGET], errors="coerce").to_numpy(float)
        fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
        for ax, (lab, cx, fn) in zip(axes, cfg["candidates"]):
            p = np.asarray(fn(d), dtype=float)
            r2, rmse, mae, n = stats(y, p)
            m = np.isfinite(y) & np.isfinite(p)
            ax.plot(y[m], p[m], ".", ms=1.5, alpha=0.25, color=COLOR[arm], rasterized=True)
            lim = np.nanpercentile(np.concatenate([y[m], p[m]]), [0, 99.5])
            ax.plot(lim, lim, "-", color="black", lw=0.8)
            ax.set_xlim(lim); ax.set_ylim(lim)
            r2lab = f"R$^2$={r2:.3f}" if r2 > -1 else "R$^2$$<$$-$1 (near-pole rows)"
            ax.set_title(f"{lab}  (c={cx})\n{r2lab}  RMSE={rmse:.2f}", fontsize=8)
            ax.set_xlabel("observed", fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel(f"{cfg['label']}: predicted", fontsize=8)
        fig.tight_layout()
        for out in (os.path.join(OUT, f"obspred_{arm.lower()}_{ts}.png"),
                    os.path.join(OUT, f"obspred_{arm.lower()}.png")):
            fig.savefig(out, dpi=300)
            print(f"  [SAVED] {out}")
        plt.close(fig)


# ── figure D: common-DAT overlay of the pooled arm ──────────────────────────
# DAT 0 = transplanting. Dates and sources, from plot_DAT_sites_v2.py:
#   Mase     2012-05-02  (Iwata et al. 2018),  harvest 2012-09-12
#   IRRI     2015-12-17  (midpoint of the 14-21 Dec 2015 machine transplanting)
#   Cheorwon 2018-04-27  (Hwang et al. 2020),  harvest 2018-08-28
TRANSPLANT = {"JP-MSE": ("Mase", "2012-05-02", "2012-09-12"),
              "PH-IR": ("IRRI", "2015-12-17", None),
              "SK-CRK": ("Cheorwon", "2018-04-27", "2018-08-28")}


def fig_dat_overlay(ts):
    """v1.6: three panels, one per site, shared DAT axis and shared y axis."""
    cfg = ARMS["POOLED"]
    if not os.path.isfile(cfg["csv"]):
        print(f"  [SKIP DAT] {cfg['csv']} not found"); return
    d = load(cfg)
    if "site" not in d.columns:
        print("  [SKIP DAT] no 'site' column in the pooled CSV"); return
    _, fn, cx = cfg["operative"]
    d = d.assign(__pred__=np.asarray(fn(d), dtype=float))
    order = [c for c in ["JP-MSE", "PH-IR", "SK-CRK"] if c in set(d["site"])]
    fig, axes = plt.subplots(1, len(order), figsize=(11.6, 3.9), sharey=True)
    if len(order) == 1:
        axes = [axes]
    roll = lambda v: pd.Series(v).rolling(336, min_periods=48, center=True).mean()
    for j, (ax, code) in enumerate(zip(axes, order)):
        name, t0, harvest = TRANSPLANT[code]
        sub = d[d["site"] == code]
        key = "MASE" if name == "Mase" else ("IRRI" if name == "IRRI" else "CHEORWON")
        y = pd.to_numeric(sub[TARGET], errors="coerce").to_numpy(float)
        r2 = stats(y, sub["__pred__"].to_numpy(float))[0]
        tt, (yy, pp) = _grid(sub["__date__"], y, sub["__pred__"].to_numpy(float))
        dat = ((tt - pd.Timestamp(t0)) / pd.Timedelta(days=1)).to_numpy(float)
        w = (dat >= -25) & (dat <= 165)
        ax.plot(dat[w], roll(yy).to_numpy(float)[w], color=COLOR[key], lw=1.7)
        ax.plot(dat[w], roll(pp).to_numpy(float)[w], color=COLOR[key], lw=1.4,
                ls="--", alpha=0.9)
        ax.axvline(0, color="#555555", ls="--", lw=1.1)
        if harvest:
            hv = (pd.Timestamp(harvest) - pd.Timestamp(t0)).days
            ax.axvline(hv, color="#555555", ls=":", lw=1.2)
        ax.annotate(f"{name}  (R$^2$={r2:.2f})", xy=(0.04, 0.92),
                    xycoords="axes fraction", fontsize=8.5, fontweight="semibold")
        if j == 0:
            ax.set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$), 7-day rolling mean",
                          fontsize=8.5)
            ax.annotate("transplanting", xy=(0, 0.03),
                        xycoords=("data", "axes fraction"), fontsize=7.2,
                        color="#555555", ha="left", xytext=(3, 0),
                        textcoords="offset points")
        ax.set_xlim(-25, 165)
        ax.set_xlabel("days after transplanting (DAT)", fontsize=9)
        ax.tick_params(labelsize=7.5)
    axes[-1].plot([], [], color="#555555", lw=1.7, label="observed, 7-day mean")
    axes[-1].plot([], [], color="#555555", lw=1.4, ls="--", label="pooled equation")
    axes[-1].plot([], [], color="#555555", lw=1.2, ls=":", label="harvest")
    axes[-1].legend(fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    for out in (os.path.join(OUT, f"ch4_DAT_overlay_{ts}.png"),
                os.path.join(OUT, "ch4_DAT_overlay.png")):
        fig.savefig(out, dpi=300)
        print(f"  [SAVED] {out}")
    plt.close(fig)


def main():
    args = [a.upper() for a in sys.argv[1:]]
    figs = [a for a in args if a in ("A", "B", "C", "D")] or ["A", "B", "C", "D"]
    arms = [a for a in args if a in ARMS] or list(ARMS)
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"figures {figs} for arms {arms}\n")
    if "A" in figs: fig_timeseries(arms, ts)
    if "B" in figs: fig_pareto(arms, ts)
    if "C" in figs: fig_obspred(arms, ts)
    if "D" in figs: fig_dat_overlay(ts)


if __name__ == "__main__":
    main()
