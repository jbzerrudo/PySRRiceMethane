#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MethaneGrapherSolo.py — Single-site time-series scatter plotter for rice methane data.

Produces 2 files:
  1. Multi-panel PNG  (F_CH4_F + all requested environmental variables)
  2. Multi-panel PDF  (same)
  + Focused 3-panel (CH4, depth, AUC)

Usage:
  - Edit INPUT_CSV, SITE_NAME, and OUTPUT_DIR below.
  - Run: python MethaneGrapherSolo.py
"""

from pathlib import Path
from datetime import datetime
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import ScalarFormatter

# =====================================================================
# USER SETTINGS — EDIT THESE
# =====================================================================

INPUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\KOR-CRK_2018.0.csv"  # ← adjust path to your CSV file
SITE_NAME = "KOR-CRK_2018"  # Used in plot titles and output filenames
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\KOR\2018_correctlabels"  # ← adjust path to your desired output directory

# OUTPUT depth unit for the figures: "cm" (manuscript nomenclature) for ALL sites.
# The stored unit in each CSV is auto-detected, so leave this on "cm" regardless
# of whether a site's CSV holds depth in m (JPN, KOR) or cm (PHL). Depth and AUC
# are converted together (see make_focused_plot).
DEPTH_UNIT = "cm"

# Variables to plot (in order). F_CH4_F is always the top panel.
VARS_TO_PLOT = ["depth", "AUC", "rate", "Tair", "Tsoil", "SR", "Patm", "WS", "WD", "RH", "VPD"]

# CH4 column name
CH4_COL = "F_CH4_F"

# Rolling mean: 7 days at half-hourly resolution = 336 points
ROLL_WINDOW = 7 * 48
ROLL_MIN_PERIODS = 48
DEPTH_ROLL = False   # True to show 7-day rolling mean on water depth panels

# Sentinel / missing-value codes to convert to NaN on read
NA_STRINGS = ["NA", "NaN", "nan", "N/A", "", "-9999", "-9999.0", "-9999.0000"]

# Critical dry threshold in native depth unit (-15 cm)
DEPTH_DRY_THRESHOLD = -15 if DEPTH_UNIT == "cm" else -0.15

# Physical plausibility bounds
PHYS_BOUNDS = {
    "depth":   (-100, 100),
    "rate":    (-50, 50),
    "Tair":    (-40, 50),
    "Tsoil":   (-20, 50),
    "SR":      (0, 1400),
    "Patm":    (85, 110),
    "WS":      (0, 30),
    "WD":      (0, 360),
    "RH":      (0, 110),
    "VPD":     (0, 50),
    "F_CH4_F": (-60, 300),
}

# Y-axis labels (depth-related labels adapt to DEPTH_UNIT)
_du = DEPTH_UNIT
YLABELS = {
    "depth":   f"Water Depth ({_du})",
    "AUC":     f"AUC ({_du}$\\cdot$h)",
    "rate":    f"Vert. Rate ({_du} day$^{{-1}}$)",
    "Tair":    "T$_{air}$ (°C)",
    "Tsoil":   "T$_{soil}$ (°C)",
    "SR":      "Solar Radiation (W m$^{-2}$)",
    "Patm":    "P$_{atm}$ (kPa)",
    "WS":      "Wind Speed (m s$^{-1}$)",
    "WD":      "Wind Direction (°)",
    "RH":      "Relative Humidity (%)",
    "VPD":     "Vapor Pressure Deficit (kPa)",
    "F_CH4_F": "F$_{CH_4}$ (mg m$^{-2}$ h$^{-1}$)",
}

ENCODINGS = ["utf-8", "cp1252", "latin1", "iso-8859-1"]
DOT_SIZE = 1.5  # marker size for scatter dots

# =====================================================================
# FUNCTIONS
# =====================================================================

def load_and_clean(path):
    p = Path(path)
    for enc in ENCODINGS:
        try:
            df = pd.read_csv(p, encoding=enc, na_values=NA_STRINGS, keep_default_na=True)
            break
        except (UnicodeDecodeError, UnicodeError):
            continue
    else:
        raise RuntimeError(f"Cannot read {p} with encodings {ENCODINGS}")

    # ↓↓↓ CHANGED: replaces the CANDIDATE_FMTS loop + best-format pick + NaT fallback ↓↓↓
    raw = df["Date"].astype(str).str.strip()

    # day-first vs month-first: a leading field in 13–31 can only be a day
    lead = pd.to_numeric(raw.str.extract(r"^(\d+)")[0], errors="coerce")
    dayfirst = bool(((lead > 12) & (lead <= 31)).any())

    # parse each value on its own, so a mix of 4-digit and 2-digit years
    # ("04/09/2018" and "12/31/18" in the same column) is handled without
    # dropping rows  (needs pandas >= 2.0)
    df["Date"] = pd.to_datetime(raw, format="mixed", dayfirst=dayfirst, errors="coerce")

    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    # ↑↑↑ CHANGED ↑↑↑

    all_cols = [CH4_COL] + VARS_TO_PLOT
    for col in all_cols:
        if col in df.columns and col in PHYS_BOUNDS:
            lo, hi = PHYS_BOUNDS[col]
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].where(df[col].between(lo, hi), np.nan)

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df

def pct_ylim(series, lo_pct=0, hi_pct=100, pad=0.08):
    clean = series.dropna()
    if len(clean) == 0:
        return (0, 1)
    lo = np.percentile(clean, lo_pct)
    hi = np.percentile(clean, hi_pct)
    margin = max((hi - lo) * pad, 0.5)
    return (lo - margin, hi + margin)


def plain_yticks(ax):
    fmt = ScalarFormatter(useMathText=False)
    fmt.set_scientific(False)
    fmt.set_useOffset(False)
    ax.yaxis.set_major_formatter(fmt)


def find_auc_mean_crossings(dates, auc):
    """Return (crossing_dates, directions) where AUC crosses its own mean.
    direction: +1 = rising through mean, -1 = falling through mean."""
    clean = auc.dropna()
    if len(clean) < 2:
        return [], []
    mean_val = clean.mean()
    diff = clean - mean_val
    sign = np.sign(diff)
    change = sign.diff()
    cross_idx = change[change.abs() > 0].index
    crossing_dates = dates.loc[cross_idx].tolist()
    directions = [+1 if change.loc[i] > 0 else -1 for i in cross_idx]
    return crossing_dates, directions

def make_focused_plot(df, site_name):
    focus_vars = ["depth", "AUC"]
    available = [v for v in focus_vars if v in df.columns and df[v].notna().any()]
    n_panels = 1 + len(available)

    FS_TITLE, FS_LABEL, FS_LEG, FS_TICK, FS_AVG = 22, 19, 17, 16, 14

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3.6 * n_panels), sharex=True,
                             gridspec_kw={"height_ratios": [1.6] + [1] * len(available)})
    if n_panels == 1:
        axes = [axes]
    date = df["Date"]

    # --- top panel: CH4 flux ---
    ax0 = axes[0]
    ax0.plot(date, df[CH4_COL], ".-", markersize=DOT_SIZE + 1, linewidth=0.5,
             color="steelblue", alpha=0.5, label="F$_{CH_4}$ (half-hourly)")
    ch4_roll = df[CH4_COL].rolling(ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS).mean()
    ax0.plot(date, ch4_roll, linewidth=2.4, color="darkorange", label="7-day rolling mean", zorder=5)
    ax0.axhline(0, ls="--", lw=0.8, color="grey", alpha=0.5)
    ch4_mean = df[CH4_COL].mean()
    if np.isfinite(ch4_mean):
        ax0.axhline(ch4_mean, ls="--", lw=1.2, color="navy", alpha=0.6)
        ax0.text(0.02, ch4_mean, f"  avg: {ch4_mean:.2f}", transform=ax0.get_yaxis_transform(),
                 va="center", fontsize=FS_AVG, color="navy",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax0.set_ylabel(YLABELS.get(CH4_COL, CH4_COL), fontsize=FS_LABEL)
    ax0.set_ylim(pct_ylim(df[CH4_COL]))
    ax0.legend(loc="upper right", fontsize=FS_LEG, frameon=True, framealpha=0.8)
    ax0.set_title(f"{site_name}  \u2014  CH$_4$ flux, water depth, and AUC",
                  fontsize=FS_TITLE, fontweight="bold")
    plain_yticks(ax0); ax0.tick_params(labelsize=FS_TICK); ax0.grid(True, alpha=0.15, linestyle=":")

    # --- depth + AUC panels ---
    # AUC is the trapezoidal time-integral of water depth, so it carries depth's
    # unit (depth_unit * h). Derive ONE conversion factor from the stored depth
    # column and apply it to BOTH depth and AUC, so the two panels are always
    # plotted in DEPTH_UNIT and DEPTH_UNIT*h. Paddy depth is sub-metre, so a 99th
    # percentile |depth| > 1 means the CSV stores depth in cm; otherwise in m.
    if "depth" in df.columns and df["depth"].notna().any():
        stored_depth = "cm" if df["depth"].abs().quantile(0.99) > 1.0 else "m"
    else:
        stored_depth = DEPTH_UNIT
    if stored_depth == "m" and DEPTH_UNIT == "cm":
        depth_factor = 100.0
    elif stored_depth == "cm" and DEPTH_UNIT == "m":
        depth_factor = 0.01
    else:
        depth_factor = 1.0

    colours = {"depth": "royalblue", "AUC": "teal"}
    for i, var in enumerate(available):
        ax = axes[i + 1]
        col = colours.get(var, "steelblue")

        if var == "depth":
            series = df[var] * depth_factor
        elif var == "AUC":
            # Build AUC from the depth being plotted, so the bottom panel is the
            # integral of the middle panel in DEPTH_UNIT*h — independent of any
            # precomputed AUC column (which for KOR is stale, in m*h).
            h_disp = df["depth"] * depth_factor
            if "Deltime" in df.columns:
                dt_h = pd.to_numeric(df["Deltime"], errors="coerce").diff()
            else:
                dt_h = pd.Series(0.5, index=df.index)   # half-hourly fallback (hours)
            inc = dt_h * (h_disp + h_disp.shift(1)) / 2.0
            inc.iloc[0] = 0.0
            series = inc.fillna(0).cumsum()
        else:
            series = df[var]

        if var == "AUC":
            ax.fill_between(date, series, 0, color="turquoise", alpha=0.35)
        ax.plot(date, series, ".-", markersize=DOT_SIZE + 1, linewidth=0.5, color=col, alpha=0.6)
        ax.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.4)

        vmean = series.mean()
        if np.isfinite(vmean):
            ax.axhline(vmean, ls="--", lw=1.0, color="grey", alpha=0.5)
            ax.text(0.02, vmean, f"  avg: {vmean:.2f}", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=FS_AVG, color="grey",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))
        ax.set_ylabel(YLABELS.get(var, var), fontsize=FS_LABEL)
        ax.set_ylim(pct_ylim(series))
        plain_yticks(ax); ax.tick_params(labelsize=FS_TICK); ax.grid(True, alpha=0.15, linestyle=":")

    # vertical line(s) where AUC crosses its own mean
    if "AUC" in df.columns and df["AUC"].notna().any():
        cross_dates, cross_dirs = find_auc_mean_crossings(date, df["AUC"])
        for cd, direction in zip(cross_dates, cross_dirs):
            color = "green" if direction > 0 else "crimson"
            label = "above avg" if direction > 0 else "below avg"
            for ax in axes:
                ax.axvline(cd, ls="-.", lw=1.4, color=color, alpha=0.6, zorder=4)
            axes[0].text(cd, axes[0].get_ylim()[1], f"  AUC \u2192 {label}", rotation=90,
                         va="top", ha="left", fontsize=FS_AVG - 1, color=color, alpha=0.8)

    axes[-1].set_xlabel("Date", fontsize=FS_LABEL)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))
    fig.align_ylabels(axes); fig.tight_layout(h_pad=0.5)
    return fig

def make_plot(df, site_name, vars_to_plot):
    valid_vars = [v for v in vars_to_plot if v in df.columns and df[v].notna().any()]
    n_panels = 1 + len(valid_vars)

    fig, axes = plt.subplots(n_panels, 1, figsize=(14, max(8, 1.4 * n_panels)),
                             sharex=True,
                             gridspec_kw={"height_ratios": [2.5] + [1] * len(valid_vars)})
    if n_panels == 1:
        axes = [axes]

    date = df["Date"]

    ax0 = axes[0]
    ax0.plot(date, df[CH4_COL], ".-", markersize=DOT_SIZE, linewidth=0.4, color="steelblue",
             alpha=0.5, label="F$_{CH_4}$ (half-hourly)")

    ch4_roll = df[CH4_COL].rolling(ROLL_WINDOW, center=True, min_periods=ROLL_MIN_PERIODS).mean()
    ax0.plot(date, ch4_roll, linewidth=1.6, color="darkorange", label="7-day rolling mean", zorder=5)
    ax0.axhline(0, ls="--", lw=0.6, color="grey", alpha=0.5)

    ch4_mean = df[CH4_COL].mean()
    if np.isfinite(ch4_mean):
        ax0.axhline(ch4_mean, ls="--", lw=1.0, color="navy", alpha=0.6)
        ax0.text(0.02, ch4_mean, f"  avg: {ch4_mean:.2f}",
                 transform=ax0.get_yaxis_transform(), va="center", fontsize=7, color="navy",
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    ax0.set_ylabel(YLABELS.get(CH4_COL, CH4_COL), fontsize=9)
    ax0.set_ylim(pct_ylim(df[CH4_COL]))
    ax0.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.8)
    ax0.set_title(f"{site_name}  —  Methane and environmental time series",
                  fontsize=12, fontweight="bold")
    plain_yticks(ax0)
    ax0.tick_params(labelsize=7)
    ax0.grid(True, alpha=0.15, linestyle=":")

    for i, var in enumerate(valid_vars):
        ax = axes[i + 1]
        colour = "steelblue"

        if var == "AUC":
            ax.fill_between(date, df[var], 0, color="turquoise", alpha=0.35)
            colour = "teal"

        ax.plot(date, df[var], ".-", markersize=DOT_SIZE, linewidth=0.4, color=colour, alpha=0.5)

        if var in ("depth", "rate", "AUC"):
            ax.axhline(0, ls="--", lw=0.4, color="grey", alpha=0.4)

        if var == "depth" and DEPTH_ROLL:
            depth_roll = df[var].rolling(ROLL_WINDOW, center=True,
                                         min_periods=ROLL_MIN_PERIODS).mean()
            ax.plot(date, depth_roll, linewidth=1.4, color="darkorange",
                    label="7-day rolling mean", zorder=5)
            ax.axhline(DEPTH_DRY_THRESHOLD, ls=":", lw=1.2, color="red", alpha=0.7,
                       label=f"{DEPTH_DRY_THRESHOLD} {DEPTH_UNIT} threshold")
            ax.legend(loc="upper right", fontsize=7, frameon=True, framealpha=0.8)

        vmean = df[var].mean()
        if np.isfinite(vmean):
            ax.axhline(vmean, ls="--", lw=0.7, color="grey", alpha=0.5)
            ax.text(0.02, vmean, f"  avg: {vmean:.2f}",
                    transform=ax.get_yaxis_transform(), va="center", fontsize=6, color="grey",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.7))

        ax.set_ylabel(YLABELS.get(var, var), fontsize=8)
        ax.set_ylim(pct_ylim(df[var]))
        plain_yticks(ax)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15, linestyle=":")

    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator())
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(axes[-1].xaxis.get_major_locator()))

    fig.align_ylabels(axes)
    fig.tight_layout(h_pad=0.3)
    return fig


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"MethaneGrapherSolo — {SITE_NAME}")
    print(f"{'='*60}")

    df = load_and_clean(INPUT_CSV)
    print(f"  Loaded {len(df)} rows, {df['Date'].min()} → {df['Date'].max()}")

    all_cols = [CH4_COL] + VARS_TO_PLOT
    for col in all_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            print(f"  {col:>10}: {valid}/{len(df)} valid")

    # Multi-panel plot
    FOCUSED_VARS = {"depth", "AUC", "Tair", "Tsoil"}
    fig = make_plot(df, SITE_NAME, [v for v in VARS_TO_PLOT if v not in FOCUSED_VARS])
    out_png = outdir / f"{SITE_NAME}_methane_panel_{timestamp}.png"
    out_pdf = outdir / f"{SITE_NAME}_methane_panel_{timestamp}.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_png}")
    print(f"  Saved: {out_pdf}")
    plt.close(fig)

    # Focused 3-panel plot
    fig_focus = make_focused_plot(df, SITE_NAME)
    out_focus_png = outdir / f"{SITE_NAME}_CH4_depth_AUC_{timestamp}.png"
    out_focus_pdf = outdir / f"{SITE_NAME}_CH4_depth_AUC_{timestamp}.pdf"
    fig_focus.savefig(out_focus_png, dpi=300, bbox_inches="tight")
    fig_focus.savefig(out_focus_pdf, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out_focus_png}")
    print(f"  Saved: {out_focus_pdf}")
    plt.close(fig_focus)

    print(f"\n{'='*60}")
    print("Done.")