"""
Collinearity Checker — Iterative VIF elimination for rice methane datasets
===========================================================================

Performs:
  1. Replaces missing flags (-9999 etc.) and inf values
  2. Plots and saves correlation heatmap
  3. Iterative VIF elimination (drops highest-VIF variable until all ≤ threshold)
  4. Saves detailed text report with elimination log
  5. Exports retained-variables CSV (with target) ready for next pipeline step
  6. Exports final VIF table as CSV

Changes from collinearity_heatmapper2.py:
  1. [BUG FIX] Top correlates now reference only REMAINING variables, not
     already-dropped ones (v1 used the initial correlation matrix for all iterations)
  2. [BUG FIX] Missing flags (-9999, -999900, -99999) replaced with NaN before VIF
  3. [BUG FIX] inf/-inf values replaced with NaN before VIF
  4. [BUG FIX] Returned DataFrame now includes target column (ready for GAM/PySR)
  5. [BUG FIX] Heatmap saved to file instead of plt.show() (which blocks execution)
  6. [NEW] Output directory support
  7. [NEW] Timestamped outputs
  8. [NEW] Post-collinearity CSV export ({sitename}_retainedvars_postcollin.csv)
  9. [NEW] Final VIF table exported as CSV

Usage:
  - Edit INPUT_CSV, OUTPUT_DIR, and SITE_NAME below.
  - Run: python collinearity_checker.py

Author: Jef Zerrudo / Claude
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
matplotlib.use("Agg")  # Non-blocking backend — saves to file, no plt.show()
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from pathlib import Path
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

# =====================================================================
# USER CONFIG — EDIT THESE
# =====================================================================

INPUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\PooledIntensive_retvars_gam1.csv"
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\COLLINCHECK\POOLED"
SITE_NAME = "POOLED"  # used in output filenames; e.g. "SK-CRK-2016"

TARGET_COL = "F_CH4_F"
VIF_THRESHOLD = 5.0 # 5.0 or 10.0
EXCLUDE_HEADERS = ["site", "w", "Date", "Deltime", "time", "F_CH4_F_orig"]
MISSING_FLAGS = [-9999, -999900, -99999]
DPI = 300
ADD_TIMESTAMP = True

# Importance-guided VIF elimination (optional)
# When provided, drops the LEAST important variable among those above VIF threshold
# instead of the HIGHEST VIF variable. This preserves physically meaningful predictors.
# Set to None to use the original highest-VIF-first behaviour.
IMPORTANCE_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\POOLED\Metadata\Intensive\predictor_perm_importance_20260729_190125.csv"
IMPORTANCE_COL = "perm_RMSE_increase"  # column name in the importance CSV
IMPORTANCE_PREDICTOR_COL = "predictor"  # predictor name column


# =====================================================================
# FUNCTIONS
# =====================================================================

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def analyze_collinearity(df, target_col, threshold=5.0, outdir=None,  # change threshold to 5.0 for stricter elimination
                         site_name="site", timestamp="",
                         importance_rank=None):
    """
    Iterative VIF elimination with detailed reporting.

    Parameters:
        importance_rank : dict or None
            Maps predictor name → importance score. When provided, the
            elimination loop drops the LEAST important variable among
            those above the VIF threshold, instead of the highest-VIF one.
            This preserves high-importance predictors (e.g. temperature,
            depth) that would otherwise be lost to correlated-cluster
            chain elimination.

    Returns:
        retained_df  : DataFrame with retained predictors + target (complete cases)
        final_vif_df : DataFrame with feature names and final VIF values
    """
    # ── 1. Prepare Data ──
    cols_to_exclude = [target_col] + EXCLUDE_HEADERS
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])

    # FIX: Coerce string columns to numeric BEFORE filtering
    # (catches columns pandas misread as text, e.g. h*Ts, Tsoil)
    # Note: newer pandas uses dtype 'str' instead of 'object', so check dtype.kind
    for c in X.columns:
        if X[c].dtype.kind == 'O':  # 'O' covers both 'object' and 'str' dtypes
            X[c] = pd.to_numeric(X[c], errors="coerce")

    # Now keep only numeric columns (drops any that were truly non-numeric)
    X = X.select_dtypes(include=[np.number])

    # FIX: Replace missing flags with NaN
    X = X.replace(MISSING_FLAGS, np.nan)

    # FIX: Replace inf/-inf with NaN
    X = X.replace([np.inf, -np.inf], np.nan)

    X = X.dropna()

    n_obs = len(X)
    n_initial = len(X.columns)
    print(f"  [INFO] {n_obs:,} complete observations, {n_initial} predictors")

    # ── 2. Correlation Heatmap (saved to file) ──
    correlation_matrix = X.corr()

    figsize = max(10, n_initial * 0.4)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(
        correlation_matrix, annot=(n_initial <= 20), fmt=".2f",
        cmap="coolwarm", center=0, ax=ax, square=True,
        annot_kws={"size": 7} if n_initial <= 20 else {},
        linewidths=0.5 if n_initial <= 20 else 0
    )
    ax.set_title(f"Correlation Heatmap — {site_name} (pre-VIF, {n_initial} vars)", fontsize=13)
    plt.tight_layout()

    heatmap_file = f"{site_name}_correlation_heatmap_{timestamp}.png" if ADD_TIMESTAMP \
        else f"{site_name}_correlation_heatmap.png"
    fig.savefig(outdir / heatmap_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {heatmap_file}")

    # ── 3. Iterative VIF Elimination ──
    variables = X.columns.tolist()
    dropped = []
    iteration_logs = []
    iteration = 0

    # Initial VIF snapshot (before any elimination)
    initial_vif = pd.DataFrame()
    initial_vif["feature"] = variables
    initial_vif["VIF"] = [variance_inflation_factor(X[variables].values, i)
                          for i in range(len(variables))]
    initial_vif = initial_vif.sort_values("VIF", ascending=False).reset_index(drop=True)

    while True:
        iteration += 1
        vif_data = pd.DataFrame()
        vif_data["feature"] = variables
        vif_data["VIF"] = [variance_inflation_factor(X[variables].values, i)
                           for i in range(len(variables))]
        vif_data = vif_data.sort_values("VIF", ascending=False).reset_index(drop=True)

        max_vif = vif_data["VIF"].max()

        # FIX: Handle inf VIF (happens with perfectly collinear variables)
        if not np.isfinite(max_vif):
            inf_vars = vif_data.loc[~np.isfinite(vif_data["VIF"]), "feature"].tolist()

            if importance_rank and len(inf_vars) > 1:
                # Drop the least important among the inf-VIF variables
                drop_feature = min(inf_vars, key=lambda f: importance_rank.get(f, float("inf")))
            else:
                drop_feature = inf_vars[0]

            print(f"  [WARN] '{drop_feature}' has infinite VIF — dropping (perfect collinearity)")
            variables.remove(drop_feature)
            dropped.append(drop_feature)
            iteration_logs.append({
                "iteration": iteration,
                "vif_table": vif_data.copy(),
                "dropped": drop_feature,
                "dropped_vif": float("inf"),
                "top_correlates": {},
                "drop_reason": "importance-guided (inf VIF group)" if importance_rank else "highest VIF (inf)"
            })
            continue

        if max_vif > threshold:
            above = vif_data.loc[vif_data["VIF"] > threshold, "feature"].tolist()

            if importance_rank and len(above) > 1:
                # Drop the LEAST important variable among those above threshold
                drop_feature = min(above, key=lambda f: importance_rank.get(f, float("inf")))
                drop_vif = float(vif_data.loc[vif_data["feature"] == drop_feature, "VIF"].iloc[0])
            else:
                # Original behaviour: drop highest VIF
                drop_feature = vif_data.iloc[0]["feature"]
                drop_vif = max_vif

            # FIX: Compute correlates using only REMAINING variables
            remaining_corr = X[variables].corr()
            correlations = remaining_corr[drop_feature].drop(drop_feature).abs().sort_values(ascending=False)
            top_correlates = correlations.head(3)

            drop_reason = (
                f"importance-guided (least important above VIF threshold; "
                f"importance={importance_rank.get(drop_feature, 0):.2f})"
                if importance_rank else "highest VIF"
            )

            iteration_logs.append({
                "iteration": iteration,
                "vif_table": vif_data.copy(),
                "dropped": drop_feature,
                "dropped_vif": drop_vif,
                "top_correlates": top_correlates,
                "drop_reason": drop_reason
            })

            print(f"  Iter {iteration}: Dropping '{drop_feature}' (VIF={drop_vif:.2f})"
                  + (f" [least important above threshold]" if importance_rank else ""))
            variables.remove(drop_feature)
            dropped.append(drop_feature)
        else:
            # All pass — log final state
            iteration_logs.append({
                "iteration": iteration,
                "vif_table": vif_data.copy(),
                "dropped": None,
                "dropped_vif": None,
                "top_correlates": None,
                "drop_reason": None
            })
            print(f"  Iter {iteration}: All VIFs ≤ {threshold}. Done.")
            break

    # ── 4. Final VIF Table ──
    final_vif = vif_data.copy()

    # ── 5. Write Detailed Report ──
    report_file = f"{site_name}_collinearity_report_{timestamp}.txt" if ADD_TIMESTAMP \
        else f"{site_name}_collinearity_report.txt"
    report_path = outdir / report_file

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  COLLINEARITY ANALYSIS REPORT\n")
        f.write(f"  Site            : {site_name}\n")
        f.write(f"  Target variable : {target_col}\n")
        f.write(f"  VIF threshold   : {threshold}\n")
        f.write(f"  Observations    : {n_obs}\n")
        f.write(f"  Initial features: {n_initial}\n")
        f.write(f"  Retained        : {len(variables)}\n")
        f.write(f"  Dropped         : {len(dropped)}\n")
        if importance_rank:
            f.write(f"  Drop strategy   : Importance-guided (least important above VIF threshold)\n")
        else:
            f.write(f"  Drop strategy   : Standard (highest VIF first)\n")
        f.write(f"  Generated       : {get_readable_timestamp()}\n")
        f.write("=" * 72 + "\n\n")

        # --- Initial VIF Scores ---
        f.write("-" * 72 + "\n")
        f.write("  INITIAL VIF SCORES (before elimination)\n")
        f.write("-" * 72 + "\n")
        for _, row in initial_vif.iterrows():
            flag = " *** ABOVE THRESHOLD" if row["VIF"] > threshold else ""
            f.write(f"  {row['feature']:<30s}  VIF = {row['VIF']:>10.2f}{flag}\n")
        f.write("\n")

        # --- Iteration Details ---
        f.write("-" * 72 + "\n")
        f.write("  STEPWISE VIF ELIMINATION LOG\n")
        f.write("-" * 72 + "\n\n")

        for log in iteration_logs:
            f.write(f"  Iteration {log['iteration']}\n")
            f.write(f"  {'Feature':<30s}  {'VIF':>10s}\n")
            f.write(f"  {'-'*30}  {'-'*10}\n")
            for _, row in log["vif_table"].iterrows():
                marker = " <-- DROPPED" if (log["dropped"] and row["feature"] == log["dropped"]) else ""
                vif_str = f"{row['VIF']:>10.2f}" if np.isfinite(row['VIF']) else "       inf"
                f.write(f"  {row['feature']:<30s}  {vif_str}{marker}\n")

            if log["dropped"]:
                vif_str = f"{log['dropped_vif']:.2f}" if np.isfinite(log['dropped_vif']) else "inf"
                f.write(f"\n  Action: Dropped '{log['dropped']}' (VIF = {vif_str})\n")
                if log.get("drop_reason"):
                    f.write(f"  Strategy: {log['drop_reason']}\n")
                if log["top_correlates"] is not None and len(log["top_correlates"]) > 0:
                    f.write(f"  Reason: VIF exceeds threshold of {threshold}. High multicollinearity\n")
                    f.write(f"          with the following REMAINING variables:\n")
                    for var, corr_val in log["top_correlates"].items():
                        f.write(f"            - {var} (|r| = {corr_val:.3f})\n")
                else:
                    f.write(f"  Reason: Perfect collinearity (inf VIF)\n")
            else:
                f.write(f"\n  Action: None. All remaining VIFs <= {threshold}.\n")
            f.write("\n")

        # --- Dropped Variables Summary ---
        f.write("-" * 72 + "\n")
        f.write("  DROPPED VARIABLES SUMMARY\n")
        f.write("-" * 72 + "\n\n")

        if dropped:
            for i, log in enumerate([l for l in iteration_logs if l["dropped"]]):
                f.write(f"  {i+1}. {log['dropped']}\n")
                vif_str = f"{log['dropped_vif']:.2f}" if np.isfinite(log['dropped_vif']) else "inf"
                f.write(f"     VIF at removal : {vif_str}\n")
                if log["top_correlates"] is not None and len(log["top_correlates"]) > 0:
                    f.write(f"     Top correlates : ")
                    pairs = [f"{var} (|r|={corr_val:.3f})" for var, corr_val in log["top_correlates"].items()]
                    f.write(", ".join(pairs) + "\n")
                f.write(f"     Verdict        : Redundant; variance largely explained by other predictors.\n\n")
        else:
            f.write("  No variables were dropped.\n\n")

        # --- Retained Variables Summary ---
        f.write("-" * 72 + "\n")
        f.write("  RETAINED VARIABLES SUMMARY\n")
        f.write("-" * 72 + "\n\n")

        for _, row in final_vif.iterrows():
            f.write(f"  {row['feature']}\n")
            f.write(f"     Final VIF : {row['VIF']:.2f}\n")
            f.write(f"     Verdict   : Retained. VIF <= {threshold}; contributes unique variance\n")
            f.write(f"                 not captured by other predictors.\n\n")

        f.write("=" * 72 + "\n")
        f.write("  END OF REPORT\n")
        f.write("=" * 72 + "\n")

    print(f"  [SAVED] {report_file}")

    # ── 6. Export Final VIF Table ──
    vif_csv = f"{site_name}_final_vif_{timestamp}.csv" if ADD_TIMESTAMP \
        else f"{site_name}_final_vif.csv"
    final_vif.to_csv(outdir / vif_csv, index=False)
    print(f"  [SAVED] {vif_csv}")

    # ── 7. Export Retained Variables + Target as CSV ──
    # FIX: Build retained set from original df — replace flags but keep all rows
    export_cols = variables + [target_col]
    retained_df = df[["Date"] + export_cols].copy() if "Date" in df.columns \
        else df[export_cols].copy()
    retained_df[variables] = retained_df[variables].replace(MISSING_FLAGS, np.nan)
    retained_df[variables] = retained_df[variables].replace([np.inf, -np.inf], np.nan)
    retained_df[target_col] = retained_df[target_col].replace(MISSING_FLAGS, np.nan)

    postcollin_csv = f"{site_name}_retainedvars_postcollin_{timestamp}.csv" if ADD_TIMESTAMP \
        else f"{site_name}_retainedvars_postcollin.csv"
    retained_df.to_csv(outdir / postcollin_csv, index=False)
    print(f"  [SAVED] {postcollin_csv} "
          f"({len(retained_df):,} rows × {len(variables)} predictors + target)")

    # Summary
    print(f"\n  --- Final Selection ---")
    print(f"  Initial : {n_initial} predictors")
    print(f"  Dropped : {len(dropped)} ({', '.join(dropped) if dropped else 'none'})")
    print(f"  Retained: {len(variables)} ({', '.join(variables)})")

    return retained_df, final_vif


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()

    # Create timestamped subfolder to keep each run's outputs together
    run_dir = outdir / f"{SITE_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  Collinearity Checker — {SITE_NAME}")
    print(f"  {get_readable_timestamp()}")
    print(f"{'='*70}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  [INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Load importance ranking (optional)
    importance_rank = None
    if IMPORTANCE_CSV is not None:
        try:
            imp_df = pd.read_csv(IMPORTANCE_CSV)
            importance_rank = dict(zip(
                imp_df[IMPORTANCE_PREDICTOR_COL],
                imp_df[IMPORTANCE_COL]
            ))
            print(f"  [INFO] Loaded importance ranking: {len(importance_rank)} predictors")
            print(f"  [INFO] Drop strategy: importance-guided (least important above VIF threshold)")
        except Exception as e:
            print(f"  [WARN] Could not load importance CSV: {e}")
            print(f"  [WARN] Falling back to standard highest-VIF-first strategy")
            importance_rank = None
    else:
        print(f"  [INFO] Drop strategy: standard (highest VIF first)")
        print(f"         Set IMPORTANCE_CSV to enable importance-guided elimination")

    retained_df, final_vif = analyze_collinearity(
        df,
        target_col=TARGET_COL,
        threshold=VIF_THRESHOLD,
        outdir=run_dir,
        site_name=SITE_NAME,
        timestamp=timestamp,
        importance_rank=importance_rank,
    )

    print(f"\n{'='*70}")
    print(f"  Done. All outputs in: {run_dir}")
    print(f"{'='*70}\n")