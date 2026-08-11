"""
pooledsites_collinchecker.py — Collinearity checker for the POOLED arm
==============================================================================
Derived from collinearity_checker_final.py. Same outputs, same report format,
same importance-guided elimination. Five corrections and three additions.

CORRECTIONS
  C1 [BUG] VIF was computed without an intercept, which returns the UNCENTRED
           VIF. A variable whose mean dwarfs its spread is then "explained" by
           any other offset variable, which is not collinearity. Measured on the
           pooled set: Tv_K 9.6e8 uncentred against 5.1e5 centred (mean/sd = 46
           in kelvin), Patm 1.6e8 against 5.7e4 (mean/sd = 86 in hPa),
           hbar_wet 17.76 against 3.36. This is why every run lost its whole
           thermal block. VIF_ADD_CONSTANT now controls it and defaults to True.

           >>> THIS CHANGES RESULTS. The per-site runs of 23-25 July used the
           >>> uncentred version. Set VIF_ADD_CONSTANT = False to reproduce them
           >>> exactly. Applying the fix to one arm and not the others is not
           >>> defensible, so either re-run every site or record the deviation.

  C2 [BUG] A predictor absent from the importance CSV received
           importance = float("inf") in the selection, making it undroppable,
           while the report printed importance = 0.00 for the same variable. The
           two defaults disagreed and the protection was silent. Missing scores
           are now a hard stop (MISSING_IMPORTANCE_IS_FATAL), and if you switch
           that off there is ONE default everywhere and it is -inf: no score means
           no evidence of importance, so drop first rather than shield. 0 is not a
           lower bound because permutation importance can be negative, and NaN
           cannot be used because min() against NaN returns whichever element
           happened to come first.

  C3 [BUG] `if importance_rank and len(above) > 1` bypassed importance whenever
           exactly one variable sat above threshold. Now >= 1.

  C4 [BUG] The exported CSV carried only Date plus the retained predictors, so
           `site` and `w` never reached the next step. LOSO is impossible without
           `site` and site-weighted training is impossible without `w`.

  C5 [BUG] Exact identities were left for the VIF loop to discover. With an
           identity present the loop drops one member, then often the survivor
           too, and the whole family is lost. They are now removed up front and
           reported.

ADDITIONS
  A1  DROP_COLUMNS: box (0) structural de-duplication, declared in the script so
      the run is reproducible instead of depending on a hand-edited CSV.
  A2  PROTECTED: never dropped. Defaults to EMPTY so behaviour is unchanged
      unless you declare something. Declare it in the transfer log with reasons.
  A3  Automatic exact-identity detection by SVD null space, plus a coverage
      diagnostic that reports whether any water and any thermal variable
      survived. The diagnostic only reports; it does not act.

Author: Jef Zerrudo / Claude
==============================================================================
"""

import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore", message="divide by zero encountered in scalar divide")

# =====================================================================
# USER CONFIG
# =====================================================================

INPUT_CSV  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\POOLED_retvars_pass1.csv"
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\COLLINCHECK\POOLED\rerun_20260731\pass1"
SITE_NAME  = "POOLED"

TARGET_COL      = "F_CH4_F"
VIF_THRESHOLD   = 5.0
EXCLUDE_HEADERS = ["site", "w", "Date", "Deltime", "time", "F_CH4_F_orig"]
MISSING_FLAGS   = [-9999, -999900, -99999]
DPI             = 300
ADD_TIMESTAMP   = True

# C1. True = standard (centred) VIF. False = the uncentred VIF the 23-25 July
#     per-site runs used. Changing this changes which variables survive.
VIF_ADD_CONSTANT = True

# A1. Box (0): structural de-duplication, declared before the run.
#     Exact identities measured on PooledIntensive_retvars_gam1.csv:
#         depth  = hwet  - hdry     (residual 0.000e+00)  -> drop hdry
#         VPD    = es    - ea       (residual 1.1e-01, rounding from kPa) -> drop ea
#         DelTsa = Tsoil - Tair     (residual 6.0e-15)    -> drop DelTsa
#     Redundant intensive cluster, mutually |r| 0.72 to 0.95, keep hbar_wet:
#         hbar, hbar*es, hbar_wet*es
#     hdry is chosen over depth/hwet because it is identically zero at SK-CRK.
#     ea is chosen because es and VPD are the two that appear in the equations.
#     DelTsa is chosen because Tair and Tsoil are state variables and PySR can
#     form their difference itself.

#DROP_COLUMNS = ["hdry", "ea", "DelTsa", "hbar", "hbar*es", "hbar_wet*es"]
DROP_COLUMNS = ["hbar", "hbar*es", "hbar_wet*es"]

# A2. Never dropped. EMPTY by default: unchanged behaviour. If you populate it,
#     record the choice and the reason in the transfer log before running.
PROTECTED = []          # e.g. ["hbar_wet", "es"]

# A3. Coverage diagnostic only. Reports whether the retained set kept any water
#     and any thermal variable. Does not change the elimination.
WATER_FAMILY = ["hbar", "hbar_wet", "hbar_dry", "hbar*es", "hbar_wet*es", "fwet",
                "AUC", "AUC_wet", "AUC_dry", "depth", "hwet", "hdry",
                "h_inv", "h_ASINH_cm"]
THERMAL_FAMILY = ["Tair", "Tsoil", "Tv_K", "es", "ea", "VPD", "DelTsa",
                  "asinh_Ta", "asinh_Ts", "rho_moist", "Tdew"]

IMPORTANCE_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\POOLED\rerun_20260731\pass1\predictor_perm_importance_20260731_125512.csv"
IMPORTANCE_COL = "perm_RMSE_increase"
IMPORTANCE_PREDICTOR_COL = "predictor"
MISSING_IMPORTANCE_IS_FATAL = True      # C2

IDENTITY_TOL = 1e-8      # null-space tolerance for the identity detector


# =====================================================================
# HELPERS
# =====================================================================

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def vif_series(X, cols, add_constant):
    """VIF for each column. With add_constant the design includes an intercept,
    which is the standard (centred) VIF. Without it, the uncentred VIF."""
    A = X[cols].to_numpy(float)
    if add_constant:
        A = np.column_stack([np.ones(len(A)), A])
    off = 1 if add_constant else 0
    with np.errstate(all="ignore"):
        v = [variance_inflation_factor(A, i + off) for i in range(len(cols))]
    return pd.DataFrame({"feature": cols, "VIF": v}) \
             .sort_values("VIF", ascending=False).reset_index(drop=True)


def detect_identities(X, cols, tol=IDENTITY_TOL):
    """Find exact linear dependencies by SVD null space.

    Returns (identities, constants). Zero-variance columns are pulled out first
    and reported separately: they are their own kind of degeneracy (infinite VIF
    against the intercept) and leaving them in makes every null direction they
    appear in unreadable.
    """
    live = [c for c in cols if float(np.nanstd(X[c].to_numpy(float))) > 0.0]
    constants = [c for c in cols if c not in live]
    if len(live) < 2:
        return [], constants

    A = X[live].to_numpy(float)
    A = (A - A.mean(axis=0)) / A.std(axis=0)
    _, sv, Vt = np.linalg.svd(A, full_matrices=False)
    if not np.isfinite(sv[0]) or sv[0] <= 0:
        return [], constants

    out = []
    for k, val in enumerate(sv):
        if val / sv[0] < tol:
            v = Vt[k]
            idx = np.argsort(-np.abs(v))[:4]
            out.append((val / sv[0],
                        [(live[i], float(v[i])) for i in idx if abs(v[i]) > 1e-3]))
    return out, constants


# =====================================================================
# MAIN ANALYSIS
# =====================================================================

def analyze_collinearity(df, target_col, threshold, outdir, site_name, timestamp,
                         importance_rank=None):
    # ── 1. Prepare ───────────────────────────────────────────────────────────
    cols_to_exclude = [target_col] + EXCLUDE_HEADERS
    X = df.drop(columns=[c for c in cols_to_exclude if c in df.columns])

    for c in X.columns:
        if X[c].dtype.kind == "O":
            X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.select_dtypes(include=[np.number])
    X = X.replace(MISSING_FLAGS, np.nan).replace([np.inf, -np.inf], np.nan)

    # A1 / C5. Box (0) removals, before anything is measured.
    dropped_box0 = [c for c in DROP_COLUMNS if c in X.columns]
    if dropped_box0:
        X = X.drop(columns=dropped_box0)
        print(f"  [BOX 0] removed by declaration: {dropped_box0}")
    missing_box0 = [c for c in DROP_COLUMNS if c not in dropped_box0]
    if missing_box0:
        print(f"  [BOX 0] listed in DROP_COLUMNS but not present: {missing_box0}")

    X = X.dropna()
    n_obs, n_initial = len(X), len(X.columns)
    print(f"  [INFO] {n_obs:,} complete observations, {n_initial} predictors")
    print(f"  [INFO] VIF_ADD_CONSTANT = {VIF_ADD_CONSTANT} "
          f"({'standard centred VIF' if VIF_ADD_CONSTANT else 'UNCENTRED VIF, legacy behaviour'})")

    # A3. Any exact dependency left after box (0)?
    residual_identities, constant_cols = detect_identities(X, list(X.columns))
    if constant_cols:
        print(f"  [WARN] zero-variance column(s): {constant_cols}. These carry no")
        print("         information and take infinite VIF against the intercept.")
    if residual_identities:
        print(f"  [WARN] {len(residual_identities)} exact linear dependency(ies) remain "
              f"after box (0):")
        for ratio, terms in residual_identities:
            print("         " + "  ".join(f"{c}({w:+.3f})" for c, w in terms))
        print("         Add one member of each to DROP_COLUMNS rather than letting")
        print("         the VIF loop discover them, or the family will be lost.")
    else:
        print("  [OK]   no exact linear dependencies remain after box (0)")

    # ── 2. Heatmap ───────────────────────────────────────────────────────────
    corr = X.corr()
    figsize = max(10, n_initial * 0.4)
    fig, ax = plt.subplots(figsize=(figsize, figsize * 0.85))
    sns.heatmap(corr, annot=(n_initial <= 20), fmt=".2f", cmap="coolwarm", center=0,
                ax=ax, square=True, annot_kws={"size": 7} if n_initial <= 20 else {},
                linewidths=0.5 if n_initial <= 20 else 0)
    ax.set_title(f"Correlation Heatmap — {site_name} (pre-VIF, {n_initial} vars)", fontsize=13)
    plt.tight_layout()
    heatmap_file = (f"{site_name}_correlation_heatmap_{timestamp}.png" if ADD_TIMESTAMP
                    else f"{site_name}_correlation_heatmap.png")
    fig.savefig(outdir / heatmap_file, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {heatmap_file}")

    # ── 3. Iterative elimination ─────────────────────────────────────────────
    variables = X.columns.tolist()
    protected_present = [c for c in PROTECTED if c in variables]
    if protected_present:
        print(f"  [PROTECTED] never dropped: {protected_present}")

    dropped, iteration_logs, iteration = [], [], 0
    initial_vif = vif_series(X, variables, VIF_ADD_CONSTANT)

    def importance_of(f):
        """C2. ONE default everywhere, and it is -inf, not inf and not 0.

        A predictor with no score has no evidence of importance, so it is dropped
        FIRST rather than shielded. inf shielded it silently, and 0 is not a lower
        bound because permutation importance can be negative. NaN cannot be used:
        every comparison against NaN is False, so min() would return whichever
        element happened to come first. In fatal mode this is never reached.
        """
        if not importance_rank:
            return float("-inf")
        return importance_rank.get(f, float("-inf"))

    unscored = ([f for f in variables if f not in importance_rank]
                if importance_rank else [])
    if unscored:
        print(f"  [WARN] no importance score, will be dropped first if above "
              f"threshold: {unscored}")

    while True:
        iteration += 1
        vif_data = vif_series(X, variables, VIF_ADD_CONSTANT)
        max_vif = vif_data["VIF"].max()

        # perfect collinearity
        if not np.isfinite(max_vif):
            inf_vars = vif_data.loc[~np.isfinite(vif_data["VIF"]), "feature"].tolist()
            inf_vars = [f for f in inf_vars if f not in PROTECTED]        # A2
            if not inf_vars:
                print("  [STOP] only PROTECTED variables have infinite VIF. "
                      "Add one of them to DROP_COLUMNS.")
                break
            if importance_rank and len(inf_vars) >= 1:                     # C3
                drop_feature = min(inf_vars, key=importance_of)
            else:
                drop_feature = inf_vars[0]
            print(f"  [WARN] '{drop_feature}' has infinite VIF — dropping (perfect collinearity)")
            variables.remove(drop_feature)
            dropped.append(drop_feature)
            iteration_logs.append(dict(iteration=iteration, vif_table=vif_data.copy(),
                                       dropped=drop_feature, dropped_vif=float("inf"),
                                       top_correlates={},
                                       drop_reason=("importance-guided (inf VIF group)"
                                                    if importance_rank else "highest VIF (inf)")))
            continue

        if max_vif > threshold:
            above = vif_data.loc[vif_data["VIF"] > threshold, "feature"].tolist()
            above = [f for f in above if f not in PROTECTED]               # A2
            if not above:
                print(f"  [STOP] only PROTECTED variables remain above {threshold}. Done.")
                iteration_logs.append(dict(iteration=iteration, vif_table=vif_data.copy(),
                                           dropped=None, dropped_vif=None,
                                           top_correlates=None, drop_reason=None))
                break

            if importance_rank and len(above) >= 1:                        # C3
                drop_feature = min(above, key=importance_of)
                drop_vif = float(vif_data.loc[vif_data["feature"] == drop_feature, "VIF"].iloc[0])
                imp = importance_of(drop_feature)
                drop_reason = (f"importance-guided (least important above VIF threshold; "
                               f"importance={imp:.4g})")
            else:
                drop_feature = above[0]
                drop_vif = float(vif_data.loc[vif_data["feature"] == drop_feature, "VIF"].iloc[0])
                drop_reason = "highest VIF"

            remaining_corr = X[variables].corr()
            correlations = remaining_corr[drop_feature].drop(drop_feature).abs() \
                                                       .sort_values(ascending=False)
            iteration_logs.append(dict(iteration=iteration, vif_table=vif_data.copy(),
                                       dropped=drop_feature, dropped_vif=drop_vif,
                                       top_correlates=correlations.head(3),
                                       drop_reason=drop_reason))
            print(f"  Iter {iteration}: Dropping '{drop_feature}' (VIF={drop_vif:.2f})"
                  + (" [least important above threshold]" if importance_rank else ""))
            variables.remove(drop_feature)
            dropped.append(drop_feature)
        else:
            iteration_logs.append(dict(iteration=iteration, vif_table=vif_data.copy(),
                                       dropped=None, dropped_vif=None,
                                       top_correlates=None, drop_reason=None))
            print(f"  Iter {iteration}: All VIFs <= {threshold}. Done.")
            break

    final_vif = vif_series(X, variables, VIF_ADD_CONSTANT)

    # A3. coverage diagnostic
    water = [c for c in variables if c in WATER_FAMILY]
    therm = [c for c in variables if c in THERMAL_FAMILY]
    print(f"\n  [COVERAGE] water survivors  : {water or 'NONE'}")
    print(f"  [COVERAGE] thermal survivors: {therm or 'NONE'}")
    coverage_fail = (not water) or (not therm)
    if coverage_fail:
        print("  [COVERAGE] *** the declared coverage rule FIRES. Report the relaxation")
        print("             or add the lost variable to PROTECTED, and say which. ***")

    # ── 4. Report ────────────────────────────────────────────────────────────
    report_file = (f"{site_name}_collinearity_report_{timestamp}.txt" if ADD_TIMESTAMP
                   else f"{site_name}_collinearity_report.txt")
    with open(outdir / report_file, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n  COLLINEARITY ANALYSIS REPORT\n")
        f.write(f"  Site             : {site_name}\n")
        f.write(f"  Target variable  : {target_col}\n")
        f.write(f"  VIF threshold    : {threshold}\n")
        f.write(f"  VIF definition   : {'standard (intercept in design)' if VIF_ADD_CONSTANT else 'UNCENTRED (no intercept, legacy)'}\n")
        f.write(f"  Observations     : {n_obs}\n")
        f.write(f"  Initial features : {n_initial}\n")
        f.write(f"  Retained         : {len(variables)}\n")
        f.write(f"  Dropped by VIF   : {len(dropped)}\n")
        f.write(f"  Drop strategy    : {'Importance-guided (least important above VIF threshold)' if importance_rank else 'Standard (highest VIF first)'}\n")
        f.write(f"  Generated        : {get_readable_timestamp()}\n")
        f.write("=" * 72 + "\n\n")

        f.write("-" * 72 + "\n  PRE-REGISTERED DECLARATIONS\n" + "-" * 72 + "\n")
        f.write(f"  Box (0) removed by declaration : {dropped_box0 or 'none'}\n")
        f.write(f"  PROTECTED (never dropped)      : {protected_present or 'none'}\n")
        f.write(f"  Importance source              : {IMPORTANCE_CSV}\n")
        f.write(f"  Missing importance is fatal    : {MISSING_IMPORTANCE_IS_FATAL}\n\n")

        if residual_identities:
            f.write("-" * 72 + "\n  EXACT LINEAR DEPENDENCIES REMAINING AFTER BOX (0)\n" + "-" * 72 + "\n")
            for ratio, terms in residual_identities:
                f.write("  " + "  ".join(f"{c}({w:+.3f})" for c, w in terms) + "\n")
            f.write("\n")

        f.write("-" * 72 + "\n  INITIAL VIF SCORES (after box 0, before elimination)\n" + "-" * 72 + "\n")
        for _, r in initial_vif.iterrows():
            flag = " *** ABOVE THRESHOLD" if r["VIF"] > threshold else ""
            vs = f"{r['VIF']:>10.2f}" if np.isfinite(r["VIF"]) else "       inf"
            f.write(f"  {r['feature']:<30s}  VIF = {vs}{flag}\n")
        f.write("\n")

        f.write("-" * 72 + "\n  STEPWISE VIF ELIMINATION LOG\n" + "-" * 72 + "\n\n")
        for log in iteration_logs:
            f.write(f"  Iteration {log['iteration']}\n  {'Feature':<30s}  {'VIF':>10s}\n")
            f.write(f"  {'-'*30}  {'-'*10}\n")
            for _, r in log["vif_table"].iterrows():
                marker = " <-- DROPPED" if (log["dropped"] and r["feature"] == log["dropped"]) else ""
                marker += " [PROTECTED]" if r["feature"] in PROTECTED else ""
                vs = f"{r['VIF']:>10.2f}" if np.isfinite(r["VIF"]) else "       inf"
                f.write(f"  {r['feature']:<30s}  {vs}{marker}\n")
            if log["dropped"]:
                vs = f"{log['dropped_vif']:.2f}" if np.isfinite(log["dropped_vif"]) else "inf"
                f.write(f"\n  Action: Dropped '{log['dropped']}' (VIF = {vs})\n")
                f.write(f"  Strategy: {log['drop_reason']}\n")
                if log["top_correlates"] is not None and len(log["top_correlates"]) > 0:
                    f.write(f"  Reason: VIF exceeds threshold of {threshold}. High multicollinearity\n")
                    f.write("          with the following REMAINING variables:\n")
                    for var, cv in log["top_correlates"].items():
                        f.write(f"            - {var} (|r| = {cv:.3f})\n")
                else:
                    f.write("  Reason: Perfect collinearity (inf VIF)\n")
            else:
                f.write(f"\n  Action: None. All remaining VIFs <= {threshold}.\n")
            f.write("\n")

        f.write("-" * 72 + "\n  DROPPED VARIABLES SUMMARY\n" + "-" * 72 + "\n\n")
        drops = [l for l in iteration_logs if l["dropped"]]
        if drops:
            for i, log in enumerate(drops, 1):
                vs = f"{log['dropped_vif']:.2f}" if np.isfinite(log["dropped_vif"]) else "inf"
                f.write(f"  {i}. {log['dropped']}\n     VIF at removal : {vs}\n")
                if log["top_correlates"] is not None and len(log["top_correlates"]) > 0:
                    pairs = [f"{v} (|r|={c:.3f})" for v, c in log["top_correlates"].items()]
                    f.write("     Top correlates : " + ", ".join(pairs) + "\n")
                f.write("     Verdict        : Redundant; variance largely explained by other predictors.\n\n")
        else:
            f.write("  No variables were dropped by VIF.\n\n")

        f.write("-" * 72 + "\n  RETAINED VARIABLES SUMMARY\n" + "-" * 72 + "\n\n")
        for _, r in final_vif.iterrows():
            f.write(f"  {r['feature']}\n     Final VIF : {r['VIF']:.2f}\n")
            f.write(f"     Verdict   : Retained. VIF <= {threshold}; contributes unique variance\n")
            f.write("                 not captured by other predictors.\n\n")

        f.write("-" * 72 + "\n  COVERAGE DIAGNOSTIC\n" + "-" * 72 + "\n")
        f.write(f"  water   survivors : {water or 'NONE'}\n")
        f.write(f"  thermal survivors : {therm or 'NONE'}\n")
        f.write(f"  coverage rule     : {'FIRES - report the relaxation' if coverage_fail else 'satisfied'}\n\n")
        f.write("=" * 72 + "\n  END OF REPORT\n" + "=" * 72 + "\n")
    print(f"  [SAVED] {report_file}")

    # ── 5. Exports ───────────────────────────────────────────────────────────
    vif_csv = (f"{site_name}_final_vif_{timestamp}.csv" if ADD_TIMESTAMP
               else f"{site_name}_final_vif.csv")
    final_vif.to_csv(outdir / vif_csv, index=False)
    print(f"  [SAVED] {vif_csv}")

    # C4. carry site and w through, or LOSO and weighting are impossible
    keep_meta = [c for c in ["Date", "site", "w"] if c in df.columns]
    export_cols = variables + [target_col]
    retained_df = df[keep_meta + export_cols].copy()
    for c in export_cols:                       # a column coerced inside X was
        retained_df[c] = pd.to_numeric(retained_df[c], errors="coerce")
    retained_df[export_cols] = retained_df[export_cols].replace(MISSING_FLAGS, np.nan) \
                                                       .replace([np.inf, -np.inf], np.nan)

    postcollin_csv = (f"{site_name}_retainedvars_postcollin_{timestamp}.csv" if ADD_TIMESTAMP
                      else f"{site_name}_retainedvars_postcollin.csv")
    retained_df.to_csv(outdir / postcollin_csv, index=False)
    print(f"  [SAVED] {postcollin_csv} ({len(retained_df):,} rows x "
          f"{len(variables)} predictors + target, metadata kept: {keep_meta})")

    print("\n  --- Final Selection ---")
    print(f"  Box 0 removed : {len(dropped_box0)} ({', '.join(dropped_box0) or 'none'})")
    print(f"  Initial       : {n_initial} predictors")
    print(f"  Dropped by VIF: {len(dropped)} ({', '.join(dropped) or 'none'})")
    print(f"  Retained      : {len(variables)} ({', '.join(variables)})")
    return retained_df, final_vif


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    outdir = Path(OUTPUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)
    timestamp = get_timestamp()
    run_dir = outdir / f"{SITE_NAME}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}\n  Pooled-sites Collinearity Checker — {SITE_NAME}")
    print(f"  {get_readable_timestamp()}\n{'='*70}")

    df = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"  [INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    importance_rank = None
    if IMPORTANCE_CSV is not None:
        try:
            imp_df = pd.read_csv(IMPORTANCE_CSV)
            importance_rank = dict(zip(imp_df[IMPORTANCE_PREDICTOR_COL],
                                       imp_df[IMPORTANCE_COL]))
            print(f"  [INFO] Loaded importance ranking: {len(importance_rank)} predictors")
        except Exception as e:
            print(f"  [WARN] Could not load importance CSV: {e}")
            importance_rank = None

    # C2. every predictor that will enter the loop must have a score.
    #     Mirror the loop's own filter exactly, including "numeric or coercible",
    #     so a text column that never enters cannot trigger a false stop.
    if importance_rank:
        will_enter = []
        for c in df.columns:
            if c in EXCLUDE_HEADERS + [TARGET_COL] + DROP_COLUMNS:
                continue
            if pd.to_numeric(df[c], errors="coerce").notna().any():
                will_enter.append(c)
        missing = [c for c in will_enter if c not in importance_rank]
        if missing:
            msg = ("\n  [STOP] no importance score for these predictors:\n"
                   f"         {missing}\n"
                   "         They would be treated as un-droppable, so the elimination would\n"
                   "         strip the variables that DO have scores instead. Point\n"
                   "         IMPORTANCE_CSV at the union run that produced these columns.\n")
            if MISSING_IMPORTANCE_IS_FATAL:
                raise SystemExit(msg)
            print(msg + "         MISSING_IMPORTANCE_IS_FATAL is False, continuing.\n")
        else:
            print(f"  [OK]   importance covers all {len(will_enter)} entering predictors")

    retained_df, final_vif = analyze_collinearity(
        df, target_col=TARGET_COL, threshold=VIF_THRESHOLD, outdir=run_dir,
        site_name=SITE_NAME, timestamp=timestamp, importance_rank=importance_rank)

    print(f"\n{'='*70}\n  Done. All outputs in: {run_dir}\n{'='*70}\n")
