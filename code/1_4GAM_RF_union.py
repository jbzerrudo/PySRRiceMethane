"""
Customized Multivariate GAM Analysis for Rice Methane Data (v9)
================================================================

Changes from v8:
  1. [NEW] Boruta all-relevant feature selection (Kursa & Rudnicki, 2010,
     J. Stat. Softw. 36(11):1-13). Uses Random Forest shadow variables to
     identify all predictors with real signal, including collinear ones.
  2. [NEW] PIMP permutations increased from 50 → 200 (configurable) for
     sharper null distributions and more statistical power.
  3. [NEW] Ensemble feature selection consensus (Saeys et al., 2008,
     Bioinformatics 24(19):2267-2273). Combines PIMP and Boruta into a
     three-tier classification:
       • Confirmed  — both methods agree the variable is important
       • Supported  — one method flags it, the other does not
       • Rejected   — neither method flags it
     The "important" set for downstream VIF/GAM is the union of both
     methods (Confirmed + Supported).
  4. [NEW] Consensus summary report and visualization exported alongside
     existing outputs.
  5. [IMPROVEMENT] Boruta and PIMP can be independently enabled/disabled.

  All v8 bug fixes and improvements are retained.

References:
  Altmann A et al. (2010). Bioinformatics 26(10):1340-1347.  [PIMP]
  Kursa MB, Rudnicki WR (2010). J Stat Softw 36(11):1-13.   [Boruta]
  Saeys Y et al. (2008). Bioinformatics 24(19):2267-2273.    [Ensemble FS]

Author: Jef Zerrudo / Claude Optimised
"""

import os
import sys
import re
import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# =========================
# USER CONFIG
# =========================
INPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data"
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\POOLED\Metadata\Intensive"

# Specify a single CSV file to process (filename only, must be inside INPUT_DIR).
# Set to None to process ALL CSV files in INPUT_DIR.
INPUT_FILE = "POOL_3sites_INTENSIVE.csv"

EXCLUDE_HEADERS = {"Date", "Deltime", "time", "F_CH4_F_orig", "site", "w"}
TARGET_COL = "F_CH4_F"

# Data rules
MIN_COMPLETE_CASES = 30
MIN_NONNA_PER_COL = 10
DPI = 300

# GAM settings
N_SPLINES = 10
LAM = None

# Stability
STANDARDIZE_PREDICTORS = True

# Missing flags found in your dataset
MISSING_FLAGS = [-9999, -999900, -99999]

# Timestamping
ADD_TIMESTAMP = True

# Exclude interaction terms by default
EXCLUDE_INTERACTION_TERMS = False  # Set False to include engineered combos

# Focus on primary predictors
PRIMARY_PREDICTORS_ONLY = False  # Set False to include all variables

PRIMARY_PREDICTOR_LIST = [
    'depth',        # Water depth (main variable)
    'Tair',         # Air temperature
    'Tsoil',        # Soil temperature
    'SR',           # Solar radiation
    'WS',           # Wind speed
    'VPD',          # Vapour pressure deficit
    'RH',           # Relative humidity (%)
    'Patm',         # Atmospheric pressure (kPa)
    'WD',           # Wind direction 
    'AUC',          # area under the curve of water depth over time (if available)
    'rate',         # rate of change of water depth over time
]

# Train/test split
TEST_FRACTION = 0.2
RANDOM_STATE = 42

# PIMP settings (Altmann et al. 2010, Bioinformatics 26(10):1340-1347)
# Permutes response S times to build null importance distribution per feature.
# Features with p < PIMP_THRESHOLD are statistically significant predictors.
PIMP_ENABLED = True
PIMP_S = 200            # Number of response permutations (200+ recommended for power)
PIMP_THRESHOLD = 0.05   # Significance threshold for feature selection

# =========================================================================
# PARALLEL-RUN OVERRIDES (added for the two-at-once .bat; analysis unchanged)
#   * If a config file is passed as the first argument, INPUT_DIR / INPUT_FILE
#     / OUTPUT_DIR are read from it (key=value lines). Otherwise the values
#     hard-coded above are used, so running the script bare behaves exactly
#     as before.
#   * BORUTA_N_JOBS comes from the GAMRF_N_JOBS env var (default -1 = all cores)
#     so the launcher can give each of the two parallel runs half the cores.
# =========================================================================
if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
    _cfg = {}
    with open(sys.argv[1], "r", encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _k, _v = _line.split("=", 1)
            _cfg[_k.strip()] = _v.strip()
    INPUT_DIR  = _cfg.get("INPUT_DIR", INPUT_DIR)
    INPUT_FILE = _cfg.get("INPUT_FILE", INPUT_FILE)
    OUTPUT_DIR = _cfg.get("OUTPUT_DIR", OUTPUT_DIR)
    print(f"[CONFIG] Using overrides from {sys.argv[1]}")
    print(f"[CONFIG] INPUT_FILE = {INPUT_FILE}")
    print(f"[CONFIG] OUTPUT_DIR = {OUTPUT_DIR}")

BORUTA_N_JOBS = int(os.environ.get("GAMRF_N_JOBS", "-1"))  # cores for Boruta RF; -1 = all

# Boruta settings (Kursa & Rudnicki 2010, J Stat Softw 36(11):1-13)
# All-relevant feature selection using Random Forest shadow variables.
# Identifies ALL predictors carrying real signal, including collinear ones.
BORUTA_ENABLED = True
BORUTA_MAX_ITER = 100       # Maximum Boruta iterations
BORUTA_ALPHA = 0.05         # Statistical significance level
BORUTA_N_ESTIMATORS = 500   # RF trees per iteration
BORUTA_MAX_DEPTH = None     # RF max tree depth (None = unrestricted, matches original R package)
BORUTA_RANDOM_STATE = 42
BORUTA_TENTATIVE_AS_IMPORTANT = False  # False = standard Boruta (only Confirmed = important)
                                       # True  = inclusive (Confirmed + Tentative = important)

# Ensemble feature selection (Saeys et al. 2008, Bioinformatics 24(19):2267-2273)
# Consensus rule: union of PIMP + Boruta → "important" set for downstream VIF/GAM
CONSENSUS_ENABLED = True    # Requires both PIMP_ENABLED and BORUTA_ENABLED


# =========================
# UTILITY FUNCTIONS
# =========================

def safe_name(s):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(s))


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def get_readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def zscore_matrix(X):
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0)
    sd_safe = np.where(sd == 0, 1.0, sd)
    Xz = (X - mu) / sd_safe
    return Xz, mu, sd_safe


def apply_zscore(X, mu, sd):
    """Apply pre-computed standardization (train μ/σ) to new data."""
    return (X - mu) / sd


def write_text_summary(path, summary_obj):
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(summary_obj()))
    except Exception:
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(summary_obj))


def clean_missing_flags(df: pd.DataFrame, flags: list) -> pd.DataFrame:
    """Replace common missing data flags with NaN (vectorized)."""
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_clean[num_cols] = df_clean[num_cols].replace(flags, np.nan)
    return df_clean


def filter_interaction_terms(columns: list) -> list:
    """
    Remove obvious engineered interaction term columns.

    Avoid overly broad rules (e.g., dropping any 'x' substring),
    which can incorrectly remove valid columns like Tmax, flux, index.
    """
    interaction_patterns = [
        r".*\*.*",          # explicit multiplication
        r".*_x_.*",         # underscore-x-underscore
        r".*_X_.*",
        r".*_times_.*",
        r".*:\w+.*",        # colon-style (if present)
    ]

    out = []
    for c in columns:
        cs = str(c)
        if any(re.match(p, cs) for p in interaction_patterns):
            continue
        out.append(c)
    return out


def coerce_numeric_df(df: pd.DataFrame) -> pd.DataFrame:
    """FIX v8: errors='coerce' replaces deprecated errors='ignore'."""
    df2 = df.copy()
    for c in df2.columns:
        if df2[c].dtype == "object":
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2


def select_predictors(df: pd.DataFrame) -> list:
    """Choose usable numeric predictors based on your selection rules."""
    numeric_cols = [c for c in df.columns if c not in EXCLUDE_HEADERS and c != TARGET_COL]

    df_num = coerce_numeric_df(df)
    numeric_cols = [c for c in numeric_cols if pd.api.types.is_numeric_dtype(df_num[c])]

    if EXCLUDE_INTERACTION_TERMS:
        numeric_cols = filter_interaction_terms(numeric_cols)

    if PRIMARY_PREDICTORS_ONLY:
        allowed = set([c.lower() for c in PRIMARY_PREDICTOR_LIST])
        numeric_cols = [c for c in numeric_cols if c.lower() in allowed]

    return numeric_cols


def drop_excluded_headers(df: pd.DataFrame, exclude_headers: set) -> pd.DataFrame:
    """Drop non-predictor/meta columns safely."""
    cols_to_drop = [c for c in exclude_headers if c in df.columns]
    if not cols_to_drop:
        return df
    return df.drop(columns=cols_to_drop, errors="ignore")


def numeric_columns(df: pd.DataFrame) -> list:
    """Return numeric columns only."""
    return df.select_dtypes(include=[np.number]).columns.tolist()


def permutation_importance_gam(gam, X_val, y_val, predictors, n_repeats=5, random_state=42):
    rng = np.random.default_rng(random_state)
    base_pred = gam.predict(X_val)
    base_rmse = np.sqrt(mean_squared_error(y_val, base_pred))

    importances = []
    X_val = X_val.copy()

    for j, name in enumerate(predictors):
        rmses = []
        for _ in range(n_repeats):
            Xp = X_val.copy()
            Xp[:, j] = rng.permutation(Xp[:, j])
            pred = gam.predict(Xp)
            rmse = np.sqrt(mean_squared_error(y_val, pred))
            rmses.append(rmse)

        imp = np.mean(rmses) - base_rmse
        importances.append((name, imp))

    imp_df = pd.DataFrame(importances, columns=["predictor", "perm_RMSE_increase"])
    imp_df = imp_df.sort_values("perm_RMSE_increase", ascending=False).reset_index(drop=True)
    return imp_df


def pimp_test_gam(X_train, y_train, X_val, y_val, predictors, observed_imp_df,
                  n_permutations=50, n_repeats=5, random_state=42):
    """
    PIMP: Permutation Importance P-values (Altmann et al., 2010).

    Builds a null distribution of ΔRMSE for each feature by permuting
    the response variable S times, refitting the GAM each time, and
    computing permutation importance under the null (no signal).

    Parameters:
        X_train, y_train : training data (standardized)
        X_val, y_val     : validation data (standardized)
        predictors       : list of predictor names
        observed_imp_df  : DataFrame with columns [predictor, perm_RMSE_increase]
        n_permutations   : number of response permutations (S)
        n_repeats        : shuffles per feature per null model
        random_state     : seed

    Returns:
        DataFrame with columns:
            predictor, observed_importance, pimp_p_value, pimp_significant

    Reference:
        Altmann A, Toloşi L, Sander O, Lengauer T (2010).
        Permutation importance: a corrected feature importance measure.
        Bioinformatics, 26(10), 1340–1347.
    """
    from pygam import LinearGAM, s

    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]

    # Store null importances: shape (S, n_features)
    null_importances = np.zeros((n_permutations, n_features))

    print(f"  [PIMP] Running {n_permutations} null permutations (this may take a few minutes)...")

    for s_idx in range(n_permutations):
        if (s_idx + 1) % 10 == 0 or s_idx == 0:
            print(f"  [PIMP] Permutation {s_idx + 1}/{n_permutations}...")

        # 1. Permute response (breaks all feature-target relationships)
        y_train_perm = rng.permutation(y_train)
        y_val_perm = rng.permutation(y_val)

        # 2. Refit GAM on permuted response
        terms = None
        for i in range(n_features):
            term = s(i, n_splines=N_SPLINES)
            terms = term if terms is None else terms + term

        try:
            null_gam = (
                LinearGAM(terms).fit(X_train, y_train_perm)
                if LAM is None
                else LinearGAM(terms, lam=LAM).fit(X_train, y_train_perm)
            )
        except Exception:
            # If fit fails on noise, fill with zeros (no importance)
            null_importances[s_idx, :] = 0.0
            continue

        # 3. Compute permutation importance under null
        base_pred = null_gam.predict(X_val)
        base_rmse = np.sqrt(mean_squared_error(y_val_perm, base_pred))

        for j in range(n_features):
            rmses = []
            for _ in range(n_repeats):
                Xp = X_val.copy()
                Xp[:, j] = rng.permutation(Xp[:, j])
                pred = null_gam.predict(Xp)
                rmse = np.sqrt(mean_squared_error(y_val_perm, pred))
                rmses.append(rmse)
            null_importances[s_idx, j] = np.mean(rmses) - base_rmse

    # 4. Compute p-values per feature
    # Map observed importances to array aligned with predictor order
    obs_map = dict(zip(observed_imp_df["predictor"], observed_imp_df["perm_RMSE_increase"]))

    results = []
    for j, name in enumerate(predictors):
        obs_imp = obs_map.get(name, 0.0)
        null_dist = null_importances[:, j]

        # p-value: proportion of null importances >= observed (+ correction)
        # Using (count + 1) / (S + 1) to avoid p=0 (Phipson & Smyth, 2010)
        n_geq = np.sum(null_dist >= obs_imp)
        p_value = (n_geq + 1) / (n_permutations + 1)

        results.append({
            "predictor": name,
            "observed_importance": obs_imp,
            "null_mean": np.mean(null_dist),
            "null_std": np.std(null_dist),
            "pimp_p_value": p_value,
            "pimp_significant": p_value < PIMP_THRESHOLD,
        })

    pimp_df = pd.DataFrame(results)
    pimp_df = pimp_df.sort_values("observed_importance", ascending=False).reset_index(drop=True)

    n_sig = pimp_df["pimp_significant"].sum()
    print(f"  [PIMP] Done. {n_sig}/{len(predictors)} predictors significant at p < {PIMP_THRESHOLD}")

    return pimp_df, null_importances


def drop_one_term_importance(X_train, y_train, predictors, standardized_already=True):
    """
    FIX v8: Accepts pre-split, pre-standardized training data directly.
    Ensures ΔAIC is computed on the same data as the main GAM.
    """
    from pygam import LinearGAM, s

    X = X_train
    y = y_train

    # Full model
    terms = None
    for i in range(X.shape[1]):
        term = s(i, n_splines=N_SPLINES)
        terms = term if terms is None else terms + term

    full = LinearGAM(terms).fit(X, y) if LAM is None else LinearGAM(terms, lam=LAM).fit(X, y)
    full_aic = full.statistics_.get("AIC", np.nan)
    full_gcv = full.statistics_.get("GCV", np.nan)

    rows = []

    for k, name in enumerate(predictors):
        # Remove column k
        Xm = np.delete(X, k, axis=1)

        terms_m = None
        for i in range(Xm.shape[1]):
            term = s(i, n_splines=N_SPLINES)
            terms_m = term if terms_m is None else terms_m + term

        m = LinearGAM(terms_m).fit(Xm, y) if LAM is None else LinearGAM(terms_m, lam=LAM).fit(Xm, y)

        aic_m = m.statistics_.get("AIC", np.nan)
        gcv_m = m.statistics_.get("GCV", np.nan)

        rows.append({
            "predictor": name,
            "delta_AIC": aic_m - full_aic,
            "delta_GCV": gcv_m - full_gcv
        })

    out = pd.DataFrame(rows).sort_values("delta_AIC", ascending=False).reset_index(drop=True)
    return out


# =========================
# ROBUST CI HELPER
# =========================

def get_pdep_and_ci(gam, term, XX, width=0.95):
    """
    Robustly extract partial dependence and 95% CI across pyGAM versions.

    Returns:
        pdep, lower, upper
        where lower/upper are 1D arrays or None.
    """
    pdep = gam.partial_dependence(term=term, X=XX)

    ci_obj = gam.partial_dependence(term=term, X=XX, width=width)

    lower = upper = None

    # Case 1: returns (pdep, ci_matrix)
    if isinstance(ci_obj, (list, tuple)) and len(ci_obj) == 2:
        maybe_ci = ci_obj[1]
        if isinstance(maybe_ci, np.ndarray) and maybe_ci.ndim == 2:
            if maybe_ci.shape[1] == 2:
                lower, upper = maybe_ci[:, 0], maybe_ci[:, 1]
            elif maybe_ci.shape[0] == 2:
                lower, upper = maybe_ci[0], maybe_ci[1]

    # Case 2: returns ci_matrix directly
    elif isinstance(ci_obj, np.ndarray) and ci_obj.ndim == 2:
        if ci_obj.shape[1] == 2:
            lower, upper = ci_obj[:, 0], ci_obj[:, 1]
        elif ci_obj.shape[0] == 2:
            lower, upper = ci_obj[0], ci_obj[1]

    return pdep, lower, upper


# =========================
# DATA QUALITY REPORTING
# =========================

def create_data_quality_report(df: pd.DataFrame, predictors: list, target: str,
                               outdir: Path, timestamp: str):
    report = []
    report.append(f"Data Quality Report for {target}")
    report.append(f"Generated at: {get_readable_timestamp()}")
    report.append("=" * 70)

    report.append(f"Rows total: {len(df):,}")
    report.append(f"Target non-NA: {df[target].notna().sum():,}")

    report.append("\nPredictor completeness:")
    for c in predictors:
        nn = df[c].notna().sum()
        report.append(f"  {c:20s}: {nn:7d} ({nn/len(df)*100:5.1f}%)")

    report.append("\nSimple outlier counts (IQR rule):")
    for c in predictors:
        s_col = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(s_col) < 10:
            continue
        q1, q3 = np.percentile(s_col, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = ((s_col < lo) | (s_col > hi)).sum()
        report.append(f"  {c:20s}: {outliers:7d}")

    fname = f"data_quality_report_{timestamp}.txt" if ADD_TIMESTAMP else "data_quality_report.txt"
    with open(outdir / fname, "w", encoding="utf-8") as f:
        f.write("\n".join(report))


# =========================
# PLOTTING STYLE
# =========================

def setup_plot_style():
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except Exception:
        pass

    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'legend.fontsize': 11,
        'figure.titlesize': 16,
        'lines.linewidth': 2.5,
        'axes.linewidth': 1.5,
        'grid.alpha': 0.3,
        'grid.linewidth': 0.8,
    })


def inverse_asinh(y):
    return np.sinh(y)


# =========================
# PARTIAL PLOTS
# =========================

def plot_multivariate_partial_effects_improved(
    gam,
    predictors,
    X_std,
    X_raw,
    y,
    mu,
    sd,
    outdir: Path,
    timestamp: str
):
    """
    Improved partial dependence plots with:
    - robust CI extraction (fixes 'y2' not 1D)
    - centered partial effects (so 0-line is meaningful)
    - aligned rug plots (uses X_std)
    - optional secondary axis in original units
    """
    outdir.mkdir(parents=True, exist_ok=True)
    setup_plot_style()

    for i, name in enumerate(predictors):
        try:
            XX = gam.generate_X_grid(term=i)
            xgrid = XX[:, i]

            # Robust CI extraction
            pdep, lower, upper = get_pdep_and_ci(gam, i, XX, width=0.95)

            is_asinh = 'asinh' in name.lower()

            # Center effects so 0 means "average partial effect"
            pdep_mean = np.nanmean(pdep)
            pdep_plot = pdep - pdep_mean

            lower_plot = upper_plot = None
            if lower is not None and upper is not None:
                lower_plot = lower - pdep_mean
                upper_plot = upper - pdep_mean

            # Figure layout: main + rug
            fig = plt.figure(figsize=(12, 8))
            ax_main = plt.subplot2grid((6, 1), (0, 0), rowspan=5)

            # CI band
            if lower_plot is not None and upper_plot is not None:
                ax_main.fill_between(
                    xgrid, lower_plot, upper_plot, alpha=0.25,
                    color='royalblue', label='95% CI'
                )

            # Main curve
            ax_main.plot(
                xgrid, pdep_plot, linewidth=3, color='darkblue',
                label='Partial effect (centered)', zorder=10
            )

            # Reference line at 0 (meaningful because centered)
            ax_main.axhline(
                y=0, color='red', linestyle='--', linewidth=2,
                alpha=0.6, label='Mean partial effect', zorder=5
            )

            # Water-depth surface reference for asinh terms
            if is_asinh and mu is not None and sd is not None:
                try:
                    zero_depth_std = (0 - mu[i]) / sd[i]
                    if ax_main.get_xlim()[0] <= zero_depth_std <= ax_main.get_xlim()[1]:
                        ax_main.axvline(
                            x=zero_depth_std, color='green', linestyle=':',
                            linewidth=2.5, alpha=0.7, label='Water at soil surface',
                            zorder=5
                        )
                except Exception as e:
                    print(f"    [WARNING] asinh reference line for {name}: {e}")

            ax_main.set_ylabel(
                f'Centered partial effect on {TARGET_COL}',
                fontweight='bold', fontsize=15
            )
            ax_main.set_title(
                f'Partial Effect: {name}\n({TARGET_COL} GAM Analysis)',
                fontweight='bold', fontsize=16, pad=20
            )

            ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=1)
            ax_main.legend(
                loc='best', framealpha=0.95, fontsize=12,
                edgecolor='black', fancybox=True, shadow=True
            )

            # Term statistics annotation
            try:
                st = gam.statistics_
                edof_per_term = st.get('edof_per_term', None)
                edf = edof_per_term[i] if (edof_per_term is not None and i < len(edof_per_term)) else np.nan
                pval = st.get('p_values', [np.nan] * len(predictors))[i]
                aic = st.get('AIC', np.nan)

                textstr = '\n'.join([
                    f'EDF: {edf:.2f}' if np.isfinite(edf) else 'EDF: n/a',
                    f'p-value: {pval:.3g}' if np.isfinite(pval) else 'p-value: n/a',
                    f'AIC: {aic:.1f}' if np.isfinite(aic) else 'AIC: n/a',
                ])

                props = dict(
                    boxstyle='round,pad=0.8', facecolor='wheat',
                    alpha=0.8, edgecolor='black', linewidth=2
                )
                ax_main.text(
                    0.02, 0.98, textstr, transform=ax_main.transAxes,
                    fontsize=11, verticalalignment='top', bbox=props,
                    fontfamily='monospace'
                )
            except Exception as e:
                print(f"    [WARNING] Stats annotation for {name}: {e}")

            # Secondary axis with original scale
            if mu is not None and sd is not None:
                try:
                    ax_top = ax_main.twiny()
                    ax_top.set_xlim(ax_main.get_xlim())

                    n_ticks = 5
                    tick_positions = np.linspace(xgrid.min(), xgrid.max(), n_ticks)
                    original_scale = tick_positions * sd[i] + mu[i]

                    if is_asinh:
                        water_depths = inverse_asinh(original_scale)
                        tick_labels = [f'{val:.1f}' for val in water_depths]
                        ax_top.set_xlabel(
                            'Water depth (cm, via sinh transform)',
                            fontsize=13, color='darkgreen', fontweight='bold'
                        )
                        ax_top.tick_params(axis='x', labelcolor='darkgreen', labelsize=11)
                    else:
                        tick_labels = [f'{val:.2f}' for val in original_scale]
                        ax_top.set_xlabel(
                            f'{name} (original scale)',
                            fontsize=13, fontweight='bold'
                        )

                    ax_top.set_xticks(tick_positions)
                    ax_top.set_xticklabels(tick_labels)
                except Exception as e:
                    print(f"    [WARNING] Secondary axis for {name}: {e}")

            # Rug plot (aligned to standardized axis)
            ax_rug = plt.subplot2grid((6, 1), (5, 0), rowspan=1, sharex=ax_main)

            try:
                x_data = X_std[:, i]  # correct scale alignment
                ax_rug.plot(
                    x_data, np.ones_like(x_data), '|',
                    color='black', alpha=0.5, markersize=10, markeredgewidth=1.5
                )
            except Exception as e:
                print(f"    [WARNING] Rug plot for {name}: {e}")

            ax_rug.set_ylim([0.5, 1.5])
            xlab = f'{name} (standardized)' if STANDARDIZE_PREDICTORS else name
            ax_rug.set_xlabel(xlab, fontweight='bold', fontsize=14)
            ax_rug.set_yticks([])
            ax_rug.set_ylabel('Data', fontsize=11, fontweight='bold')
            ax_rug.grid(True, alpha=0.3, axis='x')

            plt.tight_layout()

            filename = f"partial_{safe_name(name)}_{timestamp}.png" if ADD_TIMESTAMP else f"partial_{safe_name(name)}.png"
            fig.savefig(outdir / filename, dpi=DPI, bbox_inches='tight')
            plt.close(fig)

        except Exception as e:
            print(f"    [WARNING] Could not plot {name}: {e}")
            continue


# =========================
# SUMMARY PLOT
# =========================

def create_summary_plot(gam, predictors, outdir: Path, timestamp: str):
    try:
        n_predictors = len(predictors)
        if n_predictors == 0:
            return

        ncols = 2
        nrows = int(np.ceil(n_predictors / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(4, nrows * 3)))
        axes = np.array(axes).reshape(-1)

        for i, name in enumerate(predictors):
            ax = axes[i]
            try:
                XX = gam.generate_X_grid(term=i)
                xgrid = XX[:, i]
                pdep = gam.partial_dependence(term=i, X=XX)
                ax.plot(xgrid, pdep)
                ax.set_title(name)
                ax.grid(True, alpha=0.3)
            except Exception as e:
                print(f"    [WARNING] Summary plot for {name}: {e}")
                ax.set_visible(False)

        for j in range(n_predictors, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        fname = f"summary_partials_{timestamp}.png" if ADD_TIMESTAMP else "summary_partials.png"
        fig.savefig(outdir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Summary plot failed: {e}")


# =========================
# GAM EQUATION PRINTER
# =========================

def print_gam_equation(gam, predictors, target, outdir: Path, timestamp: str,
                       sig_threshold=0.05):
    """
    Print and save the fitted GAM equation with term-level diagnostics.

    Outputs:
    - Full symbolic equation (all terms)
    - Parsimonious equation (significant terms only, p < sig_threshold)
    - Term table with EDF, p-value, and significance codes
    - Intercept value

    FIX v8: Uses edof_per_term directly instead of slicing edof_per_coef.
    FIX v8: Warns when predictor count > 15 (p-values unreliable).
    """
    st = gam.statistics_
    intercept = gam.coef_[-1] if hasattr(gam, 'coef_') else np.nan
    pvals = st.get('p_values', [np.nan] * (len(predictors) + 1))

    # FIX v8: Use edof_per_term directly (robust across pyGAM versions)
    edof_per_term = st.get('edof_per_term', None)
    if edof_per_term is not None and len(edof_per_term) >= len(predictors):
        edfs = [float(edof_per_term[i]) for i in range(len(predictors))]
    else:
        # Fallback: try summing edof_per_coef in chunks
        edof_per_coef = st.get('edof_per_coef', None)
        edfs = []
        if edof_per_coef is not None:
            for i in range(len(predictors)):
                start = i * N_SPLINES
                end = start + N_SPLINES
                if end <= len(edof_per_coef):
                    edfs.append(float(np.sum(edof_per_coef[start:end])))
                else:
                    edfs.append(np.nan)
        else:
            edfs = [np.nan] * len(predictors)

    lines = []
    lines.append("=" * 80)
    lines.append(f"  GAM EQUATION REPORT FOR: {target}")
    lines.append(f"  Generated at: {get_readable_timestamp()}")
    lines.append("=" * 80)

    # --- Full equation ---
    terms_str = " + ".join([f"s({p})" for p in predictors])
    full_eq = f"  {target} = {intercept:.4f} + {terms_str}"
    lines.append("")
    lines.append("  FULL MODEL EQUATION (all terms):")
    lines.append(f"  {full_eq}")
    lines.append(f"  Number of terms: {len(predictors)}")

    # --- Term table ---
    lines.append("")
    lines.append("  TERM DETAILS:")
    lines.append(f"  {'Term':<25s} {'EDF':>8s} {'p-value':>12s} {'Sig':>6s}")
    lines.append("  " + "-" * 55)

    sig_predictors = []
    for i, name in enumerate(predictors):
        edf_i = edfs[i] if i < len(edfs) else np.nan
        p_i = pvals[i] if i < len(pvals) else np.nan

        # Significance codes
        if np.isfinite(p_i):
            if p_i < 0.001:
                sig_code = "***"
            elif p_i < 0.01:
                sig_code = "**"
            elif p_i < 0.05:
                sig_code = "*"
            elif p_i < 0.1:
                sig_code = "."
            else:
                sig_code = ""
        else:
            sig_code = "n/a"

        edf_str = f"{edf_i:.2f}" if np.isfinite(edf_i) else "n/a"
        p_str = f"{p_i:.3g}" if np.isfinite(p_i) else "n/a"

        lines.append(f"  s({name:<22s}) {edf_str:>8s} {p_str:>12s} {sig_code:>6s}")

        if np.isfinite(p_i) and p_i < sig_threshold:
            sig_predictors.append(name)

    lines.append("  " + "-" * 55)
    lines.append("  Sig. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")

    # --- Parsimonious equation (significant terms only) ---
    lines.append("")
    lines.append(f"  PARSIMONIOUS EQUATION (p < {sig_threshold}):")

    # FIX v8: Warn when p-values are unreliable
    if len(predictors) > 15:
        lines.append(f"  ⚠ WARNING: {len(predictors)} predictors — p-values are unreliable")
        lines.append(f"    with this many correlated terms. Run collinearity screening")
        lines.append(f"    first, then interpret significance from the reduced model.")

    if sig_predictors:
        sig_str = " + ".join([f"s({p})" for p in sig_predictors])
        parsim_eq = f"{target} = {intercept:.4f} + {sig_str}"
        lines.append(f"  {parsim_eq}")
        lines.append(f"  Significant terms: {len(sig_predictors)} / {len(predictors)}")
    else:
        lines.append("  No individually significant terms at this threshold.")

    # --- Model-level stats ---
    lines.append("")
    lines.append("  MODEL STATISTICS:")
    for key in ['AIC', 'AICc', 'GCV', 'scale', 'loglikelihood', 'deviance']:
        val = st.get(key, None)
        if val is not None and isinstance(val, (int, float, np.floating)):
            lines.append(f"    {key:<25s}: {float(val):.4f}")

    # pseudo_r2 is an OrderedDict in pyGAM
    pr2 = st.get('pseudo_r2', None)
    if pr2 is not None and hasattr(pr2, 'items'):
        for r2_name, r2_val in pr2.items():
            label = f"R² ({r2_name})"
            lines.append(f"    {label:<25s}: {float(r2_val):.4f}")
    elif pr2 is not None:
        lines.append(f"    {'Pseudo R²':<25s}: {float(pr2):.4f}")

    edof_total = st.get('edof', None)
    if edof_total is not None:
        lines.append(f"    {'Effective DoF':<25s}: {float(edof_total):.2f}")

    lines.append("=" * 80)

    # Print to console
    report = "\n".join(lines)
    print(report)

    # Save to file
    eq_file = f"gam_equation_{timestamp}.txt" if ADD_TIMESTAMP else "gam_equation.txt"
    with open(outdir / eq_file, "w", encoding="utf-8") as f:
        f.write(report)

    return sig_predictors


# =========================
# PIMP VISUALIZATION
# =========================

def _plot_pimp_results(pimp_df, null_importances, predictors, outdir, timestamp):
    """
    Plot PIMP results: observed importance vs null distributions.
    Two plots:
      1. Bar chart with significance markers
      2. Null distribution boxplots with observed overlaid
    """
    try:
        # --- Plot 1: Bar chart with p-values ---
        fig, ax = plt.subplots(figsize=(12, max(6, len(pimp_df) * 0.35)))

        colors = ['#2E86AB' if sig else '#CCCCCC'
                  for sig in pimp_df["pimp_significant"]]

        y_pos = range(len(pimp_df))
        ax.barh(y_pos, pimp_df["observed_importance"], color=colors, edgecolor="k", lw=0.5)

        for i, (_, row) in enumerate(pimp_df.iterrows()):
            sig_marker = "***" if row["pimp_p_value"] < 0.001 else \
                         "**" if row["pimp_p_value"] < 0.01 else \
                         "*" if row["pimp_p_value"] < 0.05 else ""
            label = f"  p={row['pimp_p_value']:.3f} {sig_marker}"
            ax.text(row["observed_importance"], i, label, va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(pimp_df["predictor"], fontsize=9)
        ax.set_xlabel("Permutation Importance (ΔRMSE)", fontsize=11)
        ax.set_title(
            f"PIMP Feature Selection (Altmann et al., 2010)\n"
            f"Blue = significant (p < {PIMP_THRESHOLD}), "
            f"Grey = not significant  |  S = {PIMP_S} permutations",
            fontsize=12
        )
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()

        fname = f"pimp_significance_{timestamp}.png" if ADD_TIMESTAMP else "pimp_significance.png"
        fig.savefig(outdir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        # --- Plot 2: Null distributions vs observed (top 20) ---
        n_show = min(20, len(pimp_df))
        top_preds = pimp_df.head(n_show)

        fig, ax = plt.subplots(figsize=(14, max(6, n_show * 0.4)))

        # Get null distributions for top predictors in order
        pred_to_idx = {name: i for i, name in enumerate(predictors)}

        null_data = []
        labels = []
        observed_vals = []
        for _, row in top_preds.iterrows():
            j = pred_to_idx[row["predictor"]]
            null_data.append(null_importances[:, j])
            labels.append(row["predictor"])
            observed_vals.append(row["observed_importance"])

        bp = ax.boxplot(null_data, vert=False, labels=labels,
                        patch_artist=True, widths=0.6,
                        boxprops=dict(facecolor="#E8E8E8", edgecolor="grey"),
                        medianprops=dict(color="grey"),
                        whiskerprops=dict(color="grey"),
                        capprops=dict(color="grey"),
                        flierprops=dict(marker=".", markersize=3, color="grey", alpha=0.5))

        for i, obs in enumerate(observed_vals):
            color = "red" if top_preds.iloc[i]["pimp_significant"] else "grey"
            marker = "D" if top_preds.iloc[i]["pimp_significant"] else "o"
            ax.scatter(obs, i + 1, color=color, marker=marker, s=80, zorder=5,
                       edgecolors="k", linewidths=0.8)

        ax.set_xlabel("ΔRMSE", fontsize=11)
        ax.set_title(
            f"PIMP Null Distributions vs Observed Importance\n"
            f"Red ◆ = significant (p < {PIMP_THRESHOLD}), Grey ○ = not significant",
            fontsize=12
        )
        ax.grid(True, alpha=0.2, axis="x")
        ax.axvline(0, color="k", ls="--", lw=0.5, alpha=0.3)
        plt.tight_layout()

        fname2 = f"pimp_null_distributions_{timestamp}.png" if ADD_TIMESTAMP else "pimp_null_distributions.png"
        fig.savefig(outdir / fname2, dpi=DPI, bbox_inches="tight")
        plt.close(fig)

    except Exception as e:
        print(f"  [WARNING] PIMP plot failed: {e}")


# =========================
# BORUTA FEATURE SELECTION
# =========================

def boruta_feature_selection(X_train, y_train, predictors, max_iter=100,
                             alpha=0.05, n_estimators=500, max_depth=None,
                             tentative_as_important=False, random_state=42):
    """
    Boruta all-relevant feature selection (Kursa & Rudnicki, 2010).

    Algorithm:
      1. Create "shadow" copies of all features by permuting each column
      2. Train Random Forest on [real features | shadow features]
      3. Record feature importances; compare each real feature against
         the maximum shadow importance ("MZSA" — max z-score among shadows)
      4. Binomial test: if a real feature beats MZSA in significantly
         more iterations than expected by chance, it is "Confirmed"
      5. Repeat until all features are confirmed/rejected or max_iter reached

    Parameters:
        X_train      : ndarray (n, p), training predictors
        y_train      : ndarray (n,), training target
        predictors   : list of predictor names
        max_iter     : maximum iterations
        alpha        : significance level for two-sided binomial test
        n_estimators : RF trees per iteration
        max_depth    : RF max tree depth (None = unrestricted, per original R package)
        tentative_as_important : if False (default/standard), only Confirmed = important;
                                 if True, Confirmed + Tentative = important
        random_state : seed

    Returns:
        boruta_df : DataFrame with columns:
            predictor, n_hits, n_iterations, hit_rate, boruta_decision,
            boruta_important, mean_importance, mean_shadow_max

    Reference:
        Kursa MB, Rudnicki WR (2010). Feature Selection with the Boruta Package.
        Journal of Statistical Software, 36(11), 1-13.
    """
    from scipy.stats import binomtest

    rng = np.random.default_rng(random_state)
    n_features = X_train.shape[1]

    # Track hits: how many iterations each feature beats max shadow importance
    hits = np.zeros(n_features, dtype=int)
    n_iter_run = 0

    # Track importance values for reporting
    real_importances_sum = np.zeros(n_features)
    shadow_max_sum = 0.0

    # Status: 0 = undecided, 1 = confirmed, -1 = rejected
    status = np.zeros(n_features, dtype=int)

    print(f"  [BORUTA] Running Boruta feature selection (max {max_iter} iterations)...")

    for iteration in range(1, max_iter + 1):
        if (iteration % 20 == 0) or iteration == 1:
            print(f"  [BORUTA] Iteration {iteration}/{max_iter}...")

        # Only evaluate undecided features
        active_mask = (status == 0)
        if not active_mask.any():
            print(f"  [BORUTA] All features decided at iteration {iteration - 1}.")
            break

        # 1. Create shadow features (permute each column independently)
        X_shadow = np.column_stack([
            rng.permutation(X_train[:, j]) for j in range(n_features)
        ])

        # 2. Combine real + shadow
        X_combined = np.column_stack([X_train, X_shadow])

        # 3. Fit Random Forest
        rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            n_jobs=BORUTA_N_JOBS,
            random_state=random_state + iteration
        )
        rf.fit(X_combined, y_train)

        # 4. Extract importances
        importances = rf.feature_importances_
        real_imp = importances[:n_features]
        shadow_imp = importances[n_features:]

        # Maximum shadow importance (MZSA threshold)
        shadow_max = shadow_imp.max()

        real_importances_sum += real_imp
        shadow_max_sum += shadow_max

        # 5. Record hits for undecided features
        for j in range(n_features):
            if status[j] == 0 and real_imp[j] > shadow_max:
                hits[j] += 1

        n_iter_run = iteration

        # 6. Test for early confirmation/rejection (every 10 iterations)
        if iteration >= 20 and iteration % 10 == 0:
            for j in range(n_features):
                if status[j] != 0:
                    continue

                # Two-sided binomial test: H0: hit_rate = 0.5
                # (under null, real feature should beat shadow max ~50% of time)
                result = binomtest(hits[j], iteration, 0.5, alternative='two-sided')

                if result.pvalue < alpha:
                    if hits[j] > iteration / 2:
                        status[j] = 1   # Confirmed
                    else:
                        status[j] = -1  # Rejected

    # Final decisions for still-undecided features
    for j in range(n_features):
        if status[j] == 0:
            # Tentative: did not reach significance in either direction
            # Apply final test
            result = binomtest(hits[j], n_iter_run, 0.5, alternative='two-sided')
            if result.pvalue < alpha and hits[j] > n_iter_run / 2:
                status[j] = 1   # Confirmed
            elif result.pvalue < alpha and hits[j] < n_iter_run / 2:
                status[j] = -1  # Rejected
            # else: remains 0 = Tentative

    # Build results DataFrame
    decisions = []
    for j in range(n_features):
        if status[j] == 1:
            dec = "Confirmed"
        elif status[j] == -1:
            dec = "Rejected"
        else:
            dec = "Tentative"
        decisions.append(dec)

    boruta_df = pd.DataFrame({
        "predictor": predictors,
        "n_hits": hits,
        "n_iterations": n_iter_run,
        "hit_rate": hits / max(n_iter_run, 1),
        "boruta_decision": decisions,
        "boruta_important": [s != -1 for s in status] if tentative_as_important
                           else [s == 1 for s in status],  # Standard: only Confirmed
        "mean_importance": real_importances_sum / max(n_iter_run, 1),
        "mean_shadow_max": shadow_max_sum / max(n_iter_run, 1),
    })

    boruta_df = boruta_df.sort_values("mean_importance", ascending=False).reset_index(drop=True)

    n_confirmed = sum(1 for d in decisions if d == "Confirmed")
    n_tentative = sum(1 for d in decisions if d == "Tentative")
    n_rejected = sum(1 for d in decisions if d == "Rejected")
    print(f"  [BORUTA] Done after {n_iter_run} iterations: "
          f"{n_confirmed} Confirmed, {n_tentative} Tentative, {n_rejected} Rejected")
    if tentative_as_important:
        print(f"  [BORUTA] Tentative treated as important (inclusive mode)")
    else:
        print(f"  [BORUTA] Only Confirmed treated as important (standard Boruta)")

    return boruta_df


def _plot_boruta_results(boruta_df, outdir, timestamp):
    """Bar chart of Boruta mean importance coloured by decision."""
    try:
        color_map = {"Confirmed": "#2E86AB", "Tentative": "#F6AE2D", "Rejected": "#CCCCCC"}
        colors = [color_map.get(d, "#CCCCCC") for d in boruta_df["boruta_decision"]]

        fig, ax = plt.subplots(figsize=(12, max(6, len(boruta_df) * 0.35)))
        y_pos = range(len(boruta_df))
        ax.barh(y_pos, boruta_df["mean_importance"], color=colors, edgecolor="k", lw=0.5)

        # Shadow max reference line
        shadow_max = boruta_df["mean_shadow_max"].iloc[0]
        ax.axvline(shadow_max, color="red", ls="--", lw=1.5, alpha=0.7,
                   label=f"Mean shadow max = {shadow_max:.4f}")

        for i, (_, row) in enumerate(boruta_df.iterrows()):
            label = f"  {row['boruta_decision']} ({row['hit_rate']:.0%})"
            ax.text(row["mean_importance"], i, label, va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(boruta_df["predictor"], fontsize=9)
        ax.set_xlabel("Mean RF Feature Importance", fontsize=11)
        ax.set_title(
            f"Boruta Feature Selection (Kursa & Rudnicki, 2010)\n"
            f"Blue = Confirmed, Yellow = Tentative, Grey = Rejected  |  "
            f"{boruta_df['n_iterations'].iloc[0]} iterations",
            fontsize=12
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()

        fname = f"boruta_selection_{timestamp}.png" if ADD_TIMESTAMP else "boruta_selection.png"
        fig.savefig(outdir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Boruta plot failed: {e}")


# =========================================
# ENSEMBLE FEATURE SELECTION (CONSENSUS)
# =========================================

def consensus_feature_selection(pimp_df, boruta_df, predictors, outdir, timestamp):
    """
    Ensemble feature selection combining PIMP and Boruta.

    Classification:
        Confirmed  — both methods agree the variable is important
        Supported  — one method flags it, the other does not
        Rejected   — neither method flags it

    The "important" set for downstream analysis is the union:
    Confirmed + Supported (i.e., flagged by at least one method).

    Reference:
        Saeys Y, Inza I, Larrañaga P (2008). A review of feature selection
        techniques in bioinformatics. Bioinformatics, 24(19), 2267-2273.
    """
    # Build lookup dicts
    pimp_map = dict(zip(pimp_df["predictor"], pimp_df["pimp_significant"]))
    boruta_map = dict(zip(boruta_df["predictor"], boruta_df["boruta_important"]))
    boruta_decision_map = dict(zip(boruta_df["predictor"], boruta_df["boruta_decision"]))
    pimp_pval_map = dict(zip(pimp_df["predictor"], pimp_df["pimp_p_value"]))
    boruta_hitrate_map = dict(zip(boruta_df["predictor"], boruta_df["hit_rate"]))
    perm_imp_map = dict(zip(pimp_df["predictor"], pimp_df["observed_importance"]))

    rows = []
    for name in predictors:
        pimp_sig = pimp_map.get(name, False)
        boruta_imp = boruta_map.get(name, False)

        if pimp_sig and boruta_imp:
            consensus = "Confirmed"
        elif pimp_sig or boruta_imp:
            consensus = "Supported"
        else:
            consensus = "Rejected"

        rows.append({
            "predictor": name,
            "pimp_significant": pimp_sig,
            "pimp_p_value": pimp_pval_map.get(name, np.nan),
            "boruta_important": boruta_imp,
            "boruta_decision": boruta_decision_map.get(name, "n/a"),
            "boruta_hit_rate": boruta_hitrate_map.get(name, np.nan),
            "consensus": consensus,
            "consensus_important": consensus != "Rejected",
            "perm_importance": perm_imp_map.get(name, 0.0),
        })

    consensus_df = pd.DataFrame(rows)
    consensus_df = consensus_df.sort_values("perm_importance", ascending=False).reset_index(drop=True)

    # Summary
    n_confirmed = (consensus_df["consensus"] == "Confirmed").sum()
    n_supported = (consensus_df["consensus"] == "Supported").sum()
    n_rejected = (consensus_df["consensus"] == "Rejected").sum()
    n_important = consensus_df["consensus_important"].sum()

    print(f"\n  [CONSENSUS] Ensemble Feature Selection (PIMP ∪ Boruta):")
    print(f"    Confirmed (both methods) : {n_confirmed}")
    print(f"    Supported (one method)   : {n_supported}")
    print(f"    Rejected  (neither)      : {n_rejected}")
    print(f"    → Important set (union)  : {n_important} variables for downstream VIF/GAM")

    # Print the important set
    important = consensus_df[consensus_df["consensus_important"]]
    print(f"\n  [CONSENSUS] Important variables (carry to collinearity checker):")
    for _, row in important.iterrows():
        sources = []
        if row["pimp_significant"]:
            sources.append(f"PIMP p={row['pimp_p_value']:.3f}")
        if row["boruta_important"]:
            sources.append(f"Boruta {row['boruta_decision']} ({row['boruta_hit_rate']:.0%})")
        print(f"    {row['predictor']:25s}  [{', '.join(sources)}]")

    # Save
    csv_file = f"consensus_feature_selection_{timestamp}.csv" if ADD_TIMESTAMP \
        else "consensus_feature_selection.csv"
    consensus_df.to_csv(outdir / csv_file, index=False)

    # Save important-only list (convenience for collinearity checker input)
    imp_list_file = f"consensus_important_variables_{timestamp}.txt" if ADD_TIMESTAMP \
        else "consensus_important_variables.txt"
    with open(outdir / imp_list_file, "w", encoding="utf-8") as f:
        f.write("=" * 72 + "\n")
        f.write("  ENSEMBLE FEATURE SELECTION — IMPORTANT VARIABLES\n")
        f.write(f"  Generated at: {get_readable_timestamp()}\n")
        f.write(f"  Method: Union of PIMP (Altmann et al., 2010) and Boruta\n")
        f.write(f"          (Kursa & Rudnicki, 2010) per Saeys et al. (2008)\n")
        f.write("=" * 72 + "\n\n")
        f.write(f"  Total predictors evaluated : {len(consensus_df)}\n")
        f.write(f"  Confirmed (both methods)   : {n_confirmed}\n")
        f.write(f"  Supported (one method)     : {n_supported}\n")
        f.write(f"  Rejected  (neither)        : {n_rejected}\n")
        f.write(f"  Important set (union)      : {n_important}\n\n")
        f.write("-" * 72 + "\n")
        f.write(f"  {'Predictor':<25s} {'Consensus':<12s} {'PIMP':<14s} {'Boruta':<18s} {'PermImp':>10s}\n")
        f.write("-" * 72 + "\n")
        for _, row in consensus_df.iterrows():
            pimp_str = f"p={row['pimp_p_value']:.3f}" if np.isfinite(row['pimp_p_value']) else "n/a"
            boruta_str = f"{row['boruta_decision']} ({row['boruta_hit_rate']:.0%})"
            marker = "  ✓" if row["consensus_important"] else "  ✗"
            f.write(f"{marker} {row['predictor']:<25s} {row['consensus']:<12s} "
                    f"{pimp_str:<14s} {boruta_str:<18s} {row['perm_importance']:>10.2f}\n")
        f.write("-" * 72 + "\n")
        f.write(f"\n  → Carry the {n_important} important variables to the collinearity checker.\n")
        f.write("=" * 72 + "\n")

    print(f"  [SAVED] {csv_file}")
    print(f"  [SAVED] {imp_list_file}")

    return consensus_df


def _plot_consensus_results(consensus_df, outdir, timestamp):
    """Horizontal bar chart showing consensus classification."""
    try:
        color_map = {"Confirmed": "#2E86AB", "Supported": "#F6AE2D", "Rejected": "#CCCCCC"}
        colors = [color_map[c] for c in consensus_df["consensus"]]

        fig, ax = plt.subplots(figsize=(14, max(6, len(consensus_df) * 0.35)))
        y_pos = range(len(consensus_df))
        ax.barh(y_pos, consensus_df["perm_importance"], color=colors, edgecolor="k", lw=0.5)

        for i, (_, row) in enumerate(consensus_df.iterrows()):
            pimp_mark = "P" if row["pimp_significant"] else "·"
            boruta_mark = "B" if row["boruta_important"] else "·"
            label = f"  [{pimp_mark}|{boruta_mark}] {row['consensus']}"
            ax.text(max(row["perm_importance"], 0), i, label, va="center", fontsize=8)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(consensus_df["predictor"], fontsize=9)
        ax.set_xlabel("Permutation Importance (ΔRMSE)", fontsize=11)
        ax.set_title(
            "Ensemble Feature Selection Consensus (PIMP ∪ Boruta)\n"
            "Blue = Confirmed (both), Yellow = Supported (one), Grey = Rejected (neither)\n"
            "[P|B] = PIMP|Boruta significant",
            fontsize=11
        )
        ax.grid(True, alpha=0.2, axis="x")
        ax.invert_yaxis()
        plt.tight_layout()

        fname = f"consensus_feature_selection_{timestamp}.png" if ADD_TIMESTAMP \
            else "consensus_feature_selection.png"
        fig.savefig(outdir / fname, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"  [WARNING] Consensus plot failed: {e}")


# =========================
# MAIN FITTING LOGIC
# =========================

def fit_multivariate_gam(df: pd.DataFrame, predictors: list, target: str,
                         outdir: Path, csv_name: str, run_ts: str):
    try:
        from pygam import LinearGAM, s
    except Exception:
        print("  [ERROR] pyGAM not installed. Install with: pip install pygam")
        return

    timestamp = run_ts

    print(f"  [INFO] Cleaning missing data flags: {MISSING_FLAGS}")
    df = clean_missing_flags(df, MISSING_FLAGS)

    df[target] = pd.to_numeric(df[target], errors="coerce")

    # Clean predictors (drop low-NA or constant)
    cleaned = []
    for c in predictors:
        s_col = df[c]
        if s_col.notna().sum() < MIN_NONNA_PER_COL:
            continue
        if np.nanstd(pd.to_numeric(s_col, errors="coerce").to_numpy()) == 0:
            continue
        cleaned.append(c)
    predictors = cleaned

    if not predictors:
        print("  [SKIP] No usable numeric predictors after cleaning.")
        return

    sub = df[predictors + [target]].dropna()
    if len(sub) < MIN_COMPLETE_CASES:
        print(f"  [SKIP] Not enough complete cases ({len(sub)} rows).")
        return

    print(f"  [INFO] Complete cases: {len(sub):,} ({len(sub)/len(df)*100:.1f}%)")

    # ── FIX v8: Split FIRST, then standardize on TRAIN only ──
    X_raw = sub[predictors].to_numpy()
    y = sub[target].to_numpy()

    X_raw_train, X_raw_val, y_train, y_val = train_test_split(
        X_raw, y, test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )

    mu = sd = None
    if STANDARDIZE_PREDICTORS:
        # Compute μ/σ on TRAIN only
        X_train, mu, sd = zscore_matrix(X_raw_train)
        # Apply TRAIN μ/σ to validation set
        X_val = apply_zscore(X_raw_val, mu, sd)
        # Also standardize full data for plotting (using train μ/σ)
        X_std_full = apply_zscore(X_raw, mu, sd)

        scaler_file = f"predictor_scaler_{timestamp}.csv" if ADD_TIMESTAMP else "predictor_scaler.csv"
        pd.DataFrame({'predictor': predictors, 'mean': mu, 'std': sd}).to_csv(
            outdir / scaler_file, index=False
        )
    else:
        X_train = X_raw_train
        X_val = X_raw_val
        X_std_full = X_raw

    # ── FIX v8: Drop-one ΔAIC on same TRAIN split ──
    try:
        doi = drop_one_term_importance(X_train, y_train, predictors)
        doi_file = (
            f"predictor_drop_one_importance_{timestamp}.csv"
            if ADD_TIMESTAMP else
            "predictor_drop_one_importance.csv"
        )
        doi.to_csv(outdir / doi_file, index=False)

        print("  [INFO] Top 10 predictors by drop-one ΔAIC:")
        for _, row in doi.head(10).iterrows():
            print(f"    {row['predictor']:25s}: ΔAIC={row['delta_AIC']:.2f}, ΔGCV={row['delta_GCV']:.4f}")
    except Exception as e:
        print(f"  [WARNING] Drop-one-term importance failed: {e}")

    # Build GAM
    terms = None
    for i in range(X_train.shape[1]):
        term = s(i, n_splines=N_SPLINES)
        terms = term if terms is None else terms + term

    # Fit on TRAIN only
    gam = (
        LinearGAM(terms).fit(X_train, y_train)
        if LAM is None
        else LinearGAM(terms, lam=LAM).fit(X_train, y_train)
    )

    # Permutation importance on VALIDATION
    imp_df = None  # Initialize so PIMP can check
    try:
        imp_df = permutation_importance_gam(
            gam, X_val, y_val, predictors, n_repeats=5, random_state=RANDOM_STATE
        )
        imp_file = (
            f"predictor_perm_importance_{timestamp}.csv"
            if ADD_TIMESTAMP else
            "predictor_perm_importance.csv"
        )
        imp_df.to_csv(outdir / imp_file, index=False)

        print("  [INFO] Top 10 predictors by permutation importance:")
        for _, row in imp_df.head(10).iterrows():
            print(f"    {row['predictor']:25s}: ΔRMSE={row['perm_RMSE_increase']:.4f}")
    except Exception as e:
        print(f"  [WARNING] Permutation importance failed: {e}")

    # ── RUNTIME ESTIMATE ──
    if PIMP_ENABLED or BORUTA_ENABLED:
        parts = []
        if PIMP_ENABLED:
            parts.append(f"PIMP ({PIMP_S} permutations × {len(predictors)}-term GAM)")
        if BORUTA_ENABLED:
            parts.append(f"Boruta ({BORUTA_MAX_ITER} iterations × {BORUTA_N_ESTIMATORS}-tree RF)")
        print(f"\n  [RUNTIME] Starting feature selection: {' + '.join(parts)}")
        print(f"  [RUNTIME] This may take 30 min to several hours depending on dataset size.")
        print(f"  [RUNTIME] Observations: {len(y_train):,} train, {len(y_val):,} val, "
              f"{len(predictors)} predictors\n")

    # ── PIMP: Statistical significance of permutation importance ──
    # (Altmann et al. 2010, Bioinformatics 26(10):1340-1347)
    pimp_df = None
    if PIMP_ENABLED and imp_df is None:
        print("  [WARNING] PIMP skipped: permutation importance was not computed")
    elif PIMP_ENABLED and imp_df is not None:
        try:
            pimp_df, null_imps = pimp_test_gam(
                X_train, y_train, X_val, y_val, predictors, imp_df,
                n_permutations=PIMP_S, n_repeats=5, random_state=RANDOM_STATE
            )

            pimp_file = (
                f"predictor_pimp_pvalues_{timestamp}.csv"
                if ADD_TIMESTAMP else
                "predictor_pimp_pvalues.csv"
            )
            pimp_df.to_csv(outdir / pimp_file, index=False)

            # Print significant predictors
            sig_pimp = pimp_df[pimp_df["pimp_significant"]]
            print(f"  [PIMP] Significant predictors (p < {PIMP_THRESHOLD}):")
            for _, row in sig_pimp.iterrows():
                print(f"    {row['predictor']:25s}: ΔRMSE={row['observed_importance']:.4f}, "
                      f"p={row['pimp_p_value']:.4f}")

            print(f"  [PIMP] Non-significant (drop these):")
            nonsig = pimp_df[~pimp_df["pimp_significant"]]
            for _, row in nonsig.iterrows():
                print(f"    {row['predictor']:25s}: ΔRMSE={row['observed_importance']:.4f}, "
                      f"p={row['pimp_p_value']:.4f}")

            # Save null distribution for reproducibility
            null_file = (
                f"predictor_pimp_null_distributions_{timestamp}.csv"
                if ADD_TIMESTAMP else
                "predictor_pimp_null_distributions.csv"
            )
            null_df = pd.DataFrame(null_imps, columns=predictors)
            null_df.to_csv(outdir / null_file, index=False)

            # Plot PIMP results
            _plot_pimp_results(pimp_df, null_imps, predictors, outdir, timestamp)

        except Exception as e:
            print(f"  [WARNING] PIMP test failed: {e}")
            import traceback
            traceback.print_exc()

    # ── BORUTA: All-relevant feature selection ──
    # (Kursa & Rudnicki 2010, J Stat Softw 36(11):1-13)
    boruta_df = None
    if BORUTA_ENABLED:
        try:
            # Boruta uses raw (unstandardized) train data with RF
            # RF is scale-invariant so standardization is unnecessary
            boruta_df = boruta_feature_selection(
                X_raw_train, y_train, predictors,
                max_iter=BORUTA_MAX_ITER,
                alpha=BORUTA_ALPHA,
                n_estimators=BORUTA_N_ESTIMATORS,
                max_depth=BORUTA_MAX_DEPTH,
                tentative_as_important=BORUTA_TENTATIVE_AS_IMPORTANT,
                random_state=BORUTA_RANDOM_STATE,
            )

            boruta_file = (
                f"boruta_feature_selection_{timestamp}.csv"
                if ADD_TIMESTAMP else
                "boruta_feature_selection.csv"
            )
            boruta_df.to_csv(outdir / boruta_file, index=False)

            _plot_boruta_results(boruta_df, outdir, timestamp)

        except Exception as e:
            print(f"  [WARNING] Boruta feature selection failed: {e}")
            import traceback
            traceback.print_exc()

    # ── CONSENSUS: Ensemble feature selection (PIMP ∪ Boruta) ──
    # (Saeys et al. 2008, Bioinformatics 24(19):2267-2273)
    consensus_df = None
    if CONSENSUS_ENABLED and pimp_df is not None and boruta_df is not None:
        try:
            consensus_df = consensus_feature_selection(
                pimp_df, boruta_df, predictors, outdir, timestamp
            )
            _plot_consensus_results(consensus_df, outdir, timestamp)

        except Exception as e:
            print(f"  [WARNING] Consensus feature selection failed: {e}")
            import traceback
            traceback.print_exc()
    elif CONSENSUS_ENABLED:
        missing = []
        if pimp_df is None:
            missing.append("PIMP")
        if boruta_df is None:
            missing.append("Boruta")
        print(f"  [WARNING] Consensus skipped: {' and '.join(missing)} not available")

    # Summary
    summary_file = f"multivariate_gam_summary_{timestamp}.txt" if ADD_TIMESTAMP else "multivariate_gam_summary.txt"
    write_text_summary(outdir / summary_file, gam.summary)

    # Print and save GAM equation
    try:
        sig_preds = print_gam_equation(gam, predictors, target, outdir, timestamp)
    except Exception as e:
        print(f"  [WARNING] Equation printing failed: {e}")

    # Data quality report
    create_data_quality_report(df, predictors, target, outdir, timestamp)

    # Plots (use full data standardized with TRAIN μ/σ)
    partial_dir = outdir / 'partial_effects'
    plot_multivariate_partial_effects_improved(
        gam, predictors, X_std_full, X_raw, y, mu, sd, partial_dir, timestamp
    )

    create_summary_plot(gam, predictors, outdir, timestamp)

    # Predictor term stats
    try:
        st = gam.statistics_
        edof_per_term = st.get('edof_per_term', None)
        pvals = st.get('p_values', None)

        if edof_per_term is not None or pvals is not None:
            imp = pd.DataFrame({'predictor': predictors})
            if edof_per_term is not None:
                imp['edf'] = list(edof_per_term[:len(predictors)])
            if pvals is not None:
                imp['p_value'] = list(pvals[:len(predictors)])

            imp_file = f"predictor_term_stats_{timestamp}.csv" if ADD_TIMESTAMP else "predictor_term_stats.csv"
            imp.to_csv(outdir / imp_file, index=False)
    except Exception as e:
        print(f"  [WARNING] Term stats export failed: {e}")

    print("  [OK] Multivariate GAM fitted and outputs saved.")


# =========================
# FILE PROCESSING
# =========================

def drop_leaky_predictors(df, predictors, target, corr_thresh=0.999):
    """
    Remove predictors that are identical to or almost perfectly correlated
    with the target. Prevents target leakage leading to 'perfect fit'.

    FIX v8: Renamed loop variable from 's' to 's_col' to avoid shadowing.
    """
    safe = []
    dropped = []

    t = pd.to_numeric(df[target], errors="coerce")

    for c in predictors:
        s_col = pd.to_numeric(df[c], errors="coerce")
        pair = pd.concat([s_col, t], axis=1).dropna()

        if len(pair) < 10:
            safe.append(c)
            continue

        # Perfect equality check
        if np.allclose(pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy(), equal_nan=False):
            dropped.append((c, "identical to target"))
            continue

        # Correlation check
        corr = pair.corr().iloc[0, 1]
        if np.isfinite(corr) and abs(corr) >= corr_thresh:
            dropped.append((c, f"corr={corr:.6f}"))
            continue

        safe.append(c)

    if dropped:
        print("  [WARNING] Dropped potential leakage predictors:")
        for name, reason in dropped:
            print(f"    - {name}: {reason}")

    return safe


def process_one_csv(csv_path: Path, out_root: Path, run_ts: str):
    print(f"\n{'='*70}")
    print(f"Processing: {csv_path.name}")
    print(f"Started at: {get_readable_timestamp()}")
    print(f"{'='*70}")

    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  [INFO] Loaded {len(df):,} rows, {len(df.columns)} columns")

    df = drop_excluded_headers(df, EXCLUDE_HEADERS)
    df = coerce_numeric_df(df)

    if TARGET_COL not in df.columns:
        print(f"  [SKIP] Missing target '{TARGET_COL}'.")
        return

    # Ensure target is numeric
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")

    num_cols = numeric_columns(df)
    if TARGET_COL not in num_cols:
        print(f"  [SKIP] Target '{TARGET_COL}' is not numeric.")
        return

    predictors = [c for c in num_cols if c != TARGET_COL]

    # Leakage guard
    predictors = drop_leaky_predictors(df, predictors, TARGET_COL)

    if EXCLUDE_INTERACTION_TERMS:
        n_before = len(predictors)
        predictors = filter_interaction_terms(predictors)
        print(f"  [INFO] Excluded {n_before - len(predictors)} interaction terms")

    if PRIMARY_PREDICTORS_ONLY:
        available_primary = [p for p in PRIMARY_PREDICTOR_LIST if p in predictors]
        if available_primary:
            print(f"  [INFO] Using {len(available_primary)} primary predictors")
            predictors = available_primary
        else:
            print("  [WARNING] No primary predictors found, using all available")

    if not predictors:
        print("  [SKIP] No numeric predictors found.")
        return

    print(f"  [INFO] Selected {len(predictors)} predictors for analysis")

    file_out = out_root #/ safe_name(csv_path.stem) / "multivariate_GAM_custom"
    file_out.mkdir(parents=True, exist_ok=True)

    # Pass run_ts down
    fit_multivariate_gam(df, predictors, TARGET_COL, file_out, csv_path.name, run_ts)


# =========================
# MAIN
# =========================

def main():
    input_dir = Path(INPUT_DIR)

    # ONE timestamp for the entire run
    run_ts = get_timestamp()

    # Create a unique run folder
    out_root = Path(OUTPUT_DIR) #/ f"run_{run_ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if INPUT_FILE:
        target_file = input_dir / INPUT_FILE
        if not target_file.is_file():
            print(f"File not found: {target_file}")
            return
        csv_files = [target_file]
    else:
        csv_files = sorted([p for p in input_dir.glob("*.csv") if p.is_file()])

    if not csv_files:
        print(f"No CSV files found in: {input_dir}")
        return

    print("\n" + "=" * 80)
    print("CUSTOMIZED MULTIVARIATE GAM ANALYSIS (v9)")
    print("Optimized for Rice Methane dataset")
    print("with PIMP + Boruta + Consensus Feature Selection")
    print("=" * 80)
    print(f"Run timestamp : {run_ts}")
    print(f"Analysis started: {get_readable_timestamp()}")
    print(f"Found {len(csv_files)} CSV file(s)")
    print(f"Input  : {input_dir}")
    print(f"Output : {out_root}")
    print(f"Target : {TARGET_COL}")

    print("\nCUSTOMIZATIONS:")
    print(f"  ✓ Missing data flags: {MISSING_FLAGS}")
    print(f"  ✓ Exclude interactions: {EXCLUDE_INTERACTION_TERMS}")
    print(f"  ✓ Primary predictors only: {PRIMARY_PREDICTORS_ONLY}")
    if PRIMARY_PREDICTORS_ONLY:
        print(f"    Primary list: {', '.join(PRIMARY_PREDICTOR_LIST[:5])}...")
    print(f"  ✓ DPI: {DPI}")
    print(f"  ✓ Timestamps: {ADD_TIMESTAMP} (run-level enforced)")
    print(f"  ✓ Train/test split: {1-TEST_FRACTION:.0%}/{TEST_FRACTION:.0%} (seed={RANDOM_STATE})")
    print(f"  ✓ PIMP: {'ON' if PIMP_ENABLED else 'OFF'} (S={PIMP_S}, α={PIMP_THRESHOLD})")
    print(f"  ✓ Boruta: {'ON' if BORUTA_ENABLED else 'OFF'} (max_iter={BORUTA_MAX_ITER}, α={BORUTA_ALPHA}, "
          f"max_depth={'None (unrestricted)' if BORUTA_MAX_DEPTH is None else BORUTA_MAX_DEPTH}, "
          f"tentative_as_important={BORUTA_TENTATIVE_AS_IMPORTANT})")
    print(f"  ✓ Consensus: {'ON' if CONSENSUS_ENABLED else 'OFF'} (union of PIMP ∪ Boruta)")
    print("=" * 80 + "\n")

    for csv_path in csv_files:
        try:
            process_one_csv(csv_path, out_root, run_ts)
        except Exception as e:
            print(f"  [ERROR] {csv_path.name} -> {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"All analyses completed at: {get_readable_timestamp()}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
