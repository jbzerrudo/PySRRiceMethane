"""
Symbolic Regression for Rice Methane Equation Discovery (v6.3)
Changes from v6.2:
  - Warning handling: replaced the blanket warnings.filterwarnings("ignore")
    (top of script AND inside the generated Stage 8 driver) with category-
    scoped filters. Library chatter (FutureWarning / DeprecationWarning /
    sklearn UserWarning) is still silenced, but numerical warnings
    (RuntimeWarning: overflow in exp, divide-by-zero, invalid value) and
    curve_fit convergence warnings now SURFACE — consistent with the
    fail-loud discipline that motivated the v6.1/v6.2 fixes.
  - Reproducibility: `deterministic` is now a DETERMINISTIC config flag
    (default False, unchanged behaviour). With multithreading on, a fixed
    random_state gives STATISTICAL reproducibility, not bit-identical runs;
    set DETERMINISTIC=True (plus serial execution — see config note) only
    for a final archival run. Methods sections should state which applies.
  - Persistence: pickling retained but gated behind SAVE_PICKLE (default
    True). hall_of_fame.csv (written via equation_file) is the canonical,
    version-stable artifact; the .pkl is a best-effort convenience cache.
    (Corrects the inaccurate v2 note that claimed model.save() replaced it.)
  - Added commented, OFF-by-default hooks for weighted loss (model.fit
    weights=) and robust elementwise_loss — verify the kwarg name for your
    PySR version before enabling. No change to default scientific output.
Changes from v6.1:
  - Day-grouped 80/20 mask now uses pandas Series.isin() instead of
    np.isin(set(perm.tolist())). On Windows / older numpy the
    datetime64[ns].tolist() round-trip silently converts to ints, and the
    subsequent comparison returns all False — producing train=ALL,
    test=0 (PySR fits on 100% of data with no held-out evaluation).
    Added a hard assertion that neither split is empty so any future
    dtype mismatch fails loud instead of silent.
Changes from v6:
  - Date-column detection uses a dual-attempt parser (dayfirst=False AND
    True) and picks the better one by (monotonicity, completeness). Fixes
    silent fallback to record-level split on DD/MM/YYYY CSVs (e.g.
    Filipino convention "04/01/2016" = 4 January). Threshold for accepting
    a parse is now 95% (was 50%) so partial parses with wrong-format
    inferences fail loud instead of succeeding silently.
Changes from v5:
  - Day-grouped 80/20 in run_symbolic_regression: PySR's internal
    train/test split is now random by *day* (not by record). Closes the
    within-day autocorrelation leakage that biased the candidate-ranking
    R² optimistically. Falls back to record-level split if no date column
    is found, with a printed warning. Date column auto-detected from
    {"Date", "time", "Deltime"}.
  - Generated stage8 driver gains three diagnostics for the
    "is variance from coefficients or from structure?" question:
      (1) No-refit baseline columns ({slot}_R2_nofit / {slot}_RMSE_nofit
          / {slot}_MAE_nofit) — predictions using PySR's warm-start p0
          directly on each test fold, no curve_fit. Diff vs the refit
          column is the per-fold gain from coefficient refitting.
      (2) coefficient_stability.csv — for each (slot, parameter) the
          mean / std / |CoV| / min / max of refitted values across the
          5 folds. |CoV| > 0.5 flagged as unstable.
      (3) per_day_residuals.csv — long-format dump of every test day in
          every fold, with y_true, y_pred (refit and nofit), residual,
          and slot. Lets you locate which days drive the variance.
  - No change to PySR run config, candidate selection, or Rule A.
Changes from v4 → v5:
  - (no documented changes; v5 was a checkpoint)
Changes from v3 → v4:
  - Added NESTED_CONSTRAINTS: prevents exp(exp(...)), tanh(tanh(...)),
    log(log(...)), sqrt(sqrt(...)) — transcendental self-nesting that is
    physically unmotivated and unstable outside the training domain.
  - Added CONSTRAINTS: limits argument complexity for transcendental operators
    (exp, log, tanh capped at 8 nodes; sqrt at 10), preventing deeply nested
    arguments that overfit residual structure.
  - Both are configurable in the Configuration section and can be set to {}
    to recover v3 (unconstrained) behaviour for comparison runs.
Changes from v2:
  - Removed `abs` from unary operators (prevents sign-folding of net CH4 flux)
  - Added non-finite prediction guard in diagnostics
  - Pareto-filtered knee detection (non-dominated solutions only)
  - Adaptive mid-complexity range (quantile-based, not hardcoded)
  - Model state cached to .pkl (hall_of_fame.csv remains the canonical artifact)
  - Try/except for equation_file vs temp_equation_file (works on any PySR version)
  - Added zero-predictor guard with diagnostic column listing
  - Added pd.to_numeric coercion for object-dtype columns (fixes pandas misreads)
  - replace_x_with_names retained as defensive fallback only
Requires: pysr, numpy, pandas, matplotlib, sympy, sklearn
Author: Jef Zerrudo / Claude
"""

import os, re
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# Scoped warning control (v6.3) — replaces the former blanket
# warnings.filterwarnings("ignore"). Benign library chatter is silenced by
# category, but numerical warnings (overflow in exp, divide-by-zero, invalid
# value) and curve_fit convergence warnings are NOT suppressed, so genuine
# numerical trouble stays visible during diagnostics and per-fold refitting.
# To quiet a specific noisy source, add a narrow filter for THAT category.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\JPN\Data-Metadata"
OUTPUT_DIR = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\PYSR\JPN\rerun_20260802"
INPUT_FILE = "JPN_retvars_pass2_C.csv"
TARGET_COL = "F_CH4_F"
EXCLUDE_HEADERS = {"site", "w", "Date", "Deltime", "time", "F_CH4_F_orig"}
MISSING_FLAGS = [-9999, -999900, -99999]
NITERATIONS = 2000       # or: 10000
MAXSIZE = 35             # Note: for ORYZA embedding, winning equations are typically 10-20 nodes
POPULATION_SIZE = 50
BINARY_OPERATORS = ["+", "-", "*", "/"]
# v3: removed `abs` — CH4 fluxes can be negative (net uptake); abs would mask sign dynamics.
# If you want magnitude prediction, train on |F_CH4| explicitly.
UNARY_OPERATORS = ["exp", "log", "sqrt", "tanh"]
# For longer runs (NITERATIONS=10000), consider adding: "square", "inv"

# v4: Operator nesting constraints.
# Prevents transcendental functions from nesting inside themselves.
# Key: outer operator. Value dict: inner operator -> max nesting depth.
#   0 = inner can NEVER appear inside outer
#   1 = inner can appear once inside outer (but not deeper)
#  -1 = no restriction (default if key absent)
# Rationale: exp(exp(x)) diverges catastrophically outside training range;
# tanh(tanh(x)) compresses signal with no physical basis; log(log(x)) has
# domain issues. Self-nesting of transcendentals is mathematically valid but
# ecohydrologically unmotivated and extrapolation-hazardous.
# Set to {} to recover v3 (unconstrained) behaviour.
NESTED_CONSTRAINTS = {
    "exp":  {"exp": 1, "log": 0},        # exp can contain at most one exp (e.g.  Gompertz's); no log inside exp
    "log":  {"log": 0, "exp": 1},        # allow one nested exp inside log (e.g., log(1+exp(x)))
    "tanh": {"tanh": 0},                 # tanh cannot contain tanh
}

# v4: Argument complexity constraints.
# Limits how many expression-tree nodes the argument of each operator can have.
# Prevents deeply nested arguments like exp(tanh(x1/(x2+1.5)) + sqrt(x3*x4))
# which overfit residual structure at high complexity.
# For binary operators, tuple gives (left_max, right_max); -1 = no limit.
# Set to {} to recover v3 (unconstrained) behaviour.
CONSTRAINTS = {
    "exp":  8,        # argument to exp limited to 8 nodes
    "log":  8,
    "tanh": 8,
    "sqrt": 10,
    "+":    (-1, -1), # unconstrained
    "-":    (-1, -1),
    "*":    (-1, -1),
    "/":    (-1, -1),
}
EXTRA_SYMPY_MAPPINGS = {}
RANDOM_STATE = 42
# v6.3: reproducibility control. False (default) = multithreaded search;
# a fixed RANDOM_STATE then gives STATISTICAL reproducibility across seeds,
# NOT bit-identical runs. Set True only for a final archival run to get
# exact reproducibility — note PySR also requires serial execution for this
# (set the single-thread/serial flag for YOUR PySR version), and it is
# substantially slower.
DETERMINISTIC = False
# v6.3: write the PySR model as a .pkl cache. hall_of_fame.csv (the
# equation_file) is the canonical, version-stable artifact; the pickle is a
# convenience cache and is fragile across PySR/Julia versions. False = skip.
SAVE_PICKLE = True
N_SEEDS = 12       # Number of PySR seeds to run for multi-seed pipeline.
                   # Set to 1 for single-seed mode (no Rule A/B selection).
                   # Seeds used: RANDOM_STATE, RANDOM_STATE+1, ...
TEST_FRACTION = 0.2
PARSIMONY = 0.0032
BATCHING = True
BATCH_SIZE = 256
N_DISPLAY = 15
DPI = 300

# ─── Utilities ────────────────────────────────────────────────────────────────

def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def get_readable_timestamp():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def clean_missing_flags(df, flags):
    df_clean = df.copy()
    num_cols = df_clean.select_dtypes(include=[np.number]).columns
    if len(num_cols) > 0:
        df_clean[num_cols] = df_clean[num_cols].replace(flags, np.nan)
    return df_clean

def safe_name(s):
    return "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in str(s))

def replace_x_with_names(equation_str, predictors):
    """Defensive fallback: substitute x0, x1, ... with predictor names.

    When `variable_names` is passed to PySRRegressor (as we do), PySR natively
    writes equations with actual predictor names.  This function is kept only
    for compatibility with older PySR versions or edge cases where the sympy
    repr falls back to indexed variables.
    """
    result = str(equation_str)
    for i in sorted(range(len(predictors)), reverse=True):
        result = re.sub(rf'\bx{i}\b', predictors[i], result)
    return result

_SAFE_TRANSLATE = [
    ("*", "_STAR_"),
    ("/", "_SLASH_"),
    (".", "_DOT_"),
    ("-", "_DASH_"),
    ("+", "_PLUS_"),
    (" ", "_SP_"),
]

def _is_unsafe_predictor(name):
    return any(tok in name for tok in [c for c, _ in _SAFE_TRANSLATE])

def _sanitize_name(name):
    """Replace sympy-hostile characters in a predictor name with safe tokens."""
    out = name
    for bad, good in _SAFE_TRANSLATE:
        out = out.replace(bad, good)
    return out

def _build_translation(predictors):
    """Build (safe_predictors, eq_substitutions, reverse_substitutions).

    eq_substitutions: list of (original, safe) pairs sorted longest-first,
    suitable for str.replace() on equation strings (longest-first prevents
    partial replacements like "h*VPD" when "VPD" is also a predictor).
    """
    safe = [_sanitize_name(p) for p in predictors]
    pairs_unsafe = [(p, s) for p, s in zip(predictors, safe)
                    if _is_unsafe_predictor(p)]
    # Apply longest original first to avoid partial overlaps.
    pairs_unsafe.sort(key=lambda kv: -len(kv[0]))
    reverse = list(zip(safe, predictors))
    reverse.sort(key=lambda kv: -len(kv[0]))
    return safe, pairs_unsafe, reverse


def _apply_subs(s, pairs):
    out = s
    for a, b in pairs:
        out = out.replace(a, b)
    return out


def _safe_sympify(eq_str, predictors):
    """Try to sympify an equation string. Returns sympy expr or None.

    v4.3-fix: predictor names containing sympy-hostile characters (asterisks,
    slashes, etc.) are translated to safe placeholders BEFORE sympy parses,
    so a column literally named "h*VPD" doesn't get tokenised as h * VPD.
    """
    try:
        import sympy as sp
        safe_preds, eq_subs, _ = _build_translation(predictors)
        eq_safe = _apply_subs(eq_str, eq_subs)
        local = {sp_name: sp.Symbol(sp_name) for sp_name in safe_preds}
        # Also handle x0/x1/... style references (legacy from raw PySR output)
        for i, sp_name in enumerate(safe_preds):
            local[f"x{i}"] = sp.Symbol(sp_name)
        return sp.sympify(eq_safe, locals=local)
    except Exception:
        return None


# ─── Stage C: auto-generate day-grouped CV script ────────────────────────────

def _write_auto_stage8(forms_for_stage8, predictors, input_csv,
                       outdir, run_ts):
    """Generate a runnable stage8_cv_methane_auto.py populated with the
    candidate forms found by PySR. Each form gets its own slot:
        simple_topR2, knee_topR2, complex_topR2, best_accuracy_topR2.

    No Stage A or B — this generator builds straight from PySR seed-42
    candidates. PySR's own coefficients are used as warm-start p0.
    """
    def _slot_prefix(label):
        l = label.lower()
        if "auto-best" in l or "auto_best" in l:
            return "simple"
        if "knee" in l:
            return "knee"
        if "mid" in l:
            return "complex"
        if "best accuracy" in l or "best_accuracy" in l:
            return "best_accuracy"
        return safe_name(label).lower()

    slots = {}
    for label, forms_list in forms_for_stage8.items():
        if not forms_list:
            continue
        prefix = _slot_prefix(label)
        for info in forms_list:
            origin = info.get("origin", "topR2")
            # Accept any origin string (topR2, consensus, stageA, ruleA, ruleB, ...)
            slot = f"{prefix}_{origin}"
            if slot in slots:
                i = 2
                while f"{slot}_{i}" in slots:
                    i += 1
                slot = f"{slot}_{i}"
            slots[slot] = (label, info)

    def _expr_to_python(free_expr_str, predictors, n_params):
        s = free_expr_str
        for i in reversed(range(len(predictors))):
            s = s.replace(f"x{i}", _sanitize_name(predictors[i]))
        for pname in predictors:
            if _is_unsafe_predictor(pname):
                s = s.replace(pname, _sanitize_name(pname))
        letters = "abcdefghijklmnop"
        for i in reversed(range(n_params)):
            s = s.replace(f"p{i}", letters[i])
        for fn in ("exp", "tanh", "sqrt", "log", "sin", "cos"):
            s = s.replace(f"{fn}(", f"np.{fn}(")
        return s

    unpack_line = ", ".join(_sanitize_name(p) for p in predictors) + " = X"

    func_blocks = []
    p0_blocks = []           # v6: warm-start constants for no-refit baseline
    fit_blocks = []
    predict_blocks = []
    main_calls = []
    summary_lines = []
    csv_record_pairs = []

    canonical = []
    for prefix in ("simple", "knee", "complex", "best_accuracy"):
        for origin in ("topR2", "consensus", "stageA", "ruleA", "ruleB"):
            canonical.append(f"{prefix}_{origin}")
    extras = sorted(s for s in slots.keys() if s not in canonical)
    slot_order = [s for s in canonical if s in slots] + extras

    for slot in slot_order:
        if slot not in slots:
            continue
        label, info = slots[slot]
        n = info["n_params"]
        letters = "abcdefghijklmnop"[:n]
        args = ", ".join(letters) if letters else ""
        py = _expr_to_python(info["free_expr"], predictors, n)
        p0 = ", ".join(f"{v:.6e}" for v in info["popt"])

        origin = info.get("origin", "topR2")
        rep = info.get("representative", "(unavailable)")
        rep_short = rep if len(rep) <= 200 else rep[:197] + "..."
        source_seed = info.get("source_seed", "(multiple)")
        complexity = info.get("complexity", "?")

        docstring_lines = [
            f"    Label               : {label}",
            f"    Origin              : {origin}",
            f"    Source seed         : {source_seed}  (complexity={complexity})",
            f"    PySR representative form (constants left as found by PySR):",
            f"      CH4 = {rep_short}",
            f"",
            f"    Refit (this function): coefficients {letters or '(none)'} "
                f"→ refit per CV fold by curve_fit; warm-start p0 below.",
        ]
        docstring = "\n".join(docstring_lines)

        sig = f"X, {args}" if args else "X"
        func_blocks.append(
            f"def {slot}({sig}):\n"
            f"    \"\"\"\n{docstring}\n    \"\"\"\n"
            f"    {unpack_line}\n"
            f"    return {py}\n"
        )
        fit_blocks.append(
            f"def fit_{slot}(train_df, predictors):\n"
            f"    X = _pack_X(train_df, predictors)\n"
            f"    y = train_df[TARGET_COL].values\n"
            f"    try:\n"
            f"        popt, _ = curve_fit({slot}, X, y, "
            f"p0=[{p0}], maxfev=40000)\n"
            f"        return popt\n"
            f"    except Exception:\n"
            f"        return np.array([{p0}])\n"
        )
        predict_blocks.append(
            f"def predict_{slot}(test_df, predictors, params):\n"
            f"    return {slot}(_pack_X(test_df, predictors), *params)\n"
        )
        # v6: warm-start constants and no-refit predictor for diagnostic A.
        p0_blocks.append(
            f"P0_{slot} = [{p0}]\n"
        )
        predict_blocks.append(
            f"def predict_{slot}_nofit(test_df, predictors):\n"
            f"    \"\"\"v6: predictions using PySR's warm-start p0 directly,\n"
            f"    no fold-level curve_fit. Diff vs predict_{slot} = refit gain.\"\"\"\n"
            f"    return {slot}(_pack_X(test_df, predictors), *P0_{slot})\n"
        )
        main_calls.append(slot)
        summary_lines.append(slot)
        csv_record_pairs.append(slot)

    fit_lines = "\n        ".join(
        f"p_{s} = fit_{s}(tr, predictors)" for s in main_calls)

    # v6: predict both refit and no-refit; capture per-day residuals.
    predict_lines_parts = []
    for s in main_calls:
        predict_lines_parts.append(
            f"yp_{s}    = predict_{s}(te, predictors, p_{s})")
        predict_lines_parts.append(
            f"yp_{s}_nf = predict_{s}_nofit(te, predictors)")
        predict_lines_parts.append(
            f"r_{s}    = compute_metrics(yte, yp_{s})")
        predict_lines_parts.append(
            f"r_{s}_nf = compute_metrics(yte, yp_{s}_nf)")
    predict_lines = "\n        ".join(predict_lines_parts)

    # v6: per-fold residual frame: one row per test record, all slots side by side.
    resid_capture_parts = ["te_rec = te[[TIME_COL, 'DAY', TARGET_COL]].copy()",
                           "te_rec['fold'] = k + 1"]
    for s in main_calls:
        resid_capture_parts.append(
            f"te_rec['{s}_pred_refit'] = yp_{s}")
        resid_capture_parts.append(
            f"te_rec['{s}_pred_nofit'] = yp_{s}_nf")
        resid_capture_parts.append(
            f"te_rec['{s}_resid_refit'] = te_rec[TARGET_COL] - yp_{s}")
        resid_capture_parts.append(
            f"te_rec['{s}_resid_nofit'] = te_rec[TARGET_COL] - yp_{s}_nf")
    resid_capture_parts.append("all_residuals.append(te_rec)")
    resid_capture = "\n        ".join(resid_capture_parts)

    record_lines = []
    for s in csv_record_pairs:
        record_lines.append(
            f'            "{s}_R2": r_{s}[0], "{s}_RMSE": r_{s}[1], "{s}_MAE": r_{s}[2],')
        # v6: no-refit columns
        record_lines.append(
            f'            "{s}_R2_nofit": r_{s}_nf[0], '
            f'"{s}_RMSE_nofit": r_{s}_nf[1], "{s}_MAE_nofit": r_{s}_nf[2],')
    record_lines.append('            "fold": k+1,')
    for s in csv_record_pairs:
        record_lines.append(
            f'            "{s}_params": ",".join(f"{{v:.6e}}" for v in p_{s}),')
    record_block = "\n".join(record_lines)
    print_parts = " ".join(
        f"{s} R²={{r_{s}[0]:+.3f}}/nf{{r_{s}_nf[0]:+.3f}}"
        for s in main_calls)

    slot_w = max((len(s) for s in summary_lines), default=16)
    slot_w = max(slot_w, 16)

    summary_blocks = []
    for s in summary_lines:
        summary_blocks.append(
            f'    m = out["{s}_RMSE"].mean(); std = out["{s}_RMSE"].std()\n'
            f'    lines.append(f"  {s:<{slot_w}} mean RMSE = {{m:.3f}} ± {{std:.3f}}")'
        )
    summary_block = "\n".join(summary_blocks)
    summary_blocks_r2 = []
    for s in summary_lines:
        summary_blocks_r2.append(
            f'    m = out["{s}_R2"].mean(); std = out["{s}_R2"].std()\n'
            f'    lines.append(f"  {s:<{slot_w}} mean R²   = {{m:+.3f}} ± {{std:.3f}}")'
        )
    summary_block_r2 = "\n".join(summary_blocks_r2)

    # v6: also summarise the no-refit baseline R² (diagnostic A).
    summary_blocks_r2_nofit = []
    for s in summary_lines:
        summary_blocks_r2_nofit.append(
            f'    m = out["{s}_R2_nofit"].mean(); '
            f'std = out["{s}_R2_nofit"].std()\n'
            f'    lines.append(f"  {s:<{slot_w}} mean R²nf  = {{m:+.3f}} ± {{std:.3f}}")'
        )
    summary_block_r2_nofit = "\n".join(summary_blocks_r2_nofit)

    # v6: emit the slot list as a Python literal for the coefficient-stability
    # post-loop block.
    slot_list_literal = "[" + ", ".join(f'"{s}"' for s in csv_record_pairs) + "]"

    input_csv_doc = str(input_csv).replace("\\", "\\\\")
    outdir_doc = (str(outdir) + f"\\stage8_auto_{run_ts}").replace("\\", "\\\\")

    auto_path = outdir / f"stage8_cv_methane_auto_{run_ts}.py"

    n_emitted = len([s for s in main_calls])

    header_w = max((len(s) for s in main_calls), default=16)
    header_w = max(header_w, 16)
    header_lines = [
        f"#   {n_emitted} candidate forms emitted (PySR seed-42 candidates,",
        f"#   one slot per label, no Stage A/B):",
    ]
    for slot in slot_order:
        label, info = slots[slot]
        origin = info.get("origin", "topR2")
        rep = info.get("representative", "(unavailable)")
        rep_short = rep if len(rep) <= 130 else rep[:127] + "..."
        seed_str = f"seed={info.get('source_seed')}"
        header_lines.append(
            f"#   • {slot:<{header_w}}  [{origin:<9}]  {label}"
        )
        header_lines.append(
            f"#       {seed_str}, complexity={info.get('complexity', '?')}"
        )
        header_lines.append(f"#       CH4 = {rep_short}")
    header_block = "\n".join(header_lines)

    script = f'''"""
Auto-generated Stage 8 CV driver (from PySR_NT_stage8)
Generated: {get_readable_timestamp()}
Source:    {input_csv_doc}
Predictors (in CSV order): {predictors}
Forms emitted: {n_emitted} ({", ".join(main_calls)})

This is a self-contained day-grouped 5-fold CV script. PySR seed-42
candidate equations are emitted with their PySR-native coefficients as
warm-start p0; each fold refits via curve_fit on train days and
evaluates on held-out days.
Run directly:  python {auto_path.name}
"""

# ──────────────────────────────────────────────────────────────────────────
{header_block}
# ──────────────────────────────────────────────────────────────────────────

import os, warnings
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy import stats
# v6.3: category-scoped (was a blanket ignore). Numerical / curve_fit
# convergence warnings stay visible; only library chatter is silenced.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ── CONFIG (edit OUTPUT_DIR if desired) ───────────────────────────────────
INPUT_CSV  = r"{input_csv}"
OUTPUT_DIR = r"{outdir}\\stage8_auto_{run_ts}"
TARGET_COL = "F_CH4_F"
TIME_COL   = "Date"
MISSING_FLAGS = [-9999, -999900, -99999]
N_FOLDS      = 5
RANDOM_STATE = 42


def parse_dates_robust(series, verbose=True):
    s = series.astype(str).str.strip()
    n = len(s)
    cands = []
    for df_flag in (False, True):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(s, errors="coerce", dayfirst=df_flag, format="mixed")
        valid = int(parsed.notna().sum())
        pv = parsed.dropna()
        mono = (float((pv.diff().dropna() >= pd.Timedelta(0)).sum())
                / max(1, len(pv.diff().dropna())))
        cands.append({{"dayfirst": df_flag, "parsed": parsed,
                       "valid": valid, "mono": mono}})
    best = max(cands, key=lambda c: (c["mono"], c["valid"]))
    if verbose:
        for c in cands:
            flag = "  <-- chosen" if c is best else ""
            print(f"  dayfirst={{str(c['dayfirst']):<5}}  "
                  f"parsed={{c['valid']}}/{{n}}  "
                  f"monotonic={{c['mono']:.2%}}{{flag}}")
    return best["parsed"]


def _pack_X(train_df, predictors):
    return tuple(train_df[p].values for p in predictors)


# ── Auto-generated model functions ────────────────────────────────────────
{chr(10).join(func_blocks)}

# ── Warm-start coefficients (v6: used by *_nofit predictors) ─────────────
{chr(10).join(p0_blocks)}

# ── Auto-generated fit functions ──────────────────────────────────────────
{chr(10).join(fit_blocks)}

# ── Prediction helpers ────────────────────────────────────────────────────
{chr(10).join(predict_blocks)}

# ── Metrics ───────────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred):
    finite = np.isfinite(y_pred)
    if finite.sum() < 10:
        return np.nan, np.nan, np.nan
    res = y_true[finite] - y_pred[finite]
    ss_tot = float(np.sum((y_true[finite] - y_true[finite].mean()) ** 2))
    r2 = float(1 - np.sum(res ** 2) / ss_tot) if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(np.mean(res ** 2)))
    mae  = float(np.mean(np.abs(res)))
    return r2, rmse, mae


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(INPUT_CSV)
    for col in df.select_dtypes(include=["object"]).columns:
        if col == TIME_COL: continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            df[col] = converted
    num_cols = df.select_dtypes(include=[np.number]).columns
    df[num_cols] = df[num_cols].replace(MISSING_FLAGS, np.nan)

    df[TIME_COL] = parse_dates_robust(df[TIME_COL], verbose=True)
    df = df.dropna(subset=[TIME_COL])
    df["DAY"] = df[TIME_COL].dt.date

    predictors = {predictors!r}
    df = df.dropna(subset=predictors + [TARGET_COL])
    print(f"Complete cases: {{len(df)}}  Days: {{df['DAY'].nunique()}}")

    days = df["DAY"].unique()
    rng = np.random.RandomState(RANDOM_STATE)
    rng.shuffle(days)
    fold_size = len(days) // N_FOLDS

    results = []
    all_residuals = []   # v6: per-record residual capture for diagnostic C
    for k in range(N_FOLDS):
        if k < N_FOLDS - 1:
            test_days = days[k * fold_size:(k + 1) * fold_size]
        else:
            test_days = days[k * fold_size:]
        train_days = np.setdiff1d(days, test_days)
        tr = df[df["DAY"].isin(train_days)]; te = df[df["DAY"].isin(test_days)]
        yte = te[TARGET_COL].values

        {fit_lines}

        {predict_lines}

        # v6: capture per-record residuals (refit + nofit) for this fold
        {resid_capture}

        results.append({{
{record_block}
        }})
        print(f"Fold {{k+1}}: {print_parts}")

    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUTPUT_DIR, "cv_fold_metrics.csv"), index=False)

    # v6 — diagnostic C: per-day residual export.
    # Aggregates per-record residuals to per-day means so the file stays small
    # and AWD-transition days are identifiable. Per-record file kept as well
    # for finer inspection.
    if all_residuals:
        rec_df = pd.concat(all_residuals, axis=0, ignore_index=True)
        rec_df.to_csv(os.path.join(OUTPUT_DIR, "per_record_residuals.csv"),
                      index=False)
        # Day-aggregated: mean residual per (DAY, slot, refit/nofit)
        slot_list = {slot_list_literal}
        long_rows = []
        for slot in slot_list:
            g = rec_df.groupby(["DAY", "fold"], as_index=False).agg(
                y_mean=(TARGET_COL, "mean"),
                pred_refit=(f"{{slot}}_pred_refit", "mean"),
                pred_nofit=(f"{{slot}}_pred_nofit", "mean"),
                resid_refit=(f"{{slot}}_resid_refit", "mean"),
                resid_nofit=(f"{{slot}}_resid_nofit", "mean"),
                n_records=(TARGET_COL, "size"),
            )
            g["slot"] = slot
            long_rows.append(g)
        day_df = pd.concat(long_rows, axis=0, ignore_index=True)
        day_df = day_df[["slot", "fold", "DAY", "n_records", "y_mean",
                         "pred_refit", "pred_nofit",
                         "resid_refit", "resid_nofit"]]
        day_df.to_csv(os.path.join(OUTPUT_DIR, "per_day_residuals.csv"),
                      index=False)
        print(f"  [SAVED] per_record_residuals.csv  "
              f"({{len(rec_df)}} rows)")
        print(f"  [SAVED] per_day_residuals.csv     "
              f"({{len(day_df)}} rows = slots × folds × test-days)")

    # v6 — diagnostic B: coefficient stability across folds.
    # For each (slot, parameter), compute mean / std / |CoV| of refitted
    # values across the 5 folds. |CoV| > 0.5 flagged as unstable: the
    # parameter is absorbing fold-to-fold regime shifts.
    slot_list = {slot_list_literal}
    coef_rows = []
    LETTERS = "abcdefghijklmnop"
    for slot in slot_list:
        warm = np.array(globals().get(f"P0_{{slot}}", []), dtype=float)
        params_col = f"{{slot}}_params"
        if params_col not in out.columns or len(warm) == 0:
            continue
        fold_params = []
        for s in out[params_col]:
            try:
                vals = np.array([float(x) for x in str(s).split(",")
                                 if x.strip() != ""])
            except Exception:
                vals = np.full(len(warm), np.nan)
            if len(vals) != len(warm):
                vals = np.full(len(warm), np.nan)
            fold_params.append(vals)
        fp = np.stack(fold_params, axis=0) if fold_params else np.zeros(
            (0, len(warm)))
        for j in range(len(warm)):
            v = fp[:, j] if fp.size else np.array([np.nan])
            v_finite = v[np.isfinite(v)]
            if len(v_finite) == 0:
                mean = std = cov = vmin = vmax = np.nan
            else:
                mean = float(v_finite.mean())
                std = float(v_finite.std(ddof=1)) if len(v_finite) > 1 \
                    else float("nan")
                cov = (std / abs(mean)) if (np.isfinite(std) and
                                             abs(mean) > 1e-12) else np.nan
                vmin = float(v_finite.min())
                vmax = float(v_finite.max())
            coef_rows.append({{
                "slot":         slot,
                "param_idx":    j,
                "param_letter": LETTERS[j] if j < len(LETTERS) else f"p{{j}}",
                "warm_start":   float(warm[j]),
                "fold_mean":    mean,
                "fold_std":     std,
                "fold_cov_abs": cov,
                "fold_min":     vmin,
                "fold_max":     vmax,
                "unstable_flag": int(np.isfinite(cov) and cov > 0.5),
            }})
    if coef_rows:
        coef_df = pd.DataFrame(coef_rows)
        coef_df.to_csv(
            os.path.join(OUTPUT_DIR, "coefficient_stability.csv"),
            index=False)
        n_unstable = int(coef_df["unstable_flag"].sum())
        print(f"  [SAVED] coefficient_stability.csv "
              f"({{len(coef_df)}} rows; {{n_unstable}} flagged |CoV|>0.5)")

    lines = ["{n_emitted}-form Day-Grouped CV (auto-generated, v6 diagnostics)",
             "=" * 70]
{summary_block}

    lines.append("")
{summary_block_r2}

    lines.append("")
    lines.append("No-refit baseline R² (warm-start p0 used directly, "
                 "no curve_fit per fold):")
{summary_block_r2_nofit}

    with open(os.path.join(OUTPUT_DIR, "cv_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\\n".join(lines))
    print()
    print("\\n".join(lines))
    print(f"\\nSaved: {{OUTPUT_DIR}}")


if __name__ == "__main__":
    main()
'''
    with open(auto_path, "w", encoding="utf-8") as f:
        f.write(script)
    print(f"\n  [SAVED] {auto_path.name}")
    print(f"          → emitted {n_emitted} forms: {', '.join(main_calls)}")
    print(f"          → run with:  python {auto_path}")


def _build_forms_from_run(candidates, eqs, predictors, seed, origin="topR2"):
    """Convert one seed run's candidates into the form_dict shape that
    _write_auto_stage8 expects. No refit — uses PySR's own coefficients
    as warm-start p0 for the per-fold curve_fit in the generated stage8.

    Args:
      candidates: list of (cidx, clabel) tuples from select_candidate_equations.
      eqs: model.equations_ DataFrame (has 'equation', 'complexity' columns).
      predictors: list of predictor column names.
      seed: PySR random_state used for this run.
      origin: tag identifying which selection rule produced this set
              ("topR2", "ruleA", "ruleB", etc.). Used to namespace the
              slot in the generated stage8 script.

    Returns:
      {label: [form_dict]} ready to pass to _write_auto_stage8.
    """
    import sympy as sp
    forms = {}
    for cidx, clabel in candidates:
        eq_str = str(eqs.loc[cidx, "equation"])
        complexity = float(eqs.loc[cidx, "complexity"])

        expr = _safe_sympify(eq_str, predictors)
        if expr is None:
            print(f"    [WARNING] Could not sympify candidate #{cidx} "
                  f"({clabel}); skipping.")
            continue
        consts = [n for n in expr.atoms(sp.Number)
                  if n not in (sp.S.Zero, sp.S.One, sp.S.NegativeOne)]
        param_syms = [sp.Symbol(f"p{i}") for i in range(len(consts))]
        subs = {c: ps for c, ps in zip(consts, param_syms)}
        free_expr = expr.xreplace(subs)
        popt = [float(c) for c in consts]
        forms[clabel] = [{
            "origin":          origin,
            "n_params":        len(consts),
            "free_expr":       str(free_expr),
            "popt":            popt,
            "representative":  eq_str,
            "source_seed":     seed,
            "complexity":      complexity,
            "r2":              float("nan"),
            "rmse":            float("nan"),
            "consensus_count": 1,
            "n_seeds":         1,
            "canonical":       eq_str,
            "in_sample_r2":    float("nan"),
        }]
    return forms


# Back-compat alias for the old name (in case anything else uses it).
def _build_forms_from_seed42(candidates, eqs, predictors, seed):
    return _build_forms_from_run(candidates, eqs, predictors, seed,
                                 origin="topR2")


# ─── Candidate selection ──────────────────────────────────────────────────────

def _pareto_filter(eqs):
    """Return a copy of `eqs` containing only non-dominated rows.

    PySR should already return a Pareto front, but with batching or short runs
    dominated solutions can leak in.  We keep only rows where loss is strictly
    non-increasing with increasing complexity.
    """
    eqs_sorted = eqs.sort_values("complexity").copy()
    keep_idx = []
    best_loss = np.inf
    for idx, row in eqs_sorted.iterrows():
        if row["loss"] < best_loss:
            best_loss = row["loss"]
            keep_idx.append(idx)
    return eqs.loc[eqs.index.isin(keep_idx)]

def select_candidate_equations(eqs):
    candidates = []

    # 1. Auto-best (highest marginal score)
    if "score" in eqs.columns:
        best_score_idx = eqs["score"].idxmax()
        candidates.append((best_score_idx, "Auto-best (highest score)"))

    # 2. Knee of Pareto front (on non-dominated solutions only)
    eqs_pareto = _pareto_filter(eqs)
    complexities = eqs_pareto["complexity"].values.astype(float)
    losses = eqs_pareto["loss"].values.astype(float)

    if len(eqs_pareto) >= 3:
        c_min, c_max = complexities.min(), complexities.max()
        l_min, l_max = losses.min(), losses.max()
        c_range = max(c_max - c_min, 1.0)
        l_range = max(l_max - l_min, 1.0)
        cn = (complexities - c_min) / c_range
        ln = (losses - l_min) / l_range
        p1 = np.array([cn[0], ln[0]])
        p2 = np.array([cn[-1], ln[-1]])
        line_vec = p2 - p1
        line_len = np.linalg.norm(line_vec)
        if line_len > 0:
            line_unit = line_vec / line_len
            max_dist = -1
            knee_idx = eqs_pareto.index[0]
            for j in range(len(eqs_pareto)):
                point = np.array([cn[j], ln[j]])
                vec_to_point = point - p1
                proj = np.dot(vec_to_point, line_unit)
                closest = p1 + proj * line_unit
                dist = np.linalg.norm(point - closest)
                if dist > max_dist:
                    max_dist = dist
                    knee_idx = eqs_pareto.index[j]
            if not any(idx == knee_idx for idx, _ in candidates):
                candidates.append((knee_idx, "Knee of Pareto front"))
            else:
                # Knee coincides with auto-best — pick the next Pareto member
                pareto_indices = list(eqs_pareto.index)
                knee_pos = pareto_indices.index(knee_idx)
                if knee_pos + 1 < len(pareto_indices):
                    candidates.append((pareto_indices[knee_pos + 1], "Near-knee candidate"))

    # 3. Best accuracy (lowest loss)
    best_loss_idx = eqs["loss"].idxmin()
    if not any(idx == best_loss_idx for idx, _ in candidates):
        candidates.append((best_loss_idx, "Best accuracy (most complex)"))

    # 4. Mid-complexity candidate (adaptive quantile range)
    c_lo = eqs["complexity"].quantile(0.25)
    c_hi = eqs["complexity"].quantile(0.60)
    mid_eqs = eqs[(eqs["complexity"] >= c_lo) & (eqs["complexity"] <= c_hi)]
    if len(mid_eqs) > 0:
        mid_idx = mid_eqs["loss"].idxmin()
        if not any(idx == mid_idx for idx, _ in candidates):
            candidates.append((mid_idx, "Mid-complexity candidate"))

    return candidates

# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_pareto_front(model, candidates, outdir, timestamp):
    eqs = model.equations_
    if eqs is None or len(eqs) == 0:
        return
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    complexities = eqs["complexity"].values
    losses = eqs["loss"].values
    ax.scatter(complexities, losses, c="steelblue", s=60, zorder=3,
               edgecolors="k", linewidths=0.5, label="All equations")
    for i, (cx, lo) in enumerate(zip(complexities, losses)):
        ax.annotate(f"#{i}", (cx, lo), textcoords="offset points",
                    xytext=(5, 5), fontsize=7, color="grey")
    markers = ["*", "D", "s", "^"]
    colours = ["red", "darkorange", "green", "purple"]
    for k, (cidx, clabel) in enumerate(candidates):
        row = eqs.loc[cidx]
        ax.scatter(row["complexity"], row["loss"],
                   c=colours[k % len(colours)], s=150, zorder=5,
                   edgecolors="k", linewidths=1.5,
                   marker=markers[k % len(markers)],
                   label=f"#{cidx}: {clabel}")
    ax.set_xlabel("Complexity (expression tree nodes)", fontsize=12)
    ax.set_ylabel("Loss (MSE)", fontsize=12)
    ax.set_title("Pareto Front: Complexity vs Accuracy", fontsize=14)
    ax.set_yscale("log")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fname = outdir / f"pareto_front_{timestamp}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {fname.name}")

def plot_prediction_diagnostics(model, X_test, y_test, predictors, outdir,
                                 timestamp, equation_idx, equation_label):
    eqs = model.equations_
    row = eqs.loc[equation_idx]
    named_eq = replace_x_with_names(str(row["equation"]), predictors)

    y_pred = model.predict(X_test, index=equation_idx)

    # v3: guard against non-finite predictions (log/sqrt domain violations)
    finite_mask = np.isfinite(y_pred)
    if not finite_mask.all():
        n_bad = int((~finite_mask).sum())
        print(f"  [WARNING] {n_bad}/{len(y_pred)} non-finite predictions "
              f"dropped for Eq #{equation_idx}")
        y_pred = y_pred[finite_mask]
        y_test_eval = y_test[finite_mask]
    else:
        y_test_eval = y_test

    if len(y_pred) < 10:
        print(f"  [ERROR] Too few finite predictions ({len(y_pred)}) for "
              f"Eq #{equation_idx} — skipping diagnostics.")
        return {"r2": np.nan, "rmse": np.nan, "mae": np.nan,
                "idx": equation_idx, "label": equation_label,
                "equation": named_eq, "complexity": row["complexity"],
                "mse": row["loss"], "n_nonfinite": int((~finite_mask).sum())}

    residuals = y_test_eval - y_pred
    rmse = np.sqrt(np.mean(residuals**2))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_test_eval - np.mean(y_test_eval))**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mae = np.mean(np.abs(residuals))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    display_eq = named_eq if len(named_eq) <= 80 else named_eq[:77] + "..."
    fig.suptitle(f"Eq #{equation_idx} ({equation_label})\n"
                 f"CH4 = {display_eq}", fontsize=11, fontfamily="monospace",
                 y=1.02)

    # 1:1 scatter
    ax = axes[0]
    ax.scatter(y_test_eval, y_pred, alpha=0.3, s=10, c="steelblue")
    lims = [min(y_test_eval.min(), y_pred.min()),
            max(y_test_eval.max(), y_pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1, label="1:1 line")
    ax.set_xlabel("Observed CH4 (mg/m2/h)", fontsize=11)
    ax.set_ylabel("Predicted CH4 (mg/m2/h)", fontsize=11)
    ax.set_title(f"R² = {r2:.3f}, RMSE = {rmse:.2f}", fontsize=12)
    ax.legend(); ax.grid(True, alpha=0.3)

    # Residuals vs fitted
    ax = axes[1]
    ax.scatter(y_pred, residuals, alpha=0.3, s=10, c="coral")
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Predicted CH4 (mg/m2/h)", fontsize=11)
    ax.set_ylabel("Residual (mg/m2/h)", fontsize=11)
    ax.set_title("Residuals vs Fitted", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Residual histogram
    ax = axes[2]
    ax.hist(residuals, bins=50, color="mediumpurple", edgecolor="k", alpha=0.7)
    ax.axvline(0, color="k", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (mg/m2/h)", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(f"Residual Distribution\nMAE = {mae:.2f}", fontsize=12)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = outdir / f"diagnostics_eq{equation_idx}_{timestamp}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {fname.name}")

    n_nonfinite = int((~finite_mask).sum())
    return {"r2": r2, "rmse": rmse, "mae": mae, "idx": equation_idx,
            "label": equation_label, "equation": named_eq,
            "complexity": row["complexity"], "mse": row["loss"],
            "n_nonfinite": n_nonfinite}

def plot_comparison_table(all_metrics, outdir, timestamp):
    # Filter out any candidates that failed diagnostics entirely
    valid = [m for m in all_metrics if m is not None and np.isfinite(m["r2"])]
    if not valid:
        print("  [WARNING] No valid candidates to compare.")
        return
    fig, ax = plt.subplots(figsize=(16, 2 + len(valid) * 0.8))
    ax.axis("off")
    headers = ["#", "Label", "Cmplx", "Equation", "R²", "RMSE", "MAE"]
    table_data = []
    for m in valid:
        eq_short = (m["equation"] if len(m["equation"]) <= 60
                    else m["equation"][:57] + "...")
        table_data.append([
            f"#{m['idx']}", m["label"], f"{m['complexity']:.0f}",
            eq_short, f"{m['r2']:.3f}", f"{m['rmse']:.2f}", f"{m['mae']:.2f}"
        ])
    table = ax.table(cellText=table_data, colLabels=headers,
                     colWidths=[0.03, 0.17, 0.05, 0.45, 0.06, 0.06, 0.06],
                     loc="center", cellLoc="left")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#2E5E86")
        cell.set_text_props(color="white", fontweight="bold")
    best_r2_row = max(range(len(valid)), key=lambda i: valid[i]["r2"])
    for j in range(len(headers)):
        table[best_r2_row + 1, j].set_facecolor("#D1FAE5")
    ax.set_title("Candidate Equation Comparison", fontsize=14,
                 fontweight="bold", pad=20)
    plt.tight_layout()
    fname = outdir / f"equation_comparison_{timestamp}.png"
    fig.savefig(fname, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {fname.name}")

# ─── Report export ────────────────────────────────────────────────────────────

def export_equations(model, predictors, outdir, timestamp, all_metrics):
    eqs = model.equations_
    eqs_export = eqs.copy()
    eqs_export["equation_named"] = eqs_export["equation"].apply(
        lambda eq: replace_x_with_names(str(eq), predictors))
    csv_path = outdir / f"pareto_equations_{timestamp}.csv"
    eqs_export.to_csv(csv_path, index=True)
    print(f"  [SAVED] {csv_path.name}")

    txt_path = outdir / f"equation_report_{timestamp}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 90 + "\n")
        f.write("SYMBOLIC REGRESSION - EQUATION DISCOVERY REPORT (v4)\n")
        f.write(f"Generated: {get_readable_timestamp()}\n")
        f.write(f"Target: {TARGET_COL}\n")
        f.write(f"Predictors: {', '.join(predictors)}\n")
        f.write(f"PySR iterations: {NITERATIONS}, Max complexity: {MAXSIZE}\n")
        f.write(f"Nested constraints: {NESTED_CONSTRAINTS if NESTED_CONSTRAINTS else 'None'}\n")
        f.write(f"Arg complexity constraints: {CONSTRAINTS if CONSTRAINTS else 'None'}\n")
        f.write("=" * 90 + "\n\n")

        f.write("VARIABLE MAPPING\n" + "-" * 40 + "\n")
        for i, name in enumerate(predictors):
            f.write(f"  x{i}  =  {name}\n")

        f.write("\nPARETO FRONT (ranked by complexity)\n" + "-" * 90 + "\n\n")
        for idx, row in eqs.iterrows():
            named_eq = replace_x_with_names(str(row["equation"]), predictors)
            f.write(f"  Equation #{idx}  [complexity = {row['complexity']:.0f}]\n")
            f.write(f"    MSE   : {row['loss']:.4f}\n")
            if "score" in row:
                f.write(f"    Score : {row['score']:.6f}\n")
            f.write(f"    CH4 = {named_eq}\n\n")

        f.write("\n" + "=" * 90 + "\n")
        f.write("CANDIDATE EQUATIONS - DETAILED COMPARISON\n")
        f.write("=" * 90 + "\n\n")

        valid = [m for m in all_metrics
                 if m is not None and np.isfinite(m.get("r2", np.nan))]

        f.write(f"  {'#':>4}  {'Label':<30}  {'Cmplx':>5}  {'R²':>7}  "
                f"{'RMSE':>8}  {'MAE':>8}  {'NaN':>4}\n")
        f.write(f"  {'-'*4}  {'-'*30}  {'-'*5}  {'-'*7}  "
                f"{'-'*8}  {'-'*8}  {'-'*4}\n")
        for m in valid:
            f.write(f"  #{m['idx']:<3d}  {m['label']:<30}  "
                    f"{m['complexity']:5.0f}  {m['r2']:7.3f}  "
                    f"{m['rmse']:8.2f}  {m['mae']:8.2f}  "
                    f"{m.get('n_nonfinite', 0):4d}\n")

        f.write("\n")
        for m in valid:
            f.write(f"\n  --- Equation #{m['idx']} ({m['label']}) ---\n")
            f.write(f"  Complexity : {m['complexity']:.0f}\n")
            f.write(f"  R²         : {m['r2']:.4f}\n")
            f.write(f"  RMSE       : {m['rmse']:.4f} mg CH4/m2/h\n")
            f.write(f"  MAE        : {m['mae']:.4f} mg CH4/m2/h\n")
            if m.get("n_nonfinite", 0) > 0:
                f.write(f"  Non-finite : {m['n_nonfinite']} predictions dropped\n")
            f.write(f"  Equation   :\n    CH4 = {m['equation']}\n")

        f.write("\n\n" + "=" * 90 + "\n")
        f.write("ORYZA INTEGRATION NOTES\n" + "=" * 90 + "\n\n")
        f.write("Variable mapping (ECS measurement -> ORYZA equivalent):\n\n")
        oryza_map = {
            "depth": "ORYZA water balance module (WL0)",
            "Tsoil": "ORYZA soil temperature module (TSOILL)",
            "Tair": "Weather input file (TMMN/TMMX)",
            "SR": "Weather input file (RDD)",
            "WS": "Weather input file (WN)",
            "VPD": "Computed from weather input (es - ea)",
            "AUC_dry": "Accumulated from ORYZA water depth (negative part)",
            "AUC_wet": "Accumulated from ORYZA water depth (positive part)",
            "hwet": "MAX(0, ORYZA water depth)",
            "h_inv": "1/(ORYZA water depth + 0.001)",
            "SR*Ts": "Weather SR x ORYZA Tsoil",
            "SR*HODsin": "Weather SR x sin(2pi x hour/24)",
            "h*sinTOD": "ORYZA depth x sin(2pi x hour/24)",
            "h*cosTOD": "ORYZA depth x cos(2pi x hour/24)",
            "h*u": "ORYZA depth x (-WS x sin(WD))",
            "h*v": "ORYZA depth x (-WS x cos(WD))",
            "h*VPD": "ORYZA depth x VPD",
        }
        for pred in predictors:
            mapped = oryza_map.get(pred,
                                   "Derive from ORYZA state or weather input")
            f.write(f"  {pred:20s} -> {mapped}\n")

        f.write("\n\nTo embed in ORYZA:\n"
                "  1. Choose candidate equation balancing accuracy and simplicity\n"
                "  2. Replace variables with ORYZA equivalents above\n"
                "  3. Code into ORYZA simulation loop\n"
                "  4. Compute CH4 flux at each timestep\n"
                "  5. Validate against EC station observations\n")
    print(f"  [SAVED] {txt_path.name}")

# ─── Main driver ──────────────────────────────────────────────────────────────

def run_symbolic_regression(csv_path, out_root, run_ts, seed=None, outdir=None):
    """Run a single PySR discovery + diagnostics pass.

    Args:
      csv_path: Path to input CSV.
      out_root: Root directory for the run (for legacy single-seed path).
      run_ts:   Run timestamp string.
      seed:     PySR random_state. Defaults to module-level RANDOM_STATE.
      outdir:   Output directory for this seed's outputs. If None, falls back
                to out_root / <csv_stem> / "symbolic_regression" (single-seed
                legacy behavior).

    Returns:
      dict with keys {seed, eqs, candidates, all_metrics, predictors, outdir}
      on success, or None on failure (e.g., no equations found).
    """
    from pysr import PySRRegressor
    from sklearn.model_selection import train_test_split

    if seed is None:
        seed = RANDOM_STATE

    print(f"\n{'=' * 70}")
    print(f"SYMBOLIC REGRESSION - EQUATION DISCOVERY (v4)")
    print(f"{'=' * 70}")
    print(f"  Input: {csv_path.name}  |  Started: {get_readable_timestamp()}")

    df = pd.read_csv(csv_path)
    print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
    df = clean_missing_flags(df, MISSING_FLAGS)

    # Coerce object-dtype columns to numeric where possible.
    # Stray non-numeric values (e.g. "NA", ".", "#VALUE!") become NaN.
    obj_cols = df.select_dtypes(include=["object"]).columns
    coerced = []
    for col in obj_cols:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Only accept the conversion if at least some values survived
        if converted.notna().sum() > 0:
            df[col] = converted
            coerced.append(col)
    if coerced:
        print(f"  Coerced to numeric: {coerced}")
        # Re-clean missing flags in newly-numeric columns
        df = clean_missing_flags(df, MISSING_FLAGS)

    drop_cols = [c for c in EXCLUDE_HEADERS if c in df.columns]

    # v6: capture date column for day-grouped 80/20 BEFORE dropping it.
    # v6.1: use a dual-attempt parser (dayfirst=False AND True) and pick the
    # one with the higher parse rate. Fixes silent fallback on DD/MM/YYYY
    # CSVs (e.g. Filipino convention) where dayfirst=False only parses ~40%
    # because rows with day > 12 fail.
    dates_full = None
    date_col_used = None
    for cand in ("Date", "time", "Deltime"):
        if cand not in df.columns:
            continue
        attempts = []
        for df_flag in (False, True):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(df[cand], errors="coerce",
                                            dayfirst=df_flag, format="mixed")
            except Exception:
                continue
            valid = int(parsed.notna().sum())
            pv = parsed.dropna()
            diffs = pv.diff().dropna()
            mono = (float((diffs >= pd.Timedelta(0)).sum()) /
                    max(1, len(diffs)))
            attempts.append({
                "dayfirst": df_flag, "parsed": parsed,
                "valid": valid, "mono": mono,
            })
        if not attempts:
            continue
        # Prefer monotonic + complete parse.
        best = max(attempts, key=lambda c: (c["mono"], c["valid"]))
        if best["valid"] >= 0.95 * len(df):
            dates_full = best["parsed"]
            date_col_used = cand
            print(f"  Date column for day-grouped split: '{cand}' "
                  f"(dayfirst={best['dayfirst']}, parsed "
                  f"{best['valid']:,}/{len(df):,}, "
                  f"monotonic={best['mono']:.1%})")
            for a in attempts:
                if a is not best:
                    print(f"    (rejected dayfirst={a['dayfirst']}: "
                          f"parsed {a['valid']:,}, mono={a['mono']:.1%})")
            break
        else:
            print(f"  [INFO] '{cand}' best parse only "
                  f"{best['valid']:,}/{len(df):,} "
                  f"(dayfirst={best['dayfirst']}); trying next candidate.")
    if date_col_used is None:
        print(f"  [WARNING] No parseable date column found; "
              f"falling back to record-level 80/20 split.")

    if drop_cols:
        print(f"  Dropping excluded columns: {drop_cols}")
        df = df.drop(columns=drop_cols)
    print(f"  Remaining columns: {list(df.columns)}")

    if TARGET_COL not in df.columns:
        print(f"  [ERROR] Target '{TARGET_COL}' not found.")
        return

    predictors = [c for c in df.columns
                  if c != TARGET_COL and pd.api.types.is_numeric_dtype(df[c])]
    print(f"  Predictors: {len(predictors)}")
    for i, name in enumerate(predictors):
        print(f"    x{i} = {name}")

    if len(predictors) == 0:
        print(f"  [ERROR] No predictors found. All columns were either "
              f"non-numeric, the target, or in EXCLUDE_HEADERS.")
        print(f"  Check that EXCLUDE_HEADERS does not contain your "
              f"retained predictor names.")
        return

    sub = df[predictors + [TARGET_COL]].dropna()
    print(f"  Complete cases: {len(sub):,} / {len(df):,}")
    if len(sub) < 50:
        print("  [ERROR] Too few complete cases.")
        return

    X = sub[predictors].values
    y = sub[TARGET_COL].values

    # v6: day-grouped 80/20. Sample TEST_FRACTION of *days*, not records.
    # Removes within-day autocorrelation leakage from candidate ranking.
    # v6.2: build the test mask via pandas Series.isin() rather than
    # np.isin(set(...tolist())) — the latter silently fails on Windows /
    # older numpy where datetime64[ns].tolist() returns ints and the
    # subsequent comparison against a datetime64 array returns all False
    # (yielding train=ALL, test=0).
    if dates_full is not None:
        sub_dates = dates_full.loc[sub.index]
        valid = sub_dates.notna().values
        if valid.sum() < len(sub):
            n_drop = int(len(sub) - valid.sum())
            print(f"  Dropping {n_drop} rows with unparseable dates "
                  f"from PySR fit data.")
            X = X[valid]
            y = y[valid]
            sub_dates = sub_dates[valid]
        days_series = sub_dates.dt.normalize()  # pandas Series, day-resolution
        unique_days_pd = pd.DatetimeIndex(days_series.drop_duplicates())
        rng = np.random.RandomState(seed)
        perm_idx = rng.permutation(len(unique_days_pd))
        n_test_days = int(round(len(unique_days_pd) * TEST_FRACTION))
        n_test_days = max(1, min(len(unique_days_pd) - 1, n_test_days))
        test_days_pd = unique_days_pd[perm_idx[:n_test_days]]
        test_mask = days_series.isin(test_days_pd).values
        train_mask = ~test_mask
        # Sanity check: both splits must be non-empty.
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            raise RuntimeError(
                f"Day-grouped split produced empty fold "
                f"(train={train_mask.sum()}, test={test_mask.sum()}). "
                f"This indicates a date dtype mismatch — please report.")
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        print(f"  Day-grouped 80/20: {len(unique_days_pd)} days "
              f"-> {len(unique_days_pd) - n_test_days} train / {n_test_days} test")
        print(f"  Records:  train={len(X_train):,}  |  test={len(X_test):,}")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_FRACTION, random_state=seed)
        print(f"  Record-level 80/20: train={len(X_train):,}  "
              f"|  test={len(X_test):,}")
    print(f"  Seed:  {seed}")

    if outdir is None:
        outdir = out_root / safe_name(csv_path.stem) / "symbolic_regression"
    outdir.mkdir(parents=True, exist_ok=True)

    _base_kwargs = dict(
        niterations=NITERATIONS,
        maxsize=MAXSIZE,
        binary_operators=BINARY_OPERATORS,
        unary_operators=UNARY_OPERATORS,
        nested_constraints=(NESTED_CONSTRAINTS
                            if NESTED_CONSTRAINTS else None),
        constraints=(CONSTRAINTS if CONSTRAINTS else None),
        extra_sympy_mappings=(EXTRA_SYMPY_MAPPINGS
                              if EXTRA_SYMPY_MAPPINGS else None),
        populations=20,
        population_size=POPULATION_SIZE,
        parsimony=PARSIMONY,
        batching=BATCHING,
        batch_size=BATCH_SIZE,
        progress=True,
        verbosity=1,
        random_state=seed,
        deterministic=DETERMINISTIC,
        tempdir=str(outdir),
        variable_names=predictors,
        # Optional robust loss instead of default MSE (flux outliers /
        # heteroscedasticity). Verify the kwarg name for YOUR PySR version
        # before enabling — newer PySR: elementwise_loss=...; older: loss=...
        # elementwise_loss="loss(pred, target) = abs(pred - target)",  # L1
    )

    # Try equation_file first (PySR ≥ 0.16); fall back to temp_equation_file
    try:
        model = PySRRegressor(
            equation_file=str(outdir / "hall_of_fame.csv"),
            **_base_kwargs,
        )
    except TypeError:
        model = PySRRegressor(
            temp_equation_file=True,
            **_base_kwargs,
        )

    print(f"\n  Running PySR...")
    print(f"  Press 'q' then Enter to stop early.\n")
    # Optional weighted loss: PySR's fit accepts weights= (array aligned to
    # X_train). Derive e.g. from per-point flux random error or u*-filtering
    # confidence, then call: model.fit(X_train, y_train, weights=w_train)
    model.fit(X_train, y_train)

    eqs = model.equations_
    n_eqs = len(eqs) if eqs is not None else 0
    print(f"\n{'=' * 70}\nRESULTS\n{'=' * 70}")
    print(f"  Found {n_eqs} equations.\n")
    if n_eqs == 0:
        print("  [ERROR] No equations. Increase NITERATIONS.")
        return

    n_show = min(N_DISPLAY, n_eqs)
    print(f"  {'#':>3} | {'Cmplx':>5} | {'MSE':>10} | {'Score':>8} | Equation")
    print(f"  {'-'*3}-+-{'-'*5}-+-{'-'*10}-+-{'-'*8}-+-{'-'*50}")
    for idx, row in eqs.head(n_show).iterrows():
        score = row.get("score", 0)
        named_eq = replace_x_with_names(str(row["equation"]), predictors)
        eq_display = (named_eq if len(named_eq) <= 60
                      else named_eq[:57] + "...")
        print(f"  {idx:3d} | {row['complexity']:5.0f} | "
              f"{row['loss']:10.4f} | {score:8.4f} | {eq_display}")

    candidates = select_candidate_equations(eqs)
    print(f"\n  Selected {len(candidates)} candidates:\n")
    for cidx, clabel in candidates:
        named = replace_x_with_names(str(eqs.loc[cidx, "equation"]),
                                     predictors)
        print(f"    #{cidx} ({clabel})")
        print(f"       CH4 = {named}\n")

    plot_pareto_front(model, candidates, outdir, run_ts)

    all_metrics = []
    for cidx, clabel in candidates:
        print(f"\n  Evaluating #{cidx} ({clabel})...")
        metrics = plot_prediction_diagnostics(
            model, X_test, y_test, predictors, outdir, run_ts,
            equation_idx=cidx, equation_label=clabel)
        all_metrics.append(metrics)
        if metrics is not None and np.isfinite(metrics["r2"]):
            print(f"    R²={metrics['r2']:.3f} | "
                  f"RMSE={metrics['rmse']:.2f} | MAE={metrics['mae']:.2f}")
        else:
            print(f"    [SKIPPED] — insufficient finite predictions")

    valid_metrics = [m for m in all_metrics
                     if m is not None and np.isfinite(m.get("r2", np.nan))]

    print(f"\n  {'='*70}\n  CANDIDATE COMPARISON\n  {'='*70}")
    print(f"  {'#':>4}  {'Label':<30}  {'Cmplx':>5}  {'R²':>7}  "
          f"{'RMSE':>8}  {'MAE':>8}")
    print(f"  {'-'*4}  {'-'*30}  {'-'*5}  {'-'*7}  {'-'*8}  {'-'*8}")
    for m in valid_metrics:
        print(f"  #{m['idx']:<3d}  {m['label']:<30}  "
              f"{m['complexity']:5.0f}  {m['r2']:7.3f}  "
              f"{m['rmse']:8.2f}  {m['mae']:8.2f}")

    plot_comparison_table(all_metrics, outdir, run_ts)
    export_equations(model, predictors, outdir, run_ts, all_metrics)

    # NOTE: Stage C generation has moved to run_multiseed_pipeline so that
    # selection rules can be applied across all seeds before generating one
    # combined stage8 script. See run_multiseed_pipeline below.

    # Save the model as a .pkl cache (optional; hall_of_fame.csv is canonical).
    if SAVE_PICKLE:
        try:
            model_path = outdir / f"pysr_model_{run_ts}.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            print(f"  [SAVED] {model_path.name}")
        except Exception as e:
            print(f"  [WARNING] Could not save model: {e}")

    print(f"\n  All outputs: {outdir}")
    print(f"  Done: {get_readable_timestamp()}")

    return {
        "seed":        seed,
        "eqs":         eqs,
        "candidates":  candidates,
        "all_metrics": all_metrics,
        "predictors":  predictors,
        "outdir":      outdir,
    }

def run_multiseed_pipeline(csv_path, out_root, run_ts, n_seeds, base_seed=42):
    """Run PySR n_seeds times with consecutive seeds. Apply Rule A (pick the
    run whose single best candidate has the highest 80/20 R²) and feed that
    run's 4 candidates into Stage 8. Also produce a cross-seed sensitivity
    table (per-category mean ± std across all seeds) for the methods section.

    Selection rule:
      Rule A — pick the run whose SINGLE BEST candidate has the highest
               80/20 test R² across all runs. Take that run's full set of
               4 candidates into Stage 8. Origin tag: "ruleA".

    Cross-seed sensitivity table (Option 3):
      For each candidate category (auto-best/knee/mid/complex) compute
      mean ± std of 80/20 R² across the n_seeds runs. Saved to
      seed_sensitivity_summary_<run_ts>.txt for the paper's methods
      section. NOT fed to Stage 8 — equations don't average.

    The generated stage8 script is written to:
        out_root / safe_name(csv_path.stem) / "symbolic_regression"

    Per-seed PySR outputs (plots, exports, model pickles) live in
    seed-specific subdirectories so nothing overwrites.
    """
    print(f"\n{'#' * 80}")
    print(f"# MULTI-SEED PIPELINE  ({n_seeds} seeds: "
          f"{base_seed}..{base_seed + n_seeds - 1})")
    print(f"# Selection rule: A = run with max(R²) across its 4 candidates")
    print(f"# Cross-seed sensitivity: per-category mean ± std reported")
    print(f"{'#' * 80}\n")

    sr_root = out_root #/ safe_name(csv_path.stem) / "symbolic_regression"
    sr_root.mkdir(parents=True, exist_ok=True)

    # Run all seeds.
    seed_results = []
    for i in range(n_seeds):
        seed = base_seed + i
        print(f"\n{'#' * 80}\n# SEED {seed}  ({i+1}/{n_seeds})\n{'#' * 80}")
        seed_outdir = sr_root / f"seed_{seed}"
        result = run_symbolic_regression(
            csv_path, out_root, run_ts, seed=seed, outdir=seed_outdir)
        if result is None:
            print(f"  [WARN] Seed {seed} produced no result; skipping.")
            continue
        seed_results.append(result)

    if not seed_results:
        print(f"\n[ERROR] No seeds produced results. Aborting Stage C.")
        return

    # Per-run summary: max(R²) across that run's candidates (for Rule A).
    summaries = []
    for r in seed_results:
        r2s = [m["r2"] for m in r["all_metrics"]
               if m is not None and np.isfinite(m.get("r2", np.nan))]
        if not r2s:
            continue
        summaries.append({
            "seed":   r["seed"],
            "max_r2": max(r2s),
            "result": r,
        })

    if not summaries:
        print(f"\n[ERROR] No seed produced any finite R²; aborting Stage C.")
        return

    # Rule A.
    rule_a_winner = max(summaries, key=lambda s: s["max_r2"])

    # ── Cross-seed sensitivity table (Option 3) ──────────────────────────
    # For each candidate label, collect R²s across all seeds and compute
    # mean ± std. Equations do not average — this is a methods-section
    # statistic, not a selection rule.
    by_label = {}  # {label: [(seed, r2, complexity), ...]}
    for r in seed_results:
        for m in r["all_metrics"]:
            if m is None or not np.isfinite(m.get("r2", np.nan)):
                continue
            by_label.setdefault(m["label"], []).append(
                (r["seed"], m["r2"], m.get("complexity", float("nan"))))

    sensitivity_rows = []
    for label, entries in by_label.items():
        r2s = np.array([e[1] for e in entries])
        sensitivity_rows.append({
            "label":   label,
            "n_seeds": len(entries),
            "mean":    float(r2s.mean()),
            "std":     float(r2s.std(ddof=1)) if len(r2s) > 1 else float("nan"),
            "min":     float(r2s.min()),
            "max":     float(r2s.max()),
        })

    # Print + persist the selection summary.
    print(f"\n{'#' * 80}")
    print(f"# SEED SELECTION SUMMARY  (Rule A)")
    print(f"{'#' * 80}")
    print(f"  {'seed':>5}  {'max(R²)':>10}  notes")
    print(f"  {'-'*5}  {'-'*10}  {'-'*30}")
    for s in summaries:
        notes = "  <-- Rule A winner" if s["seed"] == rule_a_winner["seed"] else ""
        print(f"  {s['seed']:>5}  {s['max_r2']:>+10.4f}  {notes}")

    print(f"\n{'#' * 80}")
    print(f"# CROSS-SEED SENSITIVITY  (per category, R² across seeds)")
    print(f"{'#' * 80}")
    print(f"  {'category':<32}  n   {'mean':>8}  {'std':>7}  "
          f"{'min':>7}  {'max':>7}")
    for row in sensitivity_rows:
        print(f"  {row['label']:<32}  {row['n_seeds']:>1}  "
              f"{row['mean']:>+8.4f}  {row['std']:>+7.4f}  "
              f"{row['min']:>+7.4f}  {row['max']:>+7.4f}")

    sel_path = sr_root / f"seed_selection_summary_{run_ts}.txt"
    with open(sel_path, "w", encoding="utf-8") as f:
        f.write(f"Multi-seed selection summary (Rule A only)\n")
        f.write(f"Run timestamp: {run_ts}\n")
        f.write(f"Seeds run: {[s['seed'] for s in summaries]}\n\n")
        f.write(f"  {'seed':>5}  {'max(R²)':>10}  notes\n")
        f.write(f"  {'-'*5}  {'-'*10}  {'-'*30}\n")
        for s in summaries:
            notes = "  <-- Rule A winner" if s["seed"] == rule_a_winner["seed"] else ""
            f.write(f"  {s['seed']:>5}  {s['max_r2']:>+10.4f}  {notes}\n")
        f.write(f"\nRule A: pick run whose single highest-R² candidate "
                f"is highest across runs.\n")
        f.write(f"Selected seed for Stage 8: {rule_a_winner['seed']}\n")
    print(f"\n  [SAVED] {sel_path.name}")

    sens_path = sr_root / f"seed_sensitivity_summary_{run_ts}.txt"
    with open(sens_path, "w", encoding="utf-8") as f:
        f.write(f"Cross-seed sensitivity summary\n")
        f.write(f"Run timestamp: {run_ts}\n")
        f.write(f"Per-category R² (80/20 hold-out) across "
                f"{len(seed_results)} seeds.\n\n")
        f.write(f"  {'category':<32}  n   {'mean':>8}  {'std':>7}  "
                f"{'min':>7}  {'max':>7}\n")
        for row in sensitivity_rows:
            f.write(f"  {row['label']:<32}  {row['n_seeds']:>1}  "
                    f"{row['mean']:>+8.4f}  {row['std']:>+7.4f}  "
                    f"{row['min']:>+7.4f}  {row['max']:>+7.4f}\n")
        f.write(f"\nNote: equations do not average. These statistics "
                f"summarize PySR's seed-to-seed variability for the\n"
                f"methods section. Selection (Rule A) drives Stage 8; "
                f"these means do NOT determine which equations are\n"
                f"validated by day-grouped CV.\n")
    print(f"  [SAVED] {sens_path.name}")

    # Build forms for stage8 from Rule A's winning run.
    predictors = rule_a_winner["result"]["predictors"]
    forms = _build_forms_from_run(
        rule_a_winner["result"]["candidates"],
        rule_a_winner["result"]["eqs"],
        predictors, rule_a_winner["seed"], origin="ruleA")

    print(f"\n  Generating Stage 8 CV script (auto)...")
    if forms:
        _write_auto_stage8(forms, predictors, csv_path, sr_root, run_ts)
    else:
        print(f"  [WARNING] No forms could be sympified; "
              f"stage8 generator skipped.")


def main():
    input_dir = Path(INPUT_DIR)
    run_ts = get_timestamp()
    out_root = Path(OUTPUT_DIR) / f"run_{run_ts}"
    out_root.mkdir(parents=True, exist_ok=True)

    if INPUT_FILE:
        csv_path = input_dir / INPUT_FILE
    else:
        csv_files = sorted(input_dir.glob("*.csv"))
        if not csv_files:
            print(f"No CSV files in: {input_dir}")
            return
        print("Available CSV files:")
        for i, f in enumerate(csv_files):
            print(f"  [{i}] {f.name}")
        choice = int(input("Select file number: "))
        csv_path = csv_files[choice]

    if not csv_path.is_file():
        print(f"Not found: {csv_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"SYMBOLIC REGRESSION FOR RICE METHANE (v4)")
    print(f"{'=' * 80}")
    print(f"  Timestamp : {run_ts}")
    print(f"  Input     : {csv_path}")
    print(f"  Output    : {out_root}")
    print(f"  Target    : {TARGET_COL}")
    print(f"  Iters     : {NITERATIONS}")
    print(f"  MaxSize   : {MAXSIZE}")
    print(f"  Operators : {BINARY_OPERATORS + UNARY_OPERATORS}")
    print(f"  Parsimony : {PARSIMONY}")
    print(f"  N seeds   : {N_SEEDS}")
    print(f"  Determ.   : {DETERMINISTIC}  (False = statistical reproducibility, not bit-identical)")
    print(f"  Nesting   : {NESTED_CONSTRAINTS if NESTED_CONSTRAINTS else 'unconstrained'}")
    print(f"  Arg limits: {CONSTRAINTS if CONSTRAINTS else 'unconstrained'}")
    print("=" * 80 + "\n")

    if N_SEEDS <= 1:
        # Single-seed legacy path. Stage C runs once on this seed's results.
        result = run_symbolic_regression(csv_path, out_root, run_ts,
                                         seed=RANDOM_STATE)
        if result is not None:
            sr_root = result["outdir"]
            forms = _build_forms_from_run(
                result["candidates"], result["eqs"], result["predictors"],
                result["seed"], origin="topR2")
            if forms:
                _write_auto_stage8(forms, result["predictors"],
                                   csv_path, sr_root, run_ts)
    else:
        run_multiseed_pipeline(csv_path, out_root, run_ts, N_SEEDS,
                               base_seed=RANDOM_STATE)

    print(f"\n{'=' * 80}")
    print(f"Done. {get_readable_timestamp()}")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()