"""
10_sobol.py -- variance-based (Sobol) driver sensitivity of the four equations
==============================================================================
REPLACES the old Sobol analysis, which was computed on the superseded June
equations. Everything here is evaluated on the OPERATIVE equations of the
August 2026 runs, at their PUBLISHED coefficients, no refitting:

  Mase      complexity 22   run_20260805_122821  seed 49
  Cheorwon  complexity 40   run_20260805_170113  seed 45
  IRRI      complexity  8   run_20260806_104044  seed 53   (post-screen form)
  POOLED    complexity 40   run_20260810_111532  seed 45

WHAT IT COMPUTES, per arm

  1. GIVEN-DATA first-order index  S_i = Var(E[F|x_i]) / Var(F),
     estimated by quantile-binning x_i over the OBSERVED rows. Real driver
     correlations are retained, so these indices need not sum to 1.

  2. INDEPENDENT-INPUT Sobol indices S_i and S_Ti (Jansen 1999 / Saltelli 2010
     estimators) by Monte-Carlo sampling each input UNIFORMLY over its observed
     [min, max], inputs independent. This is the "footprint" of each driver on
     the equation's output over its observed range, correlations removed.

     The Mase and IRRI equations have poles (a denominator can cross zero when
     inputs are sampled independently). Non-finite and beyond-observed-output
     samples are masked, and the kept fraction is reported. If kept < 90% the
     independent indices should be quoted with that caveat; the given-data
     indices are unaffected, because observed rows do not reach the poles
     (Mase: 4 of 6,762 rows).

USAGE
    python 10_sobol.py                 # all four arms
    python 10_sobol.py MASE POOLED     # named arms only

OUTPUT (next to this script's OUT folder)
    sobol_results_<ARM>_<ts>.csv       one row per input:
                                       given_data_Si, sobol_Si, sobol_STi
    sobol_sensitivity_<ts>.png         four-panel bar chart for the paper

Author: Jef Zerrudo / Claude.  Requires numpy, pandas, matplotlib.
==============================================================================
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGET_COL = "F_CH4_F"
MISSING_FLAGS = [-9999, -999900, -99999]
N_BINS = 50            # quantile bins for the given-data index
N_MC = 2 ** 14         # Monte-Carlo base sample per matrix (A and B)
SEED = 42

BASE = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV"
OUT = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\SOBOL"


# -- the four equations, verbatim published coefficients ---------------------
def eq_mase(v):
    z = 2.1130118e-7 * v["AUC_wet"] * v["Tv_C"] ** 2
    return np.exp(z) - (v["u*VPD"] / v["VPD*WS"]) / (3.0412269 - z)


def eq_cheorwon(v):
    inner = ((v["dayhr"] * np.log(v["Tv_C"])) + (-9.83549 / v["h_inv"])) \
        + ((v["SR*u"] * -0.019528149)
           - (((v["AUC_wet"] - v["SR*u"]) + (v["SR*HODsin"] * -7.484453))
              / (v["AUC_wet"] - (v["h_inv"] * 6.249428))))
    return (((v["AUC_wet"] - v["SR*v"]) + v["SR"]) * 6.3612692e-6) \
        * (np.exp(np.sqrt(v["Tv_C"])) - inner) + 0.4964769


def eq_irri(v):
    return -1676.962 / (np.sqrt(v["SR*Ts"]) + (-216.37347 - v["AUC_dry"]))


def eq_pooled(v):
    inner = (((v["SR*HODsin"] + (v["SR*v"] / 1.6380328))
              + (((v["h*v"] - v["h*sinTOD"]) - (v["hbar_wet"] * v["Tv_C"])) * v["Tv_C"]))
             * -1.7110912e-7) * (v["hbar_wet"] - 5.6869154)
    return (v["hbar_wet"] - 0.40055007) \
        * ((((v["h_inv"] * 0.00034019744) - (v["Tv_C"] * (v["Tv_C"] * inner)))
            * v["hbar_wet"]) + np.exp(v["VPD*WS"] * -39.338554))


ARMS = {
    "MASE": dict(
        csv=os.path.join(BASE, r"JPN\Data-Metadata\JPN_retvars_pass2_C.csv"),
        fn=eq_mase, inputs=["AUC_wet", "Tv_C", "u*VPD", "VPD*WS"],
        label="Mase, complexity 22"),
    "CHEORWON": dict(
        csv=os.path.join(BASE, r"KOR\Data_Metadata\Papale_hampel_cleaned\KOR_retvars_pass2_C.csv"),
        fn=eq_cheorwon,
        inputs=["AUC_wet", "SR", "SR*v", "SR*u", "SR*HODsin", "Tv_C", "dayhr", "h_inv"],
        label="Cheorwon, complexity 40"),
    "IRRI": dict(
        csv=os.path.join(BASE, r"PHL\Data_Metadata\PHL_retvars_pass2_C.csv"),
        fn=eq_irri, inputs=["SR*Ts", "AUC_dry"],
        label="IRRI, complexity 8"),
    "POOLED": dict(
        csv=os.path.join(BASE, r"POOLED\Data-Metadata\POOLED_retvars_pass2_ISO_C.csv"),
        fn=eq_pooled,
        inputs=["hbar_wet", "Tv_C", "h_inv", "SR*HODsin", "SR*v", "h*v",
                "h*sinTOD", "VPD*WS"],
        label="Pooled, complexity 40"),
}


def load(csv, inputs):
    d = pd.read_csv(csv, low_memory=False)
    for c in inputs:
        if c not in d.columns:
            raise SystemExit(f"  [STOP] column '{c}' not in {csv}")
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[inputs].replace(MISSING_FLAGS, np.nan).dropna().reset_index(drop=True)
    return d


def given_data_Si(fn, d, inputs):
    """Var(E[F|x_i])/Var(F) by quantile binning on the observed rows."""
    v = {c: d[c].to_numpy(float) for c in inputs}
    F = np.asarray(fn(v), dtype=float)
    ok = np.isfinite(F)
    F = F[ok]
    varF = F.var()
    out = {}
    for c in inputs:
        x = d[c].to_numpy(float)[ok]
        # rank -> quantile bins; duplicates collapse harmlessly
        q = pd.qcut(pd.Series(x).rank(method="first"), N_BINS, labels=False)
        m = pd.Series(F).groupby(q).mean()
        n = pd.Series(F).groupby(q).size()
        out[c] = float(np.average((m - F.mean()) ** 2, weights=n) / varF)
    return out, float(varF), int((~ok).sum())


def sobol_indices(fn, d, inputs, rng):
    """Jansen estimators on uniform-in-observed-range independent inputs."""
    lo = {c: d[c].min() for c in inputs}
    hi = {c: d[c].max() for c in inputs}

    def sample(n):
        return {c: rng.uniform(lo[c], hi[c], n) for c in inputs}

    A, B = sample(N_MC), sample(N_MC)
    fA = np.asarray(fn(A), dtype=float)
    fB = np.asarray(fn(B), dtype=float)

    # mask non-finite AND wild values beyond the observed output range x10,
    # so a near-pole sample cannot dominate the variance
    vobs = {c: d[c].to_numpy(float) for c in inputs}
    Fobs = np.asarray(fn(vobs), dtype=float)
    Fobs = Fobs[np.isfinite(Fobs)]
    lim = 10 * max(abs(Fobs.min()), abs(Fobs.max()))

    keep = np.isfinite(fA) & np.isfinite(fB) & (np.abs(fA) < lim) & (np.abs(fB) < lim)
    fAB = {}
    for c in inputs:
        AB = {k: A[k].copy() for k in inputs}
        AB[c] = B[c]
        f = np.asarray(fn(AB), dtype=float)
        fAB[c] = f
        keep &= np.isfinite(f) & (np.abs(f) < lim)

    fA, fB = fA[keep], fB[keep]
    n = keep.sum()
    varY = np.concatenate([fA, fB]).var()
    Si, STi = {}, {}
    for c in inputs:
        f = fAB[c][keep]
        Si[c] = float((varY - 0.5 * np.mean((fB - f) ** 2)) / varY)
        STi[c] = float(0.5 * np.mean((fA - f) ** 2) / varY)
    return Si, STi, float(n / len(keep))


def main():
    wanted = [a.upper() for a in sys.argv[1:]] or list(ARMS)
    bad = [a for a in wanted if a not in ARMS]
    if bad:
        raise SystemExit(f"unknown arm(s) {bad}; choose from {list(ARMS)}")
    os.makedirs(OUT, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    rng = np.random.RandomState(SEED)

    fig, axes = plt.subplots(1, len(wanted), figsize=(4.2 * len(wanted), 4.4))
    if len(wanted) == 1:
        axes = [axes]

    for ax, arm in zip(axes, wanted):
        cfg = ARMS[arm]
        print("=" * 74)
        print(f"  {arm}   {cfg['label']}")
        print("=" * 74)
        if not os.path.isfile(cfg["csv"]):
            print(f"  [SKIP] not found: {cfg['csv']}\n")
            continue
        d = load(cfg["csv"], cfg["inputs"])
        print(f"  {len(d):,} complete rows, {len(cfg['inputs'])} equation inputs")

        gsi, varF, nbad = given_data_Si(cfg["fn"], d, cfg["inputs"])
        Si, STi, kept = sobol_indices(cfg["fn"], d, cfg["inputs"], rng)
        print(f"  independent-input MC: kept {kept:.1%} of {N_MC:,} triples"
              + ("" if kept > 0.9 else "   [CAVEAT: pole masking above 10%]"))
        print(f"\n  {'input':<14}{'given-data Si':>15}{'Sobol Si':>12}{'Sobol STi':>12}")
        rows = []
        for c in sorted(cfg["inputs"], key=lambda c: -gsi[c]):
            print(f"  {c:<14}{gsi[c]:>15.3f}{Si[c]:>12.3f}{STi[c]:>12.3f}")
            rows.append({"arm": arm, "input": c, "given_data_Si": gsi[c],
                         "sobol_Si": Si[c], "sobol_STi": STi[c],
                         "mc_kept_fraction": kept})
        out_csv = os.path.join(OUT, f"sobol_results_{arm}_{ts}.csv")
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\n  [SAVED] {out_csv}\n")

        order = sorted(cfg["inputs"], key=lambda c: -gsi[c])
        ypos = np.arange(len(order))
        ax.barh(ypos + 0.22, [gsi[c] for c in order], height=0.2, label="given-data $S_i$")
        ax.barh(ypos, [Si[c] for c in order], height=0.2, label="Sobol $S_i$")
        ax.barh(ypos - 0.22, [STi[c] for c in order], height=0.2, label="Sobol $S_{Ti}$")
        ax.set_yticks(ypos, order, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("index")
        ax.set_title(cfg["label"], fontsize=10)
        ax.set_xlim(0, 1.05)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    out_png = os.path.join(OUT, f"sobol_sensitivity_{ts}.png")
    fig.savefig(out_png, dpi=300)
    print(f"  [SAVED] {out_png}")


if __name__ == "__main__":
    main()
