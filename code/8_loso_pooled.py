"""
8_loso_pooled.py — leave-one-site-out validation of the pooled CH4 equations
==============================================================================
REPLACES loso_pooled_validation.py, which could no longer run.

That file read COLS = ["site", "AUC", "es", "VPD", "vmerid", "h*VPD", "F_CH4_F"].
Every one of those predictors is gone from the current pooled design:

  es        eliminated at box 3 by the centred VIF
  AUC       replaced by hbar_wet when the pooled file went intensive, because a
            depth-hour total encodes season length (148, 175 and 76 days) rather
            than water
  vmerid    rejected by the union
  VPD       not retained; VPD*WS is
  h*VPD     not retained

All seven of its model functions also used tanh, which was removed from
UNARY_OPERATORS in v6.4 after it was measured 95.85% saturated at Mase.

WHY LOSO AND NOT THE CV NUMBER

Day-grouped CV inside the pool cannot tell a genuine common equation from a
pooled average, because every fold still contains all three sites. LOSO fits on
two sites and predicts the third, unseen. That is the claim the paper makes.

For scale, decomposing the pooled run's own CV predictions by site gives 0.371
at Mase, 0.553 at IRRI and 0.577 at Cheorwon for best_accuracy_ruleA. Those are
out-of-sample per record but NOT per site, since every site was in training.
Expect LOSO to be lower, most at Mase.

CANDIDATES

Six forms from the pooled run of 10 August 2026, run_20260810_111532, seed 45.
The first two recur in 12 of 12 seeds; the complexity-8 knee is the one with the
water term and a fold coefficient CoV of 0.0031.

CONVENTION, unchanged from the old script and from stage 8: one free parameter
per distinct numeric literal in the printed equation, repeated literals tied to
one parameter. Training uses inverse-frequency site weights over the two
training sites so the larger site does not dominate. Held-out metrics are
unweighted.

USAGE
    python 8_loso_pooled.py                       # uses CSV below
    python 8_loso_pooled.py <pooled csv>

Author: Jef Zerrudo / Claude.  Requires numpy, pandas, scipy.
==============================================================================
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

np.seterr(all="ignore")

CSV = (r"C:\Users\zerru001\OneDrive - Wageningen University & Research"
       r"\Paper1\RUN2\CSV\POOLED\Data-Metadata\POOLED_retvars_pass2_ISO_C.csv")
TARGET = "F_CH4_F"
SITE_COL = "site"

# Every predictor any candidate below needs.
NEEDED = ["Tv_C", "hbar_wet", "SR*v", "VPD*WS", "h_inv",
          "SR*HODsin", "h*v", "h*sinTOD"]


def pack(d):
    return tuple(d[c].to_numpy(float) for c in NEEDED)


# ── the pooled candidates, verbatim from seed 45 ────────────────────────────
def f_c04(X, a):                                    # 12/12 seeds, MSE 38.76
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return np.exp(Tv * a)


def f_c05(X, a):                                    # 12/12 seeds, MSE 33.19
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return (Tv * a) * hb


def f_c07(X, a):                                    # auto-best, MSE 26.62
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return np.exp(np.sqrt(Tv)) * (hb * a)


def f_c08(X, a):                                    # KNEE, MSE 25.73
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return hb * (np.exp(Tv * a) / Tv)


def f_c10(X, a, b):                                 # MSE 25.46
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return ((hb + a) / Tv) * np.exp(Tv * b)


def f_c21(X, a, b, c, d):                           # mid-complexity, MSE 21.60
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    return (hb - SRv * a) * ((np.exp(VW / b) - hi * c) + np.exp(-3.8196046 - Tv * d))


def f_c40(X, a, b, c, d, e, f, g):
    """Equation #27, complexity 40, the highest pooled CV at +0.596 +/- 0.028.

    Verbatim from seed 45:
      (hbar_wet - 0.40055007)
      * ( ( (h_inv*0.00034019744
             - Tv_C*(Tv_C*( ((SR*HODsin + SR*v/1.6380328
                              + ((h*v - h*sinTOD - hbar_wet*Tv_C)*Tv_C)) * -1.7110912e-7)
                            * (hbar_wet - 5.6869154) )) ) * hbar_wet )
          + exp(VPD*WS * -39.338554) )
    """
    Tv, hb, SRv, VW, hi, SRH, hv, hs = X
    inner = ((SRH + SRv / b) + (((hv - hs) - hb * Tv) * Tv)) * c
    return (hb - a) * ((((hi * d) - Tv * (Tv * (inner * (hb + e)))) * hb)
                       + np.exp(VW * f)) * g


MODELS = {
    "c04  exp(b*Tv)      12/12": (f_c04, [0.06823276]),
    "c05  a*Tv*hbar      12/12": (f_c05, [0.08884082]),
    "c07  exp(sqrt Tv)*hbar  ": (f_c07, [0.013211074]),
    "c08  hbar*exp/Tv   KNEE ": (f_c08, [0.15075056]),
    "c10  (hbar+a)/Tv*exp    ": (f_c10, [-0.43145436, 0.15555266]),
    "c21  mid-complexity     ": (f_c21, [0.0003887753, -0.039896015,
                                         -0.0012095191, -0.1589963]),
    "c40  BEST ACCURACY CV.596": (f_c40, [0.40055007, 1.6380328, -1.7110912e-07,
                                          0.00034019744, -5.6869154, -39.338554,
                                          1.0]),
}

# v2: local calibration. Fit the shape on the two training sites, then allow ONE
# multiplicative constant to be refitted on the held-out site. This separates
# "the form does not transfer" from "the form transfers but the level is
# site-specific", which is what the bias column of the first run suggested:
# shape r2 of 0.39 to 0.51 with biases of +2.8, +0.9 and -3.2.
# A calibration this cheap is what a user of the equation would actually do.
CALIBRATE = True


def r2(y, p):
    m = np.isfinite(p)
    if m.sum() < 10:
        return np.nan
    return 1 - np.sum((y[m] - p[m]) ** 2) / np.sum((y[m] - y[m].mean()) ** 2)


def shape_r2(y, p):
    """Pearson r squared: agreement in shape with bias removed. A form can
    transfer in shape and fail in level, and the two need separating."""
    m = np.isfinite(p)
    if m.sum() < 10 or np.std(p[m]) == 0:
        return np.nan
    return float(np.corrcoef(y[m], p[m])[0, 1] ** 2)


def main():
    csv = sys.argv[1] if len(sys.argv) > 1 else CSV
    if not os.path.isfile(csv):
        raise SystemExit(f"\n  [STOP] not found: {csv}\n")

    d = pd.read_csv(csv, low_memory=False)
    missing = [c for c in NEEDED + [TARGET, SITE_COL] if c not in d.columns]
    if missing:
        raise SystemExit(
            f"\n  [STOP] these columns are not in the CSV: {missing}\n"
            f"         present: {[c for c in d.columns]}\n"
            f"         The candidates below were fitted on a different design.\n")

    d = d[NEEDED + [TARGET, SITE_COL]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != SITE_COL else s)
    d = d.dropna().reset_index(drop=True)
    sites = sorted(d[SITE_COL].unique())
    print(f"  {csv}")
    print(f"  {len(d):,} complete rows   {d[SITE_COL].value_counts().to_dict()}\n")
    if len(sites) < 2:
        raise SystemExit("  [STOP] need at least two sites for LOSO.\n")

    rows = []
    for name, (fn, p0) in MODELS.items():
        losos, cals = [], []
        for hold in sites:
            tr = d[d[SITE_COL] != hold]
            te = d[d[SITE_COL] == hold]
            # inverse-frequency weights over the TRAINING sites only
            n = tr.groupby(SITE_COL)[TARGET].transform("size").to_numpy(float)
            w = (len(tr) / tr[SITE_COL].nunique()) / n
            try:
                pp, _ = curve_fit(fn, pack(tr), tr[TARGET].to_numpy(float),
                                  p0=p0, sigma=1 / np.sqrt(w), maxfev=300000)
                obs = te[TARGET].to_numpy(float)
                pred = np.asarray(fn(pack(te), *pp), dtype=float)
                # one-parameter local calibration: least-squares scale + offset
                cal = np.nan
                if CALIBRATE:
                    m = np.isfinite(pred)
                    if m.sum() > 10 and np.std(pred[m]) > 0:
                        A = np.vstack([pred[m], np.ones(m.sum())]).T
                        k, c0 = np.linalg.lstsq(A, obs[m], rcond=None)[0]
                        cal = r2(obs[m], k * pred[m] + c0)
                rows.append([name, hold, r2(obs, pred), shape_r2(obs, pred),
                             float(np.nanmean(pred) - obs.mean()), cal,
                             " ".join(f"{v:.4g}" for v in pp)])
                losos.append(rows[-1][2])
                cals.append(cal)
            except Exception as e:
                rows.append([name, hold, np.nan, np.nan, np.nan, np.nan, f"FAIL {e}"])
                losos.append(np.nan); cals.append(np.nan)

        # pooled in-sample, for reference only
        n = d.groupby(SITE_COL)[TARGET].transform("size").to_numpy(float)
        try:
            pp, _ = curve_fit(fn, pack(d), d[TARGET].to_numpy(float), p0=p0,
                              sigma=1 / np.sqrt((len(d) / len(sites)) / n),
                              maxfev=300000)
            pred = np.asarray(fn(pack(d), *pp), dtype=float)
            obs = d[TARGET].to_numpy(float)
            rows.append([name, "POOL in-sample", r2(obs, pred), shape_r2(obs, pred),
                         0.0, np.nan, " ".join(f"{v:.4g}" for v in pp)])
        except Exception as e:
            rows.append([name, "POOL in-sample", np.nan, np.nan, np.nan, np.nan,
                         f"FAIL {e}"])

        ls = np.array(losos, dtype=float)
        cs = np.array(cals, dtype=float)
        rows.append([name, ">>> LOSO mean", float(np.nanmean(ls)), np.nan, np.nan,
                     float(np.nanmean(cs)) if np.isfinite(cs).any() else np.nan,
                     f"worst site R2 = {np.nanmin(ls):.3f}"
                     if np.isfinite(ls).any() else "all folds failed"])

    out = pd.DataFrame(rows, columns=["model", "held_out", "R2_LOSO",
                                      "r2_shape", "bias", "R2_calibrated",
                                      "fitted_params"])
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(out.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    dst = os.path.join(os.path.dirname(csv) or ".", "LOSO_pooled_results.csv")
    out.to_csv(dst, index=False)
    print(f"\n  saved -> {dst}")
    print("\n  R2_calibrated = the same prediction after fitting ONE scale and one\n"
          "  offset on the held-out site. The gap between R2_LOSO and R2_calibrated is\n"
          "  how much of the failure is level rather than form.")
    print("\n  Read R2_LOSO and r2_shape together. A form that holds its shape but\n"
          "  loses R2 is transferring the physics and failing on level, which is a\n"
          "  different and more interesting result than failing outright.")


if __name__ == "__main__":
    main()
