#!/usr/bin/env python3
"""
Eq.5 -> bilinear refit, with day-grouped CV and the coefficient-stability screen.

Tests whether the SK-CRK growing-season form can drop from the 7-coefficient
Eq.5 to the bilinear in-season analogue of Eq.3:

    F_CH4 = AUC_wet * (gamma + delta * SR)
          = gamma * AUC_wet + delta * (AUC_wet * SR)      # LINEAR in (gamma, delta)

so each fold is a 2-feature, no-intercept OLS -- no PySR, no nonlinear optimiser.

Decision rule (the paper's screen):
    KEEP bilinear (replace Eq.5)  iff  day-grouped CV R2 (mean over folds) >= 0.24
                                       AND  max across-fold |CoV| < 0.5
    else  keep Eq.5  and apply the d -> k rename.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

# ---- EDIT to match your SK-CRK growing-season file (9 Apr-30 Sep 2018, n=8365) ----
CSV      = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv"
COL_F    = "F_CH4_F"    # gap-filled CH4 flux        (mg m-2 h-1)
COL_AUC  = "AUC_wet"    # cumulative wet exposure     (cm h)
COL_SR   = "SR"         # shortwave radiation         (W m-2)
COL_DAY  = "Date"       # half-hourly timestamp; floored to calendar day below
N_SPLITS = 5
R2_FLOOR = 0.24         # within 1 s.d. of Eq.5's 0.344 +/- 0.101
COV_CEIL = 0.5
# If you saved the pipeline's fold assignments, load them instead of GroupKFold
# (set FOLD_COL to the column holding 0..N_SPLITS-1) for exact comparability.
FOLD_COL = None
# -----------------------------------------------------------------------------------


def r2(y, yhat):
    ss_res = np.sum((y - yhat) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    return 1.0 - ss_res / ss_tot


def design(auc, sr):
    """Columns [AUC_wet, AUC_wet*SR]; no intercept (matches Eq.3's form)."""
    return np.column_stack([auc, auc * sr])


def cov(x):
    x = np.asarray(x, float)
    return np.std(x, ddof=1) / abs(np.mean(x))


df = pd.read_csv(CSV)
df[COL_DAY] = pd.to_datetime(df[COL_DAY], format="mixed", dayfirst=False)
df = df.dropna(subset=[COL_F, COL_AUC, COL_SR, COL_DAY])
F   = df[COL_F].to_numpy(float)
AUC = df[COL_AUC].to_numpy(float)
SR  = df[COL_SR].to_numpy(float)
day = df[COL_DAY].dt.normalize().to_numpy()   # calendar day = grouping key

print(f"n = {len(F)} half-hourly records over {len(np.unique(day))} days")

# Full-data fit -- sanity check vs warm-start (gamma ~ 1.2e-3, delta ~ 1.6e-6).
# These won't match exactly: dropping Eq.5's rational correction re-optimises both.
beta_full, *_ = np.linalg.lstsq(design(AUC, SR), F, rcond=None)
print(f"full-data fit:  gamma = {beta_full[0]:.3e}   delta = {beta_full[1]:.3e}")

# Day-grouped folds
if FOLD_COL is not None:
    fold_id = df[FOLD_COL].to_numpy(int)
    splits = [(np.where(fold_id != k)[0], np.where(fold_id == k)[0])
              for k in range(N_SPLITS)]
else:
    splits = list(GroupKFold(n_splits=N_SPLITS).split(AUC, F, groups=day))

r2s, gammas, deltas = [], [], []
for k, (tr, te) in enumerate(splits, 1):
    beta, *_ = np.linalg.lstsq(design(AUC[tr], SR[tr]), F[tr], rcond=None)
    fr2 = r2(F[te], design(AUC[te], SR[te]) @ beta)
    r2s.append(fr2); gammas.append(beta[0]); deltas.append(beta[1])
    print(f"fold {k}: R2={fr2:+.3f}  gamma={beta[0]:.3e}  delta={beta[1]:.3e}")

r2s = np.asarray(r2s)
cv_mean, cv_sd = r2s.mean(), r2s.std(ddof=1)
cov_g, cov_d = cov(gammas), cov(deltas)
max_cov = max(cov_g, cov_d)

print("\n--- screen ---")
print(f"day-grouped CV R2 = {cv_mean:.3f} +/- {cv_sd:.3f}   (Eq.5: 0.344 +/- 0.101)")
print(f"|CoV| gamma       = {cov_g:.2f}")
print(f"|CoV| delta       = {cov_d:.2f}")
print(f"max |CoV|         = {max_cov:.2f}")

keep = (cv_mean >= R2_FLOOR) and (max_cov < COV_CEIL)
print(f"\nKEEP bilinear (replace Eq.5): {keep}")
print("-> replace Eq.5 with AUC_wet*(gamma+delta*SR)" if keep
      else "-> keep Eq.5; apply the d -> k rename.")