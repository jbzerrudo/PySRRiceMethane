"""
Eq. 4 reduction test (PH-IR): does collapsing tanh(h*VPD) to a constant lose skill?
Fits FULL Eq. 4 and the REDUCED form on the SAME day-grouped 5-fold CV, then
applies your Sec. 2.6 acceptance rule.

  FULL    : FCH4 = a + [g*SR_Ts + b + (e + tanh(h*VPD))*tanh(SR_HODsin)] * exp(d*AUC)
  REDUCED : FCH4 = a + [g*SR_Ts + b +          c        *tanh(SR_HODsin)] * exp(d*AUC)
"""
import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

# =========================== CONFIG: EDIT ONLY THIS BLOCK ===========================
CSV_PATH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\PHL-IR_2016.csv"   # keep the exact path you already used

COL_FCH4 = "F_CH4_F"      # gap-filled flux target  (see note below re: F_CH4_F vs F_CH4_F_orig)
COL_AUC  = "AUC"

# your file already CONTAINS these interaction terms, so just name them (nothing is built):
COL_SR_Ts     = "SR*Ts"
COL_h_VPD     = "h*VPD"
COL_SR_HODsin = "SR*HODsin"

# (these are unused now, because the three terms above already exist as columns)
COL_SR, COL_Ts, COL_h, COL_VPD, COL_dayhr = "SR", "Tsoil", "depth", "VPD", "dayhr"

COL_DATE = "Date"         # whole-day grouping key for the day-grouped CV
# ====================================================================================

df = pd.read_csv(CSV_PATH)
print("Columns in file:", list(df.columns), "\n")

SR_Ts     = df[COL_SR_Ts].to_numpy()     if COL_SR_Ts     else (df[COL_SR]*df[COL_Ts]).to_numpy()
h_VPD     = df[COL_h_VPD].to_numpy()     if COL_h_VPD     else (df[COL_h]*df[COL_VPD]).to_numpy()
SR_HODsin = df[COL_SR_HODsin].to_numpy() if COL_SR_HODsin else (df[COL_SR]*np.sin(2*np.pi*df[COL_dayhr]/24)).to_numpy()

FCH4 = df[COL_FCH4].to_numpy()
AUC  = df[COL_AUC].to_numpy()
day  = df[COL_DATE].astype(str).to_numpy()       # one group per calendar day

ok = ~np.isnan(np.column_stack([FCH4, SR_Ts, h_VPD, SR_HODsin, AUC])).any(axis=1)
FCH4, SR_Ts, h_VPD, SR_HODsin, AUC, day = (a[ok] for a in (FCH4, SR_Ts, h_VPD, SR_HODsin, AUC, day))
print(f"Rows used: {ok.sum()}  (dropped {(~ok).sum()})\n")

t = np.tanh(h_VPD)
print(f"tanh(h*VPD):  mean={t.mean():+.3f}  sd={t.std():.3f}  frac|.|>0.99={np.mean(np.abs(t)>0.99):.3f}\n")

def full(X, a, b, g, d, e):
    SR_Ts, h_VPD, SR_HODsin, AUC = X
    return a + (g*SR_Ts + b + (e + np.tanh(h_VPD))*np.tanh(SR_HODsin)) * np.exp(d*AUC)

def reduced(X, a, b, g, d, c):
    SR_Ts, SR_HODsin, AUC = X
    return a + (g*SR_Ts + b + c*np.tanh(SR_HODsin)) * np.exp(d*AUC)

X_full, X_red = (SR_Ts, h_VPD, SR_HODsin, AUC), (SR_Ts, SR_HODsin, AUC)
p0_full = [0.96, 1.69, 2.4e-4, 3.2e-3, -1.54]
p0_red  = [0.96, 1.69, 2.4e-4, 3.2e-3, -1.54 + t.mean()]

def run_cv(model, X, p0, names):
    p_all, _ = curve_fit(model, X, FCH4, p0=p0, maxfev=200000)
    r2_refit, r2_norefit, coefs = [], [], []
    for tr, te in GroupKFold(5).split(FCH4, groups=day):
        Xtr = tuple(v[tr] for v in X); Xte = tuple(v[te] for v in X)
        p, _ = curve_fit(model, Xtr, FCH4[tr], p0=p_all, maxfev=200000)
        coefs.append(p)
        r2_refit.append(r2_score(FCH4[te], model(Xte, *p)))
        r2_norefit.append(r2_score(FCH4[te], model(Xte, *p_all)))
    coefs = np.array(coefs)
    cov = dict(zip(names, np.abs(coefs.std(0)/coefs.mean(0))))
    return np.mean(r2_refit), np.std(r2_refit), np.mean(r2_norefit), cov

mF, sF, nF, covF = run_cv(full,    X_full, p0_full, ["a","b","g","d","e"])
mR, sR, nR, covR = run_cv(reduced, X_red,  p0_red,  ["a","b","g","d","c"])

print(f"FULL Eq.4 :  CV R2 = {mF:.3f} +/- {sF:.3f}   (no-refit {nF:.3f})   max|CoV| = {max(covF.values()):.3f}")
print( "            (paper reports 0.476 +/- 0.345; this run should land near it)")
print(f"REDUCED   :  CV R2 = {mR:.3f} +/- {sR:.3f}   (no-refit {nR:.3f})   max|CoV| = {max(covR.values()):.3f}")
print( "             per-coef |CoV|: " + ", ".join(f"{k}={v:.3f}" for k,v in covR.items()))

floor = mF - sF
within, stable = mR >= floor, max(covR.values()) < 0.5
print(f"\nSec. 2.6 gates:  CV R2 >= {floor:.3f}? {within}    max|CoV| < 0.5? {stable}")
print("VERDICT:", "KEEP REDUCED (report it)" if (within and stable) else "KEEP FULL Eq. 4 (reduction fails)")