import numpy as np, pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score

# ===================== CONFIG: set these two paths =====================
PHIR_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\PHL\PHL-IR_2016.csv"                 # same file as before
PERDAY   = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\PYSR\PHL\vif5_pysrv2_phl\pysr7v2\run_20260619_113441\stage8_auto_20260619_113441\per_day_residuals.csv"               # extracted from the zip you sent
# column names (already matched to your file):
COL_FCH4, COL_AUC, COL_DATE        = "F_CH4_F", "AUC", "Date"
COL_SR_Ts, COL_h_VPD, COL_SR_HODsin = "SR*Ts", "h*VPD", "SR*HODsin"
# =======================================================================

df = pd.read_csv(PHIR_CSV)

# --- rebuild YOUR canonical folds from the stage-8 output (folds are assigned per day) ---
pdres    = pd.read_csv(PERDAY)
day2fold = pdres[pdres["slot"] == "simple_ruleA"].set_index("DAY")["fold"].to_dict()
#df["DAY"]  = pd.to_datetime(df[COL_DATE]).dt.strftime("%Y-%m-%d")
df["DAY"]  = pd.to_datetime(df[COL_DATE], dayfirst=True).dt.strftime("%Y-%m-%d")
df["fold"] = df["DAY"].map(day2fold)
assert df["fold"].notna().all(), "some days did not match the stage-8 fold map; check COL_DATE"
print(f"rows={len(df)}  folds={sorted(df['fold'].unique())}  "
      f"days/fold={df.groupby('fold')['DAY'].nunique().to_dict()}\n")

y      = df[COL_FCH4].to_numpy()
SR_Ts  = df[COL_SR_Ts].to_numpy()
h_VPD  = df[COL_h_VPD].to_numpy()
SR_HOD = df[COL_SR_HODsin].to_numpy()
AUC    = df[COL_AUC].to_numpy()
fold   = df["fold"].to_numpy()

def full(X, a, b, c, d, e):       # current Eq. 4 (= complex_ruleA), 5 coeffs
    SR_Ts, h_VPD, SR_HOD, AUC = X
    return a + (c*SR_Ts + b + (e + np.tanh(h_VPD))*np.tanh(SR_HOD))*np.exp(d*AUC)

def reduced(X, a, b, c, d, e):    # new Eq. 4: h*VPD dropped, same coeff count
    SR_Ts, SR_HOD, AUC = X
    return a + (c*SR_Ts + b + e*np.tanh(SR_HOD))*np.exp(d*AUC)

Xf, Xr  = (SR_Ts, h_VPD, SR_HOD, AUC), (SR_Ts, SR_HOD, AUC)
p0_full = [0.9636868, 1.686024, 2.388674e-4, 3.167974e-3, -1.537868]   # your canonical warm start
p0_red  = [0.9636868, 1.686024, 2.388674e-4, 3.167974e-3, -1.537868 + np.tanh(h_VPD).mean()]

def cv(model, X, p0, letters):
    pall, _ = curve_fit(model, X, y, p0=p0, maxfev=300000)             # full-data fit -> warm start
    r2, r2nf, coefs = [], [], []
    for k in sorted(np.unique(fold)):
        tr, te = fold != k, fold == k
        Xtr = tuple(v[tr] for v in X); Xte = tuple(v[te] for v in X)
        p, _ = curve_fit(model, Xtr, y[tr], p0=pall, maxfev=300000)
        coefs.append(p)
        r2.append(r2_score(y[te], model(Xte, *p)))
        r2nf.append(r2_score(y[te], model(Xte, *pall)))
    coefs = np.array(coefs)
    return np.array(r2), np.mean(r2), np.std(r2, ddof=1), np.mean(r2nf), dict(zip(letters, np.abs(coefs.std(0, ddof=1)/coefs.mean(0)))), pall

f_r2, f_m, f_s, f_nf, f_cov, _      = cv(full,    Xf, p0_full, list("abcde"))
print("FULL Eq.4 (sanity — must match your paper)")
print("  per-fold R2:", np.round(f_r2, 3), " canonical: [0.250 0.853 0.682 0.001 0.595]")
print(f"  CV R2 = {f_m:.3f} +/- {f_s:.3f}   max|CoV| = {max(f_cov.values()):.3f}   canonical: 0.476 +/- 0.345, 0.130\n")

r_r2, r_m, r_s, r_nf, r_cov, r_pall = cv(reduced, Xr, p0_red,  list("abcde"))
print("REDUCED Eq.4 (h*VPD dropped — the number you report)")
print("  per-fold R2:", np.round(r_r2, 3))
print(f"  CV R2 = {r_m:.3f} +/- {r_s:.3f}   no-refit {r_nf:.3f}   max|CoV| = {max(r_cov.values()):.3f}")
print("  per-coef |CoV|:", ", ".join(f"{k}={v:.3f}" for k, v in r_cov.items()))
print("  full-data coeffs (Table 6):", ", ".join(f"{s}={v:.4g}" for s, v in zip(["alpha","beta","gamma","delta","eps"], r_pall)))

floor = f_m - f_s
ok = (r_m >= floor) and (max(r_cov.values()) < 0.5)
print(f"\nSec.2.6 gates:  CV R2 {r_m:.3f} >= {floor:.3f}? {r_m>=floor}   max|CoV| < 0.5? {max(r_cov.values())<0.5}")
print("VERDICT:", "KEEP REDUCED (report it)" if ok else "KEEP FULL Eq. 4")