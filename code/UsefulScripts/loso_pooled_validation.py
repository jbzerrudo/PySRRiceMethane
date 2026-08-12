# =============================================================================
# LEAVE-ONE-SITE-OUT (LOSO) VALIDATION FOR THE POOLED / COMMON CH4 EQUATION
# Pooled PySR run 20260727_180150, seed_52, four Pareto candidates.
#
# Rationale: day-grouped CV inside the pool cannot distinguish a genuine COMMON
# equation from a pooled AVERAGE, because every fold still contains all three
# sites. LOSO fits on two sites and predicts the third, which is the actual
# claim being made in the paper.
#
# Convention (same as stage 8): one free parameter per DISTINCT numeric literal
# in the printed PySR equation. Repeated literals are tied to one parameter.
# Training uses inverse-frequency site weights recomputed over the two training
# sites. Held-out metrics are unweighted.
# =============================================================================
import pandas as pd, numpy as np
from scipy.optimize import curve_fit
np.seterr(all='ignore')

CSV   = r"POOLED_retainedvars_postcollin_20260727_161707.csv"   # <-- edit path
TARGET= "F_CH4_F"
SITES = ["SK-CRK", "JP-MSE", "PH-IR"]

COLS = ["site", "AUC", "es", "VPD", "vmerid", "h*VPD", TARGET]
df = pd.read_csv(CSV)[COLS].dropna().reset_index(drop=True)
print("rows:", len(df), df["site"].value_counts().to_dict(), "\n")

def pack(d):
    return (d["AUC"].values, d["es"].values, d["VPD"].values,
            d["vmerid"].values, d["h*VPD"].values)

# ---------------- the four seed_52 Pareto candidates -------------------------
def f_simple(X, a):                    # #3  cx5   R2 0.404
    AUC, es, VPD, vm, hV = X
    return AUC*a + es

def f_knee(X, a):                      # #4  cx6   R2 0.508  (Pareto knee)
    AUC, es, VPD, vm, hV = X
    return es*np.exp(AUC*a)

def f_mid(X, a, b, c):                 # #13 cx17  R2 0.622
    AUC, es, VPD, vm, hV = X
    return ((AUC*a + b) * (es - np.tanh(vm - np.exp(c - hV)))) * es

def f_best(X, a, b, c, d):             # #23 cx31  R2 0.643
    AUC, es, VPD, vm, hV = X
    t = (a*AUC) + (es - np.tanh(VPD)) - np.tanh(vm + b) - np.tanh((AUC*(hV*VPD))*a)
    return t*(np.tanh(AUC) + es + c) + d

# ---------------- ablations of #23 (are the extra terms real?) ---------------
def f_A1(X, a, b, c, d):               # vmerid replaced by a constant
    AUC, es, VPD, vm, hV = X
    t = (a*AUC) + (es - np.tanh(VPD)) - b - np.tanh((AUC*(hV*VPD))*a)
    return t*(np.tanh(AUC) + es + c) + d

def f_A2(X, a, b, c, d):               # drop vmerid and h*VPD
    AUC, es, VPD, vm, hV = X
    return ((a*AUC) + (es - np.tanh(VPD)) - b)*(np.tanh(AUC) + es + c) + d

def f_A3(X, a, b, c, d):               # water + temperature only
    AUC, es, VPD, vm, hV = X
    return ((a*AUC) + es - b)*(np.tanh(AUC) + es + c) + d

MODELS = {
 "#3  simple  cx5 ": (f_simple, [6.3583296e-4]),
 "#4  knee    cx6 ": (f_knee,   [1.258718e-4]),
 "#13 mid     cx17": (f_mid,    [5.4896216e-5, 0.34614047, 0.93306595]),
 "#23 best    cx31": (f_best,   [2.486764e-4, 0.66067696, -1.8193812, 0.6148757]),
 "A1  no vmerid   ": (f_A1,     [2.486764e-4, 0.66067696, -1.8193812, 0.6148757]),
 "A2  no vm,no hV ": (f_A2,     [2.486764e-4, 0.66067696, -1.8193812, 0.6148757]),
 "A3  AUC+es only ": (f_A3,     [2.486764e-4, 0.66067696, -1.8193812, 0.6148757]),
}

def R2(y, p):
    m = np.isfinite(p)
    return 1 - np.sum((y[m]-p[m])**2)/np.sum((y[m]-y[m].mean())**2)

def SHAPE(y, p):                       # Pearson r^2: agreement in shape, bias removed
    m = np.isfinite(p)
    return np.corrcoef(y[m], p[m])[0, 1]**2

rows = []
for name, (fn, p0) in MODELS.items():
    losos = []
    for hold in SITES:
        tr, te = df[df.site != hold], df[df.site == hold]
        n = tr.groupby("site")["es"].transform("size").values
        w = (len(tr)/2.0)/n                      # inverse frequency, 2 training sites
        try:
            pp, _ = curve_fit(fn, pack(tr), tr[TARGET].values, p0=p0,
                              sigma=1/np.sqrt(w), maxfev=300000)
            pred, obs = fn(pack(te), *pp), te[TARGET].values
            r, s, b = R2(obs, pred), SHAPE(obs, pred), np.nanmean(pred)-obs.mean()
            rows.append([name, hold, r, s, b, " ".join(f"{v:.3g}" for v in pp)])
            losos.append(r)
        except Exception as e:
            rows.append([name, hold, np.nan, np.nan, np.nan, f"FAIL {e}"]); losos.append(np.nan)
    n = df.groupby("site")["es"].transform("size").values
    pp, _ = curve_fit(fn, pack(df), df[TARGET].values, p0=p0,
                      sigma=1/np.sqrt((len(df)/3.0)/n), maxfev=300000)
    pred = fn(pack(df), *pp)
    rows.append([name, "POOL in-sample", R2(df[TARGET].values, pred),
                 SHAPE(df[TARGET].values, pred), 0.0,
                 " ".join(f"{v:.3g}" for v in pp)])
    rows.append([name, ">>> LOSO mean", np.nanmean(losos), np.nan, np.nan,
                 f"worst site R2 = {np.nanmin(losos):.3f}"])

out = pd.DataFrame(rows, columns=["model", "held_out", "R2_LOSO", "r2_shape",
                                  "bias", "fitted_params"])
pd.set_option("display.width", 200); pd.set_option("display.max_colwidth", 45)
print(out.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
out.to_csv("LOSO_pooled_results.csv", index=False)
print("\nsaved -> LOSO_pooled_results.csv")
