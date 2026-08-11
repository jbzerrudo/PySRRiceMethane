"""
fix_pooled_vpd_units.py

Corrects the VPD unit inconsistency in the pooled 3-site engineered file and
rebuilds every VPD-derived interaction from source columns.

Diagnosis (verified on POOL_3sites_growingseason.csv):
    VPD / (es - ea)  =  10.003 +/- 0.021   at JP-MSE   -> VPD is in hPa
                     =   9.997 +/- 0.005   at SK-CRK   -> VPD is in hPa
                     =   1.0000 +/- 0.0001 at PH-IR    -> VPD is in kPa
es and ea are kPa at all three sites, so only VPD and its products are affected.
Within each site the column is perfectly self-consistent, which is why this was
invisible until the sites were pooled.

Run this BEFORE GAM_RF_union on the pooled data.

Author: Jef Zerrudo / Claude
"""
import numpy as np
import pandas as pd

IN_CSV  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\POOL_3sites_growingseason_2.csv"
OUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\POOL_3sites_growingseason_VPDfix.csv"

HPA_SITES     = ["JP-MSE", "SK-CRK"]   # towers that reported VPD in hPa
MISSING_FLAGS = [-9999, -999900, -99999]

# child column -> (other factor, ) ; each is a simple product with VPD
PRODUCTS = {
    "h*VPD":  "depth",
    "u*VPD":  "uzonal",
    "v*VPD":  "vmerid",
    "SR*VPD": "SR",
    "VPD*WS": "WS",
}

d = pd.read_csv(IN_CSV, low_memory=False)
print(f"loaded {IN_CSV}: {d.shape[0]} rows x {d.shape[1]} cols")

# --- 1. sentinels -> NaN ----------------------------------------------------
num = [c for c in d.columns if c not in ("site", "Date", "time")]
d[num] = d[num].apply(pd.to_numeric, errors="coerce")
before = int(d[num].isna().sum().sum())
d[num] = d[num].replace(MISSING_FLAGS, np.nan)
after = int(d[num].isna().sum().sum())
print(f"sentinels converted to NaN: {after - before} cells")

# --- 2. verify the diagnosis before touching anything -----------------------
ratio = d["VPD"] / (d["es"] - d["ea"])
print("\nVPD / (es - ea) by site, before fix:")
print(ratio.groupby(d["site"]).agg(["mean", "std", "count"]).round(4))

# --- 3. recover d1sin while VPD*WS is still on its original scale ------------
#     VPD*WS*d1sin = (VPD*WS) * d1sin, so the diel factor divides out cleanly.
den = d["VPD*WS"].replace(0.0, np.nan)
d1sin = d["VPD*WS*d1sin"] / den

# --- 4. apply the unit fix --------------------------------------------------
hpa = d["site"].isin(HPA_SITES)
print(f"\nrescaling VPD by 1/10 for {hpa.sum()} rows "
      f"({', '.join(HPA_SITES)})")
d.loc[hpa, "VPD"] = d.loc[hpa, "VPD"] / 10.0

# --- 5. rebuild products from source columns (not by rescaling the product) --
for child, other in PRODUCTS.items():
    if child in d.columns and other in d.columns:
        d[child] = d[other] * d["VPD"]
        print(f"  rebuilt {child:8s} = {other} * VPD")
    else:
        print(f"  SKIPPED {child}: missing source column {other}")

d["VPD*WS*d1sin"] = d["VPD*WS"] * d1sin
print("  rebuilt VPD*WS*d1sin = VPD*WS * d1sin  (d1sin recovered pre-fix)")

# --- 6. verify -------------------------------------------------------------
ratio2 = d["VPD"] / (d["es"] - d["ea"])
print("\nVPD / (es - ea) by site, after fix (all should be ~1.0):")
print(ratio2.groupby(d["site"]).agg(["mean", "std"]).round(5))

print("\nVPD distribution by site, after fix (kPa):")
print(d.groupby("site")["VPD"].describe()[["min", "50%", "max"]].round(3))

bad = d.loc[d["VPD"] > d["es"] + 1e-6]
print(f"\nphysically impossible rows remaining (VPD > es): {len(bad)}")

d.to_csv(OUT_CSV, index=False)
print(f"\nwrote {OUT_CSV}: {d.shape[0]} rows x {d.shape[1]} cols")
