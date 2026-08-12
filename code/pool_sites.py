"""
pool_sites.py
=============================================================================
Stack the three growing-season libraries into ONE pooled table for the
cross-site (common-equation) search. All three are already on consistent units
(depth in cm after the Mase fix; F_CH4_F in mg m-2 h-1). A `site` label column
is added first; the 66 engineered columns follow, in the KOR column order.

Inputs (edit paths if your folders differ):
   SK-CRK : KOR growing-season library (Papale+|F|>60+Hampel)
   JP-MSE : Mase growing-season library, cm-corrected (JPNMSE_2012_cm.csv)
   PH-IR  : PHL library, disturbance-cleaned (PHLIR_2016_cleaned.csv)

IMPORTANT for the downstream pipeline: add "site" (and keep "Date") in
EXCLUDE_HEADERS so the label is never used as a predictor. Run GAM-RF ->
collinearity -> PySR on the POOL, and validate leave-one-site-out at the end.

Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import os, glob
import pandas as pd

# --------------------------- USER CONFIG -----------------------------------
FILES = {
  "SK-CRK": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR\21Jul26\KORCRK_2018_papale_hampel_growingseason.csv",
  "JP-MSE": r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\JPN\JPNMSE_2012_cm.csv",
  "PH-IR" : r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016_cleaned.csv",
}
OUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOL\POOL_3sites_growingseason.csv"
CANON_SITE = "SK-CRK"   # whose column order to use as canonical
# ---------------------------------------------------------------------------

def resolve(p):
    if os.path.exists(p): return p
    hit=glob.glob(os.path.basename(p.replace("\\","/")))
    return hit[0] if hit else p

def main():
    dfs={s: pd.read_csv(resolve(p)) for s,p in FILES.items()}
    common=set.intersection(*[set(d.columns) for d in dfs.values()])
    canon=[c for c in dfs[CANON_SITE].columns if c in common]     # keep shared cols, KOR order
    parts=[]
    for s,d in dfs.items():
        x=d[canon].copy(); x.insert(0,"site",s); parts.append(x)
    P=pd.concat(parts,ignore_index=True)
    dp=os.path.dirname(OUT_CSV)
    if dp: os.makedirs(dp,exist_ok=True)
    P.to_csv(OUT_CSV,index=False)
    print(f"[POOLED] {len(P)} rows, {len(P.columns)} cols  ->  {OUT_CSV}")
    print("per-site rows:", dict(P['site'].value_counts()))

if __name__=="__main__":
    main()
