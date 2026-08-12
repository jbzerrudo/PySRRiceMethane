"""
fix_mase_depth_units.py
=============================================================================
The Mase (JP-Mse) 2012 library was built with water depth in METRES, while the
KOR and PHL libraries use CENTIMETRES. This rescales every depth-derived column
to cm so all three sites are on one scale (needed for cross-site comparison, a
common water term, and ORYZA).

  * 17 columns are LINEAR in depth, so the correct cm value is exactly x100:
      depth, AUC, AUC_dry, AUC_wet, rate, hwet, hdry,
      h*Ta, h*Ts, h*WS, h*Pr, h*VPD, h*DelTsa, h*sinTOD, h*cosTOD, h*u, h*v
    (x100 also maps a -9999 missing flag to -999900, which is still a recognised
     flag in the GAM-RF MISSING_FLAGS list, so gaps stay gaps.)

  * 2 columns are NONLINEAR and are recomputed from the cm depth, NOT rescaled:
      h_inv      = 1 / (depth_cm + 0.001)
      h_ASINH_cm = arcsinh(depth_cm)
    (rescaling these would be wrong: 1/(100x) != (1/x)/100, and arcsinh is not
     linear - this is the whole reason a plain x100 of the file is not enough.)

Every non-depth column is preserved byte-for-byte. Only the 19 depth columns
change. Verified: x100 of the stored AUC equals a fresh cumulative recompute
from the cm depth to 1e-10, confirming the linear terms scale exactly.

Note (left as-is): Tsoil still carries 78 uncleaned -9999 values; your GAM-RF
MISSING_FLAGS absorbs those downstream, so they are not touched here.

Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import os, glob
import numpy as np, pandas as pd

# --------------------------- USER CONFIG -----------------------------------
IN_CSV  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\JPN\JPNMSE_2012.csv"   # metres version (adjust folder if needed)
OUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\JPN\JPNMSE_2012_cm.csv" # cm-corrected output
# ---------------------------------------------------------------------------

LINEAR = ['depth','AUC','AUC_dry','AUC_wet','rate','hwet','hdry',
          'h*Ta','h*Ts','h*WS','h*Pr','h*VPD','h*DelTsa','h*sinTOD','h*cosTOD','h*u','h*v']

def resolve(p):
    if os.path.exists(p): return p
    hit=glob.glob(os.path.basename(p.replace("\\","/")))
    return hit[0] if hit else p

def main():
    src = resolve(IN_CSV)
    d = pd.read_csv(src)
    num = lambda c: pd.to_numeric(d[c], errors='coerce')
    depth_cm = num('depth') * 100.0

    new = {c: num(c) * 100.0 for c in LINEAR}          # linear terms: exact x100
    new['h_inv']      = 1.0 / (depth_cm + 0.001)       # nonlinear: recompute from cm
    new['h_ASINH_cm'] = np.arcsinh(depth_cm)

    raw = pd.read_csv(src, dtype=str, keep_default_na=False)   # preserve everything else byte-for-byte
    for c, v in new.items():
        raw[c] = ['%.10g' % x if np.isfinite(x) else '-9999' for x in v.values]

    dp = os.path.dirname(OUT_CSV)
    if dp: os.makedirs(dp, exist_ok=True)
    raw.to_csv(OUT_CSV, index=False)

    print(f"converted {len(new)} depth columns to cm; other {len(d.columns)-len(new)} preserved byte-for-byte")
    print(f"depth range now [{depth_cm.min():.2f}, {depth_cm.max():.2f}] cm")
    print(f"[SAVED] {OUT_CSV}  rows={len(raw)}")

if __name__ == "__main__":
    main()
