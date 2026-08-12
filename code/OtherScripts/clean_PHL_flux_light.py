"""
clean_PHL_flux_light.py
=============================================================================
LIGHT clean of the PHL-IRRI (UY) 2016 engineered library. PHL is already clean
(off-trend scatter 3.9 mg m-2 h-1, close to Mase 1.8, far below Cheorwon 9.1):
a tight AWD baseline near zero with a real early emission pulse (up to ~46) and
only FOUR isolated outliers - one positive spike at 99 and three implausible
negatives at -25/-30/-35 (a flooded paddy does not take up 35 mg m-2 h-1).

So this does NOT use the Cheorwon 3-step. Applying Papale/|F|>60/Hampel(k=4) here
would flag ~4.5% and delete the REAL early emission pulse. Instead:

  ONLY STEP - absolute plausibility bounds on the target:
     remove F_CH4_F > POS_CAP (the 99) and F_CH4_F < NEG_CAP (the three negatives),
     4 points total (0.11%), refilled by short linear interpolation.
     No Papale (there is no separate measured-flux column in the library) and
     no Hampel (it would remove real signal on this tight baseline).

Everything else in the library is preserved byte-for-byte; only F_CH4_F and
F_CH4_F_orig change, and only at the 4 flagged half-hours. F_CH4_F_orig is kept
consistent via F_CH4_F_orig = F_CH4_F / CF_PHL (CF_PHL = 57.744; your orig is
umol m-2 s-1). Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import numpy as np, pandas as pd

# --------------------------- USER CONFIG -----------------------------------
IN_CSV  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016.csv"
OUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016_cleaned.csv"
POS_CAP =  60.0      # remove F_CH4_F above this (isolated positive spike at 99)
NEG_CAP = -20.0      # remove F_CH4_F below this (implausible negatives -25/-30/-35)
INTERP_LIMIT = 6     # max consecutive half-hours to linear-interpolate
CF_PHL  = 57.744     # F_CH4_F = F_CH4_F_orig * CF_PHL  (mg m-2 h-1 per umol m-2 s-1)
# ---------------------------------------------------------------------------

def main():
    d = pd.read_csv(IN_CSV)
    F  = pd.to_numeric(d['F_CH4_F'], errors='coerce')
    out = ((F > POS_CAP) | (F < NEG_CAP)).values
    idx = np.where(out)[0]
    # clean the target and refill by interpolation, keep orig consistent
    Fc = F.mask(out).interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
    Foc = Fc / CF_PHL
    # write back preserving every other cell byte-for-byte (only the 4 rows x 2 cols change)
    raw = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)
    for i in idx:
        raw.loc[i, 'F_CH4_F']      = '%.10g' % Fc.iloc[i]
        raw.loc[i, 'F_CH4_F_orig'] = '%.10g' % Foc.iloc[i]
    raw.to_csv(OUT_CSV, index=False)
    print(f"removed {len(idx)} outliers at rows {idx.tolist()}  (values {np.round(F.values[idx],1).tolist()})")
    print(f"F_CH4_F after: complete={Fc.notna().all()}  range [{Fc.min():.1f}, {Fc.max():.1f}] mg/m2/h")
    print(f"[SAVED] {OUT_CSV}  rows={len(raw)}")

if __name__ == "__main__":
    main()
