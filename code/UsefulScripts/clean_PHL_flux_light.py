"""
clean_PHL_flux_light.py  (v2 - adds documented field-disturbance exclusion)
=============================================================================
Light clean of the PHL-IRRI (UY) 2016 engineered library. PHL is a tight AWD
baseline near zero with a REAL early emission pulse (up to ~46 mg m-2 h-1); it
does NOT get the Cheorwon Papale/Hampel cascade, which would delete that pulse.
Only isolated, defensible removals are made, in two steps:

  STEP 1 - Absolute plausibility caps on the target:
     remove F_CH4_F > POS_CAP (an isolated +99 spike) and F_CH4_F < NEG_CAP
     (three implausible negatives, -25/-30/-35; a flooded paddy does not take up
     35 mg m-2 h-1). 4 points.

  STEP 2 - Documented field-disturbance exclusion:
     the half-hours of two Hobo water-level logger servicing visits, when walking
     the plot disturbed the sediment and released trapped CH4 by ebullition, are
     removed as sampling artifacts (recorded in the field log):
        26 Feb 2016  16:30, 17:00   (16.7, 22.2 mg; water at the surface)
        18 Mar 2016  15:30, 16:00,  (54.9, 48.9 mg)
                     21:00          (19.2 mg)
     On 18 Mar the water table was 24 cm BELOW the surface (aerobic, dry AWD
     phase), so 50+ mg m-2 h-1 cannot be natural methanogenesis - it is
     disturbance-induced. 5 points.

All 9 flagged half-hours are refilled by short linear interpolation, so the
target stays 100% complete. Everything else in the library is preserved
byte-for-byte; only F_CH4_F and F_CH4_F_orig change, and only at those rows.
F_CH4_F_orig is kept consistent as F_CH4_F / CF_PHL (CF_PHL = 57.744; your orig
is in umol m-2 s-1). The real early pulse (max 46.2) is untouched.

NOT removed (left for your call): two odd-hour highs at 15 Feb 00:00 (14 mg) and
3 Mar 07:00 (11 mg), which do not match a field visit; and a negative excursion
near 24-25 Feb reaching ~-15 mg. Add their timestamps to VISIT_EXCLUDE / the caps
if your log says otherwise.

Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import numpy as np, pandas as pd

# --------------------------- USER CONFIG -----------------------------------
IN_CSV  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016.csv"
OUT_CSV = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016_cleaned.csv"
DATE_COL = "Date"          # timestamps are DD/MM/YYYY HH:MM
POS_CAP =  60.0            # remove F_CH4_F above this (isolated +99 spike)
NEG_CAP = -20.0            # remove F_CH4_F below this (implausible negatives)
INTERP_LIMIT = 6           # max consecutive half-hours to linear-interpolate
CF_PHL  = 57.744           # F_CH4_F = F_CH4_F_orig * CF_PHL  (mg m-2 h-1 per umol m-2 s-1)

# Documented Hobo-servicing field-disturbance half-hours (mud disturbed):
VISIT_EXCLUDE = [
    "26/02/2016 16:30", "26/02/2016 17:00",
    "18/03/2016 15:30", "18/03/2016 16:00", "18/03/2016 21:00",
]
# ---------------------------------------------------------------------------

def main():
    d = pd.read_csv(IN_CSV)
    F = pd.to_numeric(d['F_CH4_F'], errors='coerce')

    cap   = ((F > POS_CAP) | (F < NEG_CAP)).values                        # step 1
    ts    = pd.to_datetime(d[DATE_COL], dayfirst=True, errors='coerce')
    visit = ts.isin(pd.to_datetime(VISIT_EXCLUDE, dayfirst=True)).values  # step 2

    out = cap | visit
    idx = np.where(out)[0]

    # remove flagged points, refill by short linear interpolation; keep orig consistent
    Fc  = F.mask(out).interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
    Foc = Fc / CF_PHL

    # write back, preserving every other cell byte-for-byte (only these rows x 2 cols change)
    raw = pd.read_csv(IN_CSV, dtype=str, keep_default_na=False)
    for i in idx:
        raw.loc[i, 'F_CH4_F']      = '%.10g' % Fc.iloc[i]
        raw.loc[i, 'F_CH4_F_orig'] = '%.10g' % Foc.iloc[i]
    raw.to_csv(OUT_CSV, index=False)

    print(f"step 1 plausibility caps : removed {int(cap.sum())}  (values {np.round(F.values[cap],1).tolist()})")
    print(f"step 2 field disturbance : removed {int(visit.sum())}  (values {np.round(F.values[visit],1).tolist()})")
    print(f"total removed            : {len(idx)}  rows {idx.tolist()}")
    print(f"F_CH4_F after: complete={Fc.notna().all()}  range [{Fc.min():.1f}, {Fc.max():.1f}] mg/m2/h")
    print(f"[SAVED] {OUT_CSV}  rows={len(raw)}")

if __name__ == "__main__":
    main()
