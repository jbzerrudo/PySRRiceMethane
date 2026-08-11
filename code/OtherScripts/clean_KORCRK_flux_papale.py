"""
clean_KORCRK_flux_papale.py
=============================================================================
Cheorwon (KR-CRK) 2018 CH4 flux cleaning for Paper 1, using an ESTABLISHED,
citable QC pipeline, then rebuilds the 66-column engineered library on the
cleaned target. Drop-in for your RUN2 cascade.

WHY THIS PIPELINE (and not |F|>60 alone, and not Papale alone):
  * |F|>60 alone removed only 28 points -- cosmetic, as you noted.
  * Papale et al. (2006) MAD double-difference despike is the community standard
    for local statistical spikes, BUT on this record it MISSES the largest
    isolated extremes (+136, +126, +113 mg m-2 h-1). Those are isolated
    night-time half-hours whose measured NEIGHBOURS are gap-filled, so their
    double-difference is undefined and the test never sees them.
  * So we combine the two, which is standard EC practice:
      (A) Papale (2006) MAD despike on the MEASURED flux, day/night split
          (local statistical outliers), z = Z.
      (B) an absolute physical-plausibility limit |F| > SPIKE_MAX
          (isolated extremes the 2nd-difference cannot test).
    Union removed and refilled by linear interpolation. Nothing else touched.

RESULT (this record): Papale 167 + absolute 28 = 186 unique (1.46%); target
100% complete; kept maximum drops from 136 to 58 mg m-2 h-1. The real
growing-season variability is preserved; only outliers and isolated extremes go.

Predictors reproduce your original sheet exactly (validated to 1e-9); ONLY the
flux target changes. Missing cells are written as your -9999 flag.
Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import os
import numpy as np, pandas as pd

# --------------------------- USER CONFIG -----------------------------------
FLUXNET_HH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\KOREA\FLX_KR-CRK_FLUXNET-CH4_2015-2018_1-1\FLX_KR-CRK_FLUXNET-CH4_HH_2015-2018_1-1.csv"
OUT_FULL   = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR\21Jul26\KOR-CRK_2018_papale_full.csv"
OUT_GS     = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR\21Jul26\KOR-CRK_2018_papale_growingseason.csv"
REC_START  = "2018-04-09 17:30"
REC_END    = "2018-12-31 23:30"
GS_END     = "2018-09-30 23:30"
Z          = 5.5      # Papale despike threshold (MAD units); Papale used 4 (strict)..7 (lenient)
DAY_SW     = 20.0     # SW_IN_F > DAY_SW  -> daytime, for the day/night split
SPIKE_MAX  = 60.0     # absolute plausibility limit, mg m-2 h-1 (isolated extremes)
INTERP_LIMIT = 6      # max consecutive half-hours to linear-interpolate
CF         = 0.0577   # nmol m-2 s-1 -> mg m-2 h-1 (your sheet's factor; exact 16.04*3600/1e6=0.057744)
DATE_FMT   = "%d/%m/%Y %H:%M"
MISSING    = "-9999"
# ---------------------------------------------------------------------------

def papale_flag(meas, day, z):
    """Papale et al. (2006) MAD double-difference despike, day/night populations.
    meas: measured flux Series (mg m-2 h-1); day: boolean Series. Returns boolean np.array."""
    x=meas.values; n=len(x); d=np.full(n,np.nan)
    for i in range(1,n-1):
        if np.isfinite(x[i-1]) and np.isfinite(x[i]) and np.isfinite(x[i+1]):
            d[i]=(x[i]-x[i-1])-(x[i+1]-x[i])
    flag=np.zeros(n,bool); dm_day=day.values
    for mask in [dm_day, ~dm_day]:
        dm=np.where(mask,d,np.nan); Md=np.nanmedian(dm); MAD=np.nanmedian(np.abs(dm-Md))
        if not np.isfinite(MAD) or MAD==0: continue
        thr=z*MAD/0.6745
        flag |= mask & np.isfinite(d) & ((d<Md-thr)|(d>Md+thr))
    return flag

def build(fl):
    d = pd.DataFrame({'Date': fl['ts']})
    start = d['Date'].iloc[0]
    d['Deltime'] = (d['Date']-start).dt.total_seconds()/3600.0
    d['time']    = (d['Date'].dt.hour + d['Date'].dt.minute/60)/24.0
    d['dayhr']   = d['Date'].dt.hour + d['Date'].dt.minute/60.0
    d['depth'] = fl['WTD']*100.0
    d['Tair']  = fl['TA']
    d['Tsoil'] = (fl['TS_1']+fl['TS_2'])/2.0
    d['SR']    = fl['SW_IN_F']
    d['Patm']=fl['PA']; d['WS']=fl['WS']; d['WD']=fl['WD']; d['RH']=fl['RH']; d['VPD']=fl['VPD']

    # ---- FLUX CLEANING: Papale (2006) despike UNION absolute limit -----------
    meas = fl['FCH4']*CF
    day  = fl['SW_IN_F'] > DAY_SW
    fp   = papale_flag(meas, day, Z)                       # (A) local statistical spikes
    F    = fl['FCH4_F']*CF
    fa   = (F.abs() > SPIKE_MAX).values                    # (B) isolated physical extremes
    spike= fp | fa
    orig = fl['FCH4_F'].copy(); orig[spike] = np.nan
    orig = orig.interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')
    d['F_CH4_F_orig'] = orig
    d['F_CH4_F']      = orig*CF
    stats = dict(papale=int(fp.sum()), absol=int(fa.sum()), union=int(spike.sum()),
                 papale_only=int((fp&~fa).sum()), abs_only=int((fa&~fp).sum()), both=int((fp&fa).sum()),
                 kept_max=float(np.nanmax(np.abs(d['F_CH4_F'].values[~spike]))))
    # -------------------------------------------------------------------------

    e=d['depth'].values; B=d['Deltime'].values; n=len(d)
    AUC=np.zeros(n); AUCw=np.zeros(n); AUCd=np.zeros(n)
    for k in range(1,n):
        dt=B[k]-B[k-1]; e0=e[k-1]; e1=e[k]
        if np.isnan(e0) or np.isnan(e1):
            AUC[k]=AUC[k-1]; AUCw[k]=AUCw[k-1]; AUCd[k]=AUCd[k-1]; continue
        AUC[k]=AUC[k-1]+dt*(e0+e1)/2
        AUCw[k]=AUCw[k-1]+(dt*(e0+e1)/2 if (e0>=0 and e1>=0) else (0.0 if (e0<=0 and e1<=0)
                 else dt*(max(e0,e1)**2)/(2*(max(e0,e1)+abs(min(e0,e1))))))
        AUCd[k]=AUCd[k-1]+((dt*(abs(e0)+abs(e1))/2) if (e0<=0 and e1<=0) else (0.0 if (e0>=0 and e1>=0)
                 else dt*(abs(min(e0,e1))**2)/(2*(max(e0,e1)+abs(min(e0,e1))))))
    d['AUC']=AUC; d['AUC_dry']=AUCd; d['AUC_wet']=AUCw
    d['rate']=d['depth'].diff()/d['Deltime'].diff()
    Ta=d['Tair']; RH=d['RH']; P=d['Patm']
    d['es']=0.6108*np.exp(17.27*Ta/(Ta+237.3)); d['ea']=(RH/100)*d['es']
    d['rate_P']=d['Patm'].diff()/d['Deltime'].diff()
    g=np.log(RH/100)+17.27*Ta/(237.3+Ta); d['Tdew']=237.3*g/(17.27-g)
    e_=d['ea']; d['q']=0.622*e_/(P-0.378*e_); d['wmix']=0.622*e_/(P-e_)
    d['DelTsa']=d['Tsoil']-d['Tair']
    d['hwet']=d['depth'].clip(lower=0); d['hdry']=(-d['depth']).clip(lower=0)
    for nm,a,b in [('h*Ta','depth','Tair'),('h*Ts','depth','Tsoil'),('h*WS','depth','WS'),
                   ('h*Pr','depth','Patm'),('h*VPD','depth','VPD'),('h*DelTsa','depth','DelTsa')]:
        d[nm]=d[a]*d[b]
    d['Tv_K']=(Ta+273.15)*(1+0.61*d['q'])
    d['rho_moist']=(P*1000)/(287.05*(Ta+273.15)*(1+0.61*d['q']))
    d['uzonal']=-d['WS']*np.sin(np.radians(d['WD'])); d['vmerid']=-d['WS']*np.cos(np.radians(d['WD']))
    d['buoy_TsTa']=9.81*(d['Tsoil']-d['Tair'])/(d['Tair']+273.15)
    d['SRxWS']=d['SR']*d['WS']; d['SR*u']=d['SR']*d['uzonal']; d['SR*v']=d['SR']*d['vmerid']
    d['SR*VPD']=d['SR']*d['VPD']; d['VPD*WS']=d['WS']*d['VPD']; d['q*WS']=d['WS']*d['q']
    d['h_ASINH_cm']=np.arcsinh(d['depth']); d['h_inv']=1/(d['depth']+0.001)
    d['u*VPD']=d['uzonal']*d['VPD']; d['v*VPD']=d['vmerid']*d['VPD']
    d['rate*Ta']=d['rate']*d['Tair']; d['rate*Ts']=d['rate']*d['Tsoil']
    d['SR*Ta']=d['SR']*d['Tair']; d['SR*Ts']=d['SR']*d['Tsoil']
    sin24=np.sin(2*np.pi*d['dayhr']/24); cos24=np.cos(2*np.pi*d['dayhr']/24)
    d['SR*HODsin']=d['SR']*sin24; d['VPD*WS*d1sin']=d['VPD']*d['WS']*sin24; d['q/rho']=d['q']/d['rho_moist']
    for nm,src in [('asinh_Ta','Tair'),('asinh_Ts','Tsoil'),('asinh_WS','WS'),
                   ('asinh_SR','SR'),('asinh_P_kPa','Patm'),('asinh_rate_h','rate')]:
        d[nm]=np.arcsinh(d[src])
    d['h*sinTOD']=d['depth']*sin24; d['h*cosTOD']=d['depth']*cos24
    d['h*u']=d['depth']*d['uzonal']; d['h*v']=d['depth']*d['vmerid']
    return d, stats

def main():
    for p in (OUT_FULL, OUT_GS):
        dp=os.path.dirname(p)
        if dp: os.makedirs(dp, exist_ok=True)
    fl=pd.read_csv(FLUXNET_HH, na_values=[-9999,'-9999'])
    fl['ts']=pd.to_datetime(fl['TIMESTAMP_START'].astype(str), format='%Y%m%d%H%M')+pd.Timedelta(minutes=30)
    fl=fl[(fl['ts']>=pd.Timestamp(REC_START))&(fl['ts']<=pd.Timestamp(REC_END))].sort_values('ts').reset_index(drop=True)
    d, st = build(fl)
    cols=['Date','Deltime','time','dayhr','depth','AUC','AUC_dry','AUC_wet','rate','Tair','Tsoil','SR',
          'Patm','WS','WD','RH','es','ea','VPD','rate_P','Tdew','q','wmix','DelTsa','hwet','hdry','h*Ta',
          'h*Ts','h*WS','h*Pr','h*VPD','h*DelTsa','Tv_K','rho_moist','uzonal','vmerid','buoy_TsTa','SRxWS',
          'SR*u','SR*v','SR*VPD','VPD*WS','q*WS','h_ASINH_cm','h_inv','u*VPD','v*VPD','rate*Ta','rate*Ts',
          'SR*Ta','SR*Ts','SR*HODsin','VPD*WS*d1sin','q/rho','asinh_Ta','asinh_Ts','asinh_WS','asinh_SR',
          'asinh_P_kPa','asinh_rate_h','h*sinTOD','h*cosTOD','h*u','h*v','F_CH4_F_orig','F_CH4_F']
    d=d[cols]
    pred=[c for c in cols if c not in ('Date','Deltime','time','F_CH4_F_orig','F_CH4_F')]
    d.to_csv(OUT_FULL, index=False, date_format=DATE_FMT, na_rep=MISSING)
    gs=d[(d['Date']>=pd.Timestamp(REC_START))&(d['Date']<=pd.Timestamp(GS_END))]
    gs.to_csv(OUT_GS, index=False, date_format=DATE_FMT, na_rep=MISSING)
    for _p in (OUT_FULL, OUT_GS):
        _chk=pd.read_csv(_p, dtype=str, keep_default_na=False)
        print(f"CHECK {_p}: blank cells = {int((_chk=='').sum().sum())} | -9999 cells = {int((_chk=='-9999').sum().sum())}")
    print(f"QC removed: Papale(z={Z})={st['papale']} + absolute(|F|>{SPIKE_MAX:.0f})={st['absol']} "
          f"= {st['union']} unique ({100*st['union']/len(d):.2f}%)  "
          f"[Papale-only {st['papale_only']}, abs-only {st['abs_only']}, both {st['both']}]")
    print(f"kept max |flux| = {st['kept_max']:.1f} mg/m2/h (was 136)")
    print(f"[SAVED] {OUT_FULL}  rows={len(d)}  F_CH4_F complete={d['F_CH4_F'].notna().sum()} ({100*d['F_CH4_F'].notna().mean():.1f}%)"
          f"  predictor-NaN rows={d[pred].isna().any(axis=1).sum()}")
    print(f"[SAVED] {OUT_GS}    rows={len(gs)} F_CH4_F complete={gs['F_CH4_F'].notna().sum()} ({100*gs['F_CH4_F'].notna().mean():.1f}%)"
          f"  predictor-NaN rows={gs[pred].isna().any(axis=1).sum()}")

if __name__=="__main__":
    main()