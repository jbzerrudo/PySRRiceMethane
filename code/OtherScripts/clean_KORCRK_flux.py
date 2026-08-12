"""
clean_KORCRK_flux.py   (FINAL)
=============================================================================
Cleans the Cheorwon (KR-CRK) CH4 flux for Paper 1 and rebuilds the 66-column
engineered library on the cleaned target. Drop-in for your RUN2 cascade:
outputs go straight into ...\\RUN2\\CSV\\KOR\\ for pass-1 GAM_RF_union.py.

METHOD (flux cleaning) -- three steps, nothing else touched:
  1. Start from the PUBLISHED FLUXNET-CH4 gap-filled flux FCH4_F (nmol m-2 s-1),
     converted to mg m-2 h-1 (x 0.0577). Complete product, no gaps to begin with.
  2. Absolute-limit spike QC: flag |flux| > SPIKE_MAX mg m-2 h-1 as physically
     implausible and set to missing. SPIKE_MAX=60 flags 28 half-hours
     (25 positive +62..+136; 3 negative -66,-83,-84). GS 99th pct ~42.
  3. Refill ONLY those flagged half-hours by time-based LINEAR INTERPOLATION
     (22 of 28 are isolated singles). Result: 100% complete flux, spikes gone,
     every other value is the original FLUXNET number.

Predictors reproduce your original sheet EXACTLY (validated to 1e-9). The
scattered NaNs in Tsoil/Patm/etc. are pre-existing FLUXNET raw gaps (TS_1/TS_2,
PA missing in the source) -- they are in your original sheet too, NOT introduced
here; your cascade's dropna handles them as always (~230 rows dropped).

Requires: numpy, pandas.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import os
import numpy as np, pandas as pd

# --------------------------- USER CONFIG -----------------------------------
FLUXNET_HH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\KOREA\FLX_KR-CRK_FLUXNET-CH4_2015-2018_1-1\FLX_KR-CRK_FLUXNET-CH4_HH_2015-2018_1-1.csv"
OUT_FULL   = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR\21Jul26\KOR-CRK_2018_cleaned_full.csv"
OUT_GS     = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR\21Jul26\KOR-CRK_2018_cleaned_growingseason.csv"
REC_START  = "2018-04-09 17:30"      # end-of-interval stamp of first half-hour
REC_END    = "2018-12-31 23:30"
GS_END     = "2018-09-30 23:30"
SPIKE_MAX  = 60.0     # absolute-limit spike threshold, mg m-2 h-1
INTERP_LIMIT = 6      # max consecutive half-hours to linear-interpolate
CF         = 0.0577   # nmol m-2 s-1 -> mg m-2 h-1 (your sheet's factor; exact 16.04*3600/1e6=0.057744)
DATE_FMT   = "%d/%m/%Y %H:%M"   # matches your existing CSVs (DD/MM/YYYY HH:MM); no Excel re-mangling
MISSING    = "-9999"            # write missing cells as your -9999 flag (NOT blank, NOT 0)
# ---------------------------------------------------------------------------

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

    # ---- FLUX CLEANING (the 3-step method above) --------------------------
    F = fl['FCH4_F']*CF                                   # published gap-filled, mg/m2/h
    spike = F.abs() > SPIKE_MAX                           # step 2: absolute-limit spike QC
    orig = fl['FCH4_F'].copy(); orig[spike] = np.nan      # remove spikes (nmol/s)
    orig = orig.interpolate(method='linear', limit=INTERP_LIMIT, limit_area='inside')  # step 3: refill
    d['F_CH4_F_orig'] = orig
    d['F_CH4_F']      = orig*CF
    # -----------------------------------------------------------------------

    # ---- AUC family (original formulas: net trapezoid + triangular crossing)
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
    return d, int(spike.sum())

def main():
    for p in (OUT_FULL, OUT_GS):
        dp = os.path.dirname(p)
        if dp:
            os.makedirs(dp, exist_ok=True)                       # create output folder if missing
    fl=pd.read_csv(FLUXNET_HH, na_values=[-9999,'-9999'])
    fl['ts']=pd.to_datetime(fl['TIMESTAMP_START'].astype(str), format='%Y%m%d%H%M')+pd.Timedelta(minutes=30)
    fl=fl[(fl['ts']>=pd.Timestamp(REC_START))&(fl['ts']<=pd.Timestamp(REC_END))].sort_values('ts').reset_index(drop=True)
    d, n_spike = build(fl)
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
    print(f"spikes removed (|F|>{SPIKE_MAX:.0f}): {n_spike}")
    print(f"[SAVED] {OUT_FULL}")
    print(f"        rows={len(d)}  F_CH4_F complete={d['F_CH4_F'].notna().sum()} ({100*d['F_CH4_F'].notna().mean():.1f}%)"
          f"  rows w/ predictor-NaN (pre-existing FLUXNET gaps)={d[pred].isna().any(axis=1).sum()}")
    print(f"[SAVED] {OUT_GS}")
    print(f"        rows={len(gs)} F_CH4_F complete={gs['F_CH4_F'].notna().sum()} ({100*gs['F_CH4_F'].notna().mean():.1f}%)"
          f"  rows w/ predictor-NaN={gs[pred].isna().any(axis=1).sum()}")

if __name__=="__main__":
    main()
