"""
plot_cheorwon_papale.py
=============================================================================
Before/after visualisation of the Cheorwon (KR-CRK) 2018 CH4 flux cleaning,
using the SAME QC as clean_KORCRK_flux_papale.py so the figure matches the
cleaned CSVs exactly:
  UNCLEANED = FLUXNET-CH4 gap-filled flux FCH4_F x 0.0577 (spikes present)
  CLEANED   = same series with Papale (2006) despike UNION |F|>60 removed
              and linear-interpolated

Outputs (saved to the folder you run this from):
  (1) Cheorwon_papale_before_after.png  two stacked panels (uncleaned / cleaned)
  (2) Cheorwon_papale_cleaned.png       single panel: cleaned + 7-day mean
Requires: numpy, pandas, matplotlib.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.dates as mdates

FLUXNET_HH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\KOREA\FLX_KR-CRK_FLUXNET-CH4_2015-2018_1-1\FLX_KR-CRK_FLUXNET-CH4_HH_2015-2018_1-1.csv"
OUT_BA   = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\KOR\21_Jul_2026_Cleaned\Cheorwon_papale_before_after.png"
OUT_ONE  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\Graphs\KOR\21_Jul_2026_Cleaned\Cheorwon_papale_cleaned.png"
REC_START="2018-04-09 17:30"; REC_END="2018-12-31 23:30"
GS_START ="2018-04-09";       GS_END ="2018-09-30"
Z=5.5; DAY_SW=20.0; SPIKE_MAX=60.0; INTERP_LIMIT=6; CF=0.0577

def papale_flag(meas, day, z):
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

fl=pd.read_csv(FLUXNET_HH, na_values=[-9999,'-9999'])
fl['ts']=pd.to_datetime(fl['TIMESTAMP_START'].astype(str),format='%Y%m%d%H%M')+pd.Timedelta(minutes=30)
fl=fl[(fl['ts']>=pd.Timestamp(REC_START))&(fl['ts']<=pd.Timestamp(REC_END))].sort_values('ts').reset_index(drop=True)
t=fl['ts']
uncleaned=fl['FCH4_F']*CF
meas=fl['FCH4']*CF; day=fl['SW_IN_F']>DAY_SW
fp=papale_flag(meas,day,Z)                     # (A) local statistical spikes
fa=(uncleaned.abs()>SPIKE_MAX).values          # (B) isolated physical extremes
spike=fp|fa
cn=fl['FCH4_F'].copy(); cn[spike]=np.nan
cn=cn.interpolate(method='linear',limit=INTERP_LIMIT,limit_area='inside')
cleaned=cn*CF
kept_pct=100*(1-spike.sum()/len(fl))
roll=pd.Series(cleaned.values,index=t).rolling('7D',min_periods=48).mean()

# ---- Figure 1: two-panel before/after ------------------------------------
fig,(ax1,ax2)=plt.subplots(2,1,figsize=(11,7),sharex=True,sharey=True)
for ax in (ax1,ax2):
    ax.axvspan(pd.Timestamp(GS_START),pd.Timestamp(GS_END),color='#f0f4e8')
    ax.axhline(0,color='0.6',lw=.8); ax.set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$)")
ax1.scatter(t,uncleaned,s=4,c='0.6',alpha=.5)
ax1.scatter(t[fp&~fa],uncleaned[fp&~fa],s=24,facecolors='none',edgecolors='#c0392b',lw=1.0,
            label=f'Papale (2006) despike: {int((fp&~fa).sum())}')
ax1.scatter(t[fa],uncleaned[fa],s=36,facecolors='none',edgecolors='#8e44ad',lw=1.2,
            label=f'absolute limit |F|>{SPIKE_MAX:.0f}: {int(fa.sum())}')
ax1.set_title("UNCLEANED: FLUXNET-CH$_4$ flux, with the two QC criteria marked",fontsize=10,loc='left')
ax1.legend(fontsize=8,loc='upper right')
ax2.scatter(t,cleaned,s=4,c='0.6',alpha=.5,label=f'cleaned ({kept_pct:.1f}% kept)')
ax2.plot(roll.index,roll.values,c='#1f6f3f',lw=2,label='7-day mean')
ax2.set_title(f"CLEANED: Papale despike + absolute limit, {int(spike.sum())} removed and interpolated",fontsize=10,loc='left')
ax2.legend(fontsize=8,loc='upper right')
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax2.set_xlabel("2018 (shaded = growing season)")
ax1.set_ylim(-95,145)
plt.tight_layout(); plt.savefig(OUT_BA,dpi=200,bbox_inches='tight'); plt.close()

# ---- Figure 2: single-panel cleaned --------------------------------------
fig,ax=plt.subplots(figsize=(11,4.2))
ax.axvspan(pd.Timestamp(GS_START),pd.Timestamp(GS_END),color='#f0f4e8'); ax.axhline(0,color='0.6',lw=.8)
ax.scatter(t,cleaned,s=4,c='0.6',alpha=.5,label=f'F_CH4_F cleaned ({kept_pct:.1f}% of FLUXNET values kept)')
ax.plot(roll.index,roll.values,c='#1f6f3f',lw=2,label='7-day mean')
ax.set_ylim(-95,145); ax.set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$)")
ax.set_title(f"Cheorwon 2018 CLEANED: Papale (2006) despike + |F|>{SPIKE_MAX:.0f}, {int(spike.sum())} of {len(fl)} removed",fontsize=10,loc='left')
ax.legend(fontsize=8,loc='upper right'); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
ax.set_xlabel("2018 (shaded = growing season)")
plt.tight_layout(); plt.savefig(OUT_ONE,dpi=200,bbox_inches='tight'); plt.close()
print(f"Papale={int(fp.sum())} absolute={int(fa.sum())} union={int(spike.sum())} kept%={kept_pct:.2f}  saved: {OUT_BA} , {OUT_ONE}")