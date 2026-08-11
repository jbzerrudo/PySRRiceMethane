"""
plot_cheorwon_DAT.py
=============================================================================
Methane, AUC, and water depth vs DAYS AFTER TRANSPLANTING (DAT) for Cheorwon
(KR-CRK) 2018. Transplant date from the site paper (Hwang et al. 2020, Agr For
Meteorol 285-286, 107933, Table 1): DOY 117 = 27 April 2018.

Methane = Papale (2006) + absolute-limit cleaned flux (same as your pipeline),
daily mean. AUC = cumulative trapezoidal integral of water depth (same formula
as your engineered library), daily mean. Depth = daily-mean water table depth.

Output: Cheorwon_DAT_profile.png
Requires: numpy, pandas, matplotlib.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FLUXNET_HH = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\KOREA\FLX_KR-CRK_FLUXNET-CH4_2015-2018_1-1\FLX_KR-CRK_FLUXNET-CH4_HH_2015-2018_1-1.csv"
OUT_PNG = r"Cheorwon_DAT_profile.png"
REC_START="2018-04-09 17:30"; REC_END="2018-12-31 23:30"
TRANSPLANT="2018-04-27"          # DOY 117 (Hwang et al. 2020, Table 1)
DRAIN=("2018-06-06","2018-06-20")# DOY 157-171 mid-season drainage
HARVEST="2018-08-28"             # DOY 240
Z=5.5; DAY_SW=20.0; SPIKE_MAX=60.0; INTERP_LIMIT=6; CF=0.0577
DAT_LO, DAT_HI = -20, 160        # focus on pre-transplant flooding through post-harvest

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
t=fl['ts']; depth=fl['WTD']*100.0
meas=fl['FCH4']*CF; day=fl['SW_IN_F']>DAY_SW
fp=papale_flag(meas,day,Z); fa=((fl['FCH4_F']*CF).abs()>SPIKE_MAX).values
ch4=(fl['FCH4_F'].mask(fp|fa).interpolate(method='linear',limit=INTERP_LIMIT,limit_area='inside'))*CF
# cumulative AUC of depth (net trapezoid, hours), same as engineered library
hh=(t-t.iloc[0]).dt.total_seconds().values/3600.0
e=depth.values; auc=np.zeros(len(e))
for k in range(1,len(e)):
    if np.isnan(e[k-1]) or np.isnan(e[k]): auc[k]=auc[k-1]
    else: auc[k]=auc[k-1]+(hh[k]-hh[k-1])*(e[k-1]+e[k])/2

df=pd.DataFrame({'day':t.dt.floor('D'),'ch4':ch4.values,'depth':depth.values,'auc':auc})
daily=df.groupby('day').mean()
DAT=(daily.index - pd.Timestamp(TRANSPLANT)).days.values
dat_dr=[(pd.Timestamp(DRAIN[0])-pd.Timestamp(TRANSPLANT)).days,(pd.Timestamp(DRAIN[1])-pd.Timestamp(TRANSPLANT)).days]
dat_hv=(pd.Timestamp(HARVEST)-pd.Timestamp(TRANSPLANT)).days
sel=(DAT>=DAT_LO)&(DAT<=DAT_HI)

fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
def marks(a):
    a.axvline(0,color='#1f6f3f',ls='--',lw=1.3)
    a.axvspan(dat_dr[0],dat_dr[1],color='#e6d5b8',alpha=.6)
    a.axvline(dat_hv,color='#8e5a3b',ls='--',lw=1.3)
    a.axhline(0,color='0.7',lw=.7); a.grid(alpha=.25)
ax[0].plot(DAT[sel],daily['ch4'].values[sel],c='#1f6f3f',lw=1.6); marks(ax[0])
ax[0].set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$)")
ax[0].set_title("Cheorwon 2018 vs days after transplanting (transplant 27 Apr, drainage 6-20 Jun, harvest 28 Aug)",fontsize=10,loc='left')
ax[1].plot(DAT[sel],daily['auc'].values[sel],c='#2b6cb0',lw=1.6); marks(ax[1])
ax[1].set_ylabel("AUC (cm h, cumulative)")
ax[2].plot(DAT[sel],daily['depth'].values[sel],c='#3182bd',lw=1.6); marks(ax[2])
ax[2].fill_between(DAT[sel],0,daily['depth'].values[sel],where=daily['depth'].values[sel]>0,color='#3182bd',alpha=.2)
ax[2].set_ylabel("water depth (cm)")
ax[2].set_xlabel("days after transplanting (DAT)")
plt.tight_layout(); plt.savefig(OUT_PNG,dpi=200,bbox_inches='tight'); plt.close()
print("transplant DAT0 = %s | drainage DAT %d-%d | harvest DAT %d"%(TRANSPLANT,dat_dr[0],dat_dr[1],dat_hv))
print("saved:", OUT_PNG)
