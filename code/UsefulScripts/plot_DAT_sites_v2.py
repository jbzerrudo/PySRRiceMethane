"""
plot_DAT_sites.py
=============================================================================
Uniform days-after-transplanting (DAT) figures for the rice-paddy sites.
Methane is the CLEANED half-hourly flux, plotted as POINTS (not daily means),
on a COMMON axis so sites compare directly; AUC and water depth are daily
context lines, auto-scaled per site (the water regimes differ). Transplant
(DAT 0), mid-season drainage (shaded) and harvest are marked where known.

Two input kinds (set per site):
  kind="fluxnet" -> raw FLUXNET-CH4 half-hourly file; cleaned here with
     Step 1 Papale + Step 2 |F|>SPIKE_MAX (+ Step 3 Hampel if hampel_k set).
  kind="library" -> a pre-built engineered library (F_CH4_F already the target);
     read as-is (PHL is delivered already light-cleaned).

Per-site cleaning notes:
  Cheorwon = Papale + |F|>60 + Hampel(k=4)   (noisy site, needs Step 3)
  Mase     = Papale + |F|>60                 (Hampel off; k=4 would delete ebullition)
  PHL-IRRI = light 4-point cap only          (clean AWD baseline; no Papale/Hampel)

Transplant sources: Cheorwon = Hwang et al. 2020; Mase = Iwata et al. 2018;
PHL-IRRI = Zerrudo methods (transplant 14-21 Dec 2015, midpoint 17 Dec used).
Requires: numpy, pandas, matplotlib.   Author: Jef Zerrudo / Claude
=============================================================================
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CF=0.0577; Z=5.5; DAY_SW=20.0; SPIKE_MAX=60.0; HAMPEL_WIN=240; INTERP_LIMIT=6
DAT_LO, DAT_HI = -20, 160
YL_CH4=(-18, 62)   # common methane axis across sites; AUC/depth auto-scale per site

SITES = {
  "Cheorwon 2018": dict(kind="fluxnet",
      hh=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\KOREA\FLX_KR-CRK_FLUXNET-CH4_2015-2018_1-1\FLX_KR-CRK_FLUXNET-CH4_HH_2015-2018_1-1.csv",
      rec=("2018-04-09 17:30","2018-12-31 23:30"),
      transplant="2018-04-27", drain=("2018-06-06","2018-06-20"), harvest="2018-08-28",
      hampel_k=4.0, out="Cheorwon2018_DAT.png"),
  "Mase 2012": dict(kind="fluxnet",
      hh=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\FLUXNET\JAPAN\FLX_JP-Mse_FLUXNET-CH4_2012-2012_1-1\FLX_JP-Mse_FLUXNET-CH4_HH_2012-2012_1-1.csv",
      rec=("2012-01-01 00:30","2012-12-31 23:30"),
      transplant="2012-05-02", drain=("2012-06-24","2012-06-28"), harvest="2012-09-12",
      hampel_k=None, out="Mase2012_DAT.png"),
  "PHL-IRRI (UY) 2016": dict(kind="library",
      hh=r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\PHL\PHLIR_2016_cleaned.csv",
      transplant="2015-12-17", drain=None, harvest=None,
      hampel_k=None, out="PHL_IRRI2016_DAT.png"),
}

def papale_flag(meas, day, z):
    x=meas.values; n=len(x); d=np.full(n,np.nan)
    for i in range(1,n-1):
        if np.isfinite(x[i-1]) and np.isfinite(x[i]) and np.isfinite(x[i+1]):
            d[i]=(x[i]-x[i-1])-(x[i+1]-x[i])
    flag=np.zeros(n,bool); dm=day.values
    for mask in [dm, ~dm]:
        dd=np.where(mask,d,np.nan); Md=np.nanmedian(dd); MAD=np.nanmedian(np.abs(dd-Md))
        if not np.isfinite(MAD) or MAD==0: continue
        thr=z*MAD/0.6745
        flag |= mask & np.isfinite(d) & ((d<Md-thr)|(d>Md+thr))
    return flag

def hampel_flag(F, win, k):
    med=F.rolling(win,center=True,min_periods=win//3).median()
    mad=(F-med).abs().rolling(win,center=True,min_periods=win//3).median()*1.4826
    return ((F-med).abs() > k*mad).fillna(False).values

def _daily(t, depth, auc, transplant):
    g=pd.DataFrame({'day':t.dt.floor('D'),'depth':depth,'auc':auc}).groupby('day').agg(
        depth=('depth','mean'),auc=('auc','mean'),dn=('depth','count'))
    return (g.index-pd.Timestamp(transplant)).days.values, g['depth'].values, g['auc'].values, g['dn'].values

def build_fluxnet(cfg):
    fl=pd.read_csv(cfg["hh"], na_values=[-9999,'-9999'])
    fl['ts']=pd.to_datetime(fl['TIMESTAMP_START'].astype(str),format='%Y%m%d%H%M')+pd.Timedelta(minutes=30)
    fl=fl[(fl['ts']>=pd.Timestamp(cfg['rec'][0]))&(fl['ts']<=pd.Timestamp(cfg['rec'][1]))].sort_values('ts').reset_index(drop=True)
    t=fl['ts']; depth=fl['WTD']*100.0
    meas=fl['FCH4']*CF; day=fl['SW_IN_F']>DAY_SW
    fp=papale_flag(meas,day,Z); F=fl['FCH4_F']*CF; fa=(F.abs()>SPIKE_MAX).values
    F1=fl['FCH4_F'].mask(fp|fa).interpolate(method='linear',limit=INTERP_LIMIT,limit_area='inside')
    if cfg.get('hampel_k') is not None:
        F1=F1.mask(hampel_flag(F1,HAMPEL_WIN,cfg['hampel_k'])).interpolate(method='linear',limit_area='inside')
    ch4=(F1*CF).values
    hh=(t-t.iloc[0]).dt.total_seconds().values/3600.0; e=depth.values; auc=np.zeros(len(e))
    for k in range(1,len(e)):
        auc[k]=auc[k-1] if (np.isnan(e[k-1]) or np.isnan(e[k])) else auc[k-1]+(hh[k]-hh[k-1])*(e[k-1]+e[k])/2
    hhDAT=(t-pd.Timestamp(cfg['transplant'])).dt.total_seconds().values/86400.0
    dDAT,depth_d,auc_d,dn=_daily(t,depth.values,auc,cfg['transplant'])
    return hhDAT, ch4, dDAT, depth_d, auc_d, dn

def build_library(cfg):
    d=pd.read_csv(cfg["hh"])
    t=pd.to_datetime(d['Date'], dayfirst=True, errors='coerce')
    ch4=pd.to_numeric(d['F_CH4_F'],errors='coerce').values
    depth=pd.to_numeric(d['depth'],errors='coerce').values
    auc=pd.to_numeric(d['AUC'],errors='coerce').values
    hhDAT=(t-pd.Timestamp(cfg['transplant'])).dt.total_seconds().values/86400.0
    dDAT,depth_d,auc_d,dn=_daily(t,depth,auc,cfg['transplant'])
    return hhDAT, ch4, dDAT, depth_d, auc_d, dn

def plot(name, cfg):
    hhDAT, ch4, dDAT, depth_d, auc_d, dn = (build_library if cfg['kind']=='library' else build_fluxnet)(cfg)
    hv = (pd.Timestamp(cfg['harvest'])-pd.Timestamp(cfg['transplant'])).days if cfg.get('harvest') else None
    dr = [(pd.Timestamp(cfg['drain'][0])-pd.Timestamp(cfg['transplant'])).days,
          (pd.Timestamp(cfg['drain'][1])-pd.Timestamp(cfg['transplant'])).days] if cfg.get('drain') else None
    selh=(hhDAT>=DAT_LO)&(hhDAT<=DAT_HI); seld=(dDAT>=DAT_LO)&(dDAT<=DAT_HI)&(dn>0)
    fig,ax=plt.subplots(3,1,figsize=(10,9),sharex=True)
    def marks(a):
        a.axvline(0,color='#1f6f3f',ls='--',lw=1.3)
        if dr: a.axvspan(dr[0],dr[1],color='#e6d5b8',alpha=.7)
        if hv is not None: a.axvline(hv,color='#8e5a3b',ls='--',lw=1.3)
        a.axhline(0,color='0.7',lw=.7); a.grid(alpha=.25)
    ax[0].scatter(hhDAT[selh],ch4[selh],s=5,c='#1f6f3f',edgecolors='none',alpha=.45)
    marks(ax[0]); ax[0].set_ylabel("CH$_4$ (mg m$^{-2}$ h$^{-1}$)"); ax[0].set_ylim(*YL_CH4)
    ax[0].set_title(f"{name} vs days after transplanting (cleaned methane, half-hourly points)",fontsize=10,loc='left')
    ax[1].plot(dDAT[seld],auc_d[seld],c='#2b6cb0',lw=1.7); marks(ax[1]); ax[1].set_ylabel("AUC (cm h, cumulative)")
    ax[2].plot(dDAT[seld],depth_d[seld],c='#3182bd',lw=1.7)
    ax[2].fill_between(dDAT[seld],0,depth_d[seld],where=depth_d[seld]>0,color='#3182bd',alpha=.2)
    marks(ax[2]); ax[2].set_ylabel("water depth (cm)")
    ax[2].set_xlabel("days after transplanting (DAT)"); ax[2].set_xlim(DAT_LO,DAT_HI)
    plt.tight_layout(); plt.savefig(cfg['out'],dpi=200,bbox_inches='tight'); plt.close()
    print(f"{name}: transplant DAT0={cfg['transplant']} -> {cfg['out']}")

if __name__=="__main__":
    for name,cfg in SITES.items():
        plot(name,cfg)
