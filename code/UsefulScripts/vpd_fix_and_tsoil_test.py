import pandas as pd, numpy as np
from scipy.optimize import curve_fit
np.seterr(all='ignore'); pd.set_option('display.width',240)

f='/root/.claude/uploads/beeeedb5-a1fb-56ac-bced-ab062bd29d1c/9311bf8d-Pooled_3sites_retvars_gam1.csv'
d=pd.read_csv(f).replace(-9999.0,np.nan).replace(-999900.0,np.nan)

# ---- UNIT FIX: VPD is hPa at JP-MSE and SK-CRK, kPa at PH-IR. Convert all to kPa.
hpa = d.site.isin(['JP-MSE','SK-CRK'])
d['VPD_kPa'] = np.where(hpa, d['VPD']/10.0, d['VPD'])
for child,src in [('h*VPD_fix','depth'),('u*VPD_fix','uzonal'),('v*VPD_fix','vmerid')]:
    d[child] = d[src]*d['VPD_kPa']

q = d.dropna(subset=['Tsoil','es','AUC','VPD','VPD_kPa','vmerid','depth','F_CH4_F']).copy()
print('rows for comparison:', len(q), q.site.value_counts().to_dict(), '\n')

def r2(y,p):
    m=np.isfinite(p)
    if m.sum()<50: return np.nan
    return 1-np.sum((y[m]-p[m])**2)/np.sum((y[m]-y[m].mean())**2)

def loso(fn,p0,cols,tag,extra=''):
    out=[]
    for hold in ['SK-CRK','JP-MSE','PH-IR']:
        tr,te=q[q.site!=hold],q[q.site==hold]
        n=tr.groupby('site')['es'].transform('size').values
        try:
            pp,_=curve_fit(fn,tuple(tr[c].values for c in cols),tr.F_CH4_F.values,
                           p0=p0,sigma=1/np.sqrt((len(tr)/2.)/n),maxfev=400000)
            out.append(r2(te.F_CH4_F.values,fn(tuple(te[c].values for c in cols),*pp)))
        except Exception: out.append(np.nan)
    n=q.groupby('site')['es'].transform('size').values
    try:
        pp,_=curve_fit(fn,tuple(q[c].values for c in cols),q.F_CH4_F.values,p0=p0,
                       sigma=1/np.sqrt((len(q)/3.)/n),maxfev=400000)
        ins=r2(q.F_CH4_F.values,fn(tuple(q[c].values for c in cols),*pp))
        ps=' '.join(f'{v:.4g}' for v in pp)
        q10=f'{np.exp(10*pp[1]):.2f}' if len(pp)>1 and 'Tsoil' in cols[0] else ''
    except Exception: ins,ps,q10=np.nan,'fail',''
    print(f'{tag:34s} LOSO {out[0]:7.3f} {out[1]:7.3f} {out[2]:7.3f} | mean {np.nanmean(out):6.3f} | pool {ins:6.3f} | Q10 {q10:>5s} | {ps}')

print('                                    SK-CRK  JP-MSE   PH-IR |  mean  | in-pool| Q10   | params')
print('--- matched skeletons, Tsoil vs es as the temperature carrier ---')
S1=lambda X,A,b: A*np.exp(b*X[0])
S2=lambda X,A,b,c: A*np.exp(b*X[0])*np.exp(c*X[1])
S3=lambda X,A,b,c: A*np.exp(b*X[0])*(np.tanh(X[1])+c)
for X,nm,p0b in [('Tsoil','Tsoil',0.10),('es','es   ',0.50)]:
    A0=float(q.F_CH4_F.mean()/np.exp(p0b*q[X].mean()))
    loso(S1,[A0,p0b],[X],              f'S1  A*exp(b*{nm})')
    loso(S2,[A0,p0b,1e-4],[X,'AUC'],   f'S2  A*exp(b*{nm})*exp(c*AUC)')
    loso(S3,[A0,p0b,1.5],[X,'AUC'],    f'S3  A*exp(b*{nm})*(tanh(AUC)+c)')

print('\n--- #23 skeleton: original VPD vs unit-corrected VPD ---')
def f23(X,a,b,c,d_):
    AUC,T,VPD,vm,hV=X
    t=(a*AUC)+(T-np.tanh(VPD))-np.tanh(vm+b)-np.tanh((AUC*(hV*VPD))*a)
    return t*(np.tanh(AUC)+T+c)+d_
P=[2.486764e-4,0.66067696,-1.8193812,0.6148757]
loso(f23,P,['AUC','es','VPD','vmerid','h*VPD'],      '#23 as published (VPD mixed units)')
loso(f23,P,['AUC','es','VPD_kPa','vmerid','h*VPD_fix'],'#23 with VPD corrected to kPa')
loso(f23,P,['AUC','Tsoil','VPD_kPa','vmerid','h*VPD_fix'],'#23 corrected, Tsoil for es')
