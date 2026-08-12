import pandas as pd, numpy as np

BASE = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV"
k = pd.read_csv(BASE + r"\KOR\Data_Metadata\Papale_hampel_cleaned\KOR_retvars_pass2_C.csv")
p = pd.read_csv(BASE + r"\POOLED\Data-Metadata\POOLED_retvars_pass2_ISO_C.csv")
p = p[p["site"] == "SK-CRK"]

a = pd.to_numeric(k["F_CH4_F"], errors="coerce").dropna().sort_values().to_numpy()
b = pd.to_numeric(p["F_CH4_F"], errors="coerce").dropna().sort_values().to_numpy()
print(f"cleaned KOR: n={len(a)}, max={a.max():.3f}, p99.9={np.percentile(a,99.9):.3f}")
print(f"pooled SK-CRK: n={len(b)}, max={b.max():.3f}, p99.9={np.percentile(b,99.9):.3f}")
if len(a) == len(b):
    print(f"max sorted-value difference: {np.abs(a - b).max()}")
print("KOR dates look like:   ", k["Date"].head(2).tolist())
print("POOLED dates look like:", p["Date"].head(2).tolist())