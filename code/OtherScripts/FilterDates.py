import pandas as pd

import pandas as pd

INPUT  = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\DATA\KOR-CRK.xlsx"
OUTPUT = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_AprSep.xlsx"

df = pd.read_excel(INPUT)
df = df.replace(-999900, -9999)                      # standardize NoData fills to -9999

d = pd.to_datetime(df["Date"], errors="coerce")
keep = (d >= pd.Timestamp("2018-04-09")) & (d < pd.Timestamp("2018-10-01"))
out = df[keep]
out.to_excel(OUTPUT, index=False)

print(f"{len(df)} -> {len(out)} rows | {d[keep].min()} to {d[keep].max()}")
num = out.drop(columns=["Date"]).apply(pd.to_numeric, errors="coerce")
hot = sorted(num.columns[(num.abs() > 1e5).any()].tolist())
print("still |value|>1e5:", hot if hot else "none")
print(f"Wrote {OUTPUT}")
