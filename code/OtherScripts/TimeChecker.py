import pandas as pd
df = pd.read_csv(r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\CSV\KOR\postgamrf2_run\KOR-CRK_2018.0_retdvars2.csv")   # set path
COL = "Date"                                        # set your date column name
dt = pd.to_datetime(df[COL], errors="coerce")
bad = df[dt.isna()]
print(f"{len(bad)} of {len(df)} rows unparseable")
print(bad[COL].head(20).tolist())