import pandas as pd
import os
import re

# ── File paths (user sets all three) ──
txt_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\KOR\gamrf_kor_newrun2\consensus_important_variables_20260622_164249.txt"  # <-- set this to your text file
csv_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\COLLINCHECK\KOR\vif5_collincheck_newrun_kor\KOR-CRK_2018_20260622_163809\KOR-CRK_2018_retainedvars_postcollin_20260622_163809.csv"  # <-- set this to your original CSV dataset
out_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\KOR-CRK_2018_updated_retvars2.csv"  # <-- set this to whatever you want

# ── Extract the variables marked as "Confirmed" or "Supported" ──
columns_to_keep = ['Date', 'F_CH4_F']

with open(txt_file, 'r', encoding='utf-8') as f:
    for line in f:
        match = re.match(r'\s*[✓✗]\s+(\S+)\s+(Confirmed|Supported|Rejected)', line)
        if match:
            var_name, status = match.groups()
            if status in ['Confirmed', 'Supported']:
                columns_to_keep.append(var_name)

# ── Load the original CSV dataset ──
df = pd.read_csv(csv_file)

# ── Filter columns while preserving their original order in the CSV ──
final_columns = [col for col in df.columns if col in columns_to_keep]
filtered_df = df[final_columns]

# ── Export ──
os.makedirs(os.path.dirname(out_file), exist_ok=True)
filtered_df.to_csv(out_file, index=False)
print(f"Saved: {out_file}")
print(f"Kept {len(final_columns)} columns: {final_columns}")