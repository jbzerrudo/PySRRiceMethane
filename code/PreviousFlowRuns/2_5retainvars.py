import pandas as pd
import os
import re

# ── File paths (user sets all three) ──
txt_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\POOLED\Metadata\Intensive\consensus_important_variables_20260729_190125.txt"
csv_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\POOL_3sites_INTENSIVE.csv"
out_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\PooledIntensive_retvars_gam1.csv"

# ── Extract the variables marked as "Confirmed" or "Supported" ──
columns_to_keep = ['site', 'w', 'Date', 'F_CH4_F']

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