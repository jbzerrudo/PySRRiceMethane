"""
2_5retainvars.py — keep the union's Confirmed + Supported variables
==============================================================================
CHANGE, 31 July 2026. Paths are no longer hard-coded.

The previous version had three absolute paths typed at the top. Running four
arms means editing them four times, and a stale path fails SILENTLY: the script
reads last week's consensus file and writes a variable set that looks fine and
is wrong. That is the highest-risk step in the chain, because nothing
downstream can detect it.

This version reads the SAME config file the union used, so each arm has exactly
one place where its paths live and the two steps cannot drift apart.

    python 2_5retainvars.py pooled_config.txt

Config keys, the first three shared with GAM_RF_union_parallel.py:

    INPUT_DIR    folder holding the union's input CSV
    INPUT_FILE   that CSV's filename
    OUTPUT_DIR   where the union wrote its outputs
    RETVARS_OUT  full path for this script's output CSV

The consensus file is found by globbing OUTPUT_DIR for
consensus_important_variables_*.txt and taking the newest. The file used is
printed, and the run STOPS if it is older than the input CSV, which is the
signature of pointing at a previous run. It also STOPS if the consensus names a
variable the CSV does not contain, which is the signature of a mismatched pair.

Running with no argument falls back to the paths below, so the old behaviour is
still available.

Author: Jef Zerrudo / Claude
==============================================================================
"""

import glob
import os
import re
import sys

import pandas as pd

# ── Fallback paths, used only when no config file is given ──
txt_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\GAMRF\POOLED\Metadata\Intensive\consensus_important_variables_20260729_190125.txt"
csv_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\POOL_3sites_INTENSIVE.csv"
out_file = r"C:\Users\zerru001\OneDrive - Wageningen University & Research\Paper1\RUN2\CSV\POOLED\Data-Metadata\PooledIntensive_retvars_gam1.csv"

ALWAYS_KEEP = ["site", "w", "Date", "F_CH4_F"]


def read_config(path):
    cfg = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


def newest_consensus(outdir):
    hits = glob.glob(os.path.join(outdir, "consensus_important_variables_*.txt"))
    if not hits:
        raise SystemExit(f"\n  [STOP] no consensus_important_variables_*.txt in\n"
                         f"         {outdir}\n"
                         f"         Has the union finished for this arm?\n")
    hits.sort(key=os.path.getmtime)
    if len(hits) > 1:
        print(f"  [INFO] {len(hits)} consensus files present, taking the newest:")
        for h in hits[-3:]:
            print(f"         {os.path.basename(h)}")
    return hits[-1]


def main():
    global txt_file, csv_file, out_file

    if len(sys.argv) > 1 and os.path.isfile(sys.argv[1]):
        cfg = read_config(sys.argv[1])
        missing = [k for k in ("INPUT_DIR", "INPUT_FILE", "OUTPUT_DIR", "RETVARS_OUT")
                   if k not in cfg]
        if missing:
            raise SystemExit(f"\n  [STOP] config {sys.argv[1]} is missing: {missing}\n")
        csv_file = os.path.join(cfg["INPUT_DIR"], cfg["INPUT_FILE"])
        txt_file = newest_consensus(cfg["OUTPUT_DIR"])
        out_file = cfg["RETVARS_OUT"]
        print(f"  [CONFIG] {sys.argv[1]}")
    else:
        print("  [CONFIG] no config file given, using the hard-coded paths")

    print(f"  [IN ] csv       : {csv_file}")
    print(f"  [IN ] consensus : {txt_file}")
    print(f"  [OUT] retvars   : {out_file}")

    for p in (csv_file, txt_file):
        if not os.path.isfile(p):
            raise SystemExit(f"\n  [STOP] not found: {p}\n")

    # Staleness guard: a consensus older than its own input means a wrong path
    if os.path.getmtime(txt_file) < os.path.getmtime(csv_file):
        raise SystemExit(
            "\n  [STOP] the consensus file is OLDER than the input CSV.\n"
            f"         consensus : {txt_file}\n"
            f"         input csv : {csv_file}\n"
            "         OUTPUT_DIR points at a previous union run. Fix the config\n"
            "         rather than letting a stale variable set through.\n")

    columns_to_keep = list(ALWAYS_KEEP)
    statuses = {"Confirmed": 0, "Supported": 0, "Rejected": 0}
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r"\s*[\u2713\u2717]\s+(\S+)\s+(Confirmed|Supported|Rejected)",
                         line)
            if m:
                var_name, status = m.groups()
                statuses[status] += 1
                if status in ("Confirmed", "Supported"):
                    columns_to_keep.append(var_name)

    if statuses["Confirmed"] + statuses["Supported"] == 0:
        raise SystemExit("\n  [STOP] the consensus file yielded no Confirmed or\n"
                         "         Supported variables. Check it is the file you think.\n")

    df = pd.read_csv(csv_file, low_memory=False)
    final_columns = [c for c in df.columns if c in columns_to_keep]

    absent = [c for c in columns_to_keep
              if c not in df.columns and c not in ALWAYS_KEEP]
    if absent:
        raise SystemExit(
            "\n  [STOP] the consensus names variables that are not in the CSV:\n"
            f"         {absent}\n"
            "         The consensus and the CSV are from different runs. Fix the config.\n")

    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    df[final_columns].to_csv(out_file, index=False)

    print(f"\n  consensus: {statuses['Confirmed']} Confirmed, "
          f"{statuses['Supported']} Supported, {statuses['Rejected']} Rejected")
    print(f"  kept {len(final_columns)} columns, {len(df):,} rows")
    print(f"  predictors: {[c for c in final_columns if c not in ALWAYS_KEEP]}")
    print(f"\n  Saved: {out_file}")


if __name__ == "__main__":
    main()
