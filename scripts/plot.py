# plot.py: reads results/bench.csv (written by bench.cu) and saves
# results/gflops.png, one line per version, GFLOP/s against N.
# Run from the repo root: python3 scripts/plot.py
import csv
from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

data = defaultdict(lambda: ([], []))  # version -> (N list, GFLOP/s list)
with open("results/bench.csv") as f:
    for row in csv.DictReader(f):
        ns, gf = data[row["version"]]
        ns.append(int(row["N"]))
        gf.append(float(row["gflops"]))

order = ["naive", "tiled", "register_tiled", "cublas"]
fig, ax = plt.subplots(figsize=(7, 4.5))
for name in order:
    if name in data:
        ns, gf = data[name]
        ax.plot(ns, gf, marker="o", linewidth=2, label=name)

ax.set_xscale("log", base=2)
ax.set_xticks(sorted({n for ns, _ in data.values() for n in ns}))
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.set_xlabel("matrix size N (N x N)")
ax.set_ylabel("GFLOP/s")
ax.set_title("FP32 SGEMM on a Tesla T4")
ax.grid(True, alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/gflops.png", dpi=150)
print("saved results/gflops.png")
