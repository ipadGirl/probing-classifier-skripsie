import json
import joblib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# --- Clean Native Font & Style Configuration ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Carlito", "Caladea", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MODEL_DIR = Path("models")
TOP = [869, 376, 4, 838, 590]
LOG_Y = True   # set False for a linear y-axis

# ---- load saved raw-feature stats + probe ----
meta = json.load(open(MODEL_DIR / "l1_probe_meta.json"))
var = np.asarray(meta["std"], dtype=np.float64) ** 2   # raw per-feature variance
clf = joblib.load(MODEL_DIR / "l1_probe.joblib")
coefs = clf.coef_.ravel()

D = len(var)
x = np.arange(D)
rank_desc = (-var).argsort().argsort() + 1              # 1 = highest variance
pctile = np.array([100 * np.mean(var < v) for v in var])

# ---- table for the top probe features ----
print(f"Variance across all {D} features: "
      f"min={var.min():.4g}, median={np.median(var):.4g}, max={var.max():.4g}")
print(f"\n{'feature':<9}{'coef':>9}{'variance':>12}{'var rank':>10}{'percentile':>12}")
for i in TOP:
    print(f"{i:<9}{coefs[i]:>9.4f}{var[i]:>12.4g}{rank_desc[i]:>7}/{D}{pctile[i]:>11.1f}%")

# ---- plot: x = feature index, y = variance ----
colors = plt.cm.tab10(np.arange(len(TOP)))
fig, ax = plt.subplots(figsize=(12, 5))

ax.bar(x, var, width=1.0, color="lightgray", label="all features")
for c, i in zip(colors, TOP):
    ax.bar(i, var[i], width=3.0, color=c, edgecolor="k", linewidth=0.5,
           label=f"feat {i} (coef {coefs[i]:+.3f})", zorder=3)
    ax.annotate(str(i), (i, var[i]), textcoords="offset points",
                xytext=(0, 4), ha="center", fontsize=9)

ax.axhline(np.median(var), color="k", ls="--", lw=0.8, label="median variance")
if LOG_Y:
    ax.set_yscale("log")
ax.set_xlim(-0.5, D - 0.5)
ax.set_xlabel("Feature index")
ax.set_ylabel("Raw variance" + (" (log)" if LOG_Y else ""))
ax.set_title("Per-feature variance (raw, un-normalised) with top probe features highlighted")
ax.legend(ncol=3, fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("feature_variances.pdf", dpi=300, bbox_inches="tight") # <-- Changed to .pdf
plt.close(fig)
print("Saved feature variances plot to: feature_variances.pdf")