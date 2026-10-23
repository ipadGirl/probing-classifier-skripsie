"""
Feature variance, raw vs. z-score standardised (single log-scale histogram).
Run from the folder containing `saved_features/`:
    python plot_feature_variance_hist.py
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

FEATURE_FILE = "saved_features/X_train.npy"
SAMPLE_SIZE = 200_000
EPS = 1e-12  # avoids log(0) for any constant features

# Load (memory-mapped) and subsample rows
X = np.load(FEATURE_FILE, mmap_mode="r")
n = X.shape[0]
rng = np.random.default_rng(42)
if n > SAMPLE_SIZE:
    idx = np.sort(rng.choice(n, size=SAMPLE_SIZE, replace=False))
    Xs = np.asarray(X[idx], dtype=np.float64)
else:
    Xs = np.asarray(X, dtype=np.float64)

# Per-feature variance before and after z-scoring
var_raw = np.clip(Xs.var(axis=0), EPS, None)
var_z = np.clip(StandardScaler().fit_transform(Xs).var(axis=0), EPS, None)

print(f"Raw variance: min={var_raw.min():.4g}, median={np.median(var_raw):.4g}, "
      f"max={var_raw.max():.4g}, max/min={var_raw.max() / var_raw.min():.1f}x")

# Shared log-spaced bins so both histograms are directly comparable
lo = min(var_raw.min(), var_z.min())
hi = max(var_raw.max(), var_z.max())
bins = np.logspace(np.log10(lo) - 0.1, np.log10(hi) + 0.1, 60)

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.hist(var_raw, bins=bins, color="#d95f02", alpha=0.7, label="Raw")
ax.hist(var_z, bins=bins, color="#1b9e77", alpha=0.9, label="Z-score standardised")

ax.set_xscale("log")
ax.set_xlabel("Per-feature variance (log scale)")
ax.set_ylabel("Number of features")
ax.set_title(f"Variance of the {Xs.shape[1]} embedding dimensions")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("feature_variance_hist.pdf", dpi=300, bbox_inches="tight")
print("Saved feature_variance_hist.pdf")