"""
PCA subspaces vs. L1-selected features: Token-F1 as a function of K.
Values are hardcoded from Table l2_verification and Table pca_verification.
    python plot_pca_vs_l1_final.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

K = [1, 2, 5, 200, 400]
pca_f1 = [51.64, 53.08, 57.41, 76.91, 78.83]  # Top-K PCA Token-F1 (%)
l1_f1 = [19.94, 35.22, 31.72, 67.33, 71.21]   # Top-K L1-selected Token-F1 (%)
FULL_F1 = 71.92                                # Full 1024-dim probe

fig, ax = plt.subplots(figsize=(7, 4.6))

ax.plot(K, pca_f1, marker="o", linewidth=2, color="#1b9e77",
        label="PCA subspace (top-$K$ PCs)")
ax.plot(K, l1_f1, marker="s", linewidth=2, color="#d95f02",
        label="L1-selected features (top-$K$)")
ax.axhline(FULL_F1, color="grey", linestyle="--", linewidth=1.2,
           label=f"Full 1024-dim probe ({FULL_F1:.2f}%)")

# Value labels: PCA above its points, L1 below its points
for k, v in zip(K, pca_f1):
    ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9, color="#1b9e77")
for k, v in zip(K, l1_f1):
    ax.annotate(f"{v:.1f}", (k, v), textcoords="offset points", xytext=(0, -14),
                ha="center", fontsize=9, color="#d95f02")

ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.minorticks_off()
ax.set_xlabel("Number of components / features $K$ (log scale)")
ax.set_ylabel("Token-F1 (%)")
ax.set_ylim(10, 88)
ax.set_title("PCA vs. L1 feature selection")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("pca_vs_l1_comparison.pdf", bbox_inches="tight")
print("Saved pca_vs_l1_comparison.pdf")