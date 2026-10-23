"""
Performance saturation curve: % of full-probe Token-F1 retained vs. top-K features.
Values are taken from Table l2_verification.
    python plot_l2_saturation.py
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

K = [1, 2, 5, 200, 400, 1024]
token_f1 = [19.94, 35.22, 31.72, 67.33, 71.21, 71.92]  # Token-F1 (%)
full_f1 = 71.92                                          # Full vector probe
baseline_f1 = 25.87                                      # Fixed 100 ms baseline

retained = [f / full_f1 * 100 for f in token_f1]
baseline_retained = baseline_f1 / full_f1 * 100          # ~35.97%

fig, ax = plt.subplots(figsize=(6.5, 4.5))

ax.plot(K, retained, marker="o", color="#1b9e77", linewidth=2, label="Top-$K$ L2 probe")
ax.axhline(100, color="grey", linestyle="--", linewidth=1, label="Full vector probe (100%)")
ax.axhline(baseline_retained, color="#d95f02", linestyle=":", linewidth=1.5,
           label=f"Fixed 100 ms baseline ({baseline_retained:.1f}%)")

# Value labels on each point
for k, r in zip(K, retained):
    ax.annotate(f"{r:.1f}", (k, r), textcoords="offset points", xytext=(0, 8),
                ha="center", fontsize=9)

ax.set_xscale("log")
ax.set_xticks(K)
ax.set_xticklabels([str(k) for k in K])
ax.minorticks_off()
ax.set_xlabel("Number of selected features $K$ (log scale)")
ax.set_ylabel("Token-F1 retained (%)")
ax.set_ylim(0, 112)
ax.set_title("Performance saturation with feature subset size")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
ax.spines[["top", "right"]].set_visible(False)

fig.tight_layout()
fig.savefig("l2_subspace_saturation.pdf", bbox_inches="tight")
print("Saved l2_subspace_saturation.pdf")