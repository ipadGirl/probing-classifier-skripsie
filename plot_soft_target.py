"""
plot_gaussian_target.py -- Reproduces the soft-Gaussian vs hard target probe plot.

Usage:
    python3 plot_gaussian_target.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# --- Clean Native Font & Style Configuration ---
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "sans-serif",
    "font.sans-serif": ["Carlito", "Caladea", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 11,
    "font.size": 10,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

def main():
    # Frame index range (t from 5 to 50)
    t = np.arange(5, 51)
    
    # Boundary locations
    boundaries = [15, 40]
    h = 1.5  # bandwidth parameter

    # 1) Hard target (0/1 square pulses around boundaries 15 and 40)
    # Width of pulse is from roughly t-1 to t+1 or similar, let's construct it precisely
    hard_target = np.zeros_like(t, dtype=float)
    for b in boundaries:
        hard_target[(t >= b - 1) & (t <= b + 1)] = 1.0

    # 2) Soft Gaussian target
    # Sum of Gaussians centered at the boundaries with std deviation related to h
    soft_target = np.zeros_like(t, dtype=float)
    for b in boundaries:
        soft_target += np.exp(-((t - b) ** 2) / (2 * (h ** 2)))

    # Set up the plot figure size matching the proportions
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Plot Hard Target (gray step/stairs look)
    ax.step(t, hard_target, where='mid', color='gray', linewidth=1.5, label="Hard target (0/1)")
    
    # Plot Soft Gaussian Target (blue curve)
    ax.plot(t, soft_target, color='#1f77b4', linewidth=2.5, label=f"Soft Gaussian target ($h = {h}$)")

    # Plot gray scatter points on the markers
    ax.scatter(t, hard_target, color='gray', s=20, zorder=3)
    ax.scatter(t, soft_target[t - t[0]], color='#1f77b4', s=0) # invisible or map points if needed

    # Red dashed boundary lines
    for b in boundaries:
        ax.axvline(b, color='#d62728', linestyle='--', linewidth=0.8, alpha=0.8)
        ax.text(b, 1.08, "boundary", color='#d62728', ha='center', va='bottom', fontsize=9)

    # Annotate bandwidth 'h' arrow near the first boundary
    # Center is at 15, let's show arrow from 15 to 15 + h (or 14.5 to 16)
    ax.annotate(text='', xy=(15, 0.6), xytext=(15 - h, 0.6),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))
    ax.annotate(text='', xy=(15, 0.6), xytext=(15 + h, 0.6),
                arrowprops=dict(arrowstyle='<->', lw=1.2, color='black'))
    ax.text(15, 0.65, "$h$", ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Formatting axes and labels
    ax.set_xlim(5, 50)
    ax.set_ylim(-0.08, 1.2)
    ax.set_xlabel("Frame index $t$")
    ax.set_ylabel("Target value $y(t)$")

    # Clean up top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Custom legend placement at the top center
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False)

    plt.tight_layout()

    # Save to PDF
    out_path = "soft_gaussian_target.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()