"""
Stacked qualitative plot: spectrogram + boundaries, word strip,
feature 376 (z-scored), PC0 (unit variance), raw L2 norm.

Usage:
    python plot_376_pc0_norm.py --utt 5338-24615-0000 --save_dir qualitative_plots
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import zerosylprobes.plot_feature_spectogram as base

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

FEAT = 376
TAG = "ctr"
PCA_FILE = Path(f"pca_analysis_{TAG}.npz")

_state = {}


def load_state():
    d = np.load(PCA_FILE)
    comp0 = d["comps_top"][:, 0].astype(np.float32)
    corr0 = float(d["corr"][0])
    flip = -1.0 if corr0 < 0 else 1.0          # PCA sign is arbitrary: make r with the norm positive
    _state.update(
        pc_mean=d["mean"].astype(np.float32),
        pc_scale=d["scale"].astype(np.float32),
        comp0=comp0 * flip,
        sd0=float(np.sqrt(d["eigvals"][0])),
        pct0=float(100 * d["eigvals"][0] / d["eigvals"].sum()),
        r0=abs(corr0),
    )
    with open(base.META_FILE) as f:
        meta = json.load(f)
    _state["f_mean"] = np.array(meta["mean"], dtype=np.float32)
    _state["f_std"] = np.array(meta["std"], dtype=np.float32)


_orig_load = base._load_utterance_data
_orig_draw = base._draw_utterance_block


def _load_two(utt_id, top_k_indices, mean, std):
    data = _orig_load(utt_id, np.array([FEAT]), _state["f_mean"], _state["f_std"])
    X_dev = np.load(base.OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt_dev = np.load(base.OUTPUT_DIR / "utt_dev.npy")
    raw = np.asarray(X_dev[utt_dev == utt_id], dtype=np.float32)
    pc0 = (((raw - _state["pc_mean"]) / _state["pc_scale"]) @ _state["comp0"]) / _state["sd0"]
    f376 = data["feats_z"][:, 0]
    data["feats_z"] = np.column_stack([f376, pc0])      # panel 1 = feat 376, panel 2 = PC0
    print(f"[{utt_id}] r(feat376, norm)={np.corrcoef(f376, data['norm'])[0,1]:+.3f}  "
          f"r(PC0, norm)={np.corrcoef(pc0, data['norm'])[0,1]:+.3f}  "
          f"r(feat376, PC0)={np.corrcoef(f376, pc0)[0,1]:+.3f}")
    return data


def _draw_two(axes_block, utt_id, data, top_k_indices, top_k_coefs, max_time=3.0):
    _orig_draw(axes_block, utt_id, data, top_k_indices, top_k_coefs, max_time)
    axes_block[2].set_ylabel(f"feat_{FEAT}\n(z-scored)", fontsize=8)
    axes_block[3].set_ylabel(f"PC0\n({_state['pct0']:.1f}% var,\nr={_state['r0']:+.2f})", fontsize=8)


base._load_utterance_data = _load_two
base._draw_utterance_block = _draw_two


# --- Custom plot function to save with a unique filename ---
def plot_custom_utterance(utt_id, top_k_indices, top_k_coefs, mean, std, save_dir=None, max_time=3.0):
    data = base._load_utterance_data(utt_id, top_k_indices, mean, std)
    k = len(top_k_indices)

    fig, axes = plt.subplots(
        k + 3, 1, figsize=(12, base._block_height(k)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.5] + [1] * k + [1.2]},
    )
    base._draw_utterance_block(axes, utt_id, data, top_k_indices, top_k_coefs, max_time)
    axes[-1].set_xlabel("Time (s)")
    base._add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{utt_id}_feat376_pc0_norm.pdf"  # <-- Unique filename
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt", type=str, default="5338-24615-0000")
    ap.add_argument("--max_time", type=float, default=3.0)
    ap.add_argument("--save_dir", type=str, default="qualitative_plots")
    args = ap.parse_args()

    load_state()
    max_time = args.max_time if args.max_time > 0 else None
    
    plot_custom_utterance(args.utt, np.array([FEAT, 0]), np.zeros(2), None, None,
                          save_dir=args.save_dir, max_time=max_time)


if __name__ == "__main__":
    main()