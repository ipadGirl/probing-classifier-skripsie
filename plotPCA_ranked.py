"""
Same stacked qualitative plot as plot_feature_norm_spectrogram.py, but the
feature panels show the top-K PRINCIPAL COMPONENTS (by variance) instead of
L1-selected raw features.

    spectrogram + syllable boundaries
    word strip
    PC0 ... PC{k-1}   (projected, scaled by 1/sqrt(eigenvalue))
    raw L2 norm

Usage:
    python plot_pc_norm_spectrogram.py --utt 5338-24615-0000 --k 5
    python plot_pc_norm_spectrogram.py --utts <id1>,<id2> --k 5 --save_dir figs
    python plot_pc_norm_spectrogram.py --utt <id> --k 5 --align_sign

Assumes pca_analysis_ctr.npz (from your PCA script) and the same
saved_features/ files as the original plotting script. k must be <= 50,
since only the first 50 components are saved in the .npz.
"""

import argparse
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

TAG = "ctr"
PCA_FILE = Path(f"pca_analysis_{TAG}.npz")

_state = {}  # filled in main(), read by the patched functions


def load_pcs(k, align_sign):
    d = np.load(PCA_FILE)
    assert k <= d["comps_top"].shape[1], "k exceeds the number of saved components (50)"
    eig = d["eigvals"]
    comps = d["comps_top"][:, :k].astype(np.float32)
    corr = d["corr"][:k].copy()
    if align_sign:  # PCA sign is arbitrary: flip so every PC correlates positively with the norm
        flip = np.where(corr < 0, -1.0, 1.0)
        comps = comps * flip.astype(np.float32)
        corr = corr * flip
    _state.update(
        mean=d["mean"].astype(np.float32),
        scale=d["scale"].astype(np.float32),
        comps=comps,
        sd=np.sqrt(eig[:k]).astype(np.float32),
        pct=100 * eig[:k] / eig.sum(),
        corr=corr,
    )


# ---- patched loader: reuse the original, then swap in PC activations ----
_orig_load = base._load_utterance_data


def _load_with_pcs(utt_id, top_k_indices, mean, std):
    data = _orig_load(utt_id, np.array([0]),
                      np.zeros(1024, np.float32), np.ones(1024, np.float32))
    X_dev = np.load(base.OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt_dev = np.load(base.OUTPUT_DIR / "utt_dev.npy")
    raw = np.asarray(X_dev[utt_dev == utt_id], dtype=np.float32)
    scores = ((raw - _state["mean"]) / _state["scale"]) @ _state["comps"]
    data["feats_z"] = scores / _state["sd"]          # unit-variance PC activations
    return data


# ---- patched drawer: reuse the original, then relabel the y-axes ----
_orig_draw = base._draw_utterance_block


def _draw_with_pc_labels(axes_block, utt_id, data, top_k_indices, top_k_coefs, max_time=3.0):
    _orig_draw(axes_block, utt_id, data, top_k_indices, top_k_coefs, max_time)
    k = len(top_k_indices)
    for i, ax in enumerate(axes_block[2:2 + k]):
        ax.set_ylabel(f"PC{i}\n({_state['pct'][i]:.1f}% var,\nr={_state['corr'][i]:+.2f})",
                      fontsize=8)


base._load_utterance_data = _load_with_pcs
base._draw_utterance_block = _draw_with_pc_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utt", type=str, default=None)
    ap.add_argument("--utts", type=str, default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--max_time", type=float, default=3.0)
    ap.add_argument("--save_dir", type=str, default="qualitative_plots") # Default to saving directly
    ap.add_argument("--align_sign", action="store_true")
    args = ap.parse_args()

    load_pcs(args.k, args.align_sign)
    idx = np.arange(args.k)
    coefs = _state["corr"]          # passed through; labels are overwritten above
    max_time = args.max_time if args.max_time > 0 else None

    if args.utts:
        targets = [u.strip() for u in args.utts.split(",") if u.strip()]
        base.plot_utterance_comparison(targets, idx, coefs, None, None,
                                       save_dir=args.save_dir, max_time=max_time)
    else:
        utt = args.utt or np.unique(np.load(base.OUTPUT_DIR / "utt_dev.npy"))[0]
        base.plot_utterance(utt, idx, coefs, None, None,
                            save_dir=args.save_dir, max_time=max_time)


if __name__ == "__main__":
    main()