"""
Combined qualitative plot: for each utterance, stack (top to bottom)
1. mel/linear spectrogram with ground-truth syllable boundaries
    2. word transcript strip
    3. top-K L1-selected features (z-scored)
    4. RAW WavLM feature-vector L2 norm (the NormSeg baseline signal)

so the L1-probe features and the norm signal can be compared directly
against the same boundaries and words.

Usage:
    python plot_feature_norm_spectrogram.py --utt <utt_id> --k 5
    python plot_feature_norm_spectrogram.py --utts <id1>,<id2> --k 5   # comparison figure
    python plot_feature_norm_spectrogram.py --random 3 --k 5           # separate figures

Assumes:
    saved_features/X_dev.npy, utt_dev.npy, tg_dev.npy
    models/l1_probe.joblib, models/l1_probe_meta.json

Note: the top-K feature panels use z-scored features (as the probe sees them),
while the norm panel uses the RAW (unstandardized) 1024-dim vector, since
"vector length" is only meaningful on the original WavLM output scale.
"""

import argparse
import json
import re
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torchaudio
import torchaudio.transforms as T

from zerosylprobes.utils import parse_textgrid_syllables

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

OUTPUT_DIR = Path("saved_features")
MODEL_DIR = Path("models")
L1_MODEL_FILE = MODEL_DIR / "l1_probe.joblib"
META_FILE = MODEL_DIR / "l1_probe_meta.json"

FRAME_RATE = 50.0  # Hz
FRAME_DUR = 1.0 / FRAME_RATE

TIER_NAME = "syllables"
WORDS_TIER_NAME = "words"
BASE_DATA_DIR = "data"

BOUNDARY_COLOR = "red"
WORD_COLOR = "goldenrod"
FEATURE_COLOR = "tab:blue"
NORM_COLOR = "darkgreen"

SILENCE_LABELS = {"", "sil", "sp", "spn", "<unk>", "noise"}


# --------------------------------------------------------------------------
# Parsing / path helpers
# --------------------------------------------------------------------------
def parse_textgrid_words(tg_path, tier_name=WORDS_TIER_NAME):
    """Minimal Praat long-format TextGrid parser -> [(start, end, word), ...]
    for non-silence intervals in the named tier."""
    text = Path(tg_path).read_text(encoding="utf-8", errors="ignore")

    tier_chunks = re.split(r"item \[\d+\]:", text)[1:]
    target_chunk = None
    for chunk in tier_chunks:
        name_match = re.search(r'name\s*=\s*"([^"]*)"', chunk)
        if name_match and name_match.group(1).strip().lower() == tier_name.lower():
            target_chunk = chunk
            break

    if target_chunk is None:
        return []

    intervals = []
    for m in re.finditer(
        r'xmin\s*=\s*([\d.]+)\s*'
        r'xmax\s*=\s*([\d.]+)\s*'
        r'text\s*=\s*"([^"]*)"',
        target_chunk,
    ):
        xmin, xmax, word = float(m.group(1)), float(m.group(2)), m.group(3).strip()
        if word.lower() not in SILENCE_LABELS:
            intervals.append((xmin, xmax, word))
    return intervals


def find_audio_path(tg_path, base_data_dir=BASE_DATA_DIR):
    tg_path = Path(tg_path).resolve()
    stem = tg_path.stem
    parts = stem.split("-")

    if len(parts) >= 3:
        speaker_id, chapter_id = parts[0], parts[1]
        for split in ["dev-clean", "train-clean-100", "train-clean-360", "test-clean"]:
            for ext in [".flac", ".wav"]:
                cand = Path(base_data_dir) / "LibriSpeech" / split / speaker_id / chapter_id / f"{stem}{ext}"
                if cand.exists():
                    return str(cand)

    for ext in ["*.flac", "*.wav"]:
        matches = list(Path(base_data_dir).glob(f"**/{stem}{ext[1:]}"))
        if matches:
            return str(matches[0])

    return None


def get_boundaries(tg_path: Path):
    """Ground-truth syllable boundaries, via the same parser used for training labels."""
    return parse_textgrid_syllables(tg_path, tier_name=TIER_NAME)


def load_top_k_features(k: int):
    clf = joblib.load(L1_MODEL_FILE)
    coefs = clf.coef_.ravel()
    with open(META_FILE) as f:
        meta = json.load(f)
    mean = np.array(meta["mean"], dtype=np.float32)
    std = np.array(meta["std"], dtype=np.float32)

    top_k_indices = np.argsort(np.abs(coefs))[::-1][:k]
    top_k_coefs = coefs[top_k_indices]
    return top_k_indices, top_k_coefs, mean, std


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def _load_utterance_data(utt_id, top_k_indices, mean, std):
    """Load everything needed for one utterance: boundaries, words,
    spectrogram, z-scored top-K features, and raw L2 norm."""
    X_dev = np.load(OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    tg_dev = np.load(OUTPUT_DIR / "tg_dev.npy")

    mask = utt_dev == utt_id
    if not mask.any():
        raise ValueError(f"Utterance {utt_id} not found in utt_dev.npy")

    tg_path = Path(tg_dev[mask][0]).resolve()
    boundaries = get_boundaries(tg_path)
    words = parse_textgrid_words(tg_path)
    if words:
        print(f"[{utt_id}] transcript: " + " ".join(w for _, _, w in words))
    else:
        print(f"[{utt_id}] no '{WORDS_TIER_NAME}' tier found or it was empty -- "
              f"skipping word annotations (check WORDS_TIER_NAME).")

    audio_path = find_audio_path(tg_path)
    if audio_path is None:
        raise FileNotFoundError(
            f"Could not locate audio for {tg_path} under {BASE_DATA_DIR}. "
            f"Check BASE_DATA_DIR / your LibriSpeech layout."
        )

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total_duration = waveform.shape[1] / sr

    hop_length = int(round(sr / FRAME_RATE))
    spec = T.Spectrogram(n_fft=400, hop_length=hop_length)(waveform).squeeze(0).numpy()
    spec_db = 10 * np.log10(spec + 1e-10)

    feats_raw_full = np.asarray(X_dev[mask], dtype=np.float32)          # (T, 1024)
    norm = np.linalg.norm(feats_raw_full, axis=1)                        # raw scale
    feats_z = (feats_raw_full[:, top_k_indices] - mean[top_k_indices]) / std[top_k_indices]
    times_feat = np.arange(feats_raw_full.shape[0]) * FRAME_DUR

    return {
        "boundaries": boundaries,
        "words": words,
        "spec_db": spec_db,
        "sr": sr,
        "total_duration": total_duration,
        "feats_z": feats_z,
        "norm": norm,
        "times_feat": times_feat,
    }


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def _draw_utterance_block(axes_block, utt_id, data, top_k_indices, top_k_coefs, max_time=3.0):
    boundaries = data["boundaries"]
    words = data["words"]
    total_duration = data["total_duration"]
    times_feat = data["times_feat"]
    k = len(top_k_indices)

    xlim_max = min(max_time, total_duration) if max_time is not None else total_duration

    # --- Filter boundaries and words to stay within the time window ---
    if xlim_max is not None:
        boundaries = [b for b in boundaries if b <= xlim_max]
        # Only keep words that fully end before or at xlim_max
        words = [(w_start, w_end, w) for w_start, w_end, w in words if w_end <= xlim_max]

    ax0 = axes_block[0]
    word_ax = axes_block[1]
    feature_axes = axes_block[2:2 + k]
    norm_ax = axes_block[-1]

    # --- spectrogram ---
    ax0.imshow(
        data["spec_db"], aspect="auto", origin="lower",
        extent=[0, total_duration, 0, data["sr"] / 2], cmap="viridis",
    )
    ax0.set_ylabel("Frequency (Hz)")
    ax0.set_title(f"Utterance: {utt_id}")
    for b in boundaries:
        ax0.axvline(b, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)

    # --- word strip ---
    word_ax.set_ylim(0, 1)
    word_ax.set_yticks([])
    for spine in word_ax.spines.values():
        spine.set_visible(False)
    for i, (w_start, w_end, word) in enumerate(words):
        mid = (w_start + w_end) / 2
        y = 0.75 if i % 2 == 0 else 0.25
        word_ax.text(mid, y, word, fontsize=8, ha="center", va="center")
        word_ax.axvline(w_start, color=WORD_COLOR, linestyle=":", linewidth=0.6, alpha=0.6)

    # --- top-K feature panels (z-scored) ---
    for i, (feat_idx, coef) in enumerate(zip(top_k_indices, top_k_coefs)):
        ax = feature_axes[i]
        ax.plot(times_feat, data["feats_z"][:, i], color=FEATURE_COLOR, linewidth=1.2)
        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.5)
        for b in boundaries:
            ax.axvline(b, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
        for w_start, _, _ in words:
            ax.axvline(w_start, color=WORD_COLOR, linestyle=":", linewidth=0.6, alpha=0.3)
        direction = "+" if coef > 0 else "-"
        ax.set_ylabel(f"feat_{feat_idx}\n(w={coef:.2f}{direction})", fontsize=8)

    # --- raw L2 norm panel ---
    norm_ax.plot(times_feat, data["norm"], color=NORM_COLOR, linewidth=1.2)
    for b in boundaries:
        norm_ax.axvline(b, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    for w_start, _, _ in words:
        norm_ax.axvline(w_start, color=WORD_COLOR, linestyle=":", linewidth=0.6, alpha=0.3)
    norm_ax.set_ylabel(r"$\Vert h_t \Vert_2$" + "\n(raw)", fontsize=9)

    for ax in axes_block:
        ax.set_xlim(0, xlim_max)


def _add_legend(fig):
    handles = [
        Line2D([0], [0], color=BOUNDARY_COLOR, linestyle="--", linewidth=1.2,
               label="Syllable boundary (ground truth)"),
        Line2D([0], [0], color=WORD_COLOR, linestyle=":", linewidth=1.2,
               label="Word boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), frameon=False, fontsize=9)


def _block_height(k):
    return 2.6 + 1 * k + 1.2


def plot_utterance(utt_id, top_k_indices, top_k_coefs, mean, std, save_dir=None, max_time=3.0):
    data = _load_utterance_data(utt_id, top_k_indices, mean, std)
    k = len(top_k_indices)

    fig, axes = plt.subplots(
        k + 3, 1, figsize=(12, _block_height(k)), sharex=True,
        gridspec_kw={"height_ratios": [2.2, 0.5] + [1] * k + [1.2]},
    )
    _draw_utterance_block(axes, utt_id, data, top_k_indices, top_k_coefs, max_time)
    axes[-1].set_xlabel("Time (s)")
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{utt_id}_feature_norm_spectrogram.pdf"  # <-- Changed to .pdf
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_utterance_comparison(utt_ids, top_k_indices, top_k_coefs, mean, std,
                              save_dir=None, max_time=3.0):
    k = len(top_k_indices)
    rows_per_block = k + 3
    n_utts = len(utt_ids)

    height_ratios = ([2.2, 0.5] + [1] * k + [1.2]) * n_utts
    fig, all_axes = plt.subplots(
        rows_per_block * n_utts, 1,
        figsize=(12, n_utts * _block_height(k)),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.6},
    )
    all_axes = np.atleast_1d(all_axes)

    for u_idx, utt_id in enumerate(utt_ids):
        block = all_axes[u_idx * rows_per_block: (u_idx + 1) * rows_per_block]
        data = _load_utterance_data(utt_id, top_k_indices, mean, std)
        _draw_utterance_block(block, utt_id, data, top_k_indices, top_k_coefs, max_time)
        block[0].set_title(f"({chr(ord('a') + u_idx)}) Utterance: {utt_id}")
        block[-1].set_xlabel("" if u_idx < n_utts - 1 else "Time (s)")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joined = "_".join(utt_ids)
        out_path = save_dir / f"compare_{joined}_feature_norm_spectrogram.pdf"  # <-- Changed to .pdf
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt", type=str, default=None,
                        help="Specific utterance ID to plot (single-utterance mode)")
    parser.add_argument("--utts", type=str, default=None,
                        help="Comma-separated utterance IDs for one comparison figure, "
                             "e.g. --utts 5338-24615-0000,1272-128104-0000")
    parser.add_argument("--random", type=int, default=0,
                        help="Plot N random utterances as SEPARATE figures")
    parser.add_argument("--k", type=int, default=5, help="Number of top features to show")
    parser.add_argument("--max_time", type=float, default=3.0,
                        help="Zoom to the first N seconds (default 3.0). Use 0 or negative for the full utterance.")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="If set, save PDFs here instead of showing")
    args = parser.parse_args()

    top_k_indices, top_k_coefs, mean, std = load_top_k_features(args.k)
    print(f"Top-{args.k} feature indices (by |weight|): {top_k_indices.tolist()}")

    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    unique_utts = np.unique(utt_dev)
    max_time = args.max_time if args.max_time and args.max_time > 0 else None

    if args.utts:
        targets = [u.strip() for u in args.utts.split(",") if u.strip()]
        plot_utterance_comparison(targets, top_k_indices, top_k_coefs, mean, std,
                                  save_dir=args.save_dir, max_time=max_time)
        return

    if args.utt:
        targets = [args.utt]
    elif args.random > 0:
        rng = np.random.default_rng(42)
        targets = rng.choice(unique_utts, size=min(args.random, len(unique_utts)), replace=False)
    else:
        targets = [unique_utts[0]]
        print(f"No --utt/--utts/--random given; defaulting to first utterance: {targets[0]}")

    for utt_id in targets:
        plot_utterance(utt_id, top_k_indices, top_k_coefs, mean, std,
                       save_dir=args.save_dir, max_time=max_time)


if __name__ == "__main__":
    main()