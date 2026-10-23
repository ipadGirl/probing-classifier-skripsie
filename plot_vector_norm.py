"""
Plot the RAW WavLM feature vector's L2 norm (vector length) over time
against the spectrogram, ground-truth boundaries, and words. Fully
self-contained -- no other local files needed besides your existing
zerosylprobes package.

This is the exact quantity your NormSeg baseline (generate_normseg_preds,
via prominence-based peak detection) is built on -- this plot lets you
visually inspect that same signal directly, rather than only seeing its
aggregate Token-F1 score in the results table.

Uses the RAW (unstandardized) 1024-dim feature vector, not the z-scored one
-- "vector length" is meaningful on the original WavLM output scale;
z-scoring first would change what the norm represents.

Usage:
    python plot_vector_norm_spectrogram.py --utt <utt_id>
    python plot_vector_norm_spectrogram.py --utts <id1>,<id2>   # comparison figure
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torchaudio
import torchaudio.transforms as T

from zerosylprobes.utils import parse_textgrid_syllables

OUTPUT_DIR = Path("saved_features")

FRAME_RATE = 50.0
FRAME_DUR = 1.0 / FRAME_RATE

TIER_NAME = "syllables"
WORDS_TIER_NAME = "words"
BASE_DATA_DIR = "data"

BOUNDARY_COLOR = "red"
WORD_COLOR = "goldenrod"

SILENCE_LABELS = {"", "sil", "sp", "spn", "<unk>", "noise"}


def parse_textgrid_words(tg_path, tier_name=WORDS_TIER_NAME):
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
        r'xmin\s*=\s*([\d.]+)\s*xmax\s*=\s*([\d.]+)\s*text\s*=\s*"([^"]*)"',
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
    return parse_textgrid_syllables(tg_path, tier_name=TIER_NAME)


def compute_spec_db(waveform, sr):
    hop_length = int(round(sr / FRAME_RATE))
    spec_transform = T.Spectrogram(n_fft=400, hop_length=hop_length)
    spec = spec_transform(waveform).squeeze(0).numpy()
    return 10 * np.log10(spec + 1e-10)


def _load_utterance_data(utt_id):
    tg_dev = np.load(OUTPUT_DIR / "tg_dev.npy")
    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    mask = utt_dev == utt_id
    if not mask.any():
        raise ValueError(f"Utterance {utt_id} not found in utt_dev.npy")
    tg_path = Path(tg_dev[mask][0]).resolve()

    boundaries = get_boundaries(tg_path)
    words = parse_textgrid_words(tg_path)
    if words:
        print(f"[{utt_id}] transcript: " + " ".join(w for _, _, w in words))
    else:
        print(f"[{utt_id}] no '{WORDS_TIER_NAME}' tier found or it was empty -- skipping word annotations.")

    audio_path = find_audio_path(tg_path)
    if audio_path is None:
        raise FileNotFoundError(f"Could not locate audio for {tg_path} under {BASE_DATA_DIR}.")

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total_duration = waveform.shape[1] / sr
    spec_db = compute_spec_db(waveform, sr)

    return {
        "boundaries": boundaries, "words": words, "spec_db": spec_db,
        "sr": sr, "total_duration": total_duration,
    }


def _compute_norm(utt_id):
    """Raw (unstandardized) per-frame L2 norm of the 1024-dim feature vector."""
    X_dev = np.load(OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    mask = utt_dev == utt_id
    feats_raw = np.asarray(X_dev[mask], dtype=np.float32)
    return np.linalg.norm(feats_raw, axis=1)


def _draw_utterance_block(axes_block, utt_id, data, max_time=3.0):
    norm = _compute_norm(utt_id)
    times_feat = np.arange(len(norm)) * FRAME_DUR

    total_duration = data["total_duration"]
    xlim_max = min(max_time, total_duration) if max_time is not None else total_duration

    ax0, word_ax, norm_ax = axes_block[0], axes_block[1], axes_block[2]

    ax0.imshow(data["spec_db"], aspect="auto", origin="lower",
               extent=[0, total_duration, 0, data["sr"] / 2], cmap="viridis")
    ax0.set_ylabel("Frequency (Hz)")
    ax0.set_title(f"Utterance: {utt_id}")
    for b in data["boundaries"]:
        ax0.axvline(b, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.8)

    word_ax.set_ylim(0, 1)
    word_ax.set_yticks([])
    for spine in word_ax.spines.values():
        spine.set_visible(False)
    for i, (w_start, w_end, word) in enumerate(data["words"]):
        mid = (w_start + w_end) / 2
        y = 0.75 if i % 2 == 0 else 0.25
        word_ax.text(mid, y, word, fontsize=8, ha="center", va="center")
        word_ax.axvline(w_start, color=WORD_COLOR, linestyle=":", linewidth=0.6, alpha=0.6)

    norm_ax.plot(times_feat, norm, color="darkgreen", linewidth=1.2)
    for b in data["boundaries"]:
        norm_ax.axvline(b, color=BOUNDARY_COLOR, linestyle="--", linewidth=0.8, alpha=0.6)
    for w_start, _, _ in data["words"]:
        norm_ax.axvline(w_start, color=WORD_COLOR, linestyle=":", linewidth=0.6, alpha=0.3)
    norm_ax.set_ylabel(r"$\Vert h_t \Vert_2$", fontsize=9)

    for ax in axes_block:
        ax.set_xlim(0, xlim_max)


def _add_legend(fig):
    handles = [
        Line2D([0], [0], color=BOUNDARY_COLOR, linestyle="--", linewidth=1.2,
               label="Syllable boundary (ground truth)"),
        Line2D([0], [0], color=WORD_COLOR, linestyle=":", linewidth=1.2, label="Word boundary"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0),
               frameon=False, fontsize=9)


def plot_utterance(utt_id, save_dir=None, max_time=3.0):
    data = _load_utterance_data(utt_id)
    fig, axes = plt.subplots(3, 1, figsize=(12, 4.5), sharex=True,
                              gridspec_kw={"height_ratios": [2.2, 0.5, 1.2]})
    _draw_utterance_block(axes, utt_id, data, max_time)
    axes[-1].set_xlabel("Time (s)")
    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{utt_id}_vector_norm_spectrogram.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def plot_utterance_comparison(utt_ids, save_dir=None, max_time=3.0):
    rows_per_block = 3
    n_utts = len(utt_ids)
    height_ratios = [2.2, 0.5, 1.2] * n_utts
    fig, all_axes = plt.subplots(rows_per_block * n_utts, 1,
                                  figsize=(12, n_utts * 4.5),
                                  gridspec_kw={"height_ratios": height_ratios, "hspace": 0.6})
    all_axes = np.atleast_1d(all_axes)

    for u_idx, utt_id in enumerate(utt_ids):
        block = all_axes[u_idx * rows_per_block: (u_idx + 1) * rows_per_block]
        data = _load_utterance_data(utt_id)
        _draw_utterance_block(block, utt_id, data, max_time)
        block[0].set_title(f"({chr(ord('a') + u_idx)}) Utterance: {utt_id}")
        block[-1].set_xlabel("" if u_idx < n_utts - 1 else "Time (s)")

    _add_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        joined = "_".join(utt_ids)
        out_path = save_dir / f"compare_{joined}_vector_norm_spectrogram.png"
        plt.savefig(out_path, dpi=150)
        print(f"Saved: {out_path}")
        plt.close(fig)
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--utt", type=str, default=None)
    parser.add_argument("--utts", type=str, default=None,
                         help="Comma-separated utterance IDs for a comparison figure")
    parser.add_argument("--random", type=int, default=0)
    parser.add_argument("--max_time", type=float, default=3.0)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()

    utt_dev = np.load(OUTPUT_DIR / "utt_dev.npy")
    unique_utts = np.unique(utt_dev)
    max_time = args.max_time if args.max_time and args.max_time > 0 else None

    if args.utts:
        targets = [u.strip() for u in args.utts.split(",") if u.strip()]
        plot_utterance_comparison(targets, save_dir=args.save_dir, max_time=max_time)
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
        plot_utterance(utt_id, save_dir=args.save_dir, max_time=max_time)


if __name__ == "__main__":
    main()
