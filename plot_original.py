import os
import re
from pathlib import Path
import numpy as np
import torch
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

import torchaudio
import torchaudio.transforms as T
from zerosylprobes.utils import parse_textgrid_syllables


def find_audio_path(tg_path, base_data_dir="data"):
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


def plot_qualitative_predictions(audio_path, tg_path, utt_probs, save_path, frame_rate=50.0, max_time=3.0):
    gt_boundaries = parse_textgrid_syllables(tg_path, tier_name="syllables")
    total_frames = len(utt_probs)
    time_sec = np.arange(total_frames) / frame_rate

    if audio_path is None or not os.path.exists(audio_path):
        print(f"Skipping plot for {tg_path}: Audio file not found.")
        return

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    total_duration = waveform.shape[1] / sr

    spec_transform = T.Spectrogram(n_fft=400, hop_length=160)
    spec = spec_transform(waveform).squeeze(0).numpy()
    spec_db = 10 * np.log10(spec + 1e-10)

    fig, ax1 = plt.subplots(figsize=(10, 4.5))

    ax1.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        extent=[0, total_duration, 0, sr / 2],
        cmap="viridis"
    )
    ax1.set_xlabel("Time (s)", fontsize=11)
    ax1.set_ylabel("Frequency (Hz)", fontsize=11)

    ax2 = ax1.twinx()
    ax2.plot(time_sec, utt_probs, color="magenta", linewidth=2.0, label=r"Probe $P(\mathrm{boundary})$")
    ax2.axhline(y=0.5, color="white", linestyle=":", alpha=0.7, label="Threshold (0.5)")
    ax2.set_ylabel("Probe Probability", fontsize=11)
    ax2.set_ylim([-0.05, 1.05])

    for i, gt in enumerate(gt_boundaries):
        ax1.axvline(
            x=gt, 
            color="white", 
            linestyle="--", 
            linewidth=1.8, 
            label="GT Syllable Boundary" if i == 0 else ""
        )

    xlim_max = min(max_time, total_duration) if max_time is not None else total_duration
    ax1.set_xlim(0, xlim_max)
    ax2.set_xlim(0, xlim_max)

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    legend = ax1.legend(
        lines_1 + lines_2, labels_1 + labels_2,
        loc="upper center", bbox_to_anchor=(0.5, -0.18),
        ncol=3, framealpha=0.9,
        facecolor="0.25", edgecolor="none", labelcolor="white",
    )

    plt.title(f"Syllable Boundary Probe: {Path(tg_path).stem}", fontsize=12, pad=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative plot to: {save_path}")


def generate_variant_plots(variant_name, num_samples=3):
    probs_dir = Path(f"pred_probs_{variant_name}").resolve()
    tg_dir = Path(f"eval_textgrids_{variant_name}").resolve()
    plots_dir = Path(f"qualitative_plots_{variant_name}").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not probs_dir.exists() or not tg_dir.exists():
        print(f"Directory for variant '{variant_name}' not found. Run training script first.")
        return

    prob_files = sorted(list(probs_dir.glob("*_probs.npy")))[:num_samples]

    for prob_file in prob_files:
        utt_id = prob_file.stem.replace("_probs", "")
        tg_path = tg_dir / f"{utt_id}.TextGrid"
        
        if not tg_path.exists():
            continue

        utt_probs = np.load(prob_file)
        audio_path = find_audio_path(tg_path)
        save_path = plots_dir / f"{utt_id}_plot.pdf"  

        plot_qualitative_predictions(audio_path, tg_path, utt_probs, save_path)


def main():
    variants = ["full_vector", "magnitude", "direction", "fixed_100ms"]
    for variant in variants:
        print(f"\nGenerating plots for variant: {variant}")
        generate_variant_plots(variant, num_samples=3)

if __name__ == "__main__":
    main()