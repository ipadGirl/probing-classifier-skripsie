import os
import re
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torchaudio
import torchaudio.transforms as T
from zerosylprobes.utils import parse_textgrid_syllables


# ---------------------------------------------------------------------------
# EDIT THIS: map your internal variant folder names -> display row labels,
# in the order you want them to appear (top to bottom, under "Words").
# ---------------------------------------------------------------------------
VARIANT_DISPLAY_NAMES = {
    "full_vector": "Full Vector",
    "magnitude": "Magnitude",
    "direction": "Direction",
    "fixed_100ms": "Fixed 100ms",
}

# --- Match LaTeX Document Fonts & Styling ---
plt.rcParams.update({
    "text.usetex": False,  # Disable external LaTeX compilation
    "font.family": "sans-serif",
    "font.sans-serif": ["Carlito", "Caladea", "DejaVu Sans", "Helvetica", "Arial"],
    "axes.labelsize": 10,
    "font.size": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,      # Ensures TrueType/OpenType fonts are embedded properly in PDF
    "ps.fonttype": 42,
})


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


def parse_textgrid_all_intervals(tg_path, tier_name):
    """
    Same as parse_textgrid_intervals, but keeps empty-label intervals too
    (these are the silence gaps between words in a standard word tier).
    """
    with open(tg_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    intervals = []
    in_tier = False
    cur_start, cur_end = None, None

    name_re = re.compile(r'name\s*=\s*"(.*)"')
    xmin_re = re.compile(r'xmin\s*=\s*([\d.eE+-]+)')
    xmax_re = re.compile(r'xmax\s*=\s*([\d.eE+-]+)')
    text_re = re.compile(r'text\s*=\s*"(.*)"')

    for line in lines:
        line = line.strip()

        m = name_re.match(line)
        if m:
            in_tier = (m.group(1) == tier_name)
            continue

        if not in_tier:
            continue

        m = xmin_re.match(line)
        if m:
            cur_start = float(m.group(1))
            continue

        m = xmax_re.match(line)
        if m:
            cur_end = float(m.group(1))
            continue

        m = text_re.match(line)
        if m:
            label = m.group(1).strip()
            if cur_start is not None and cur_end is not None:
                intervals.append((cur_start, cur_end, label))
            cur_start, cur_end = None, None

    return intervals


def parse_textgrid_intervals(tg_path, tier_name):
    """
    Minimal parser for Praat long-form TextGrid files, returning labeled
    (non-empty) intervals for a given tier as a list of (start, end, label)
    tuples. Empty-label intervals (silence) are skipped.

    If you already have an equivalent helper in zerosylprobes.utils for
    interval tiers (as opposed to boundary-only tiers), swap this out for it.
    """
    return [(s, e, l) for s, e, l in parse_textgrid_all_intervals(tg_path, tier_name) if l]


def probs_to_boundaries(utt_probs, frame_rate, threshold=0.5, total_duration=None):
    """
    Convert a per-frame probability curve into a sorted list of boundary
    times (seconds), taken at each rising edge crossing of `threshold`.
    Always includes 0.0 and the final time as the segment start/end.
    """
    utt_probs = np.asarray(utt_probs)
    above = utt_probs >= threshold
    crossings = np.where(np.diff(above.astype(int)) == 1)[0] + 1  # rising edges
    times = crossings / frame_rate

    end_time = total_duration if total_duration is not None else len(utt_probs) / frame_rate
    boundaries = [0.0] + sorted(t for t in times if 0.0 < t < end_time) + [end_time]
    return boundaries


def _draw_tier_row(ax, segments, value_fn, rotate=True, silence_regions=None):
    """
    Draw one tier row: grey shading over actual silence regions (shared
    across all rows so silence lines up vertically between tiers), vertical
    boundary lines at each segment edge, and a centered label (from
    value_fn) in each segment.
    `segments` is a list of (start, end, raw_label_or_None).
    `rotate` controls whether labels are drawn vertically (good for short
    numeric duration labels in narrow segments) or horizontally (good for
    word text, which reads better upright).
    """
    ax.set_xlim(ax.get_xlim())  # inherit shared xlim from spectrogram axis
    ax.set_ylim(0, 1)
    ax.set_yticks([])

    for s, e in (silence_regions or []):
        ax.axvspan(s, e, color="0.82", zorder=0)

    for i, (start, end, raw_label) in enumerate(segments):
        ax.axvline(x=start, color="black", linewidth=0.8, zorder=2)
        label = value_fn(start, end, raw_label)
        if label:
            ax.text(
                (start + end) / 2, 0.5, label,
                ha="center", va="center", fontsize=7,
                rotation=90 if rotate else 0,
                clip_on=True,
                zorder=3,
            )
    if segments:
        ax.axvline(x=segments[-1][1], color="black", linewidth=0.8, zorder=2)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)


def plot_multitier_comparison(
    audio_path,
    tg_path,
    variant_probs,          # dict: display_name -> utt_probs (np.array)
    save_path,
    frame_rate=50.0,
    threshold=0.5,
    words_tier_name="words",
    syllable_tier_name="syllables",
    max_time=3.0,
):
    """
    Build one figure: grayscale spectrogram on top, then one row per tier
    (Words, then one row per entry in variant_probs, in insertion order),
    each row showing that tier's segment boundaries and a label per segment.
    """
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

    all_word_intervals = parse_textgrid_all_intervals(tg_path, tier_name=words_tier_name)
    word_intervals = [(s, e, l) for s, e, l in all_word_intervals if l]
    silence_regions = [(s, e) for s, e, l in all_word_intervals if not l]

    # NEW: ground-truth syllable boundaries from the TextGrid
        # NEW: ground-truth syllable boundaries from the TextGrid
    syllable_boundaries = parse_textgrid_syllables(tg_path, tier_name=syllable_tier_name)

    tier_rows = [("Words", word_intervals, "word")]
    for display_name, utt_probs in variant_probs.items():
        boundaries = probs_to_boundaries(
            utt_probs, frame_rate, threshold=threshold, total_duration=total_duration
        )
        segments = [(boundaries[i], boundaries[i + 1], None) for i in range(len(boundaries) - 1)]
        tier_rows.append((display_name, segments, "duration_ms"))

    n_tiers = len(tier_rows)
    fig, axes = plt.subplots(
        n_tiers + 1, 1, figsize=(10, 4.5 + 0.5 * n_tiers), sharex=True,
        gridspec_kw={"height_ratios": [3] + [1] * n_tiers, "hspace": 0.15},
    )

    ax_spec = axes[0]
    ax_spec.imshow(
        spec_db, aspect="auto", origin="lower",
        extent=[0, total_duration, 0, sr / 2],
        cmap="Greys",
    )
    ax_spec.set_ylabel("Hz", fontsize=10)
    xlim_max = min(max_time, total_duration) if max_time is not None else total_duration
    ax_spec.set_xlim(0, xlim_max)

    # NEW: overlay ground-truth syllable boundaries on the spectrogram
    for b in syllable_boundaries:
        ax_spec.axvline(x=b, color="red", linewidth=1.0, alpha=0.85, zorder=4)

    for ax, (row_name, segments, kind) in zip(axes[1:], tier_rows):
        if kind == "word":
            value_fn = lambda s, e, raw: raw
            rotate = False
        else:
            value_fn = lambda s, e, raw: str(int(round((e - s) * 1000)))
            rotate = True
        _draw_tier_row(ax, segments, value_fn, rotate=rotate, silence_regions=silence_regions)
        ax.set_ylabel(row_name, rotation=0, ha="right", va="center", fontsize=9)

    axes[-1].set_xlabel("Time (s)", fontsize=11)
    plt.suptitle(f"Syllable Boundary Comparison: {Path(tg_path).stem}", fontsize=12, y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved qualitative plot to: {save_path}")


def generate_comparison_plots(variants, num_samples=3, reference_variant=None):
    """
    For each of the first `num_samples` utterances (matched by utt_id across
    all variant folders), build one combined multi-tier comparison plot.
    """
    reference_variant = reference_variant or variants[0]
    tg_dir = Path(f"eval_textgrids_{reference_variant}").resolve()
    plots_dir = Path("qualitative_plots_comparison").resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not tg_dir.exists():
        print(f"Reference TextGrid dir '{tg_dir}' not found. Run training/eval script first.")
        return

    probs_dirs = {v: Path(f"pred_probs_{v}").resolve() for v in variants}
    ref_probs_dir = probs_dirs[reference_variant]
    if not ref_probs_dir.exists():
        print(f"Directory for variant '{reference_variant}' not found.")
        return

    prob_files = sorted(list(ref_probs_dir.glob("*_probs.npy")))[:num_samples]

    for prob_file in prob_files:
        utt_id = prob_file.stem.replace("_probs", "")
        tg_path = tg_dir / f"{utt_id}.TextGrid"
        if not tg_path.exists():
            continue

        variant_probs = {}
        for v in variants:
            pf = probs_dirs[v] / f"{utt_id}_probs.npy"
            if pf.exists():
                variant_probs[VARIANT_DISPLAY_NAMES.get(v, v)] = np.load(pf)

        if not variant_probs:
            continue

        audio_path = find_audio_path(tg_path)
        save_path = plots_dir / f"{utt_id}_comparison.pdf"
        plot_multitier_comparison(audio_path, tg_path, variant_probs, save_path)


def main():
    variants = ["full_vector", "magnitude", "direction", "fixed_100ms"]
    generate_comparison_plots(variants, num_samples=3)


if __name__ == "__main__":
    main()
