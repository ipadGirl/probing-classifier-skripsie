"""
Does PC0 just track loudness?

This file is related to Hypothesis 1 and computes the RMS energy of the raw audio, frame-aligned to the WavLM
features, and correlates it with PC0. A high correlation means PC0 is
an energy signal and a low one means it's picking up something beyond
loudness.

Audio lookup uses check_audio_path.audio_path_from_tg, which finds the file
by name. Run check_audio_path.py first and confirm it reports all files found.
Make sure AUDIO_ROOT is set correctly there before running this script.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import librosa  # pip install librosa soundfile --break-system-packages

from zerosylprobes.PCA import fit_exact_pca, project
from zerosylprobes.check_audio_path import audio_path_from_tg

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

FRAME_RATE = 50.0   # must match the frame rate your WavLM features use


def rms_energy_for_utt(wav_path, n_frames, frame_rate=FRAME_RATE, sr=16000):
    # number of audio samples between frames (16000/50 = 320)
    hop_length = int(sr / frame_rate)
    # load the audio at 16kHz
    y, _ = librosa.load(wav_path, sr=sr)
    # RMS energy per frame (window is twice the hop size)
    rms = librosa.feature.rms(y=y, frame_length=hop_length * 2,
                              hop_length=hop_length)[0]
    # align lengths: WavLM and librosa framing rarely match exactly
    n = min(len(rms), n_frames)
    # fill with NaN first so any leftover frames are marked as missing
    out = np.full(n_frames, np.nan, dtype=np.float32)
    out[:n] = rms[:n]
    return out


def main():
    d = "saved_features"
    # load the dev set features, utterance ids and TextGrid paths
    X_dev = np.load(os.path.join(d, "X_dev.npy"))
    utt_dev = np.load(os.path.join(d, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(d, "tg_dev.npy"))

    # recompute PC0 on dev (reuse the same centring used for the main PCA run)
    # PCA is fit on the training set only, then applied to dev
    X_tr = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
    mean, scale, eigvals, comps = fit_exact_pca(X_tr, standardize=False)
    # keep just the first component
    pc0 = project(X_dev, mean, scale, comps, k=1)[:, 0]

    # frame-aligned RMS energy, utterance by utterance
    # start with NaN everywhere so skipped utterances are easy to spot 
    energy = np.full(len(X_dev), np.nan, dtype=np.float32)
    missing = 0
    for utt in np.unique(utt_dev):
        # pick out the frames belonging to this utterance
        mask = utt_dev == utt
        # finds its audio file using the first frame's TextGrid path
        audio_path = audio_path_from_tg(tg_dev[mask][0])
        # skip (and count) utterances with no audio 
        if audio_path is None:
            missing += 1
            continue
        energy[mask] = rms_energy_for_utt(audio_path, mask.sum())

    if missing:
        print(f"[warn] {missing} utterances had no matching .wav file "
              f"(check wav_path_from_tg) and were skipped")

    # only use frames where an energy value was actually received
    valid = ~np.isnan(energy)
    # correlation between PC0 and energy
    r = np.corrcoef(pc0[valid], energy[valid])[0, 1]
    if pc0[valid].mean() == 0 or r < 0:
        # PC sign is arbitrary - report magnitude and note the sign
        print(f"(PC0 sign is arbitrary correlation is negative -> "
              f"flip sign for a 1-to-1 reading)")
    print(f"\nCorrelation of PC0 with raw audio RMS energy: r = {r:+.3f} "
          f"(n={valid.sum()} frames)")
    # r squared = share of PC0's variance that energy explains
    print(f"  r^2 = {r**2*100:.1f}% of PC0's variance explained by energy alone")

    # quick visual check
    fig, ax = plt.subplots(figsize=(5, 5))
    # plot a random sample of up to 20000 frames
    idx = np.random.default_rng(0).choice(np.where(valid)[0],
                                          size=min(20000, valid.sum()),
                                          replace=False)
    ax.scatter(energy[idx], pc0[idx], s=3, alpha=0.15, color="#1D9E75")
    ax.set_xlabel("RMS energy (raw audio)")
    ax.set_ylabel("PC0 score")
    ax.set_title(f"PC0 vs. audio energy (r = {r:+.2f})")
    
    # save the figure as a PDF and make the folder if it does not already exist
    os.makedirs("report_figures", exist_ok=True)
    fig.savefig("report_figures/pc0_vs_energy.pdf", dpi=300, bbox_inches="tight")
    print("Saved report_figures/pc0_vs_energy.pdf")


if __name__ == "__main__":
    main()