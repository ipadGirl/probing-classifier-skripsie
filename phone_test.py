"""
Does PC0 track manner of articulation (fricative, stop, nasal, ...)?

Breaks the earlier "consonant" bucket down into its ARPABET sub-classes and
compares PC0 across all of them, plus vowels and silence.

SET THESE:
  PHONE_TIER_NAME : exact (or partial, case-insensitive) name of your phone
                     tier, e.g. "phones", "phone", "phoneme".
  MANNER / VOWELS / SILENCE : edit if your aligner doesn't use ARPABET
                     (MFA/Prosodylab typically do -- AA, IH, EY, ... with an
                     optional trailing stress digit 0/1/2).
"""
import os
import re
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from zerosylprobes.PCA import fit_exact_pca, project

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

FRAME_RATE = 50.0            # frames per second of the WavLM features
PHONE_TIER_NAME = "phones"   # substring match, case-insensitive

# phone labels that count as vowels
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
          "IH", "IY", "OW", "OY", "UH", "UW"}
# labels the aligner uses for pauses / non-speech
SILENCE = {"sil", "sp", "spn", "", "pau", "noise", "<unk>"}

# ARPABET consonants grouped by manner of articulation
MANNER = {
    "fricative":  {"S", "Z", "SH", "ZH", "F", "V", "TH", "DH", "HH"},
    "stop":       {"P", "B", "T", "D", "K", "G"},
    "affricate":  {"CH", "JH"},
    "nasal":      {"M", "N", "NG"},
    "approximant": {"L", "R", "W", "Y"},
}
# order and colours used in the plot (they line up one to one)
CAT_ORDER = ["silence", "stop", "fricative", "affricate", "nasal",
             "approximant", "vowel"]
CAT_COLORS = ["#B4B2A9", "#D85A30", "#E8A33D", "#C9856B",
              "#7F77DD", "#5FB3D9", "#1D9E75"]


# ------------------------------------------------------------------
# Minimal Praat TextGrid parser (handles the standard long-form format).
# Returns {tier_name: [(xmin, xmax, label), ...]}
# ------------------------------------------------------------------
def parse_textgrid(path):
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    tiers = {}
    # split on tier blocks
    tier_blocks = re.split(r'item \[\d+\]:', text)[1:]
    for block in tier_blocks:
        # the tier's name, e.g. "words" or "phones"
        name_m = re.search(r'name\s*=\s*"([^"]*)"', block)
        if not name_m:
            continue
        name = name_m.group(1)
        intervals = []
        # each interval has a start time, end time and a label
        for m in re.finditer(
            r'xmin\s*=\s*([\d.eE+-]+)\s*'
            r'xmax\s*=\s*([\d.eE+-]+)\s*'
            r'text\s*=\s*"([^"]*)"', block):
            intervals.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        if intervals:
            tiers[name] = intervals
    return tiers


def find_phone_tier(tiers, name_hint=PHONE_TIER_NAME):
    # return the first tier whose name contains the hint
    for name in tiers:
        if name_hint.lower() in name.lower():
            return tiers[name]
    raise KeyError(f"No tier matching '{name_hint}' found. "
                    f"Available tiers: {list(tiers.keys())}")


def strip_stress(label):
    # remove the trailing stress digit, e.g. "AH1" -> "AH"
    return re.sub(r'\d$', '', label.strip())


# labels we couldn't place in any category (printed as a warning later)
UNSEEN_LABELS = set()


def classify(label):
    # turn a phone label into one category: silence, vowel, a manner class, or "other"
    base = strip_stress(label).upper()
    if label.strip().lower() in SILENCE or base == "":
        return "silence"
    if base in VOWELS:
        return "vowel"
    for manner, members in MANNER.items():
        if base in members:
            return manner
    UNSEEN_LABELS.add(label)
    return "other"   # anything not in ARPABET's usual set -- check your labels


def frame_labels(intervals, n_frames, frame_rate=FRAME_RATE):
    """One category per frame, using the frame's centre time."""
    # end time and category of every interval
    xmax = np.array([iv[1] for iv in intervals])
    labels = [classify(iv[2]) for iv in intervals]
    out = []
    for i in range(n_frames):
        # time at the middle of frame i
        t = (i + 0.5) / frame_rate
        # find the interval that contains this time
        j = np.searchsorted(xmax, t, side="left")
        # frames past the last interval get the last label
        j = min(j, len(labels) - 1)
        out.append(labels[j])
    return np.array(out)


def main():
    d = "saved_features"
    # load the dev set features, utterance ids and TextGrid paths
    X_dev = np.load(os.path.join(d, "X_dev.npy"))
    utt_dev = np.load(os.path.join(d, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(d, "tg_dev.npy"))

    # fit PCA on train only then get PC0 for every dev frame
    X_tr = np.load(os.path.join(d, "X_train.npy"), mmap_mode="r")
    mean, scale, eigvals, comps = fit_exact_pca(X_tr, standardize=False)
    pc0 = project(X_dev, mean, scale, comps, k=1)[:, 0]

    # give every dev frame a category, utterance by utterance
    cats = np.empty(len(X_dev), dtype=object)
    bad = 0
    for utt in np.unique(utt_dev):
        mask = utt_dev == utt
        try:
            tiers = parse_textgrid(tg_dev[mask][0])
            intervals = find_phone_tier(tiers)
            cats[mask] = frame_labels(intervals, mask.sum())
        except Exception as e:
            # couldn't read this utterance's phone tier, so count it and carry on
            bad += 1
            cats[mask] = "silence"   # placeholder; excluded if unknown below
            if bad <= 3:
                print(f"[warn] {utt}: {e}")
    if bad:
        print(f"[warn] {bad} utterances had a problem reading the phone tier")

    # only keep categories that actually have frames
    present = [c for c in CAT_ORDER if (cats == c).sum() > 0]
    other_n = int((cats == "other").sum())
    if other_n:
        print(f"[warn] {other_n} frames had a phone label not in VOWELS/MANNER/"
              f"SILENCE (excluded from the plot/summary): "
              f"{sorted(UNSEEN_LABELS)[:20]}")

    # does the raw norm follow the same ordering as PC0?
    norm = np.linalg.norm(X_dev.astype(np.float32), axis=1)
    for cat in CAT_ORDER:
        v = norm[cats == cat]
        if len(v): print(f"  {cat:<12} norm mean={v.mean():.3f}")

    # PC0 mean and spread for each category
    print()
    for cat in present:
        v = pc0[cats == cat]
        print(f"  {cat:<12} n={len(v):>8}  mean={v.mean():+.3f}  std={v.std():.3f}")

    # one-vs-rest correlation for every category, so you can rank them
    # (indicator = 1 for frames in the category, 0 for all others)
    print("\nCorrelation of PC0 with each category (one-vs-rest):")
    for cat in present:
        ind = (cats == cat).astype(np.float32)
        r = np.corrcoef(pc0, ind)[0, 1]
        print(f"  {cat:<12} r = {r:+.3f}")

    # visual check, ordered silence -> stop -> fricative -> ... -> vowel
    fig, ax = plt.subplots(figsize=(8, 4.5))
    data = [pc0[cats == c] for c in present]
    colors = [CAT_COLORS[CAT_ORDER.index(c)] for c in present]
    # one box per category (outliers hidden to keep the plot readable)
    bp = ax.boxplot(data, tick_labels=present, showfliers=False, patch_artist=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_ylabel("PC0 score")
    ax.set_title("PC0 by manner of articulation")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    
    # save the figure as a vector PDF
    os.makedirs("report_figures", exist_ok=True)
    fig.savefig("report_figures/pc0_vs_manner.pdf", dpi=300, bbox_inches="tight")
    print("Saved report_figures/pc0_vs_manner.pdf")


if __name__ == "__main__":
    main()