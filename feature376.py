"""
feature376.py  -- feature 376 vs L2 norm (NormSeg-style peak detector)

Run from the ZeroSyl folder, e.g.:
    python3 -m zerosylprobes.feature376 --plot                       # first dev utterance -> figs/
    python3 -m zerosylprobes.feature376 --plot --utts 5338-24615-0000
    python3 -m zerosylprobes.feature376 --plot --max_time 0          # full utterance length
    python3 -m zerosylprobes.feature376 --eval
    python3 -m zerosylprobes.feature376 --eval --sweep
"""
import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # always render to file (no display needed)
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from zerosylprobes.normseg import _smooth_signal          # exact NormSeg smoother
from zerosylprobes.plot_feature_spectogram import (_load_utterance_data, OUTPUT_DIR, FRAME_DUR,
                                                   BOUNDARY_COLOR, META_FILE)

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
FRAME_RATE = 50.0


def smooth(x, window=3, gauss_sigma=0.0):
    """Default = NormSeg's moving average. gauss_sigma>0 switches to Gaussian."""
    if gauss_sigma > 0:
        return gaussian_filter1d(x, sigma=gauss_sigma, mode="nearest")
    return _smooth_signal(x, window_size=window)


def detect(x, window=3, prom_mult=0.45, gauss_sigma=0.0, min_dist=1):
    """Identical logic to compute_l2norm_peaks, but on an arbitrary 1-D signal."""
    pred = np.zeros(len(x), dtype=int)
    if len(x) == 0:
        return pred
    s = smooth(x, window, gauss_sigma)
    peaks, _ = find_peaks(s, prominence=prom_mult * np.std(s), distance=min_dist)
    pred[peaks] = 1
    return pred


def get_signals(X, utt, mean, std):
    for u in np.unique(utt):
        Xu = np.asarray(X[utt == u], dtype=np.float32)
        yield u, np.linalg.norm(Xu, axis=1), (Xu[:, FEAT] - mean[FEAT]) / std[FEAT]


def load_meta():
    meta = json.load(open(META_FILE))
    return np.array(meta["mean"], np.float32), np.array(meta["std"], np.float32)


def run_detector(X, utt, mean, std, signal, **kw):
    return {u: detect(f if signal == "feat" else n, **kw)
            for u, n, f in get_signals(X, utt, mean, std)}


def run_norm_excluding(X, utt, dims, **kw):
    """L2 norm computed over all dims EXCEPT `dims`, then the same detector."""
    preds = {}
    for u in np.unique(utt):
        Xu = np.asarray(X[utt == u], dtype=np.float32)
        sq = (Xu ** 2).sum(axis=1) - (Xu[:, dims] ** 2).sum(axis=1)
        preds[u] = detect(np.sqrt(np.clip(sq, 0, None)), **kw)
    return preds


def do_eval(a):
    # Imported here so --plot works even if these module names are wrong.
    # CHANGE this module name to the file that holds these two functions.
    from zerosylprobes.train_and_evaluate import evaluate_normseg_dynamically, save_checkpoint

    X = np.load(OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt = np.load(OUTPUT_DIR / "utt_dev.npy")
    tg = np.load(OUTPUT_DIR / "tg_dev.npy")
    mean, std = load_meta()
    results = {}

    # 1) does the single-utterance observation hold across the dev set?
    cors = [np.corrcoef(smooth(n, a.window, a.gauss), smooth(f, a.window, a.gauss))[0, 1]
            for _, n, f in get_signals(X, utt, mean, std) if len(n) > 5]
    cors = np.array(cors)
    print(f"\nPearson r (norm vs feat {FEAT}, smoothed, per utterance, n={len(cors)}): "
          f"mean={np.nanmean(cors):.3f}  median={np.nanmedian(cors):.3f}  "
          f"frac>0={np.nanmean(cors > 0):.2%}")

    kw = dict(window=a.window, prom_mult=a.prom, gauss_sigma=a.gauss)

    # 2) reference: norm through the identical detector (should equal NormSeg row)
    p = run_detector(X, utt, mean, std, "norm", **kw)
    results["Norm (same detector)"] = evaluate_normseg_dynamically(p, utt, tg, FRAME_RATE)

    # norm without feature 376
    p = run_norm_excluding(X, utt, [FEAT], **kw)
    results[f"Norm minus feat {FEAT}"] = evaluate_normseg_dynamically(p, utt, tg, FRAME_RATE)

    # control: norm minus a random single dimension (5 draws)
    rng = np.random.default_rng(0)
    for d in rng.choice([j for j in range(X.shape[1]) if j != FEAT], 5, replace=False):
        p = run_norm_excluding(X, utt, [int(d)], **kw)
        results[f"Norm minus random dim {d}"] = evaluate_normseg_dynamically(p, utt, tg, FRAME_RATE)

    # how much of the squared norm does feature 376 account for?
    num = den = 0.0
    for u in np.unique(utt):
        Xu = np.asarray(X[utt == u], dtype=np.float32)
        num += (Xu[:, FEAT] ** 2).sum()
        den += (Xu ** 2).sum()
    print(f"\nFeat {FEAT} share of total squared norm: {num/den:.4%}  "
          f"(uniform would be {1/X.shape[1]:.4%})")

    # 3) feature 376
    windows = [a.window] if not a.sweep else [1, 3, 5, 7]
    proms = [a.prom] if not a.sweep else [0.1, 0.25, 0.45, 0.7]
    for w in windows:
        for pm in proms:
            p = run_detector(X, utt, mean, std, "feat", window=w,
                             prom_mult=pm, gauss_sigma=a.gauss)
            results[f"Feat{FEAT} smoothed (win={w}, prom={pm})"] = \
                evaluate_normseg_dynamically(p, utt, tg, FRAME_RATE)
            save_checkpoint(results)

    print("\n" + "=" * 92)
    print(f"{'Method':<46}| {'P':>7} | {'R':>7} | {'F1':>7} | {'Tok-F1':>7} | {'R-val':>6}")
    print("-" * 92)
    for n, r in results.items():
        print(f"{n:<46}| {r['P']*100:7.2f} | {r['R']*100:7.2f} | {r['F1']*100:7.2f} | "
              f"{r['Token-F1']*100:7.2f} | {r['R-val']:6.3f}")


def zs(x):
    return (x - x.mean()) / (x.std() + 1e-8)


def plot_utt(utt_id, a):
    X = np.load(OUTPUT_DIR / "X_dev.npy", mmap_mode="r")
    utt = np.load(OUTPUT_DIR / "utt_dev.npy")
    mean, std = load_meta()

    # Works with either version of _load_utterance_data
    try:
        data = _load_utterance_data(utt_id, [FEAT], mean, std)
    except TypeError:
        data = _load_utterance_data(utt_id)

    m = utt == utt_id
    _, norm, f = next(get_signals(X[m], utt[m], mean, std))
    t = np.arange(len(norm)) * FRAME_DUR
    kw = dict(window=a.window, gauss_sigma=a.gauss)
    norm_s, f_s = smooth(norm, a.window, a.gauss), smooth(f, a.window, a.gauss)
    xmax = min(a.max_time, data["total_duration"]) if a.max_time > 0 else data["total_duration"]

    fig, ax = plt.subplots(3, 1, figsize=(12, 8), sharex=True,
                           gridspec_kw={"height_ratios": [2.2, 1.3, 1.3]})
    ax[0].imshow(data["spec_db"], aspect="auto", origin="lower", cmap="viridis",
                 extent=[0, data["total_duration"], 0, data["sr"] / 2])
    ax[0].set_ylabel("Frequency (Hz)")
    ax[0].set_title(f"Utterance: {utt_id}")

    for axis, raw, sm, color, name in [(ax[1], norm, norm_s, "tab:orange", "L2 norm"),
                                       (ax[2], f, f_s, "tab:blue", f"feat {FEAT}")]:
        axis.plot(t, zs(raw), color="lightgray", lw=0.8, label=f"{name} (raw)")
        axis.plot(t, zs(sm), color=color, lw=1.6, label=f"{name} (smoothed)")
        pk = np.where(detect(raw, prom_mult=a.prom, **kw) == 1)[0]
        axis.plot(t[pk], zs(sm)[pk], "v", color="k", ms=5, label="detected peaks")
        axis.set_ylabel(f"{name} (z)")
        axis.legend(loc="upper right", fontsize=8, ncol=3, frameon=False)
    ax[2].set_xlabel("Time (s)")

    for axis in ax:
        for b in data["boundaries"]:
            axis.axvline(b, color=BOUNDARY_COLOR, ls="--", lw=0.8, alpha=0.7)
        axis.set_xlim(0, xmax)
    plt.tight_layout()

    Path(a.save_dir).mkdir(parents=True, exist_ok=True)
    out = Path(a.save_dir) / f"{utt_id}_feat{FEAT}_vs_norm.pdf"  # Saved as PDF
    plt.savefig(out, dpi=300, bbox_inches="tight")              # High-res settings
    plt.close(fig)
    print(f"Saved: {out.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--utts", type=str, default="", help="comma-separated ids; default = first dev utterance")
    ap.add_argument("--window", type=int, default=3, help="moving-average length (NormSeg = 3)")
    ap.add_argument("--gauss", type=float, default=0.0, help=">0 uses Gaussian sigma (frames) instead")
    ap.add_argument("--prom", type=float, default=0.45)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--max_time", type=float, default=3.0, help="seconds to show; 0 = full utterance")
    ap.add_argument("--save_dir", type=str, default="figs", help="Directory to save output PDFs")
    a = ap.parse_args()

    if not (a.eval or a.plot):
        print("Nothing to do: pass --plot and/or --eval")

    if a.eval:
        do_eval(a)

    if a.plot:
        utts = [u.strip() for u in a.utts.split(",") if u.strip()]
        if not utts:
            all_utts = np.unique(np.load(OUTPUT_DIR / "utt_dev.npy"))
            utts = [str(all_utts[0])]
            print(f"No --utts given; using first dev utterance: {utts[0]}")
        for u in utts:
            plot_utt(u, a)