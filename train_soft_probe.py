import os

# Must be set before numpy/sklearn import BLAS. Prevents thread oversubscription
# (joblib workers x BLAS threads) which can spike CPU/RAM and hang the machine
# on high-core-count boxes. Tune these to taste.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import gc
import json
import shutil
import sys
import tempfile
import traceback
import resource
from pathlib import Path

import numpy as np
import torch
import time
from scipy.signal import find_peaks
from sklearn.linear_model import Ridge

from zerosyl.eval.boundaries import evaluate_boundary_metrics


#keep per-utterance intermediate files outside the VS Code workspace.
# Override with: export ZEROSYL_TMP_DIR=/some/other/path
OUTPUT_BASE = Path(os.environ.get("ZEROSYL_TMP_DIR", tempfile.gettempdir())) / "zerosyl_soft_probe_outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

#partial results are saved here after every h-value
CHECKPOINT_FILE = OUTPUT_BASE / "soft_results_checkpoint.json"


def log_mem(tag=""):
    """Cheap, dependency-free peak-RSS logger (stdlib `resource`, Unix only)."""
    try:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"  [mem] peak RSS so far{(' @ ' + tag) if tag else ''}: {peak_mb:.1f} MB")
    except Exception:
        pass


def save_checkpoint(results):
    """Write partial results to disk after every h-value so a crash never
    loses everything that already ran."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(results, f, indent=2, default=float)
    except Exception as e:
        print(f"  [warn] could not write checkpoint: {e}")


def frames_to_zerosyl_segments(pred_frames):
    """
    Converts binary 0/1 predictions per frame into a (N_segments, 3) tensor
    expected by ZeroSyl: [start_frame, end_frame, 0].
    """
    boundary_indices = np.where(pred_frames == 1)[0]
    total_frames = len(pred_frames)

    if len(boundary_indices) > 0:
        runs = np.split(boundary_indices, np.where(np.diff(boundary_indices) != 1)[0] + 1)
        boundary_indices = np.array([r[len(r) // 2] for r in runs])
    else:
        boundary_indices = np.array([])

    points = list(boundary_indices)
    if 0 not in points:
        points.insert(0, 0)
    if total_frames not in points:
        points.append(total_frames)
    points = sorted(set(points))

    segments = [[points[i], points[i + 1], 0] for i in range(len(points) - 1)]
    if not segments:
        segments.append([0, total_frames, 0])

    return torch.tensor(segments, dtype=torch.long)



# Soft-label construction
def build_gaussian_soft_labels(num_frames, boundary_frames, h):
    """
    Builds one utterance's soft target curve.

    num_frames: number of frames in this utterance only
    boundary_frames: array of local frame indices (0..num_frames-1) where a
                      syllable boundary occurs in this utterance
    h: width parameter (std, in frames) of the Gaussian bump

    y[t] = exp( -(t - b)^2 / (2 h^2) )  maxed over every boundary b

    We take the max rather than the sum across nearby boundaries so that two
    boundaries close together don't produce a target value above 1 -- we
    want a clean [0, 1] curve for Ridge to regress onto, not a target that
    depends on how many boundaries happen to be nearby.
    """
    t = np.arange(num_frames)
    y = np.zeros(num_frames, dtype=np.float32)

    for b in boundary_frames:
        bump = np.exp(-((t - b) ** 2) / (2.0 * h ** 2))
        y = np.maximum(y, bump)

    return y


def build_soft_targets_per_utterance(y_hard, utt_ids, h):
    """
    Rebuilds soft Gaussian targets from the flat hard-label arrays that
    extract_dataset_split() already saved (y_train.npy / utt_train.npy),
    without letting boundaries from one utterance leak into another.
    """
    y_soft = np.zeros_like(y_hard, dtype=np.float32)

    unique_utts = np.unique(utt_ids)
    for utt in unique_utts:
        mask = (utt_ids == utt)
        local_hard = y_hard[mask]
        num_frames = len(local_hard)

        # local (per-utterance) boundary indices, derived from that
        # utterance's own slice of the hard labels only
        boundary_frames = np.where(local_hard == 1)[0]

        local_soft = build_gaussian_soft_labels(num_frames, boundary_frames, h)
        y_soft[mask] = local_soft

    return y_soft


def peak_pick_preds(scores, prominence=0.1, distance=1):
    """
    Turns a continuous predicted curve back into discrete 0/1 boundary
    predictions by picking local maxima. prominence and distance are in the
    same units as `scores` / frames respectively -- worth sweeping too if
    the h-sweep results look sensitive to these.
    """
    scores = np.asarray(scores, dtype=np.float64)
    peaks, _ = find_peaks(scores, prominence=prominence, distance=distance)
    out = np.zeros(len(scores), dtype=int)
    out[peaks] = 1
    return out


# Experiment runner 

def run_soft_experiment(variant_name, X_tr, y_tr_soft, X_ev, utt_ev, tg_ev,
                         alpha=1.0, peak_prominence=0.1, peak_distance=3,
                         frame_rate=50.0):
    print(f"\n==================================================")
    print(f" Running Experiment Variant: {variant_name}")
    print(f"==================================================")

    pred_dir = OUTPUT_BASE / f"pred_segments_{variant_name}"
    probs_dir = OUTPUT_BASE / f"pred_probs_{variant_name}"
    matched_tg_dir = OUTPUT_BASE / f"eval_textgrids_{variant_name}"

    for d in [pred_dir, probs_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # Explicitly cast to float32 rather than letting sklearn silently
    # upcast float16 -> float64 internally. float64 on a ~900k x 1024
    # array is ~4x the memory of float32, and doing that upcast fresh on
    # every h-value in the sweep (with nothing freed in between) is what
    # was driving the RAM usage that crashed VS Code -- float32 keeps
    # this bounded and is plenty of precision for this regression.
    X_tr32 = np.asarray(X_tr, dtype=np.float32)
    y_tr32 = np.asarray(y_tr_soft, dtype=np.float32)

    clf = Ridge(alpha=alpha, random_state=42)
    clf.fit(X_tr32, y_tr32)
    del X_tr32, y_tr32

    X_ev32 = np.asarray(X_ev, dtype=np.float32)
    y_scores = clf.predict(X_ev32)
    y_scores = np.clip(y_scores, 0.0, None)  # curve shouldn't go negative
    del X_ev32

    unique_utts = np.unique(utt_ev)
    for utt in unique_utts:
        mask = (utt_ev == utt)
        utt_tg_path = Path(tg_ev[mask][0]).resolve()

        pt_file = pred_dir / f"{utt}.pt"
        prob_file = probs_dir / f"{utt}_probs.npy"
        tg_file = matched_tg_dir / f"{utt}.TextGrid"

        utt_scores = y_scores[mask]
        utt_preds = peak_pick_preds(utt_scores, prominence=peak_prominence, distance=peak_distance)

        torch.save(frames_to_zerosyl_segments(utt_preds), pt_file)
        np.save(prob_file, utt_scores)

        if not tg_file.exists():
            shutil.copy(utt_tg_path, tg_file)

    metrics = evaluate_boundary_metrics(
        segments_dir=pred_dir,
        textgrid_dir=matched_tg_dir,
        segments_pattern="*.pt",
        textgrid_pattern="*.TextGrid",
        frame_rate=frame_rate,
        constant_shift=0.0,
        tolerance=0.05,
    )

    p, r, f1, os_val, rval, tp, tr, tf1 = metrics
    print(f"  --> Variant '{variant_name}' Token-F1: {tf1 * 100:.2f}% | Frame F1: {f1 * 100:.2f}%")

    result = {
        "P": p, "R": r, "F1": f1,
        "OS": os_val, "R-val": rval,
        "Token-P": tp, "Token-R": tr, "Token-F1": tf1,
    }

    del clf, y_scores
    gc.collect()
    log_mem(variant_name)

    return result


def print_summary(results):
    print("\n" + "=" * 104)
    print(f"{'SOFT-TARGET GAUSSIAN RIDGE PROBE RESULTS (dev-clean)':^104}")
    print("=" * 104)
    header = f"{'Method / Variant':<24} | {'Prec (%)':<9} | {'Rec (%)':<9} | {'F1 (%)':<9} | {'Token-F1 (%)':<13} | {'OS':<8} | {'R-val':<8}"
    print(header)
    print("-" * len(header))
    for name, res in results.items():
        print(
            f"{name:<24} | "
            f"{res['P']*100:>8.2f} | "
            f"{res['R']*100:>8.2f} | "
            f"{res['F1']*100:>8.2f} | "
            f"{res['Token-F1']*100:>12.2f} | "
            f"{res['OS']:>8.3f} | "
            f"{res['R-val']:>8.3f}"
        )
    print("=" * 104 + "\n")


def main():
    start = time.perf_counter()
    output_dir = "saved_features"

    print(f"Writing per-utterance intermediate files to: {OUTPUT_BASE}")
    print("(kept outside your VS Code workspace to avoid file-watcher overload)")

    print("\nLoading extracted feature arrays...")
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train_hard = np.load(os.path.join(output_dir, "y_train.npy"))
    utt_train = np.load(os.path.join(output_dir, "utt_train.npy"))

    X_dev = np.load(os.path.join(output_dir, "X_dev.npy"))
    utt_dev = np.load(os.path.join(output_dir, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(output_dir, "tg_dev.npy"))
    log_mem("after loading features")

    results = {}

    # Width parameter sweep, in frames. At 50 fps (20ms/frame) this range is
    # roughly 20ms to 260ms. Adjust to whatever range makes sense for your
    # syllable durations.
    h_values = [1, 1.5, 2, 2.5, 3, 5, 8, 13]

    for h in h_values:
        print(f"\nBuilding per-utterance Gaussian soft targets for h={h}...")
        y_train_soft = build_soft_targets_per_utterance(y_train_hard, utt_train, h)

        variant_name = f"gaussian_ridge_h{h}"
        results[f"Soft Probe (h={h})"] = run_soft_experiment(
            variant_name,
            X_train, y_train_soft,
            X_dev, utt_dev, tg_dev,
        )
        del y_train_soft
        gc.collect()
        save_checkpoint(results)

    elapsed = time.perf_counter() - start
    print(f"\nSoft probe runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    print_summary(results)
    save_checkpoint(results)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n[FATAL] Uncaught exception -- partial results (if any) were still",
              f"checkpointed to {CHECKPOINT_FILE}\n", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)