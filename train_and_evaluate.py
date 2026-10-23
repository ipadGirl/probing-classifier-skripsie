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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from zerosyl.eval.boundaries import evaluate_boundary_metrics
from zerosylprobes.normseg import generate_normseg_preds
from zerosylprobes.train_linear_probe import select_threshold_by_boundary_metric  # or copy the function in

OUTPUT_BASE = Path(os.environ.get("ZEROSYL_TMP_DIR", tempfile.gettempdir())) / "zerosyl_probe_outputs"
OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

# partial results are saved here after every experiment
CHECKPOINT_FILE = OUTPUT_BASE / "results_checkpoint.json"


def log_mem(tag=""):
    """Cheap, dependency-free peak-RSS logger (stdlib `resource`, Unix only)."""
    try:
        peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(f"  [mem] peak RSS so far{(' @ ' + tag) if tag else ''}: {peak_mb:.1f} MB")
    except Exception:
        # memory logging is only for monitoring, so never let it stop the run
        pass


def save_checkpoint(results):
    """Write partial results to disk after every experiment so a crash never
    loses everything that already ran."""
    try:
        with open(CHECKPOINT_FILE, "w") as f:
            json.dump(results, f, indent=2, default=float)
    except Exception as e:
        print(f"  [warn] could not write checkpoint: {e}")


def frames_to_zerosyl_segments(pred_frames):
    """
    Converts binary 0/1 predictions per frame into a (N_segments, 3) tensor
    expected by ZeroSyl: [start_frame, end_frame, dummy_val].
    """
    boundary_indices = np.where(pred_frames == 1)[0]
    total_frames = len(pred_frames)

    if len(boundary_indices) > 0:
        # neighbouring boundary frames count as one boundary, so keep the middle of each run
        runs = np.split(boundary_indices, np.where(np.diff(boundary_indices) != 1)[0] + 1)
        boundary_indices = np.array([r[len(r) // 2] for r in runs])
    else:
        boundary_indices = np.array([])

    # the utterance start and end are always cut points
    points = list(boundary_indices)
    if 0 not in points:
        points.insert(0, 0)
    if total_frames not in points:
        points.append(total_frames)
    points = sorted(set(points))

    # each segment runs from one cut point to the next
    segments = [[points[i], points[i + 1], 0] for i in range(len(points) - 1)]
    # fallback: treat the whole utterance as one segment
    if not segments:
        segments.append([0, total_frames, 0])

    return torch.tensor(segments, dtype=torch.long)


def generate_fixed_interval_preds(total_frames, frame_rate=50.0, interval_ms=100):
    # control baseline: a boundary every interval_ms, ignoring the audio entirely
    # (100 ms at 50 frames/s = every 5th frame)
    stride_frames = int(round((interval_ms / 1000.0) * frame_rate))
    pred_frames = np.zeros(total_frames, dtype=int)
    if stride_frames > 0:
        pred_frames[::stride_frames] = 1
    return pred_frames


def check_feature_scaling(X, feature_names=None, top_n=10, sample_size=50_000):
    # checks whether the features are on similar scales (matters for reading probe weights)
    n = X.shape[0]
    # use a random sample of frames if the dataset is big, to keep it quick
    if n > sample_size:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size, replace=False)
        X_sample = np.asarray(X[idx], dtype=np.float64)
    else:
        X_sample = np.asarray(X, dtype=np.float64)

    # mean and spread of every feature
    means = X_sample.mean(axis=0)
    stds = X_sample.std(axis=0)

    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(X_sample.shape[1])]

    # ratio of the biggest to the smallest spread across features
    ratio = stds.max() / (stds.min() + 1e-12)
    print(f"\nFeature scale diagnostic (n_features={X_sample.shape[1]}, "
          f"n_sampled={X_sample.shape[0]}/{n}):")
    print(f"  std  -> min={stds.min():.4f}, max={stds.max():.4f}, "
          f"mean={stds.mean():.4f}, max/min ratio={ratio:.2f}")
    print(f"  mean -> min={means.min():.4f}, max={means.max():.4f}")
    # more than 5x difference = features are on different scales
    if ratio > 5:
        print(f"  NOTE: std ratio > 5x -> features are NOT on a comparable scale. "
              f"Raw coefficients would be misleading for interpretability.")
    else:
        print(f"  NOTE: std ratio is modest -> features are roughly comparable already, "
              f"but z-scoring is still good practice before interpreting coefficients.")

    # list the features with the largest spread
    order = np.argsort(stds)[::-1]
    print(f"\n  Top {top_n} highest-std raw features (would dominate an unscaled coefficient):")
    for i in order[:top_n]:
        print(f"    {feature_names[i]:<12} std={stds[i]:.4f}  mean={means[i]:.4f}")

    del X_sample
    return means, stds


def zscore_normalize(X_tr, X_ev):
    # z-score both sets using the TRAIN mean/std only (no leakage from eval)
    scaler = StandardScaler()
    X_tr_z = scaler.fit_transform(X_tr.astype(np.float32, copy=False))
    X_ev_z = scaler.transform(X_ev.astype(np.float32, copy=False))
    return X_tr_z, X_ev_z, scaler


def inspect_logreg_coefficients(clf, feature_names=None, top_n=15):
    # prints the features with the biggest weights in a trained logistic regression
    coefs = clf.coef_.ravel()
    if feature_names is None:
        feature_names = [f"feat_{i}" for i in range(len(coefs))]

    # sort by weight size, biggest first
    order = np.argsort(np.abs(coefs))[::-1]
    print(f"\nTop {top_n} most influential features (fit on z-normalized inputs):")
    print(f"  {'feature':<12} | {'coef':>10} | direction")
    print(f"  {'-'*12}-+-{'-'*10}-+----------")
    for i in order[:top_n]:
        # positive weight pushes towards "boundary", negative towards "no boundary"
        direction = "-> boundary" if coefs[i] > 0 else "-> no boundary"
        print(f"  {feature_names[i]:<12} | {coefs[i]:>10.4f} | {direction}")


def run_experiment(variant_name, X_tr, y_tr, X_ev, y_ev, utt_ev, tg_ev,
                    utt_tr=None, tg_tr=None,
                    mode="probe", interval_ms=100, frame_rate=50.0,
                    return_clf=False, val_frac=0.1, threshold=None,
                    metric="F1", random_state=42):
    # runs one experiment end to end and returns its scores.
    # mode="probe" = logistic regression, mode="fixed_interval" = the fixed 100 ms baseline
    print(f"\n==================================================")
    print(f" Running Experiment Variant: {variant_name}")
    print(f"==================================================")

    # folders for predicted segments, probability curves, and matching TextGrids
    pred_dir = OUTPUT_BASE / f"pred_segments_{variant_name}"
    probs_dir = OUTPUT_BASE / f"pred_probs_{variant_name}"
    matched_tg_dir = OUTPUT_BASE / f"eval_textgrids_{variant_name}"

    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, probs_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    # sklearn needs 2D input (frames x features), so flatten anything bigger
    if X_tr is not None and X_tr.ndim > 2:
        print(f"  [warn] X_tr has shape {X_tr.shape}, flattening to 2D")
        X_tr = X_tr.reshape(X_tr.shape[0], -1)
    if X_ev is not None and X_ev.ndim > 2:
        print(f"  [warn] X_ev has shape {X_ev.shape}, flattening to 2D")
        X_ev = X_ev.reshape(X_ev.shape[0], -1)

    clf = None
    if mode == "probe":
        # balanced class weights because boundary frames are rare
        clf = LogisticRegression(
            max_iter=500, class_weight="balanced",
            random_state=42, n_jobs=1,
        )
        clf.fit(X_tr, y_tr)

        #threshold selection on a held-out slice of TRAIN, not dev
        if threshold is None:
            # pick a random 10% of the training utterances to tune the threshold on
            rng = np.random.RandomState(random_state)
            unique_tr_utts = np.unique(utt_tr)
            val_utts = rng.choice(
                unique_tr_utts,
                size=max(1, int(len(unique_tr_utts) * val_frac)),
                replace=False
            )
            val_mask = np.isin(utt_tr, val_utts)
            # probability of "boundary" for each of those frames
            val_scores = clf.predict_proba(X_tr[val_mask])[:, 1]

            threshold = select_threshold_by_boundary_metric(
                val_scores, utt_tr[val_mask], tg_tr[val_mask], metric=metric
            )
        else:
            print(f"Using fixed threshold={threshold:.4f}")

        # probabilities for every eval frame, then 0/1 using the chosen threshold
        y_probs = clf.predict_proba(X_ev)[:, 1]
        y_pred = (y_probs > threshold).astype(int)

    # save each utterance's predictions and copy its TextGrid
    unique_utts = np.unique(utt_ev)
    for utt in unique_utts:
        mask = (utt_ev == utt)
        utt_tg_path = Path(tg_ev[mask][0]).resolve()

        pt_file = pred_dir / f"{utt}.pt"
        prob_file = probs_dir / f"{utt}_probs.npy"
        tg_file = matched_tg_dir / f"{utt}.TextGrid"

        if mode == "probe":
            utt_preds = y_pred[mask]
            utt_probs = y_probs[mask]
        elif mode == "fixed_interval":
            total_utt_frames = np.sum(mask)
            utt_preds = generate_fixed_interval_preds(total_utt_frames, frame_rate=frame_rate, interval_ms=interval_ms)
            utt_probs = utt_preds.astype(np.float32)

        torch.save(frames_to_zerosyl_segments(utt_preds), pt_file)
        np.save(prob_file, utt_probs)

        if not tg_file.exists():
            shutil.copy(utt_tg_path, tg_file)

    # compare predicted boundaries to the true ones (50 ms tolerance)
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

    log_mem(variant_name)

    # optionally hand back the trained model too (used to inspect its weights)
    if return_clf:
        return result, clf
    return result


def evaluate_normseg_dynamically(normseg_preds_dict, utt_ev, tg_ev, frame_rate=50.0):
    # scores NormSeg predictions (a dict of {utterance id: 0/1 frame predictions})
    # with the same evaluation used for the probes
    print(f"\n==================================================")
    print(f" Running Dynamic Evaluation: NormSeg")
    print(f"==================================================")

    pred_dir = OUTPUT_BASE / "pred_segments_normseg"
    matched_tg_dir = OUTPUT_BASE / "eval_textgrids_normseg"

    # start from empty folders so old results don't get mixed in
    for d in [pred_dir, matched_tg_dir]:
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    unique_utts = np.unique(utt_ev)
    missing_count = 0

    for utt in unique_utts:
        mask = (utt_ev == utt)
        utt_tg_path = Path(tg_ev[mask][0]).resolve()

        pt_file = pred_dir / f"{utt}.pt"
        tg_file = matched_tg_dir / f"{utt}.TextGrid"

        # only utterances that have predictions get saved and scored
        if utt in normseg_preds_dict:
            utt_preds = normseg_preds_dict[utt]
            torch.save(frames_to_zerosyl_segments(utt_preds), pt_file)
            if not tg_file.exists():
                shutil.copy(utt_tg_path, tg_file)
        else:
            missing_count += 1

    if missing_count > 0:
        print(f"Warning: {missing_count} utterances were missing from the NormSeg predictions dictionary.")

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
    print(f"  --> NormSeg Token-F1: {tf1 * 100:.2f}% | Frame F1: {f1 * 100:.2f}%")

    return {
        "P": p, "R": r, "F1": f1,
        "OS": os_val, "R-val": rval,
        "Token-P": tp, "Token-R": tr, "Token-F1": tf1,
    }


def print_summary(results):
    # results table, one row per method; P, R and F1 are shown as percentages
    print("\n" + "=" * 104)
    print(f"{'SYLLABLE BOUNDARY EVALUATION RESULTS (dev-clean)':^104}")
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

    # load all the saved features, labels, utterance ids and TextGrid paths
    print("\nLoading extracted feature arrays...")
    X_train = np.load(os.path.join(output_dir, "X_train.npy"))
    y_train = np.load(os.path.join(output_dir, "y_train.npy"))
    utt_train = np.load(os.path.join(output_dir, "utt_train.npy"))
    tg_train = np.load(os.path.join(output_dir, "tg_train.npy"))
    X_dev = np.load(os.path.join(output_dir, "X_dev.npy"))
    y_dev = np.load(os.path.join(output_dir, "y_dev.npy"))
    utt_dev = np.load(os.path.join(output_dir, "utt_dev.npy"))
    tg_dev = np.load(os.path.join(output_dir, "tg_dev.npy"))
    log_mem("after loading features")

    results = {}

    # Fixed-Interval Baseline Control (100 ms)
    # no features or training needed, so those inputs are None
    results["Fixed Baseline (100ms)"] = run_experiment(
        "fixed_100ms", None, None, None, y_dev, utt_dev, tg_dev, mode="fixed_interval", interval_ms=100
    )
    save_checkpoint(results)

    #NormSeg Baseline (peak detection on the L2 norm of frame embeddings)
    normseg_dict = generate_normseg_preds(
        X_dev, utt_dev,
        prominence_multiplier=0.45,
        window_size=3,
    )
    results["NormSeg (L2 Norm)"] = evaluate_normseg_dynamically(normseg_dict, utt_dev, tg_dev)
    del normseg_dict
    save_checkpoint(results)

    # logistic regression on the raw (unscaled) features
    results["Full Vector Probe"] = run_experiment(
    "full_vector", X_train, y_train, X_dev, y_dev, utt_dev, tg_dev,
    utt_tr=utt_train, tg_tr=tg_train,
    mode="probe"
    )
    save_checkpoint(results)

    elapsed = time.perf_counter() - start
    print(f"\nLogistic probe runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    #Z-Score Normalized Probe - for interpretability of coefficients
    check_feature_scaling(X_train)

    print("\nApplying z-score normalization (StandardScaler, fit on train only)...")
    X_train_z, X_dev_z, scaler = zscore_normalize(X_train, X_dev)

    # free the raw copies, only the z-scored ones are needed from here
    del X_train, X_dev
    gc.collect()
    log_mem("after freeing raw features")

    zscore_result, zscore_clf = run_experiment(
        "zscore_full_vector", X_train_z, y_train, X_dev_z, y_dev, utt_dev, tg_dev, utt_tr=utt_train, tg_tr=tg_train,
        mode="probe", return_clf=True
    )
    results["Z-Norm Full Vector Probe"] = zscore_result
    save_checkpoint(results)

    del X_train_z, X_dev_z
    gc.collect()

    # show which features the z-scored probe relies on most
    inspect_logreg_coefficients(zscore_clf, top_n=15)

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