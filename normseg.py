"""
NormSeg: reproduces ZeroSyl's own boundary detector as a
baseline, so it can be compared directly against the probes.

As described in the ZeroSyl paper (Sec. 3.1), boundaries are found by
running prominence-based peak detection on the L2 norm of frame embeddings,
after smoothing with a short moving-average filter. This is the same
smoothing/prominence machinery as PromSeg (promseg.py), but applied to the
L2 norm signal instead of the cosine distance between adjacent frames.
"""

import numpy as np
from scipy.signal import find_peaks


def _smooth_signal(signal_1d, window_size=3):
    """
    Simple moving-average smoothing over a 1D signal (matches the paper's
    "3-point moving average filter" applied before peak detection).
    window_size=1 is a no-op (no smoothing).
    """
    # a window of 1 means no smoothing so just hand the signal back
    if window_size <= 1:
        return signal_1d
    # kernel of equal weights that add up to 1 = a plain average 
    kernel = np.ones(window_size) / window_size
    # "same" keeps the output the same length as the input
    return np.convolve(signal_1d, kernel, mode="same")


def compute_l2norm_peaks(X_utt, prominence_multiplier=0.45, window_size=3, min_distance_frames=1):
    """
    NormSeg boundary detection for a single utterance.

    - window_size: length of the moving-average smoothing filter applied
      to the L2-norm signal before peak detection (paper uses 3).
    - prominence_multiplier: a peak counts as a boundary if its prominence
      is at least `prominence_multiplier * sigma`, where sigma is the
      standard deviation of the (smoothed) signal for this utterance
      (paper uses delta = 0.45 * sigma).
    - min_distance_frames: minimum spacing enforced between detected
      peaks; not specified explicitly in the paper, defaults to 1 (i.e.
      no extra constraint beyond what prominence already enforces).

    Returns a binary frame-level prediction array of length T.
    """
    X_utt = X_utt.astype(np.float32)
    total_frames = X_utt.shape[0]
    # start with "no boundary" for every frame
    pred_frames = np.zeros(total_frames, dtype=int)

    # nothing to do for an empty utterance
    if total_frames == 0:
        return pred_frames

    # n_t = ||h_t||_2 for each frame
    norms = np.linalg.norm(X_utt, axis=1)

    #smooth the norm signal, then set prominence threshold from its spread
    smoothed = _smooth_signal(norms, window_size=window_size)
    sigma = np.std(smoothed)
    prominence = prominence_multiplier * sigma

    # every peak that stands out enoigh is marked as a boundary
    peak_indices, _ = find_peaks(smoothed, prominence=prominence, distance=min_distance_frames)
    pred_frames[peak_indices] = 1
    return pred_frames


def generate_normseg_preds(X_ev, utt_ev, prominence_multiplier=0.45, window_size=3, min_distance_frames=1):
    """
    Runs the NormSeg peak-detection baseline over every utterance in the
    evaluation set, using layer-13 features (X_ev is assumed to already be
    layer 13, matching ZeroSyl's boundary-detection layer). Returns a dict
    utt_id -> binary frame-level predictions, directly compatible with
    evaluate_normseg_dynamically / frames_to_zerosyl_segments.
    """
    preds_dict = {}
    unique_utts = np.unique(utt_ev)
    #run the detector on each utterance seperately
    for utt in unique_utts:
        # pick out just this utterance's frames
        mask = (utt_ev == utt)
        X_utt = X_ev[mask]
        preds_dict[utt] = compute_l2norm_peaks(
            X_utt,
            prominence_multiplier=prominence_multiplier,
            window_size=window_size,
            min_distance_frames=min_distance_frames,
        )
    return preds_dict