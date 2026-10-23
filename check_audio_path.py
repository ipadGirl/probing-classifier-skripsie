"""
before running check_PC0_energy: run this to check that TextGrid alignmnts can be matched to raw audio files


AUDIO_ROOT, the top of wherever your raw LibriSpeech audio
lives (not the alignments package). If you're not sure, try:
    find data -iname "*.flac" | head -3
from your project root, and set AUDIO_ROOT to the common parent of those
results (e.g. "data/LibriSpeech").
"""
import os
from pathlib import Path

AUDIO_ROOT = "data/waveforms/LibriSpeech"   # can confirm this via: find data -iname "*.flac" | head -3
AUDIO_EXTS = (".flac", ".wav", ".sph") #audio file types counted as audio in this project

_INDEX = None  # built once


def _build_index(root):
    #makes a lookup of {filename without extension: full path to audio file}
    index = {}
    n_files = 0
    # go through every folder under root
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            stem, ext = os.path.splitext(fn)
            #only keep files with an audio extension
            if ext.lower() in AUDIO_EXTS:
                index[stem] = os.path.join(dirpath, fn)
                n_files += 1
    print(f"  indexed {n_files} audio files under {root}")
    return index


def audio_path_from_tg(tg_path):
    """Returns the matching audio path, or None if not found."""
    global _INDEX
    #build index the first time only
    if _INDEX is None:
        print(f"Indexing audio under {AUDIO_ROOT} (first call only)")
        _INDEX = _build_index(AUDIO_ROOT)
    #textGrid and audio files share the same name so we match on that
    stem = Path(tg_path).stem
    return _INDEX.get(stem)


def main():
    import numpy as np
    # load the saved list of TextGrid paths for the dev set
    tg_dev = np.load(os.path.join("saved_features", "tg_dev.npy"))
    # take the first 10 unique paths as a quick test or all of them if fewer than 10
    sample = np.unique(tg_dev)[:10] if len(np.unique(tg_dev)) >= 10 else np.unique(tg_dev)

    found = 0
    # try to find the Textgrid path in the sample from above
    for tg_path in sample:
        audio_path = audio_path_from_tg(tg_path)
        ok = audio_path is not None
        found += ok
        print(f"{'found    ' if ok else 'NOT FOUND'}  {Path(tg_path).stem}  ->  {audio_path}")

    #summary of matches
    print(f"\n{found}/{len(sample)} found.")
    if found == 0:
        print(f"Nothing found under AUDIO_ROOT = '{AUDIO_ROOT}'. Check that path exists and "
              f"actually contains audio (try: find {AUDIO_ROOT} -iname '*.flac' | head).")
    elif found < len(sample):
        print("Some missing -- check whether those specific utterances have audio at all.")
    else:
        print("All found -- safe to run check_pc0_energy.py now.")


if __name__ == "__main__":
    main()