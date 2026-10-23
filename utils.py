import os 
import glob
import torch
import torchaudio
import textgrid
import numpy as np

##syllables = index 2

## textgrid = similar to spreadsheet: syllable 1 starts at 0.1 seconds and ends at 0.35s etc


## extracting timestamp boundaries from the text grid (where one syllable ends and another begins)
## outputs list of floats: every syllable boundary time in seconds sorted and duplicates removed
def parse_textgrid_syllables(textgrid_path, tier_name="syllables"):
    tg = textgrid.TextGrid.fromFile(textgrid_path)
    selected_tier = None
    
    ##search tier name
    for tier in tg:
        if tier_name.lower() in tier.name.lower():
            selected_tier = tier
            break
            
    ## If named tier isn't found pick first available tier
    if selected_tier is None and len(tg) > 0:
        selected_tier = tg[0]

    if selected_tier is None:
        return []

    boundary_times = []
    for interval in selected_tier:
        if interval.mark.strip(): ## skipping empty silence intervals
            boundary_times.append(interval.minTime) ## start of syllable
            boundary_times.append(interval.maxTime) ## end of syllable (in seconds)

    return sorted(list(set(boundary_times))) ##remove any duplicate timestamps where one syllable end and the next begins at the exact same millisecond

## matches the audio with its answer key (.Textgrid files)
## pairs every recording with the correct syllable-timing sheet
## outputs list of tuples (audio_path, textgrid_path): one tuple per matched recording
def get_aligned_dataset_pairs(data_dir="data", split="dev-clean"):
    """
    Recursively finds matching .flac and .TextGrid files for a given split
    by matching their utterance IDs (e.g., '1089-134686-0001').
    """
    ##debug
    print(f"Searching for '{split}' pairs in '{data_dir}/'...")

    ## Search for all TextGrids belonging to the dev-clean split
    tg_pattern = os.path.join(data_dir, "**", split, "**", "*.TextGrid")
    tg_files = glob.glob(tg_pattern, recursive=True)
    
    # Fallback search across all TextGrids if split is not in subpath name
    if not tg_files:
        tg_files = glob.glob(os.path.join(data_dir, "**", "*.TextGrid"), recursive=True)

    ## Index TextGrids by their base ID (e.g. "1089-134686-0001")
    tg_map = {os.path.splitext(os.path.basename(f))[0]: f for f in tg_files}

    # Search for all audio FLAC files for dev-clean
    audio_pattern = os.path.join(data_dir, "**", split, "**", "*.flac")
    audio_files = glob.glob(audio_pattern, recursive=True)
    
    # Fallback search across all FLAC files
    if not audio_files:
        audio_files = glob.glob(os.path.join(data_dir, "**", "*.flac"), recursive=True)

    ## Match audio files with TextGrids
    file_pairs = []
    for audio_path in audio_files:
        file_id = os.path.splitext(os.path.basename(audio_path))[0]
        if file_id in tg_map:
            file_pairs.append((audio_path, tg_map[file_id]))

    print(f"Successfully matched {len(file_pairs)} audio-alignment pairs!") ##debug
    return file_pairs


## converts seconds to 20 ms frame 
## WavLM downsamples 16kHz through the 7 CNN layers by a factor of 320: 320/16000 = 0.02 seconds per feature vector
## determines if a small slice of audio is a syllable boundary or not for each fram
## ensure that for every slice of recording we know the correct answer (boundary, no boundary) to compare the AI's guesses against
def build_frame_labels(num_frames, boundary_times_sec, frame_shift=0.02, tolerance_sec=0.025):
    y = np.zeros(num_frames, dtype=int)
    for b_time in boundary_times_sec:
        center_frame = int(b_time/frame_shift)

        window_radius = int(tolerance_sec/frame_shift)

        start_f = max(0, center_frame - window_radius)
        end_f = min(num_frames, center_frame + window_radius +1)

        y[start_f: end_f] = 1

    return y