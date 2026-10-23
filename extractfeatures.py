import os
import torch
import torchaudio
import soundfile as sf
import numpy as np

from zerosylprobes.utils import (
    get_aligned_dataset_pairs,
    parse_textgrid_syllables,
    build_frame_labels,
)

## runs each recording through WavLM to get numerical description of the sound for every timestamp
## pairs that up with the boundary/no boundary answer sheet
## saves everything to disk

## limits the duration of the utterances to 5 hours
def get_duration_limited_pairs(pairs, max_hours):
    """
    Select utterances until a target duration (hours) is reached.
    """
    selected_pairs = []
    total_seconds = 0.0
    target_seconds = max_hours * 3600

    for audio_path, tg_path in pairs:
        #check how long this file is without loading the whole thing
        duration = sf.info(audio_path).duration
        selected_pairs.append((audio_path, tg_path))
        total_seconds += duration

        #stop once we have enough audio 
        if total_seconds >= target_seconds:
            break

    print(
        f"Selected {len(selected_pairs)} files "
        f"({total_seconds/3600:.2f} hours)"
    )

    return selected_pairs


def extract_dataset_split(
    pairs,
    model,
    device,
    target_layer=13,
):
    """
    Extract WavLM features and frame labels.
    """

    # one entry per utterance, joined together at the end
    features_list = []
    targets_list = []
    utt_ids_list = []
    tg_paths_list = []

    total = len(pairs)

    for idx, (audio_path, tg_path) in enumerate(pairs):

        #print progress every 50 files and then the last one
        if (idx + 1) % 50 == 0 or idx == total - 1:
            print(f"Processed {idx+1}/{total}")

        ##loads audio file 
        data, sr = sf.read(audio_path)

        # Ensure waveform is float32 tensor of shape (1, num_samples) (making sure audio is in right form)
        if data.ndim == 1:
            data = data[None, :]  # add channel dim -> (1, N)
        else:
            data = data.T  # (channels, N)

        waveform = torch.from_numpy(data).float()

        # WavLM expects 16kHz so resample if needed
        if sr != 16000:
            waveform = torchaudio.functional.resample(
                waveform,
                sr,
                16000,
            )
        ## feeds audio into WavLM so that WavLM can listen to audio and give back a large vector of numbers 
        waveform = waveform.to(device)

        # no_grad because only reading features not training
        with torch.no_grad():
            features, _ = model.extract_features(waveform)

        # take the chosen layer, drop the batch dimension and move to numpy
        h = (
            features[target_layer - 1]
            .squeeze(0)
            .cpu()
            .numpy()
            .astype(np.float16)      # <-- halves storage
        )

        # number of frames WavLM gave us for this utterance
        num_frames = h.shape[0]

        ## fetches correct answer sheet for this recording 
        boundaries = parse_textgrid_syllables(
            tg_path,
            tier_name="syllables",
        )

        ## target
        ## one boundary/no boundary label for each frames
        y = build_frame_labels(num_frames, boundaries)

        #utterance name = audio filename without extension
        utt = os.path.splitext(os.path.basename(audio_path))[0]

        ## keep track of which recording each utterance came from
        features_list.append(h)
        targets_list.append(y)
        utt_ids_list.append(np.array([utt] * num_frames))
        tg_paths_list.append(np.array([tg_path] * num_frames))


    #stack all utterances into single big arrays (one row per frame)
    X = np.concatenate(features_list, axis=0)
    y = np.concatenate(targets_list)
    utt_ids = np.concatenate(utt_ids_list)
    tg_paths = np.concatenate(tg_paths_list)

    return X, y, utt_ids, tg_paths


def main():

    data_dir = "data"
    output_dir = "saved_features"

    #make output folder if it doesn't exist yet
    os.makedirs(output_dir, exist_ok=True)

    # use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"\nLoading WavLM Large on {device}")

    # load pretrained WavLM model and put into eval mode
    bundle = torchaudio.pipelines.WAVLM_LARGE
    model = bundle.get_model().eval().to(device)

    # TRAIN (5 HOURS)
    print("\nLoading train-clean-100")

    # list of audio file, TG file pairs for the training split
    train_pairs = get_aligned_dataset_pairs(
        data_dir=data_dir,
        split="train-clean-100",
    )

    # keep only the first 5 hours
    train_pairs = get_duration_limited_pairs(
        train_pairs,
        max_hours=5.0,
    )

    X_train, y_train, utt_train, tg_train = extract_dataset_split(
        train_pairs,
        model,
        device,
    )

    # DEV (30 MINUTES)
    print("\nLoading dev-clean")

    dev_pairs = get_aligned_dataset_pairs(
        data_dir=data_dir,
        split="dev-clean",
    )

    dev_pairs = get_duration_limited_pairs(
        dev_pairs,
        max_hours=0.5,      # Change to 1.0 if desired
    )

    X_dev, y_dev, utt_dev, tg_dev = extract_dataset_split(
        dev_pairs,
        model,
        device,
    )

    print("\nSaving...")
    
    #traing: features, labels, utterance ids, textGrid paths
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "utt_train.npy"), utt_train)
    np.save(os.path.join(output_dir, "tg_train.npy"), tg_train)
   
    #dev
    np.save(os.path.join(output_dir, "X_dev.npy"), X_dev)
    np.save(os.path.join(output_dir, "y_dev.npy"), y_dev)
    np.save(os.path.join(output_dir, "utt_dev.npy"), utt_dev)
    np.save(os.path.join(output_dir, "tg_dev.npy"), tg_dev)

    print("\nDone!")

    ##shape check (number of frames, feature size)
    print(f"Train: {X_train.shape}")
    print(f"Dev:   {X_dev.shape}")


if __name__ == "__main__":
    main()