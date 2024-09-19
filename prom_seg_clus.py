"""
Main script to pre-process, cluster, and save speech word segments and their cluster assingments.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: August 2024
"""

import numpy as np
import argparse
import random
from pathlib import Path
import os
from tqdm import tqdm
from utils import data_process
from wordseg import cluster
from sklearn.decomposition import PCA

def get_data(data, args, speaker):
    """
    Get all the supporting data for the input utterances.

    Parameters
    ----------
    data : Features
        The data object, containing the features and the alignments.
    args : Namespace
        The arguments for the script.
    speaker : list (str)
        The speaker being processed.
        default: [None] (speaker-independet)

    Return
    ------
    lexicon_builder : 
        The segmentations of the downsampled utterances
    samples : list (str)
        The list of the sampled file paths to the features of the utterances.
    wavs : list (str)
        The list of the sampled file paths to the audio files of the utterances.
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances.
    """

    samples, wavs = data.sample_features(speaker) # sample file paths from the speech features
       
    pca = None
    if args.model not in ["mfcc", "melspec"]:
        print('Fitting PCA')
        pca = PCA(n_components=250)
        pca.fit(np.concatenate(data.load_features(random.sample(samples, int(0.8*len(samples)))), axis=0))

    if len(samples) == 0:
        print('No features to segment, sampled a file with only one frame.')
        exit()
    
    # get landmarks, lenths, and segments
    landmarks, lengths = get_landmarks(data, args, wavs)

    segments = []
    for landmark in landmarks: 
        segment = np.concatenate(([0], landmark))
        segment = [(segment[i], segment[i+1]) for i in range(len(segment)-1)]
        segments.append(segment)

    # use fixed K_max
    # K_max = 43000 # TODO 43000 for zrc english
    # K_max = 29000 # TODO 29000 for zrc french
    # K_max = 3000 # TODO 3000 for zrc mandarin
    # K_max = 29000 # TODO 29000 for zrc german
    # K_max = 3500 # TODO 3500 for zrc wolof
    K_max = 13967 # TODO 13967 for librispeech

    # load the utterance data into the lexicon builder
    lexicon_builder = cluster.CLUSTER(data, samples, segments, lengths, K_max, pca)

    return lexicon_builder, samples, wavs, landmarks

def get_landmarks(data, args, wavs):
    """
    Sample pre-saved landmarks for the utterances.

    Parameters
    ----------
    data : Features
        The data object, containing the features and the alignments.
    args : Namespace
        The arguments for the script.
    wavs : list (str)
        The list of the sampled file paths to the audio files of the utterances.

    Return
    ------
    landmarks : list (list (int))
        The landmarks (in frames) of the utterances.
    lengths : list (int)
        The length of each utterance's possible segments.
    """

    landmarks = []
    lengths = []
    for wav in tqdm(wavs, desc="Getting landmarks"):
        landmark_details = os.path.split(wav)
        landmark_dir = Path(args.load_landmarks / landmark_details[-1]).with_suffix(".list")
        with open(landmark_dir) as f:
            landmark = []
            for line in f:
                landmark.append(float(line.strip()))
            landmark = data.get_frame_num(np.array(landmark)).astype(np.int32).tolist()
            if landmark[-1] == 0:
                landmark = [1] # fix rounding error (0.5 -> 0.0)
            landmarks.append(landmark)
        lengths.append(len(landmarks[-1]))
    
    return landmarks, lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering prominence-segmented speech.")
    parser.add_argument(
        "model",
        help="available models",
        default="mfcc",
    )
    parser.add_argument(
        "layer", # -1 for no layer
        type=int,
    )
    parser.add_argument(
        "wav_dir",
        metavar="wav-dir",
        help="path to the audio waveform directory.",
        type=Path,
    )
    parser.add_argument(
        "feature_dir",
        metavar="feature-dir",
        help="path to the speech feature directory.",
        type=Path,
    )
    parser.add_argument(
        "load_landmarks",
        help="root landmark directory to load landmarks.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "save_segments",
        help="root directory to save word boundaries.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio waveform files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--sample_size",
        metavar="sample-size",
        help="number of features to sample (-1 to sample all available data).",
        default=-1,
        type=int,
    )
    parser.add_argument(
        "--speaker",
        help="Speaker list if speaker dependent sampling must be done.",
        default=None,
        type=Path,
    )
    args = parser.parse_args()

    if args.model in ["mfcc", "melspec"]:
        mfcc = True
        frame_len = 10
    else:
        mfcc = False
        frame_len = 20

    random.seed(42)
    np.random.seed(42)

    # ~~~~~~~~~~ Setup data ~~~~~~~~~~
    data = data_process.Features(wav_dir=args.wav_dir, root_dir=args.feature_dir, model_name=args.model, layer=args.layer, extension=args.extension, alignment_format=args.align_format, num_files=args.sample_size, frames_per_ms=frame_len)

    if args.speaker is not None:
        speakerlist = data.get_speakers(args.speaker)
    else:
        speakerlist = [None]

    for speaker in tqdm(speakerlist, desc="Speaker"):
        # ~~~~~~~~~~ Get data for all utterances without saving the features ~~~~~~~~~~~
        lexicon_builder, samples, wavs, landmarks = get_data(data, args, speaker)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Cluster segment embeddings ~~~~~~~~~~~~~~~~~~~~~~~~~~
        classes = lexicon_builder.cluster()

        # ~~~~~~~~~~~~~~~~~~~ Save utterance segments and assignments ~~~~~~~~~~~~~~~~~~~~
        for i in tqdm(range(lexicon_builder.D), desc='Getting boundary frames and classes'): # for each utterance
            segmentation_frames = landmarks[i]
            if len(classes[i]) == 1:
                class_i = classes[i]
            else:
                class_i = [x for x in classes[i] if x != -1]
            save_dir = (args.save_segments / os.path.split(wavs[i])[-1]).with_suffix(".list")
            save_dir.parent.mkdir(parents=True, exist_ok=True)
            with open(save_dir, "w") as f:
                for t, c in zip(segmentation_frames, class_i):
                    f.write(f"{data.get_sample_second(t)} {int(c)}\n")