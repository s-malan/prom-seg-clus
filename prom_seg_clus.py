"""
Main script to segment audio using ES-KMeans, and evaluate the resulting segmentation.

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
from wordseg import cluster, sylseg
from sklearn.decomposition import PCA
import pickle

# sys.path.append(str(Path("..")/".."/"src"/"eskmeans"/"utils")) use something like this to import the data_process scripts

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

    samples, wavs = data.sample_embeddings(speaker) # sample file paths from the feature embeddings
       
    pca = None
    if args.model not in ["mfcc", "melspec"]:
        print('Fitting PCA')
        pca = PCA(n_components=250)
        pca.fit(np.concatenate(data.load_embeddings(random.sample(samples, int(0.8*len(samples)))), axis=0))
        # with open('french_zrc_pca.pkl', 'wb') as file:
        #     pickle.dump(pca, file)
        # with open('french_zrc_pca.pkl', 'rb') as file:
        #     pca = pickle.load(file)

    if len(samples) == 0:
        print('No embeddings to segment, sampled a file with only one frame.')
        exit()
    
    # Get landmarks
    landmarks, lengths = get_landmarks(data, args, wavs)

    segments = []
    for landmark in landmarks: # for each utterance get the initial data
        # Get durations and active segments
        segment = np.concatenate(([0], landmark)) # get all possible segments alignment_end_frames))
        segment = [(segment[i], segment[i+1]) for i in range(len(segment)-1)]
        segments.append(segment)

    # use fixed K_max
    # K_max = 43000 # TODO 43000 for zrc english!!!
    # K_max = 29000 # TODO 29000 for zrc french!!!
    # K_max = 3000 # TODO 3000 for zrc mandarin!!!
    # K_max = 29000 # TODO 29000 for zrc german!!!
    # K_max = 3500 # TODO 3500 for zrc wolof!!!
    K_max = 13967 # TODO 13967 for librispeech!!!

    # load the utterance data into the ESKMeans segmenter
    lexicon_builder = cluster.CLUSTER(data, samples, segments, lengths, K_max, pca)

    return lexicon_builder, samples, wavs, landmarks

def get_landmarks(data, args, wavs):
    """
    Sample pre-saved landmarks for the utterances.

    Parameters
    ----------
    data : 
        The downsampled embeddings of the encoded utterances
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
    for wav in tqdm(wavs, desc="Getting landmarks"): # for each utterance
        
        # Get and load/save landmarks
        landmark_details = os.path.split(wav)

        # if args.load_landmarks is not None:
        landmark_dir = Path(args.load_landmarks / landmark_details[-1]).with_suffix(".list")
        with open(landmark_dir) as f:
            landmark = []
            for line in f:
                landmark.append(float(line.strip())) # loaded into frames
            landmark = data.get_frame_num(np.array(landmark)).astype(np.int32).tolist()
            if landmark[-1] == 0:
                landmark = [1] # fix rounding error (0.5 -> 0.0)
            landmarks.append(landmark)
        # else:
        #     landmark_root_dir = os.path.split(args.embeddings_dir)
        #     landmark_root_dir = os.path.join(os.path.commonpath([args.embeddings_dir, wav]), 'segments', 'sylseg', os.path.split(landmark_root_dir[0])[-1], landmark_root_dir[-1])
        #     if args.layer != -1:
        #         landmark_dir = Path(os.path.join(landmark_root_dir, args.model, str(args.layer), landmark_details[-1])).with_suffix(".list")
        #     else:
        #         landmark_dir = Path(os.path.join(landmark_root_dir, args.model, landmark_details[-1])).with_suffix(".list")

        #     if os.path.isfile(landmark_dir):
        #         with open(landmark_dir) as f: # if the landmarks are already saved to a file
        #             landmark = []
        #             for line in f:
        #                 landmark.append(data.get_frame_num(line.strip())) # load into frames
        #             landmarks.append(landmark)
        #     else:
        #         landmarks.append(data.get_frame_num(sylseg.get_boundaries(wav, fs=16000)).astype(np.int32).tolist()[1:]) # get the boundaries in frames
        #         landmark_dir.parent.mkdir(parents=True, exist_ok=True)
        #         with open(landmark_dir, "w") as f: # save the landmarks to a file
        #             for l in landmarks[-1]:
        #                 f.write(f"{data.get_sample_second(l)}\n") # save in seconds
        lengths.append(len(landmarks[-1]))
    
    return landmarks, lengths

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clustering prominence-segmented speech.")
    parser.add_argument(
        "model",
        help="available models (MFCCs)",
        choices=["mfcc", "hubert_shall", "hubert_fs", "mhubert", "c_hubert", "f_hubert", "w2v2_hf", "wavlm", "wavlm_shall"],
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
        "embeddings_dir",
        metavar="embeddings-dir",
        help="path to the embeddings directory.",
        type=Path,
    )
    parser.add_argument(
        "alignments_dir",
        metavar="alignments-dir",
        help="path to the alignments directory.",
        type=Path,
    )
    parser.add_argument(
        "sample_size",
        metavar="sample-size",
        help="number of embeddings to sample (-1 to sample all available data).",
        type=int,
    )
    parser.add_argument(
        "--wav_format",
        help="extension of the audio waveform files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    parser.add_argument(
        "--align_format",
        help="extension of the alignment files (defaults to .TextGrid).",
        default=".TextGrid",
        type=str,
    )
    parser.add_argument(
        "--load_landmarks",
        help="root landmark directory to load landmarks.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--save_segments",
        help="root directory to save word boundaries.",
        default=None,
        type=Path,
    )
    parser.add_argument(
        "--speaker",
        help="Speaker list if speaker dependent sampling must be done.",
        default=None,
        type=Path,
    )
    parser.add_argument( # optional argument to make the evaluation strict
        '--strict',
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    if args.model in ["mfcc", "melspec"]:
        mfcc = True
        frame_len = 10
    else:
        mfcc = False
        frame_len = 20

    # ~~~~~~~~~~ Sample a audio file and its alignments ~~~~~~~~~~
    data = data_process.Features(wav_dir=args.wav_dir, root_dir=args.embeddings_dir, model_name=args.model, layer=args.layer, data_dir=args.alignments_dir, wav_format=args.wav_format, alignment_format=args.align_format, num_files=args.sample_size, frames_per_ms=frame_len)
    # python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/buckeye_segments/test /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/buckeye_segments/buckeye_test_speakers.list --strict
    # python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/zrc/zrc2017_train_segments/english /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/zrc/zrc2017-train-dataset/index.json --strict
    # python3 eskmeans_dynamic.py hubert_shall 10 /media/hdd/data/librispeech /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments -1  --load_landmarks=/media/hdd/segments/tti_wordseg/librispeech/dev_clean/hubert_shall/10 --strict
    # --load_landmarks=/media/hdd/segments/sylseg/buckeye/test/mfcc OR /media/hdd/segments/sylseg/zrc2017_train_segments/english/mfcc
    # --load_landmarks=/media/hdd/segments/tti_wordseg/buckeye/test/hubert_shall/10 OR /media/hdd/segments/tti_wordseg/zrc2017_train_segments/english/hubert_shall/10
    # --save_segments=/media/hdd/segments/eskmeans/sylseg/buckeye/test OR /media/hdd/segments/eskmeans/sylseg/zrc2017_train_segments/english
    # --save_segments=/media/hdd/segments/eskmeans/tti/buckeye/test OR /media/hdd/segments/eskmeans/tti/zrc2017_train_segments/english

    if args.speaker is not None:
        speakerlist = data.get_speakers(args.speaker)
    else:
        speakerlist = [None]

    num_hit = 0
    num_ref = 0
    num_seg = 0
    random.seed(42)
    np.random.seed(42)

    for speaker in tqdm(speakerlist, desc="Speaker"):
        # ~~~~~~~~~~ Get data for all utterances without saving the embeddings ~~~~~~~~~~~
        lexicon_builder, samples, wavs, landmarks = get_data(data, args, speaker)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~ Cluster segment embeddings ~~~~~~~~~~~~~~~~~~~~~~~~~~
        classes = lexicon_builder.cluster()

        # ~~~~~~~~~~~~~~~~~~~ Save utterance segments and assignments ~~~~~~~~~~~~~~~~~~~~
        seg_list = []
        for i in tqdm(range(lexicon_builder.D), desc='Getting boundary frames and classes'): # for each utterance
            segmentation_frames = landmarks[i]
            seg_list.append(segmentation_frames)
            
            if args.save_segments is not None:
                if len(classes[i]) == 1:
                    class_i = classes[i]
                else:
                    class_i = [x for x in classes[i] if x != -1]
                save_dir = (args.save_segments / os.path.split(wavs[i])[-1]).with_suffix(".list")
                save_dir.parent.mkdir(parents=True, exist_ok=True)
                with open(save_dir, "w") as f:
                    for t, c in zip(segmentation_frames, class_i):
                        f.write(f"{data.get_sample_second(t)} {int(c)}\n") # save in seconds