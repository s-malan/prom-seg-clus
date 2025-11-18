# Prominence Segmentation with Clustering for Word Discovery and Lexicon Learning

[![IEEE](https://img.shields.io/badge/IEEE-Paper-<COLOR>.svg)](https://ieeexplore.ieee.org/document/10890719)

This repository contains code that builds a lexicon on prominence-based word segments. The lexicon is built by clustering word segment embeddings found through manipulated input speech features. We use [FAISS](https://github.com/facebookresearch/faiss) K-means clustering, each word segment is assigned to the class centroid closest to its segment embedding. 

## Preliminaries

**Datasets**

- [ZeroSpeech](https://download.zerospeech.com/) Challenge Corpus (Track 2).
- [LibriSpeech](https://www.openslr.org/12) Corpus (dev-clean split) with alignments found [here](https://zenodo.org/records/2619474).
- [Buckeye](https://buckeyecorpus.osu.edu/) Corpus with splits found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits) and alignments found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0).

**Pre-Process and Encode Speech Data**

Use VAD to extract utterances from long speech files (specifically for ZeroSpeech and BuckEye) by cloning and following the recipes in the repository at [https://github.com/s-malan/data-process](https://github.com/s-malan/data-process).

**Encode Utterances**

Use pre-trained speech models or signal processing methods to encode speech utterances. Example code can be found here [https://github.com/bshall/hubert/blob/main/encode.py](https://github.com/bshall/hubert/blob/main/encode.py) using HuBERT-base for self-supervised audio encoding.
Save the feature encodings as .npy files with the file path as: 

    model_name/layer_#/relative/path/to/input/audio.npy

where # is replaced with an integer of the self-supervised model layer used for encoding, and as:

    model_name/relative/path/to/input/audio.npy

when signal processing methods like MFCCs are used.

**Extract Word Boundaries**

Clone and follow the recipe of the unsupervised prominence-based word segmentation repository [https://github.com/s-malan/prom-word-seg](https://github.com/s-malan/prom-word-seg).

**ZeroSpeech Repository**

Clone the ZeroSpeech repository at [https://github.com/zerospeech/benchmarks](https://github.com/zerospeech/benchmarks) to use the ZeroSpeech toolkit used for benchmark resources and evaluation scripts.

## Example Usage

**Feature Manipulation and Clustering**

    python3 prom_seg_clus.py model_name layer path/to/audio path/to/features path/to/boundaries path/to/output k_max --extension --sample_size --speaker

The naming and format conventions used requires the preliminary scripts to be followed. 
The `layer` argument selects a specific model layer, use `-1` if the model has no layers (such as MFCC features).
The `k_max` argument specifies the number of K-means clusters used.
The `extension` argument specifies the format of the audio files (.wav or the default `.flac`). 
The `sample_size` argument controls how many utterances are sampled, the default is -1 to sample all utterances. 
The `speaker` argument let's you supply a `.json` file with speaker names whereafter the script will run speaker specific clustering on all the provided speakers, the default of `None` selects all speakers in a speaker-independent setting.

## Evaluation

Note: all segmentation, alignment, and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Boundary Evaluation

Script name: `evaluation/boundary_eval.py`

This script evaluates boundary metrics namely: Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, and R-Value.

**Example Usage**

    python3 boundary_eval.py path/to/segment/files path/to/alignment/files --alignment_format=.TextGrid --alignment_type=words --ms_per_frame=20 --tolerance=1 --strict=True

The `alignment_format` argument specifies the extension of the alignment files (options: `.TextGrid`, or `.txt`).
The `frames_per_ms` argument specifies the number of milliseconds contained in one frame of encoded audio.
The `tolerance` argument specifies the number of frames (to both sides) that the hypothesized boundary can be from the ground truth boundary to still count.
The `strict` argument determines if the boundary hit count is strict or lenient as described by D. Harwath in the following paper [https://ieeexplore.ieee.org/abstract/document/10022827](https://ieeexplore.ieee.org/abstract/document/10022827), default `True`.

The input file format is a `.list` file containing offset boundaries (and optional space-separated class assignment values) with each new value (or pair of values) on a new line.

### Cluster Evaluation

Python script name: `evaluation/clust_eval.py`

This script evaluates clustering metrics namely: NED, coverage, type scores, and cluster purity.

**Example Usage**

    python3 clust_eval.py path/to/segment/files path/to/alignment/files --alignment_format=.TextGrid

with the same argument definitions as above.

The input file format is a .list file containing boundaries and a  space-separated class assignment value with each new pair of values on a new line.

## Contributors

- [Simon Malan](https://scholar.google.com/citations?user=rxKKwFAAAAAJ&hl=en)
- [Benjamin van Niekerk](https://scholar.google.com/citations?user=zCokvy8AAAAJ&hl=en&oi=ao)
- [Herman Kamper](https://www.kamperh.com/)
