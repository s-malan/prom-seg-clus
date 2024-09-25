# Prominence Segmentation with Clustering for Word Discovery and Lexicon Learning

[![arXiv](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://arxiv.org/abs/2409.14486)

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
The **layer** argument selects a specific model layer, use -1 if the model has no layers (such as MFCC features).
The **k_max** argument specifies the number of K-means clusters used.
The **extension** argument specifies the format of the audio files (.wav or the default .flac). 
The **sample_size** argument controls how many utterances are sampled, the default is -1 to sample all utterances. 
The **speaker** argument let's you supply a .json file with speaker names whereafter the script will run speaker specific clustering on all the provided speakers, the default of None selects all speakers in a speaker-independent setting.

**Evaluation**

To evaluate the resultant hypothesized word boundaries and cluster assignments, clone and follow the recipe of the evaluation repository at [https://github.com/s-malan/evaluation](https://github.com/s-malan/evaluation).

For the ZeroSpeech Challenge dataset, use the ZeroSpeech toolkit's built in evaluation script.