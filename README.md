# Prominence Segmentation with Clustering for Word Discovery and Lexicon Learning

This repository contains code that builds a lexicon on prominence-based word segments. The lexicon is built by clustering word segment embeddings found through manipulated input speech features. We use [FAISS](https://github.com/facebookresearch/faiss) K-means clustering, each word segment is assigned to the class centroid closest to its segment embedding. 

## Preliminaries

**Pre-Process and Encode Speech Data**
Use VAD to extract utterances from long speech files (specifically for ZeroSpeech and BuckEye) by cloning and following the recipes in the repository at [https://github.com/s-malan/data-process](https://github.com/s-malan/data-process).
Use this repository and follow the recipe to encode speech audio by extracting features from various models (and their layers, where applicable).

**Extract Word Boundaries**
Clone and follow the recipe of the unsupervised prominence-based word segmentation repository [https://github.com/s-malan/prom-word-seg](https://github.com/s-malan/prom-word-seg).

**ZeroSpeech Repository**
Clone the ZeroSpeech repository at [https://github.com/zerospeech/benchmarks](https://github.com/zerospeech/benchmarks) to use the ZeroSpeech toolkit used for benchmark resources and evaluation scripts.

## Example Usage

The following script runs the entire feature manipulation, clustering, and assignment pipeline:

    python3 prom_seg_clus.py model_name layer path/to/audio path/to/audio/features path/to/boundaries path/to/output --extension --sample_size --speaker

The naming and format conventions used requires the preliminary scripts to be followed. 
The **layer** parameter selects a specific model layer, use -1 if the model has no layers (such as MFCC features).
The **extension** parameter specifies the format of the audio files (.wav or the default .flac). 
The **sample_size** parameter controls how many utterances are sampled, the default is -1 to sample all utterances. 
The **speaker** parameter let's you supply a .json file with speaker names whereafter the script will run speaker specific clustering on all the provided speakers, the default of None selects all speakers in a speaker-independent setting.
## Datasets

- [ZeroSpeech](https://download.zerospeech.com/) Challenge Corpus (Track 2).
- [LibriSpeech](https://www.openslr.org/12) Corpus (dev-clean split) with alignments found [here](https://zenodo.org/records/2619474).
- [Buckeye](https://buckeyecorpus.osu.edu/) Corpus with splits found [here](https://github.com/kamperh/vqwordseg?tab=readme-ov-file#about-the-buckeye-data-splits) and alignments found [here](https://github.com/kamperh/vqwordseg/releases/tag/v1.0).

## Evaluation 

The ZeroSpeech toolkit has built in evaluation scripts used for all of my results ran on this data.