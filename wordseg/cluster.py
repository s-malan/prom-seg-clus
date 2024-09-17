"""
Main functions and class the lexicon building step built on word-like segments.

Author: Simon Malan
Contact: 24227013@sun.ac.za
Date: April 2024
"""

import random
import numpy as np
from tqdm import tqdm
import faiss

class CLUSTER():
    """
    Clustering for prominence-detected word-like segments.

    Clustering of all segments of all utterances is carried out in this class.
    The features of each segment is reduced to a lower dimension using PCA.
    Embeddings using a simple average and L2 normalization are found for each segment.
    The embeddings are clustered using K-means clustering.

    Parameters
    ----------
    data : Features
        The data object, containing the features and the alignments.
    samples : list
        The list of the sampled file paths to the utterances.
    segments : list of lists (of tuples)
        All possible segments (start, end) for each utterance.
    lengths : list
        The length of each utterance's possible segments.
    K_max : int
        Maximum number of K-means components.
    pca : PCA
        The PCA object to use for dimensionality reduction.
        default: None

    Attributes
    ----------
    D : int
        The number of utterances in the dataset.
    """

    def __init__(self, data, samples, segments, lengths, K_max, pca=None):
        # Attributes from parameters
        self.data = data
        self.samples = samples
        self.segments = segments
        self.lengths = lengths
        self.K_max = K_max
        self.pca = pca

        self.D = len(self.lengths) # number of utterances

    def assign_utt_i(self, i):
        """
        Assign each word segment of utterance `i` to the closest K-means cluster centroid.

        Parameters
        ----------
        i : int
            The index of the utterance used in the cluster assignment.

        Return
        ------
        classes : list (int)
            The classes assigned to each segment of utterance `i`.
        """

        # sample the current utterance and downsample all possible segmentations
        embeddings = self.data.load_embeddings([self.samples[i]])[0]

        if self.pca is not None:
            embeddings = self.pca.transform(embeddings)
            
            # Use dense downsample method:
            # embeddings = downsample.downsample([embeddings], [self.segments[i]], n=10)

            # Use mean pooling as downsampling method:
            # embeddings = np.stack([embeddings[a:b, :].mean(0) for a, b in self.segments[i]])
            downsampled_embeddings = []
            for a, b in self.segments[i]:
                if b > embeddings.shape[0]: # if the segment is longer than the utterance, stop at last frame
                    if a < embeddings.shape[0]:
                        downsampled_embeddings.append(embeddings[a:, :].mean(0))
                    elif a == embeddings.shape[0]:
                        downsampled_embeddings.append(embeddings[a-1:, :].mean(0))
                elif a == b: # if the segment is empty, add one frame
                    downsampled_embeddings.append(embeddings[a-1:b+1, :].mean(0))
                else:
                    downsampled_embeddings.append(embeddings[a:b, :].mean(0))

            embeddings = np.stack(downsampled_embeddings)
            del downsampled_embeddings
        # else:
            # embeddings = downsample.downsample([embeddings], [self.segments[i]], n=10)

        # normalize embedding
        downsampled_utterance = []
        for frame_i in range(embeddings.shape[0]): # for each frame we normalize over the 130 dimensions
            cur_embed = embeddings[frame_i, :]
            norm = np.linalg.norm(cur_embed)
            downsampled_utterance.append(cur_embed / np.linalg.norm(cur_embed))
            assert norm != 0.
        embeddings = np.stack(downsampled_utterance)

        # downsampled_utterance = [] # MNV per utt
        # for frame_i in range(embeddings.shape[0]):
        #     mean = np.mean(embeddings[frame_i, :], axis=0)
        #     std = np.std(embeddings[frame_i, :], axis=0)
        #     downsampled_utterance.append(embeddings[frame_i, :] - mean / std)
        # embeddings = np.stack(downsampled_utterance)

        classes = [-1]*len(self.segments[i])
        for i_index in range(len(classes)):
            # find negative L2 distance to closest centroid
            _, index = self.acoustic_model.index.search(embeddings[i_index].reshape(1, embeddings.shape[-1]), 1)
            classes[i_index] = index[0][0]

        return classes

    def cluster(self):
        """
        Create and cluster feature embeddings for all word segments of all utterances.

        Return
        ------
        classes : list (list (int))
            The classes assigned to each segment of each utterance.
        """

        # Cluster:
        embeddings = []
        # embedding_shapes = []
        for i_utt, (sample, segment) in enumerate(zip(self.samples, self.segments)): # for each utterance sample file path
            if self.pca is not None:
                embedding = self.pca.transform(self.data.load_embeddings([sample])[0])
            else:
                embedding = self.data.load_embeddings([sample])[0]

            # Use dense downsample method:
            # embeddings.append(downsample.downsample([embedding], [segment], n=10)) # TODO use downsample_utterance if only downsample one utterance
            
            # Use mean pooling as downsampling method:
            # embeddings.append(np.stack([embedding[a:b, :].mean(0) for a, b in segment]))
            downsampled_embedding = []
            for a, b in segment:
                if b > embedding.shape[0]: # if the segment is longer than the utterance, stop at last frame
                    if a < embedding.shape[0]:
                        downsampled_embedding.append(embedding[a:, :].mean(0))
                    elif a == embedding.shape[0]:
                        downsampled_embedding.append(embedding[a-1:, :].mean(0))
                elif a == b: # if the segment is empty, add one frame
                    downsampled_embedding.append(embedding[a-1:b+1, :].mean(0))
                else:
                    downsampled_embedding.append(embedding[a:b, :].mean(0))
            embeddings.append(np.stack(downsampled_embedding))

        print("Normalizing embeddings")
        for i_utt in range(len(embeddings)):
            downsampled_utterance = []
            for i in range(embeddings[i_utt].shape[0]): # for each frame we normalize over the 130 dimensions
                cur_embed = embeddings[i_utt][i, :]
                norm = np.linalg.norm(cur_embed)
                downsampled_utterance.append(cur_embed / np.linalg.norm(cur_embed))
                assert norm != 0.
            embeddings[i_utt] = np.stack(downsampled_utterance)

        # for i_utt in range(len(embeddings)): # MNV per utt
        #     downsampled_utterance = []
        #     for frame_i in range(embeddings[i_utt].shape[0]):
        #         mean = np.mean(embeddings[i_utt][frame_i, :], axis=0)
        #         std = np.std(embeddings[i_utt][frame_i, :], axis=0)
        #         downsampled_utterance.append(embeddings[i_utt][frame_i, :] - mean / std)
        #     embeddings[i_utt] = np.stack(downsampled_utterance)
        
        embeddings = np.concatenate(embeddings, axis=0)
        print('Clustering sizes', self.K_max, embeddings.shape)
        self.acoustic_model = faiss.Kmeans(embeddings.shape[1], self.K_max, niter=15, nredo=3, verbose=True)
        self.acoustic_model.train(embeddings)

        # Segment:
        print("Assigning clusters...")
        classes = [None]*self.D
        utt_order = list(range(self.D))
        random.shuffle(utt_order)

        for i_utt in tqdm(utt_order, desc="Cluster Assignments"): # TODO do in parallel
            i_classes = self.assign_utt_i(i_utt)
            classes[i_utt] = i_classes
            
        return classes