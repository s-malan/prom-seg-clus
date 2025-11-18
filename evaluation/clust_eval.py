"""
How ZeroSpeech does clustering evaluation.

Author: Benjamin van Niekerk, Simon Malan
Contact: benjamin.l.van.niekerk@gmail.com, 24227013@sun.ac.za
Date: June 2024
"""

import argparse
from pathlib import Path

import dataclasses
import re
import itertools
from typing import Iterable, List, Tuple
import statistics
from sklearn.metrics.cluster import contingency_matrix
import numpy as np

import editdistance
from intervaltree import IntervalTree, Interval
from textgrid import TextGrid, IntervalTier

@dataclasses.dataclass(frozen=True)
class Fragment:
    speaker: str
    interval: Interval


@dataclasses.dataclass(frozen=True)
class Transcription:
    intervals: List[Interval]

    @property
    def tokens(self) -> Tuple[str, ...]:
        return tuple(
            interval.data
            for interval in self.intervals
            # if interval.data != "sil" and interval.data != "spn" # This is the same as ZRC, but they don't have <unk> and "sp"
            if interval.data.lower() not in ["sil","spn","sp","<unk>",""] # Librispeech equivalent of above
        )

    @property
    def bounds(self) -> Interval:
        return Interval(self.intervals[0].begin, self.intervals[-1].end)


def distance(p: Tuple[str, ...], q: Tuple[str, ...]) -> float:
    length = max(len(p), len(q))
    return editdistance.eval(p, q) / length if length > 0 else 1


def ned(discovered: Iterable[Tuple[Fragment, int, Transcription]]) -> float:
    discovered = sorted(discovered, key=lambda x: x[1])
    distances = [
        distance(p[2].tokens, q[2].tokens)
        for _, group in itertools.groupby(discovered, key=lambda x: x[1])
        for p, q in itertools.combinations(group, 2)
    ]

    return statistics.mean(distances)


def purity(C, reverse=False):
    N = np.sum(C)
    axis = 1 if not reverse else 0
    count = np.sum([np.max(C[:,k]) if not reverse else np.max(C[k,:]) for k in range(C.shape[axis])])
    return count / N if N > 0 else 0.0


def coverage(
    disc: Iterable[Tuple[Fragment, Transcription]],
    gold: Iterable[Transcription],
):
    covered = {
        (fragment.speaker, interval.begin, interval.end, interval.data)
        for fragment, transcription in disc
        for interval in transcription.intervals
        if interval.data != "sil" and interval.data != "spn"
    }
    total = [
        interval.data
        for transcription in gold
        for interval in transcription.intervals
        if interval.data != "sil" and interval.data != "spn"
    ]
    return len(covered) / len(total)


def types(
    gold: Iterable[Transcription],
    disc: Iterable[Transcription],
) -> Tuple[float, float, float]:
    gold_types = {transcription.tokens for transcription in gold}
    disc_types = {transcription.tokens for transcription in disc}
    intersection = gold_types & disc_types
    precision = len(intersection) / len(disc_types)
    recall = len(intersection) / len(gold_types)
    fscore = 2 * (precision * recall) / (precision + recall)
    return precision, recall, fscore


def check_boundary(gold: Interval, disc: Interval) -> bool:
    if gold.contains_interval(disc):
        return True

    gold_duration = round(gold.end - gold.begin, 2)
    overlap_duration = round(gold.overlap_size(disc), 2)
    overlap_percentage = overlap_duration / gold_duration
    duration_condition = gold_duration >= 0.06 and overlap_duration >= 0.03
    percentage_condition = gold_duration < 0.06 and overlap_percentage > 0.5
    return duration_condition or percentage_condition


def treeify(grid: TextGrid, tier=1, sub = "\d") -> IntervalTree:
    intervals = [
        (interval.minTime, interval.maxTime, re.sub(sub, "", interval.mark))
        for interval in grid.tiers[tier]
    ]
    return IntervalTree.from_tuples(intervals)


def words(grid: TextGrid, tree: IntervalTree) -> List[Transcription]:
    overlaps = [
        tree.overlap(interval.minTime, interval.maxTime)
        for interval in grid.tiers[0]
        if interval.mark != "<eps>"
    ]
    overlaps = [
        sorted(intervals, key=lambda x: x.begin)
        for intervals in overlaps
        if all(interval.data != "spn" for interval in intervals)
    ]
    overlaps = [Transcription(intervals) for intervals in overlaps]
    return overlaps


def transcribe(fragment: Fragment, tree: IntervalTree, words: bool = False) -> Transcription:
    transcription = sorted(tree.overlap(fragment.interval), key=lambda x: x.begin)
    if words:
        if len(transcription) == 0: return Transcription([])
        if len(transcription) == 1: return Transcription(transcription)
        # Return the maximally overlapping word
        overlaps = [fragment.interval.overlap_size(interval) for interval in transcription]
        transcription = [transcription[overlaps.index(max(overlaps))]]
        return Transcription(transcription)
    
    transcription = [
        interval
        for interval in transcription
        if check_boundary(interval, fragment.interval)
    ]
    return Transcription(transcription)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=".")
    parser.add_argument(
        "disc_path",
        metavar="disc-path",
        help="path to the discovered fragments.",
        type=Path,
    )
    parser.add_argument(
        "gold_dir",
        metavar="gold-dir",
        help="path to the directory of alignments.",
        type=Path,
    )
    parser.add_argument(
        "--alignment_format",
        metavar="--alignment-format",
        help="extension of the alignment files.",
        default=".TextGrid",
        type=str,
    )
    args = parser.parse_args()

    files = args.disc_path.rglob("**/*" + ".list")
    fragments = []
    for file in files:
        with open(file, "r") as f:
            start_time = 0.0
            for line in f:
                if len(line.split(" ")) == 2: # end_time class
                    end_time, cluster = line.split(" ")
                    speaker = file.stem
                    fragments.append((speaker, Interval(float(start_time), float(end_time)), int(cluster)))
                    start_time = float(end_time)

    disc_fragments = [
        Fragment(speaker, interval) for speaker, interval, _ in fragments
    ]
    disc_clusters = [cluster for _, _, cluster in fragments]

    print("Number of clusters:", len(set(disc_clusters)))

    grids = {}
    files = args.gold_dir.rglob("**/*" + args.alignment_format)
    sub = "" if "mandarin" in str(args.gold_dir).lower() else "\d" # Add stress factors for Mandarin (same as ZRC)
    for file in files: # alignment files
        if args.alignment_format == '.TextGrid':
            grids[file.stem] = TextGrid.fromFile(file)
        elif args.alignment_format == '.txt': # ZRC format
            tier = 0
            with open(file, 'r') as f:
                if file.stem not in grids: # Create a new TextGrid for each file
                    grids[file.stem] = TextGrid()
                
                if "phone" in str(file):
                    interval_tier = IntervalTier(name='phones')
                elif "word" in str(file):
                    interval_tier = IntervalTier(name='words')
                for line in f:
                    line = line.split()
                    interval_tier.add(float(line[0]), float(line[1]), line[2])
                grids[file.stem].append(interval_tier)
                
    trees = {speaker: treeify(grid, tier=1, sub=sub) for speaker, grid in grids.items()}
    word_trees = {speaker: treeify(grid, tier=0, sub=sub) for speaker, grid in grids.items()}

    disc_transcriptions = [
        transcribe(fragment, trees[fragment.speaker]) for fragment in disc_fragments
    ]

    word_transcriptions = [
        transcribe(fragment, word_trees[fragment.speaker], words=True) for fragment in disc_fragments
    ] # The word that mostly overlaps with discovered fragments
    all_words = [" ".join(list(item.tokens)) for item in word_transcriptions]

    gold_words = {speaker: words(grids[speaker], trees[speaker]) for speaker in grids.keys()}
    gold_fragments = [
        Fragment(speaker, word.bounds)
        for speaker, words in gold_words.items()
        for word in words
    ]
    gold_transcriptions = [word for words in gold_words.values() for word in words]

    C_words = contingency_matrix(all_words, disc_clusters)

    print('NED', ned(zip(disc_fragments, disc_clusters, disc_transcriptions)))
    print('Coverage', coverage(zip(disc_fragments, disc_transcriptions), gold_transcriptions))
    print('Types', types(gold_transcriptions, disc_transcriptions))
    print("Word Purity", purity(C_words))