# prom-seg-clus
Prominence Segmentation with Clustering for Word Discovery and Lexicon Learning

# python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/buckeye_segments/test /media/hdd/embeddings/buckeye/test /media/hdd/data/buckeye_alignments/test -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/buckeye_segments/buckeye_test_speakers.list --strict
# python3 eskmeans_dynamic.py mfcc -1 /media/hdd/data/zrc/zrc2017_train_segments/english /media/hdd/embeddings/zrc/zrc2017_train_segments/english /media/hdd/data/zrc_alignments/zrc2017_train_alignments/english -1 --wav_format=.wav --align_format=.txt --load_landmarks= --save_segments= --speaker=/media/hdd/data/zrc/zrc2017-train-dataset/index.json --strict
# python3 eskmeans_dynamic.py hubert_shall 10 /media/hdd/data/librispeech /media/hdd/embeddings/librispeech /media/hdd/data/librispeech_alignments -1  --load_landmarks=/media/hdd/segments/tti_wordseg/librispeech/dev_clean/hubert_shall/10 --strict
# --load_landmarks=/media/hdd/segments/sylseg/buckeye/test/mfcc OR /media/hdd/segments/sylseg/zrc2017_train_segments/english/mfcc
# --load_landmarks=/media/hdd/segments/tti_wordseg/buckeye/test/hubert_shall/10 OR /media/hdd/segments/tti_wordseg/zrc2017_train_segments/english/hubert_shall/10
# --save_segments=/media/hdd/segments/eskmeans/sylseg/buckeye/test OR /media/hdd/segments/eskmeans/sylseg/zrc2017_train_segments/english
# --save_segments=/media/hdd/segments/eskmeans/tti/buckeye/test OR /media/hdd/segments/eskmeans/tti/zrc2017_train_segments/english