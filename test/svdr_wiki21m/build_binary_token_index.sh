TEXT=/export/data/jzhoubu/workspace/vsearch/data/corpus/wiki21m.jsonl
SAVE=/export/data/jzhoubu/workspace/vsearch/data/index/wiki21m_bow.npz
python -m inference.build_index.binary_token_index --text_file=$TEXT --save_file=$SAVE --batch_size=32 --num_shift=999


# Expect output, validated on May 7th on lccpu28
"""
100%|█████████████████████████████████████████████████████████████████████████████████████| 656729/656729 [1:22:35<00:00, 132.52it/s]
INFO:root:***** Finish Indexing *****
INFO:root:***** Time for indexing (exclude i/o): 1756 s *****
INFO:root:***** Time for indexing (include i/o): 4987 s *****

INFO:root:***** Index save to: /export/data/jzhoubu/workspace/vsearch/data/index/wiki21m_bow.npz *****
INFO:root:***** Index matrix shape: (21015324, 29523) *****
INFO:root:***** Index sparsity rate: 0.29% *****
"""