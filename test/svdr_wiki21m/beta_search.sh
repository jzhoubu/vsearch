TEXT=/export/data/jzhoubu/workspace/vsearch/data/corpus/wiki21m.jsonl
INDEX=/export/data/jzhoubu/workspace/vsearch/data/index/wiki21m_bow.npz
SAVE=/export/data/jzhoubu/workspace/vsearch/data/results/svdr-beta-wiki21m

python -m inference.search.beta_search --checkpoint=vsearch/vdr-nq \
 --query_file=/export/data/jzhoubu/workspace/vsearch/data/eval/wiki21m/nq-test-questions.jsonl \
 --text_file=${TEXT} \
 --index_file=${INDEX} \
 --save_file=${SAVE} \
 --device=cuda:0