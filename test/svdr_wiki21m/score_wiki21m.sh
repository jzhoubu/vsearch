TEXT=/export/data/jzhoubu/workspace/vsearch/data/corpus/wiki21m.jsonl
RESULT=/export/data/jzhoubu/workspace/vsearch/data/results/svdr-beta-wiki21m
QA=/export/data/jzhoubu/workspace/vsearch/data/eval/wiki21m/nq-test.qa.csv
python -m inference.score.eval_wiki21m --result_file=$RESULT --text_file=$TEXT --qa_file=$QA
