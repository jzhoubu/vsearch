# DPR (Dense Passage Retrieval) Usage Guide

This guide will walk you through how to utilize the Dense Passage Retrieval (DPR) retriever available in our repository. 

### Loading the DPR Model

Start by loading the DPR model with the `Retriever.from_pretrained` method, and then proceed to embed a query and corresponding passages:

```python
from src.ir import Retriever

# Initialize the DPR retriever
dpr = Retriever.from_pretrained("vsearch/dpr-nq")

# Define your query and passages collection
query = "Who first proposed the theory of relativity?"
passages = [
    "Albert Einstein (14 March 1879 – 18 April 1955) was a German-born theoretical physicist who is widely held to be one of the greatest and most influential scientists of all time. He is best known for developing the theory of relativity.",
    "Sir Isaac Newton FRS (25 December 1642 – 20 March 1727) was an English polymath active as a mathematician, physicist, astronomer, alchemist, theologian, and author who was described in his time as a natural philosopher.",
    "Nikola Tesla (10 July 1856 – 7 January 1943) was a Serbian-American inventor, electrical engineer, mechanical engineer, and futurist. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system."
]

# Embed the query and passages
q_emb = dpr.encoder_q.embed(query)  # Shape: [1, V]
p_embs = dpr.encoder_p.embed(passages)  # Shape: [3, V]

# Calculate relevance scores
relevance_scores = q_emb @ p_embs.t() # Q-P relevance
```



### Retrieve from Large Datastore

The large-scale retrieval inference process generally involves three key steps:

1. Building retrieval index
2. Searching on the index
3. Scoring the search results

### 1. Building a Dense Index

#### Creating the Text File for Indexing

Prepare your text data in a JSONL format:

```python
import json

wiki_passages = [...]  # wiki21m passages here, list of string
with open("data/corpus/wiki21m.jsonl", "w") as file:
    for passage in wiki_passages:
        file.write(json.dumps(passage) + "\n")

nq_test_questions = [...]
with open("data/eval/wiki21m/nq-test-questions.jsonl", "w") as file:
    for query in nq_test_questions:
        file.write(json.dumps(query) + "\n")

```

#### Building the Dense Index

Execute the following command to build the dense index:

```bash
python -m inference.build_index.dense_index \
        --checkpoint=vsearch/dpr-nq \
        --text_file=data/corpus/wiki21m.jsonl \
        --save_file=experiment/inference_dense_example/index/index.pt \
        --batch_size=64 \
        --device=cuda
```
- `--checkpoint`: Path the retriever checkpoint dir.
- `--text_file`: Path to the corpus file to be indexed (`.jsonl` format).
- `--save_file`: Path where the index file will be saved (`.pt` format, which consists of a large `torch.Tensor`).
- `--batch_size`: Batch size for processing.

This will generate a single index file as `experiment/inference_dense_example/index/index.pt`, which contains a PyTorch tensor with dimensions [num_text, dim]


### Sharding and Parallel Index Building [Optional]

For larger corpora, shard the indexing process to parallelize and accelerate it:

```bash
NUM_SHARDS=4
for SHARD_ID in $(seq 0 $(($NUM_SHARDS - 1)))
do
python -m inference.build_index.dense_index \
        --checkpoint=vsearch/dpr-nq \
        --text_file=data/corpus/wiki21m.jsonl \
        --save_file=experiment/inference_dense_example/index/index${SHARD_ID}.pt \
        --batch_size=64 \
        --num_shard=$NUM_SHARDS \
        --shard_id=$SHARD_ID \
        --device=cuda:$SHARD_ID &
done
```
- `--num_shard`: total number of shards
- `--shard_id`: shard id (start from 0 to `num_shard`-1) for the current indexing job


This will ganerate separate index files (e.g., `index0.pt`, `index1.pt`, `index2.pt`, `index3.pt`) under dir `experiment/inference_dense_example/index`.

### 2. Searching the Dense Index

Once the index is built, perform searches to find relevant passages based on queries:

```bash
python -m inference.search.search_dense_index \
        --checkpoint=vsearch/dpr-nq \
        --query_file=data/eval/wiki21m/nq-test-questions.jsonl \
        --index_file=experiment/inference_dense_example/index/index*.pt \
        --save_file=experiment/inference_dense_example/results/search_result.json  \
        --batch_size_q=32 \
        --device=cuda
```
- `--query_file`: Path to file containing questions, with each question as a separate line (`.jsonl` format). 
- `--index_file`: Path to pre-computed index files (`.pt` format, supports glob patterns).
- `--save_file`: Path where the search results will be stored (`.json` format).
- `--batch_size`: Number of queries per batch.


### 3. Scoring the Search Result

Evaluate and score the search results for wiki21m benchmark:

```bash
python -m inference.score.eval_wiki21m \
    --result_file=experiment/inference_dense_example/results/search_result.json \
    --text_file=data/corpus/wiki21m.jsonl \
    --qa_file=data/eval/wiki21m/nq-test.qa.csv \
```
- `--result_file`: Path to search results (`.json` format).
- `--qa_file`: Path to DPR-provided qa file (`.csv` format, provided by DPR repo)


The retrieval accuracy of the `vsearch/dpr-nq` checkpoint on NQ test set are shown below:

| Checkpoint | Top-1 | Top-5 | Top-10 | Top-20 | Top-100 |
|:----------:|:-----:|:-----:|:------:|:------:|:-------:|
| `dpr-nq`   | 43.49 | 67.84 |  74.96 | 80.14  | 86.48   |


