# DPR (Dense Passage Retrieval) Usage Guide

This guide will walk you through how to utilize the Dense Passage Retrieval (DPR) retriever available in our repository. 

### Loading the DPR Model

Start by loading the DPR model with the Retriever.from_pretrained method, and then proceed to embed a query and corresponding passages:

```python
from src.vdr import Retriever

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

For efficient retrieval from a large corpus, you need to build a dense index. Below are the steps to create and save a dense index of your text data.

#### Creating the Text File for Indexing

Prepare your text data in a JSONL format:

```python
import json

passages = [...]  # Your passages here
with open("path/to/your/text_file.jsonl", "w") as file:
    for passage in passages:
        file.write(json.dumps({"text": passage}) + "\n")
```

#### Building the Dense Index

Execute the following command to build the dense index:

```bash
python -m inference.build_index.dense_index \
        --checkpoint=vsearch/dpr-nq \
        --text_file=path/to/your/text_file.jsonl \
        --save_dir=path/to/save_dir/ \
        --batch_size=64 \
        --device=cuda
```
This will generate one index `path/to/save_dir/shard0.pt`. 

Parameters:
- `--checkpoint`: Path the retriever checkpoint dir.
- `--text_file`: Path to the corpus file to be indexed (`.jsonl` format).
- `--save_file`: Path where the index file will be saved (`.npz` format).
- `--batch_size`: Batch size for processing.


#### Sharding and Parallel Index Building [Optional]

For larger corpora, shard the indexing process to parallelize and accelerate it:

```bash
NUM_SHARDS=4
for SHARD_ID in {0..3}
do
python -m inference.build_index.dense_index \
        --checkpoint=vsearch/dpr-nq \
        --text_file=path/to/your/text_file.jsonl \
        --save_dir=path/to/save_dir \
        --batch_size=64 \
        --num_shard=$NUM_SHARDS \
        --shard_id=$SHARD_ID \
        --device=cuda:$SHARD_ID &
done
```
Parameters:
- `--num_shard`: total number of shards
- `--shard_id`: shard id (start from 0 to `num_shard`-1) for the current indexing job

This command creates multiple index files under `path/to/save_dir/shard*.pt`.


#### 2. Searching the Dense Index

Once the index is built, perform searches to find relevant passages based on queries:


```bash
python -m inference.search.search_dense_index \
        --checkpoint=vsearch/dpr-nq \
        --query_file=path/to/your/query_file.jsonl \
        --index_file=path/to/save_dir/*.pt \
        --save_file=path/to/result_file.json  \
        --batch_size_q=32 \
        --device=cuda
```
Parameters:
- `--query_file`: Path to file containing questions, with each question as a separate line (`.jsonl` format). 
- `--index_file`: Path to pre-computed index files (`.pt` format, supports glob patterns).
- `--save_file`: Path where the search results will be stored (`.json` format).
- `--batch_size`: Number of queries per batch.


#### 3. Scoring the Search Result

Evaluate and score the search results for wiki21m benchmark:

```bash
python -m inference.score.eval_wiki21m \
    --result_file=path/to/result_file.json \
    --text_file=path/to/your/text_file.jsonl \
    --qa_file=path/to/dpr/qa_file.csv
```

Parameters:
- `--result_file`: Path to search results (`.json` format).
- `--qa_file`: Path to DPR-provided qa file (`.csv` format)

The result presents top-k retrieval accuracy on NQ, triviaQA, or webQA dataset.


The following table displays the retrieval accuracy of the `vsearch/dpr-nq` checkpoint on NQ test set:

| Checkpoint | Top-1 | Top-5 | Top-20 |
|:----------:|:-----:|:-----:|:------:|
| `dpr-nq`   | 43.49 | 67.84 |  80.14 |


