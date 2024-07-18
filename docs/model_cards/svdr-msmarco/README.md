# Model Card: svdr-msmarco

Supervised sparse lexical retriever (SVDR architecture) trained on MS MARCO training split.


### To reproduce the training
Below code produce our checkpoint in `vsearch/svdr-msmarco`
```bash
python -m torch.distributed.launch \
--nnodes=1 \
--nproc_per_node=4 \
train_retriever.py \
hydra.run.dir=./experiments/svdr-msmarco/train \
data_stores=msmarco \
train_datasets=[msmarco_train] dev_datasets=[] \
biencoder=vdr \
train.train_insert_title=False \
train.num_train_epochs=20 \
train.hard_negatives=1 \
train.num_warmup_epochs=1 \
train.batch_size=64
```

### Performance

NDCG@10 on BEIR and MRR@10 on MS MARCO dev split.

|       Dataset       | beta search with rerank=100 | parametric search |
|:-------------------:|:---------------------------:|:-----------------:|
| ArguAna             | 51.5                        | 52.4              |
| climate-fever       | 16.7                        | 17.3              |
| DBPedia             | 38.6                        | 40.3              |
| FEVER               | 69.7                        | 73.0              |
| FiQA                | 30.3                        | 31.3              |
| HotpotQA            | 63.2                        | 66.2              |
| NFCorpus            | 33.2                        | 33.3              |
| NQ                  | 47.4                        | 51.9              |
| SCIDOCS             | 14.4                        | 14.6              |
| SciFact             | 66.9                        | 67.1              |
| TREC-COVID          | 65.6                        | 67.2              |
| Touche-2020         | 28.3                        | 28.3              |
| **Avg**             | **43.8**                    | **45.3**          |
| MSMARCO-dev (MRR@10)| 34.2                        | 36.0              |