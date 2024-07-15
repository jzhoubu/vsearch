# Model Card: svdr-hlp

Unsupervised sparse lexical retriever (SVDR architecture) pre-trained on 20 million Wikipedia hyperlink-induced pseudo question-passage pairs (HLP dataset [1]).

[1] Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering

### To reproduce the training
Below code produce our checkpoint in `vsearch/svdr-hlp`
```bash
python -m torch.distributed.launch \
--nnodes=1 \
--nproc_per_node=4 \
train_retriever.py \
hydra.run.dir=./experiments/svdr-hlp/train \
data_stores=ir \
train_datasets=[dl,cm] dev_datasets=[] \
biencoder=vdr \
biencoder.shared_encoder=True \
train=svdr_hlp \
train.other_negatives=1 \
train.batch_size=128
```

### Performance on NQ-test

We tested various configurations, such as the use of additional negative (neg=1) and whether to employ a shared encoder (shared). The results showed that models with one negative and a shared encoder perform better than those without. These findings also align with results of `dpr-hlp`. 

Therefore, we've chosen to use one negative and a shared encoder for the final configuration of `svdr-hlp`.

**Beta Search with Re-ranking**

| Model       |    Search Mode     | shared | neg |  Top-1  |  Top-5  | Top-10 | Top-20 | Top-100 |
|-------------|:------------------:|:------:|:---:|:-------:|:-------:|:------:|:------:|:-------:|
| svdr-hlp (ablation) | beta (rerank=100)  | False  |  0  | 22.77%  | 48.78%  | 59.70% | 68.86% |  78.89% |
| svdr-hlp (ablation)| beta (rerank=100)  | False  |  1  | 25.43%  | 51.05%  | 60.75% | 69.28% |  78.81% |
| `vsearch/svdr-hlp`  | beta (rerank=100)  | True   |  1  | 26.12%  | 52.63%  | 63.13% | 70.80%  | 79.78% |


**Parametric Search**

| Model       |  Search Mode | shared | neg |  Top-1  |  Top-5  | Top-10 | Top-20 | Top-100 |
|-------------|:------------:|:------:|:---:|:-------:|:-------:|:------:|:------:|:-------:|
| svdr-hlp (ablation)|  parametric  | False  |  0  | 20.64%  | 46.54%  | 58.50% | 67.67% |  82.41% |
| svdr-hlp (ablation)|  parametric  | False  |  1  | 23.68% | 49.61% | 60.69% | 69.31% |  82.63% | 
| `vsearch/svdr-hlp`  |  parametric  | True   |  1  | 24.96% | 51.63% | 62.96% | 71.47% | 83.66% |

