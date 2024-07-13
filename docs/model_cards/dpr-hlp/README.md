# Model Card for `vsearch/dpr-hlp`

### Model Description

Unsupervised dense retriever (DPR architecture) pre-trained on 20 million Wikipedia hyperlink-induced pseudo question-passage pairs (HLP dataset [1]).

[1] Hyperlink-induced Pre-training for Passage Retrieval in Open-domain Question Answering

# Usage
```python
import torch
from src.ir import Retriever
dpr_hlp = Retriever.from_pretrained("vsearch/dpr-hlp")
dpr_hlp = dpr_hlp.to("cuda")
```

### To reproduce the training
```bash
python -m torch.distributed.launch \
--nnodes=1 \
--nproc_per_node=4 \
train_retriever.py \
hydra.run.dir=./experiments/dpr-hlp/train \
data_stores=_ir \
train_datasets=[dl,cm] dev_datasets=[] \
biencoder=dpr \
biencoder.shared_encoder=True \
train=dpr_hlp \
train.batch_size=128
```

### Performance

| Model             |     Search Mode          | Top-1  | Top-5  | Top-10 | Top-20  | Top-100 |
|-------------------|:------------------------:|:------:|:------:|:------:|:-------:|:-------:|
| dpr-hlp  (no neg) |  full parametric         | 16.04% | 40.91% | 54.27% | 65.18%  |  80.14% |
| dpr-hlp (neg = 1) | full parametric          | 22.13% | 47.89% | 59.20% | 68.28%  |  81.88% |
| `vsearch/dpr-hlp` (neg=1 + shared) | full parametric | 22.35% | 48.12% | 59.97% | 68.70%  |  82.35% |