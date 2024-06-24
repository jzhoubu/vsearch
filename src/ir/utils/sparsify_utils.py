import torch
import torch.nn.functional as F
from typing import Tuple, List, Union
import numpy as np

elu1p = lambda x: F.elu(x) + 1

def build_topk_mask(embs: Union[torch.Tensor, np.ndarray], k: int = 768, dim: int = -1):        
    if isinstance(embs, np.ndarray):
        embs = torch.Tensor(embs)
    values, indices = torch.topk(embs, k, dim=dim)
    topk_mask = torch.zeros_like(embs)
    topk_mask.scatter_(dim=-1, index=indices, value=1)
    return topk_mask

def topk_sparsify(emb_dense: torch.Tensor, k: int, dim: int = -1):
    topk_mask = build_topk_mask(emb_dense, k=k, dim=dim)
    emb_sparse = emb_dense * topk_mask
    return emb_sparse


def build_bow_mask(text_ids, vocab_size=30522, shift_num=0, norm=False):
    N = text_ids.shape[0]
    V = vocab_size
    bow_mask = torch.zeros([N, V]).to(text_ids.device).scatter_(-1, text_ids, 1).bool().float()
    bow_mask = bow_mask[:, shift_num:].contiguous()
    if norm:
        bow_mask = F.normalize(bow_mask)
    return bow_mask


def init_cts_mask_like(embs):
    batch_size, vocab_size = embs.size()
    indices = torch.arange(vocab_size) % batch_size
    cts_mask = (indices.unsqueeze(0) == torch.arange(batch_size).unsqueeze(1))
    return cts_mask.to(embs.device)


def build_cts_mask(bow_embs):
    bow_batch = bow_embs.sum(0).bool()
    cts_mask_init = init_cts_mask_like(bow_embs)
    cts_mask = (cts_mask_init & ~bow_batch.unsqueeze(0))
    return cts_mask.bool().to(bow_embs.device)




