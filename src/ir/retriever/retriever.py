import logging
import torch
import random
from torch import Tensor as T
import torch.nn.functional as F
from typing import Any, Tuple, List, Union
import numpy as np
from ..biencoder.biencoder import BiEncoder, BiEncoderConfig
from ..utils.qa_utils import has_answer
from ..data.biencoder_dataset import _normalize
from ..index.base import Index

logger = logging.getLogger(__name__)

class RetrieverConfig(BiEncoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class Retriever(BiEncoder):
    config_class = RetrieverConfig

    def __init__(
        self, config: RetrieverConfig, 
        index: Index = None, 
        **kwargs
    ):
        super().__init__(config, **kwargs)
        self.config = config
        self.index = index

    def retrieve(
        self, 
        queries: Union[List[str], np.ndarray], 
        k: int = 5, 
        dropout: float = 0, 
        a: int = None, 
        index: Index = None
    ):
        index = index or self.index
        a = a or self.encoder_q.config.topk
        if isinstance(queries, list) and isinstance(queries[0], str):
            if a == 0:
                q_embs = self.encoder_q.embed(queries)
                q_embs = F.dropout(q_embs, p=dropout)
            else:
                q_embs = self.encoder_q.embed(queries, topk=a)
                q_embs = F.dropout(q_embs, p=dropout)
        elif isinstance(queries, np.ndarray):
            q_embs = queries
            q_embs = torch.Tensor(q_embs)
            if dropout:
                q_embs = F.dropout(q_embs, p=dropout)
        elif isinstance(queries, T):
            q_embs = queries
            if dropout:
                q_embs = F.dropout(q_embs, p=dropout)
        else:
            raise NotImplementedError
        results = self.index.search(q_embs, k=k)
        return results
    
    def retireve_negatives(
        self, 
        q_emb: Union[np.ndarray, T],
        answers: List[List[str]], 
        ret_neg_num: int = 1, 
        ret_topk: int = 100, 
        pool_size: int = 20, 
        ret_dropout: float = 0, 
        index: Index = None, 
    ) -> List[List[str]]:
        """
        In-training negative retrieval based on given query embeddings.

        Args:
            q_emb (Union[np.ndarray, T]): The query embeddings as a NumPy array or a pytorch tensor.
            answers (List[List[str]]): The lists of correct answers for each query, used to distinguish negatives from positives.
            ret_neg_num (int, optional): The number of negative samples to return for each query. 
                Defaults to 1.
            ret_topk (int, optional): The top-k results to retrieve from the index for each query. 
                Defaults to 100.
            pool_size (int, optional): Maximum size of the candidate pool for negative selection. 
                The process of identifying negatives stops when this size is reached. Defaults to 20.
            ret_dropout (float, optional): Dropout rate applied to the query embeddings to introduce variation.
            index (Index, optional): The index from which to retrieve negatives. 
                If not provided, the default retriever's index is used.

        Returns:
            List[List[str]]: A list of strings representing the negative samples for each query in the batch.
        """
        index = self.index or index
        assert index, "No index Found"
        assert answers, "No answer strings Found"
        ret_indices, _ = self.retrieve(q_emb, a=768, k=ret_topk, dropout=ret_dropout, index=index)

        batch_neg_texts = []
        num_neg_retrieved = []
        for sample_id, sample_ret_indices in enumerate(ret_indices):
            sample_neg_pool_indices = []
            for ret_ind in sample_ret_indices:
                ret_text = index.data[ret_ind]
                if not has_answer(answers[sample_id], ret_text, 'string'):
                    sample_neg_pool_indices.append(ret_ind)
                if len(sample_neg_pool_indices) >= pool_size:
                    break
            num_neg_retrieved.append(len(sample_neg_pool_indices))
            if len(sample_neg_pool_indices) < ret_neg_num:
                num_to_pad = ret_neg_num - len(sample_neg_pool_indices)
                sample_neg_pool_indices += random.sample(range(len(index)), num_to_pad)

            sample_neg_indices = random.sample(sample_neg_pool_indices, ret_neg_num)
            sample_neg_texts = [_normalize(index.data[i]) for i in sample_neg_indices]                
            batch_neg_texts.append(sample_neg_texts)

        logger.debug(f"Retrieved {np.mean(num_neg_retrieved)} negatives within batch ({q_emb.shape[0]} samples)")

        return batch_neg_texts

    def forward(
        self,
        cfg,
        q_ids: T,
        q_segments: T,
        q_attn_mask: T,
        p_ids: T,
        p_segments: T,
        p_attn_mask: T,
        answers: List[List[str]] = None,
        return_ids: bool = False,
    ) -> Tuple[T, T]:
        
        q_emb = self.encoder_q(q_ids, q_segments, q_attn_mask)
        p_emb = self.encoder_p(p_ids, p_segments, p_attn_mask)
        
        if cfg.train.ret_negatives > 0:
            q_emb = self.encoder_q(q_ids, q_segments, q_attn_mask)

            batch_negatives = self.retireve_negatives(
                q_emb.detach(), 
                ret_neg_num=cfg.train.ret_negatives,
                ret_topk = cfg.train.ret_topk,
                pool_size = cfg.train.negative_pool_size,
                ret_dropout = cfg.train.ret_dropout,
                answers = answers, 
            )

            batch_negatives_flat = [neg for sample_negatives in batch_negatives for neg in sample_negatives]
            encoding = self.encoder_p.encode(batch_negatives_flat)
            p_ids_all = torch.cat([p_ids, encoding.input_ids], dim=0)
            p_segments_all = torch.cat([p_segments, encoding.token_type_ids], dim=0)
            attn_mask_all = torch.cat([p_attn_mask, encoding.attention_mask], dim=0)
            p_emb = self.encoder_p(p_ids_all, p_segments_all, attn_mask_all)
            p_ids = p_ids_all

        if return_ids:
            return q_emb, p_emb, q_ids, p_ids 
        else:
            return q_emb, p_emb

