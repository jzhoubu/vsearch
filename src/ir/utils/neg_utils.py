import logging
import random
from torch import Tensor as T
from typing import Any, Tuple, List, Union
import numpy as np
from ..retriever.retriever import Retriever
from ..utils.qa_utils import has_answer
from ..data.biencoder_dataset import _normalize
from ..index.base import Index

logger = logging.getLogger(__name__)

def retireve_negatives(
    model: Retriever, 
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
    index = model.index or index
    ret_indices, _ = model.retrieve(q_emb, a=768, k=ret_topk, dropout=ret_dropout, index=index)

    in_batch_negs = []
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
        in_batch_negs.append(sample_neg_texts)

    logger.debug(f"Retrieved {np.mean(num_neg_retrieved)} negatives within batch ({q_emb.shape[0]} samples)")

    return in_batch_negs
