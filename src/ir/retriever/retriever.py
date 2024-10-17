import logging
import torch
import random
from tqdm import tqdm
from torch import Tensor as T
import torch.nn.functional as F
from typing import Any, Tuple, List, Union
import numpy as np
from scipy.sparse import save_npz, csr_array, vstack
from .index import SearchResults, Index, SparseIndex, IndexType
from .index_utils import get_first_unique_n
from ..biencoder.biencoder import BiEncoder, BiEncoderConfig
from ..utils.qa_utils import has_answer
from ..data.biencoder_dataset import _normalize
# from ..index.base import Index, SearchResults
# from ..index.bag_of_token_index import Index, SearchResults

logger = logging.getLogger(__name__)

class RetrieverConfig(BiEncoderConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class Retriever(BiEncoder):
    config_class = RetrieverConfig

    def __init__(self, config: RetrieverConfig, index: Index = None, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.index = index

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
        
        if cfg.train.ret_negatives and cfg.train.ret_negatives > 0:
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
            p_emb_neg = self.encoder_p(**encoding)
            max_length = max(p_ids.size(1), encoding.input_ids.size(1))
            p_ids_padded = F.pad(p_ids, (0, max_length - p_ids.size(1)))
            input_ids_padded = F.pad(encoding.input_ids, (0, max_length - encoding.input_ids.size(1)))
            p_ids = torch.cat([p_ids_padded, input_ids_padded], dim=0) 
            p_emb = torch.cat([p_emb, p_emb_neg], dim=0)

        if return_ids:
            return q_emb, p_emb, q_ids, p_ids 
        else:
            return q_emb, p_emb

    def process_query(self, queries: Union[str, List[str], np.ndarray, T], dropout: float, a: int = None) -> T:
        """
        Process the input queries into proper embeddings for retrieval.

        Args:
            queries (Union[str, List[str], np.ndarray, T]): queries to be processed.
            dropout (float): dropout probability to be applied to the embeddings.
            a (int, optional): number of dimension that activated.

        Returns:
            T: Query embedding in a torch tensor.
        """
        if isinstance(queries, str):
            q_emb = self.encoder_q.embed([queries], topk=a or self.encoder_q.config.topk)
        elif isinstance(queries, list) and isinstance(queries[0], str):
            q_emb = self.encoder_q.embed(queries, topk=a or self.encoder_q.config.topk)
        elif isinstance(queries, np.ndarray):
            q_emb = torch.Tensor(queries)
        elif isinstance(queries, T):
            q_emb = queries
        else:
            raise NotImplementedError("Query type not supported")
        if dropout:
            q_emb = F.dropout(q_emb, p=dropout)
        return q_emb


    def retrieve(
        self, 
        queries: Union[List[str], np.ndarray, T], 
        k: int = 5, 
        dropout: float = 0, 
        a: int = None, 
        index: Index = None
    ) -> SearchResults:
        index = index or self.index
        a = a or self.encoder_q.config.topk
        q_emb = self.process_query(queries, dropout, a)
        results = self.index.search(q_emb, k=k)
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


    def _build_bot_vectors(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_len: int = 128,
        max_token: int = None,
        num_shift: int = 999
    ) -> csr_array:
        """
        Builds bag-of-token (BoT) vectors of text.

        Parameters:
            texts (List[str]): List of input text strings.
            batch_size (int): Number of texts to process in each batch.
            max_len (int): Maximum length of tokenized sequences.
            max_token (int, optional): Maximum number of unique tokens to consider.
            num_shift (int): Number to shift tokens by; set to '999' for bert-based model.

        Returns:
            scipy.sparse.csr_array: A compressed sparse row array representing the BoT index.
        """
        tokenizer = self.encoder_p.tokenizer
        vocab_size = len(tokenizer.vocab)
        bot_embs = []
        batch_bot_emb = np.zeros([batch_size, vocab_size])
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Building Bag-of-Token Index"):
            batch_bot_emb[:, :] = 0
            batch_texts = texts[batch_start: batch_start+batch_size]
            input_ids = tokenizer(batch_texts, max_length=max_len, truncation=True)['input_ids']
            for i, token_ids in enumerate(input_ids):
                if max_token:
                    token_ids = list(get_first_unique_n(token_ids, max_token))
                batch_bot_emb[i, token_ids] = 1
            if len(batch_texts) < batch_size:
                batch_bot_emb_trimmed = batch_bot_emb[:len(batch_texts), num_shift:]
            else:
                batch_bot_emb_trimmed = batch_bot_emb[:, num_shift:]            
            bot_embs.append(csr_array(batch_bot_emb_trimmed))
        bot_emb_csr = vstack(bot_embs)
        return bot_emb_csr

    def _build_embedding_vectors(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_len: int = 128,
        num_shift: int = 0
    ) -> T:
        """
        Builds embedding-based vectors of text.

        Parameters:
            texts (List[str]): List of input text strings.
            batch_size (int): Number of texts to process in each batch.
            max_len (int): Maximum length of tokenized sequences.
            num_shift (int): Number to shift tokens by; set to '999' for bert-based and Sparse IR models.

        Returns:
            torch.Tensor: A tensor containing the embedding vectors.
        """
        embs = []
        for batch_start in tqdm(range(0, len(texts), batch_size), desc="Building Embedding Index"):
            batch_texts = texts[batch_start: batch_start+batch_size]
            batch_emb = self.encode_corpus(batch_texts, batch_size=batch_size, max_len=max_len, convert_to_tensor=True)
            batch_emb = batch_emb[:, num_shift:]
            embs.append(batch_emb)
        emb = torch.cat(embs, dim=0)
        return emb

    def build_index(self, texts: List[str], batch_size=32, index_type=IndexType.DENSE, bag_of_token=False):
        if isinstance(index_type, str):
            index_type = IndexType(str(index_type).lower())
        elif not isinstance(index_type, IndexType):
            raise TypeError("index_type must be an instance of IndexType, int, or str.")

        self.index_type = index_type

        if index_type == IndexType.DENSE:
            # build dense index
            self.index = Index()
            self.index.data = texts
            dense_vector = self._build_embedding_vectors(texts, batch_size=batch_size)
            self.index.vector = dense_vector

        elif index_type == IndexType.SPARSE:
            # Build sparse index
            self.index = SparseIndex()
            self.index.data = texts
            dense_vector = self._build_embedding_vectors(texts, batch_size=batch_size)
            sparse_vector = dense_vector.to_sparse_csr()
            self.index.vector = sparse_vector

        elif index_type == IndexType.BAG_OF_TOKEN:
            # Build bag-of-token index
            self.index = SparseIndex()
            self.index.data = texts
            bot_vector = self._build_bot_vectors(texts, batch_size=batch_size)
            self.index.vector = bot_vector
     
        else:
            raise NotImplementedError
        
        self.index.move_to_device(self.device)
    
    def save_index(self, path):
        self.index.save(path)

    def load_index(self, index_file=None, data_file=None, index_type=None):
        if index_type is None:
            if index_file.endswith(".pt"):
                index_type = IndexType.DENSE
            elif index_file.endswith(".npz"):
                index_type = IndexType.SPARSE
            else:
                raise ValueError(
                    "Cannot infer index type from file extension. "
                    "Please provide 'index_type' explicitly."
                )
        else:
            if isinstance(index_type, str):
                index_type = IndexType(index_type.lower())
            else: 
                raise TypeError(
                    "index_type must be an instance of IndexType, int, or str."
                )    
        self.index_type = index_type
        if index_type == IndexType.DENSE:
            self.index = Index(index_file, data_file, device=self.device)
        elif index_type in [IndexType.SPARSE, IndexType.BAG_OF_TOKEN]:
            self.index = SparseIndex(index_file, data_file, device=self.device)
        else:
            raise NotImplementedError(f"Unknown index type: {index_type}")

