import logging
import numpy as np
import torch
from typing import Tuple, List, Dict, Union
from torch import Tensor as T
from transformers import PreTrainedModel, PretrainedConfig
from ..utils.visualize_utils import wordcloud_from_dict


from ..data.biencoder_dataset import BiEncoderSample
from ..encoder.types import ENCODER_TYPES, CONFIG_TYPES

logger = logging.getLogger(__name__)

class BiEncoderConfig(PretrainedConfig):
    """
    Configuration class for a Bi-Encoder.
    Inherits from PretrainedConfig which provides basic configuration for Pretrained Models.
    
    Args:
    - encoder_q (Dict[str, any]): Configuration dictionary for the query encoder.
    - encoder_p (Dict[str, any]): Configuration dictionary for the passage encoder.
    - max_len (int): The maximum length of the input sequences.
    - shared_encoder (bool): Whether to use a shared encoder for bi-encoder.
    - device (str): The device to use for model training or inference.
    """
    def __init__(
        self, 
        encoder_q: Dict[str, any] = None, 
        encoder_p: Dict[str, any] = None, 
        max_len=512, 
        shared_encoder=False,
        device=None, 
        **kwargs
    ):
        self.encoder_q = encoder_q
        self.encoder_p = encoder_p
        self.max_length = max_len
        self.shared_encoder = shared_encoder
        self.device = device
        super().__init__(**kwargs)


class BiEncoder(PreTrainedModel):
    """Bi-Encoder model components"""
    config_class = BiEncoderConfig

    def __init__(self, config: BiEncoderConfig, **kwargs):        
        super().__init__(config)
        self.config = config
        encoder_q_cfg = CONFIG_TYPES[config.encoder_q['type']](**config.encoder_q)
        encoder_p_cfg = CONFIG_TYPES[config.encoder_p['type']](**config.encoder_p)
        self.encoder_q = ENCODER_TYPES[encoder_q_cfg.type](encoder_q_cfg)
        self.encoder_p = ENCODER_TYPES[encoder_p_cfg.type](encoder_p_cfg)
        self.default_batch_size = None
        # TODO: hot-fix
        if self.config.shared_encoder:
            self.encoder_q.config.max_len = max(encoder_q_cfg.max_len, encoder_p_cfg.max_len)
            self.encoder_p = self.encoder_q

    def forward(
        self,
        q_ids: T,
        q_segments: T,
        q_attn_mask: T,
        p_ids: T,
        p_segments: T,
        p_attn_mask: T,
    ) -> Tuple[T, T]:
        
        q_emb = self.encoder_q(q_ids, q_segments, q_attn_mask)
        p_emb = self.encoder_p(p_ids, p_segments, p_attn_mask)
        return q_emb, p_emb

    def encode_queries(self, queries: List[str], batch_size=None, convert_to_tensor=True, **kwargs) -> Union[List[np.ndarray], List[torch.Tensor]]:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            queries: List of sentences to encode

        Returns:
            List of embeddings for the given sentences
        """
        batch_size = batch_size or self.default_batch_size
        q_emb = self.encoder_q.embed(queries, batch_size, convert_to_tensor=convert_to_tensor, activate_lexical=False, **kwargs)
        return q_emb

    def encode_corpus(self, corpus: Union[List[str], List[Dict[str, str]]], batch_size=None, max_len=None, to_cpu=False, convert_to_tensor=True, **kwargs) -> Union[List[np.ndarray], List[torch.Tensor]]:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            corpus: List of sentences to encode
                or list of dictionaries with keys "title" and "text"

        Returns:
            List of embeddings for the given sentences
        """
        batch_size = batch_size or self.default_batch_size
        processed_corpus = []
        for p in corpus:
            if isinstance(p, str):
                processed_corpus.append(p)
            elif isinstance(p, dict):
                if "title" in p and p["title"]:
                    processed_corpus.append(f"{p['title']} [SEP] {p['text']}")
                else:
                    processed_corpus.append(p['text'])
        p_emb = self.encoder_p.embed(processed_corpus, batch_size, max_len=max_len, to_cpu=to_cpu, convert_to_tensor=convert_to_tensor, activate_lexical=False, **kwargs)
        return p_emb

    def explain(self, q, p, topk=768, visual=False, max_words=100, log_scale=True, save_file=None):
        q_dst = self.encoder_q.dst(q, topk=topk)
        p_dst = self.encoder_p.dst(p, topk=topk)
        result_dict = {
            key: q_dst.get(key, 0) * p_dst.get(key, 0)
            for key in set(q_dst) | set(p_dst)
            if q_dst.get(key, 0) * p_dst.get(key, 0) != 0
        }
        sorted_keys = sorted(result_dict, key=result_dict.get, reverse=True)
        results = {key: result_dict[key] for key in sorted_keys}
        if visual:
            wordcloud_from_dict(results, max_words=max_words, log_scale=log_scale, save_file=save_file)
        return results

