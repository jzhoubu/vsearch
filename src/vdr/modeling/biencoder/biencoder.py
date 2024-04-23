import logging
import random
import collections
import numpy as np
import torch
from typing import Tuple, List, Dict, Union
from torch import Tensor as T
from transformers import PreTrainedModel, PretrainedConfig
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from ...data.biencoder_dataset import BiEncoderSample
from ..encoder._types import ENCODER_TYPES, CONFIG_TYPES

logger = logging.getLogger(__name__)


BiEncoderBatch = collections.namedtuple(
    "BiEncoderBatch",
    [
        "q_tensor",
        "p_tensor",
        "answers",
    ],  
)


class BiEncoderConfig(PretrainedConfig):
    """
    Configuration class for a Bi-Encoder.
    Inherits from PretrainedConfig which provides basic configuration for Pretrained Models.
    
    Args:
    - type (str): The type of the model.
    - encoder_q (Dict[str, any]): Configuration dictionary for the query encoder.
    - encoder_p (Dict[str, any]): Configuration dictionary for the passage encoder.
    - max_len (int): The maximum length of the input sequences.
    - shared_encoder (bool): Whether to use a shared encoder for bi-encoder.
    - device (str): The device to use for model training or inference.
    """
    def __init__(
        self, 
        type: str = None,
        encoder_q: Dict[str, any] = None, 
        encoder_p: Dict[str, any] = None, 
        max_len=512, 
        shared_encoder=False,
        device=None, 
        **kwargs
    ):
        self.type = type
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
        encoder_q_cfg = CONFIG_TYPES[config.encoder_q['type']](**config.encoder_q)
        encoder_p_cfg = CONFIG_TYPES[config.encoder_p['type']](**config.encoder_p)
        self.config = config
        self.encoder_q = ENCODER_TYPES[encoder_q_cfg.type](encoder_q_cfg)
        self.encoder_p = ENCODER_TYPES[encoder_p_cfg.type](encoder_p_cfg) if not self.config.shared_encoder else self.encoder_q

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


    def create_biencoder_input(
        self,
        samples: List[BiEncoderSample],
        insert_title: bool = False,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
    ) -> BiEncoderBatch:
            """
            Creates a batch of the biencoder training tuple.
            :param samples: list of BiEncoderSample-s to create the batch for
            :param insert_title: enables title insertion at the beginning of the context sequences
            :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
            :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
            :param shuffle: shuffles negative passages pools
            :param shuffle_positives: shuffles positive passages pools
            :return: BiEncoderBatch tuple
            """
            assert isinstance(samples[0], BiEncoderSample)
            batch_answers = []
            q_tensors = []
            p_pos_tensors = []
            p_neg_tensors = []
            q_texts = []
            p_pos_texts = []
            p_neg_texts = []
            for sample in samples:
                if shuffle and shuffle_positives:
                    positive_ctxs = sample.positive_passages
                    positive_ctx = positive_ctxs[np.random.choice(len(positive_ctxs))]
                else:
                    positive_ctx = sample.positive_passages[0]

                neg_ctxs = sample.negative_passages                
                hard_neg_ctxs = sample.hard_negative_passages
                question = sample.query

                if shuffle:
                    random.shuffle(neg_ctxs)
                    random.shuffle(hard_neg_ctxs)

                neg_ctxs = neg_ctxs[0:num_other_negatives]
                hard_neg_ctxs = hard_neg_ctxs[0:num_hard_negatives]

                sample_q_tensor = self.encoder_q.tokenizer.encode(question, max_length=512, padding='max_length', truncation=True)

                if insert_title:
                    assert positive_ctx.title is not None and all([ctx.title is not None for ctx in neg_ctxs + hard_neg_ctxs])
                    sample_p_pos_tensor = self.encoder_p.tokenizer.encode(positive_ctx.title, positive_ctx.text, max_length=512, padding='max_length', truncation=True)
                    sample_p_neg_tensors = [self.encoder_p.tokenizer.encode(ctx.title, ctx.text, max_length=512, padding='max_length', truncation=True) for ctx in neg_ctxs + hard_neg_ctxs]
                else:
                    sample_p_pos_tensor = self.encoder_p.tokenizer.encode(positive_ctx.text, max_length=512, padding='max_length', truncation=True)
                    sample_p_neg_tensors = [self.encoder_p.tokenizer.encode(ctx.text, max_length=512, padding='max_length', truncation=True) for ctx in neg_ctxs + hard_neg_ctxs]

                batch_answers.append(sample.answers)

                q_tensors.append(sample_q_tensor)
                p_pos_tensors.append(sample_p_pos_tensor)
                p_neg_tensors.extend(sample_p_neg_tensors)
                
                q_texts.append(question)
                p_pos_texts.append(positive_ctx.text)
                p_neg_texts.extend(neg_ctxs + hard_neg_ctxs)
                p_texts = p_pos_texts + p_neg_texts
            
            batch_q_tensor = torch.stack([torch.LongTensor(q) for q in q_tensors], dim=0)
            batch_p_tensor = torch.stack([torch.LongTensor(p) for p in p_pos_tensors + p_neg_tensors], dim=0)

            return BiEncoderBatch(
                    batch_q_tensor,
                    batch_p_tensor,
                    batch_answers,
                )
    
    def encode_queries(self, queries: list[str], batch_size=32, **kwargs) -> Union[List[np.ndarray], List[torch.Tensor]]:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            queries: List of sentences to encode

        Returns:
            List of embeddings for the given sentences
        """
        q_emb = self.encoder_q.embed(queries, batch_size, **kwargs).cpu().numpy()
        q_embs = [q_emb[i] for i in range(q_emb.shape[0])]
        return q_embs

    def encode_corpus(self, corpus: Union[List[str], List[dict[str, str]]], batch_size=32, **kwargs) -> Union[List[np.ndarray], List[torch.Tensor]]:
        """
        Returns a list of embeddings for the given sentences.
        Args:
            corpus: List of sentences to encode
                or list of dictionaries with keys "title" and "text"

        Returns:
            List of embeddings for the given sentences
        """
        if isinstance(corpus[0], dict):
            corpus = [f"{x['title']} [SEP] {x['text']}" for x in corpus]
        p_emb = self.encoder_p.embed(corpus, batch_size, **kwargs).cpu().numpy()
        p_embs = [p_emb[i] for i in range(p_emb.shape[0])]
        return p_embs

    def explain(self, q, p, k=768, visual=False, visual_width=800, visual_height=800):
        q_dst = self.encoder_q.dst(q, k=k)
        p_dst = self.encoder_p.dst(p, k=k)
        result_dict = {
            key: q_dst.get(key, 0) * p_dst.get(key, 0)
            for key in set(q_dst) | set(p_dst)
            if q_dst.get(key, 0) * p_dst.get(key, 0) != 0
        }
        sorted_keys = sorted(result_dict, key=result_dict.get, reverse=True)
        results = {key: result_dict[key] for key in sorted_keys}
        if visual:
            wordcloud = WordCloud(max_words=k, width=visual_width, height=visual_height).generate_from_frequencies(results)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.tight_layout(pad=0)
            plt.show()

        return results

