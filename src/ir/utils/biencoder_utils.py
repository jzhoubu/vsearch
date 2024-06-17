import random
import torch
import collections
import numpy as np

from ..data.biencoder_dataset import BiEncoderSample
from ..biencoder.biencoder import BiEncoder

BiEncoderBatch = collections.namedtuple(
    "BiEncoderBatch",
    [
        "q_tensor",
        "p_tensor",
        "q_texts",
        "p_texts",
        "answers",
    ],  
)

def create_biencoder_batch(
        biencoder: BiEncoder,
        samples: list[BiEncoderSample],
        insert_title: bool = False,
        num_hard_negatives: int = 0,
        num_other_negatives: int = 0,
        shuffle: bool = True,
        shuffle_positives: bool = False,
    ) -> BiEncoderBatch:
        """
        Creates a batch of the biencoder training tuple.
        :param samples: list of BiEncoderSample to create the BiEncoderBatch
        :param insert_title: enables title insertion at the beginning of the context sequences
        :param num_hard_negatives: amount of hard negatives per question (taken from samples' pools)
        :param num_other_negatives: amount of other negatives per question (taken from samples' pools)
        :param shuffle: shuffles negative passages pools
        :param shuffle_positives: shuffles positive passages pools
        :return: BiEncoderBatch tuple
        """
        assert isinstance(samples[0], BiEncoderSample), f"The input type is not <BiEncoderSample> but <{type(samples[0])}>"
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

            sample_q_tensor = biencoder.encoder_q.tokenizer.encode(question, max_length=512, padding='max_length', truncation=True)

            if insert_title:
                assert positive_ctx.title is not None and all([ctx.title is not None for ctx in neg_ctxs + hard_neg_ctxs])
                sample_p_pos_tensor = biencoder.encoder_p.tokenizer.encode(positive_ctx.title, positive_ctx.text, max_length=512, padding='max_length', truncation=True)
                sample_p_neg_tensors = [biencoder.encoder_p.tokenizer.encode(ctx.title, ctx.text, max_length=512, padding='max_length', truncation=True) for ctx in neg_ctxs + hard_neg_ctxs]
            else:
                sample_p_pos_tensor = biencoder.encoder_p.tokenizer.encode(positive_ctx.text, max_length=512, padding='max_length', truncation=True)
                sample_p_neg_tensors = [biencoder.encoder_p.tokenizer.encode(ctx.text, max_length=512, padding='max_length', truncation=True) for ctx in neg_ctxs + hard_neg_ctxs]

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
                q_texts,
                p_texts,
                batch_answers,
            )