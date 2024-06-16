import logging
import torch
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor as T
from torch import nn

from ..biencoder.biencoder import BiEncoderBatch
from .ddp_utils import GatherLayer
from .model_utils import move_to_device
from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, build_cts_mask
from .info_card import InfoCard

logger = logging.getLogger(__name__)

def fetch_global_vectors(v_local, bow_local, k=768):
    topk_mask = build_topk_mask(v_local, k=k)
    topk_mask = torch.logical_or(topk_mask, bow_local)
    v_topk_local = v_local * topk_mask
    v_topk_global = torch.cat(GatherLayer.apply(v_topk_local), dim=0)
    v_global = torch.cat(GatherLayer.apply(v_local), dim=0)
    bow_global = torch.cat(GatherLayer.apply(bow_local), dim=0)
    return v_global, v_topk_global, bow_global

def _do_biencoder_fwd_pass(
    cfg,
    model: nn.Module,
    input: BiEncoderBatch,
    answers = None, 
    verbose = False,
    logger = None,
) -> Tuple[torch.Tensor, int]:

    input = BiEncoderBatch(**move_to_device(input._asdict(), cfg.device))
    
    q_ids = input.q_tensor[:, :cfg.biencoder.encoder_q.max_len]
    p_ids = input.p_tensor[:, :cfg.biencoder.encoder_p.max_len]
    q_segments = torch.zeros_like(q_ids)
    p_segments = torch.zeros_like(p_ids)
    q_attn_mask = q_ids != 0 
    p_attn_mask = p_ids != 0

    if cfg.biencoder.encoder_q.type == "vdr":
        v_q, v_p, q_ids, p_ids = model(
            cfg,
            q_ids=q_ids, 
            q_segments=q_segments,
            q_attn_mask=q_attn_mask,
            p_ids=p_ids,
            p_segments=p_segments,
            p_attn_mask=p_attn_mask,
            answers=answers,
            return_ids=True,
        )

        vocab_size = model.module.encoder_q.config.vocab_size
        shift_vocab_num = model.module.encoder_q.config.shift_vocab_num 
        norm = model.module.encoder_q.config.norm
        bow_q = build_bow_mask(q_ids, vocab_size=vocab_size, shift_num=shift_vocab_num, norm=norm).float()
        bow_p = build_bow_mask(p_ids, vocab_size=vocab_size, shift_num=shift_vocab_num, norm=norm).float()

        loss, is_correct_1, is_correct_2 = compute_vdr_loss(
            cfg,
            v_q_local=v_q,
            v_p_local=v_p,
            bow_q_local=bow_q,
            bow_p_local=bow_p,
            print_info=verbose,
            qids=q_ids,
            pids=p_ids,
            tokenizer=model.module.encoder_q.tokenizer,
            logger=logger,
            answers=input.answers
        )

    elif cfg.biencoder.encoder_q.type == "dpr":

        v_q, v_p= model(
            cfg,
            q_ids=q_ids, 
            q_segments=q_segments,
            q_attn_mask=q_attn_mask,
            p_ids=p_ids,
            p_segments=p_segments,
            p_attn_mask=p_attn_mask,
            answers=answers,
        )
                
        loss, is_correct_1, is_correct_2 = compute_dpr_loss(
                cfg,
                h_q_local=v_q,
                h_p_local=v_p,
                logger=logger,
            )
    
    else:
        raise NotImplementedError


    is_correct_1 = is_correct_1.sum().item()
    is_correct_2 = is_correct_2.sum().item()

    if cfg.n_gpu > 1:
        loss = loss.mean()

    return loss, is_correct_1, is_correct_2



def compute_vdr_loss(
    cfg,
    v_q_local, 
    v_p_local, 
    bow_q_local,
    bow_p_local,
    print_info: bool = False,
    qids = None,
    pids = None,
    tokenizer=None,
    logger=None,
    answers=None,
) -> Tuple[T, bool]:

    N, V = v_q_local.shape

    v_p_local = v_p_local.view(-1, N, V).permute(1, 0, 2).contiguous()
    bow_p_local = bow_p_local.view(-1, N, V).permute(1, 0, 2).contiguous()

    v_q_global, v_q_topk_global, bow_q_global = fetch_global_vectors(v_q_local, bow_q_local)
    v_p_global, v_p_topk_global, bow_p_global = fetch_global_vectors(v_p_local, bow_p_local)
    
    v_p_global = v_p_global.permute(1, 0, 2).contiguous().view(-1, V)
    v_p_topk_global = v_p_topk_global.permute(1, 0, 2).contiguous().view(-1, V)
    bow_p_global = bow_p_global.permute(1, 0, 2).contiguous().view(-1, V)

    N_global = v_q_global.shape[0]

    if cfg.local_rank in [-1,0] and print_info:
        sample_id = 0
        q_emb = v_q_local[sample_id]
        p_emb = v_p_local[sample_id, 0, :]
        q_text = tokenizer.decode([i for i in qids[sample_id] if i != tokenizer.pad_token_id])
        p_text = tokenizer.decode([i for i in pids[sample_id] if i != tokenizer.pad_token_id])
        answer = " | ".join(answers[sample_id])
        if cfg.train.ret_negatives == 1 or cfg.train.hard_negatives == 1:
            p_neg_text = tokenizer.decode([i for i in pids[sample_id + N] if i != tokenizer.pad_token_id])
            p_neg_emb = v_p_global[sample_id + N_global]
            texts = [q_text, p_text, p_neg_text, answer]
            descs = ['[Q_TEXT]', '[P_TEXT1]', '[P_TEXT2]', '[ANSWER]']
        else:
            p_neg_text = None
            p_neg_emb = None
            texts = [q_text, p_text, answer]
            descs = ['[Q_TEXT]', '[P_TEXT1]', '[ANSWER]']

        info_card = InfoCard(tokenizer=tokenizer, shift_vocab_num=cfg.biencoder.encoder_q.shift_vocab_num)
        info_card.add_stat_info(v_q_global, title=' v_q_global ')
        info_card.add_stat_info(v_p_global, title=' v_p_global ')
        info_card.add_stat_info(bow_q_global, title=' bow_q_global ')
        info_card.add_stat_info(bow_p_global, title=' bow_p_global ')
        info_card.add_texts_info(texts=texts, descs=descs, title=' EXAMPLE ')
        info_card.add_interaction_info(q_emb, p_emb, p_neg_emb, k=20, title=None)
        info_card.wrap_info()
        logger.info(info_card.info)

    retrieval_loss_func = SymmetryBiEncoderNllLoss() if cfg.train.sym_loss else BiEncoderNllLoss()

    if cfg.train.semi:
        loss_1, is_correct_1 = retrieval_loss_func.calc(v_q_topk_global, v_p_global)
        loss_2, is_correct_2 = retrieval_loss_func.calc(v_q_global, v_p_topk_global)
        
        if cfg.train.cts_mask:
            cts_mask_activate = build_cts_mask(bow_q_global)
            cts_mask_deactivate = torch.ones_like(v_p_global).to(cfg.device)
            cts_mask_deactivate[:N_global] = ~cts_mask_activate
            cts_mask_activate_norm = F.normalize(cts_mask_activate.float()) if cfg.train.cts_mask_norm else cts_mask_activate.float()
            bow_q_global = bow_q_global + cts_mask_activate_norm * cfg.train.cts_mask_weight
            v_p_global = v_p_global * cts_mask_deactivate

            cts_mask_activate = build_cts_mask(bow_p_global)
            cts_mask_deactivate = ~cts_mask_activate[:N_global]
            cts_mask_activate_norm = F.normalize(cts_mask_activate.float()) if cfg.train.cts_mask_norm else cts_mask_activate.float()            
            bow_p_global = bow_p_global + cts_mask_activate_norm * cfg.train.cts_mask_weight
            v_q_global = v_q_global * cts_mask_deactivate

        loss_3, is_correct_3 = retrieval_loss_func.calc(bow_q_global, v_p_global)
        loss_4, is_correct_4 = retrieval_loss_func.calc(v_q_global, bow_p_global)

        loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4
        is_correct_parametric = (is_correct_1 + is_correct_2) / 2
        is_correct_semiparametric = (is_correct_3 + is_correct_4) / 2
        
    else:
        loss_1, is_correct_1 = retrieval_loss_func.calc(v_q_topk_global, v_p_global)
        loss_2, is_correct_2 = retrieval_loss_func.calc(bow_q_global, v_p_global)
        loss = (loss_1 + loss_2) / 2
        is_correct_parametric = is_correct_1
        is_correct_semiparametric = is_correct_2

    return loss, is_correct_semiparametric, is_correct_parametric



def compute_dpr_loss(
    cfg,
    h_q_local, 
    h_p_local, 
    logger=None,
) -> Tuple[T, bool]:

    N, D = h_q_local.shape
    h_p_local = h_p_local.view(-1, N, D).permute(1, 0, 2)
    h_q_global = torch.cat(GatherLayer.apply(h_q_local.contiguous()), dim=0)
    h_p_global = torch.cat(GatherLayer.apply(h_p_local.contiguous()), dim=0)
    h_p_global = h_p_global.permute(1, 0, 2).contiguous().view(-1, D)
    retrieval_loss_func = SymmetryBiEncoderNllLoss() if cfg.train.sym_loss else BiEncoderNllLoss()
    loss, is_correct = retrieval_loss_func.calc(h_q_global, h_p_global)
    return loss, is_correct, is_correct


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        """
        positive_idx_per_question = list(range(q_vectors.shape[0]))
        
        scores = self.get_scores(q_vectors, ctx_vectors)

        if len(q_vectors.size()) > 1:
            q_num = q_vectors.size(0)
            scores = scores.view(q_num, -1)

        softmax_scores = F.log_softmax(scores, dim=1)

        loss = F.nll_loss(
            softmax_scores,
            torch.tensor(positive_idx_per_question).to(softmax_scores.device),
            reduction="mean",
        )

        max_score, max_idxs = torch.max(softmax_scores, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_question).to(max_idxs.device)).sum()

        return loss, correct_predictions_count

    @staticmethod
    def get_scores(q_vector: T, ctx_vectors: T) -> T:
        return q_vector @ ctx_vectors.t()


class SymmetryBiEncoderNllLoss(object):
    def calc(
        self,
        q_vectors: T,
        ctx_vectors: T,
        temperature: float = 1, 
        positive_idx_per_q = None,
    ) -> Tuple[T, int]:
        """
        Computes symmetry nll loss for the given lists of question and ctx vectors.
        """
        positive_idx_per_q = positive_idx_per_q or list(range(q_vectors.shape[0]))
        scores = q_vectors @ ctx_vectors.t() # [N, 2N]
        scores_t = scores.t()[positive_idx_per_q, :] # [2N, N] - > [N, N]
        
        logits_per_q = F.log_softmax(scores / temperature, dim=1) # [N, 2N]
        target = torch.tensor(positive_idx_per_q).to(logits_per_q.device)
        loss1 = F.nll_loss(
            logits_per_q,
            target,
            reduction="mean",
        )
        
        max_score, max_idxs = torch.max(logits_per_q, 1)
        correct_predictions_count = (max_idxs == torch.tensor(positive_idx_per_q).to(max_idxs.device)).sum()

        logits_per_p = F.log_softmax(scores_t / temperature, dim=1) 
        target = torch.range(0, q_vectors.size(0)-1).long().to(logits_per_p.device)
        loss2 = F.nll_loss(
            logits_per_p,
            target,
            reduction="mean",
        )

        loss = (loss1 + loss2) / 2

        return loss, correct_predictions_count
