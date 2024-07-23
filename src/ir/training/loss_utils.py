import logging
import torch
from typing import Tuple
import torch.nn.functional as F
from torch import Tensor as T

from ..utils.biencoder_utils import BiEncoderBatch
from .ddp_utils import GatherLayer
from .model_utils import move_to_device
from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, build_cts_mask
from .info_card import InfoCard

logger = logging.getLogger(__name__)




def fetch_global_vectors(emb_local, bow_local, k=768):
    topk_mask = build_topk_mask(emb_local, k=k)
    topk_mask = torch.logical_or(topk_mask, bow_local)
    emb_sparse_local = emb_local * topk_mask
    emb_sparse_global = torch.cat(GatherLayer.apply(emb_sparse_local), dim=0)
    emb_dense_global = torch.cat(GatherLayer.apply(emb_local), dim=0)
    bow_global = torch.cat(GatherLayer.apply(bow_local), dim=0)
    return emb_dense_global, emb_sparse_global, bow_global

def _do_biencoder_fwd_pass(
    cfg,
    model: torch.nn.Module,
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
        q_emb, p_emb, q_ids, p_ids = model(
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
        q_bin = build_bow_mask(q_ids, vocab_size=vocab_size, shift_num=shift_vocab_num, norm=norm).float()
        p_bin = build_bow_mask(p_ids, vocab_size=vocab_size, shift_num=shift_vocab_num, norm=norm).float()

        loss, is_correct_1, is_correct_2 = compute_vdr_loss(
            cfg,
            q_emb_local=q_emb,
            p_emb_local=p_emb,
            q_bin_local=q_bin,
            p_bin_local=p_bin,
            verbose=verbose,
            qids=q_ids,
            pids=p_ids,
            tokenizer=model.module.encoder_q.tokenizer,
            logger=logger,
            answers=input.answers
        )

    elif cfg.biencoder.encoder_q.type == "dpr":

        q_emb, p_emb= model(
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
                q_emb_local=q_emb,
                p_emb_local=p_emb,
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
    q_emb_local, 
    p_emb_local, 
    q_bin_local,
    p_bin_local,
    verbose: bool = False,
    qids = None,
    pids = None,
    q_texts = None,
    p_texts = None,
    tokenizer=None,
    logger=None,
    answers=None,
) -> Tuple[T, bool]:

    assert q_emb_local.requires_grad_ and p_emb_local.requires_grad_
    N, V = q_emb_local.shape

    p_emb_local = p_emb_local.view(-1, N, V).permute(1, 0, 2).contiguous()
    p_bin_local = p_bin_local.view(-1, N, V).permute(1, 0, 2).contiguous()

    q_emb_global, q_emb_topk_global, q_bin_global = fetch_global_vectors(q_emb_local, q_bin_local)
    p_emb_global, p_emb_topk_global, p_bin_global = fetch_global_vectors(p_emb_local, p_bin_local)
    
    p_emb_global = p_emb_global.permute(1, 0, 2).contiguous().view(-1, V)
    p_emb_topk_global = p_emb_topk_global.permute(1, 0, 2).contiguous().view(-1, V)
    p_bin_global = p_bin_global.permute(1, 0, 2).contiguous().view(-1, V)

    N_global = q_emb_global.shape[0]

    if cfg.local_rank in [-1,0] and verbose:
        sample_id = 0
        q_emb = q_emb_local[sample_id]
        p_emb = p_emb_local[sample_id, 0, :]
        if q_texts is not None and p_texts is not None:
            q_text = q_texts[sample_id]
            p_text = p_texts[sample_id]
        elif qids is not None and pids is not None:
            q_text = tokenizer.decode([i for i in qids[sample_id] if i != tokenizer.pad_token_id])
            p_text = tokenizer.decode([i for i in pids[sample_id] if i != tokenizer.pad_token_id])
        answer = " | ".join(answers[sample_id]) if answers else None
        
        if getattr(cfg.train, "ret_negatives", None) or getattr(cfg.train, "hard_negatives", None) or getattr(cfg.train, "num_score", None):
            p_neg_text = p_texts[sample_id + N] if p_texts else tokenizer.decode([i for i in pids[sample_id + N] if i != tokenizer.pad_token_id])
            p_neg_emb = p_emb_global[sample_id + N_global]
            texts = [q_text, p_text, p_neg_text, answer] if answer else [q_text, p_text, p_neg_text]
            descs = ['[Q_TEXT]', '[P_TEXT1]', '[P_TEXT2]', '[ANSWER]'] if answer else ['[Q_TEXT]', '[P_TEXT1]', '[P_TEXT2]']
        else:
            p_neg_text = None
            p_neg_emb = None
            texts = [q_text, p_text, answer] if answer else [q_text, p_text]
            descs = ['[Q_TEXT]', '[P_TEXT1]', '[ANSWER]'] if answer else ['[Q_TEXT]', '[P_TEXT1]']

        info_card = InfoCard(tokenizer=tokenizer, shift_vocab_num=cfg.biencoder.encoder_q.shift_vocab_num)
        info_card.add_stat_info(q_emb_global, title=' q_emb_global ')
        info_card.add_stat_info(p_emb_global, title=' p_emb_global ')
        info_card.add_stat_info(q_bin_global, title=' q_bin_global ')
        info_card.add_stat_info(p_bin_global, title=' p_bin_global ')
        info_card.add_texts_info(texts=texts, descs=descs, title=' EXAMPLE ')
        info_card.add_interaction_info(q_emb, p_emb, p_neg_emb, k=20, title=None)
        info_card.wrap_info()
        logger.info(info_card.info)

    retrieval_loss_func = SymmetryBiEncoderNllLoss() if cfg.train.sym_loss else BiEncoderNllLoss()

    if getattr(cfg.train, "semi", True):
        loss_1, is_correct_1 = retrieval_loss_func.calc(q_emb_topk_global, p_emb_global)
        loss_2, is_correct_2 = retrieval_loss_func.calc(q_emb_global, p_emb_topk_global)
        
        if getattr(cfg.train, "cts_mask", None):
            cts_mask_activate = build_cts_mask(q_bin_global)
            cts_mask_deactivate = torch.ones_like(p_emb_global).to(cfg.device)
            cts_mask_deactivate[:N_global] = ~cts_mask_activate
            cts_mask_activate_norm = F.normalize(cts_mask_activate.float()) if cfg.train.cts_mask_norm else cts_mask_activate.float()
            q_bin_global = q_bin_global + cts_mask_activate_norm * cfg.train.cts_mask_weight
            p_emb_global = p_emb_global * cts_mask_deactivate

            cts_mask_activate = build_cts_mask(p_bin_global)
            cts_mask_deactivate = ~cts_mask_activate[:N_global]
            cts_mask_activate_norm = F.normalize(cts_mask_activate.float()) if cfg.train.cts_mask_norm else cts_mask_activate.float()            
            p_bin_global = p_bin_global + cts_mask_activate_norm * cfg.train.cts_mask_weight
            q_emb_global = q_emb_global * cts_mask_deactivate

        loss_3, is_correct_3 = retrieval_loss_func.calc(q_bin_global, p_emb_global)
        loss_4, is_correct_4 = retrieval_loss_func.calc(q_emb_global, p_bin_global)

        loss = (loss_1 + loss_2 + loss_3 + loss_4) / 4
        is_correct_parametric = (is_correct_1 + is_correct_2) / 2
        is_correct_semiparametric = (is_correct_3 + is_correct_4) / 2
        
    else:
        loss_1, is_correct_1 = retrieval_loss_func.calc(q_emb_topk_global, p_emb_global)
        loss_2, is_correct_2 = retrieval_loss_func.calc(q_bin_global, p_emb_global)
        loss = (loss_1 + loss_2) / 2
        is_correct_parametric = is_correct_1
        is_correct_semiparametric = is_correct_2

    return loss, is_correct_semiparametric, is_correct_parametric



def compute_dpr_loss(
    cfg,
    q_emb_local, 
    p_emb_local, 
    logger=None,
) -> Tuple[T, bool]:

    N, D = q_emb_local.shape
    p_emb_local = p_emb_local.view(-1, N, D).permute(1, 0, 2)
    q_emb_global = torch.cat(GatherLayer.apply(q_emb_local.contiguous()), dim=0)
    p_emb_global = torch.cat(GatherLayer.apply(p_emb_local.contiguous()), dim=0)
    p_emb_global = p_emb_global.permute(1, 0, 2).contiguous().view(-1, D)
    retrieval_loss_func = SymmetryBiEncoderNllLoss() if cfg.train.sym_loss else BiEncoderNllLoss()
    loss, is_correct = retrieval_loss_func.calc(q_emb_global, p_emb_global)
    return loss, is_correct, is_correct


class BiEncoderNllLoss(object):
    def calc(
        self,
        q_emb: T,
        p_emb: T,
    ) -> Tuple[T, int]:
        """
        Computes nll loss for the given lists of question and ctx vectors.
        """
        positive_idx_per_question = list(range(q_emb.shape[0]))
        
        scores = self.get_scores(q_emb, p_emb)

        if len(q_emb.size()) > 1:
            q_num = q_emb.size(0)
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
    def get_scores(q_emb: T, p_emb: T) -> T:
        return q_emb @ p_emb.t()


class SymmetryBiEncoderNllLoss(object):
    def calc(
        self,
        q_emb: T,
        p_emb: T,
        temperature: float = 1, 
        positive_idx_per_q = None,
    ) -> Tuple[T, int]:
        """
        Computes symmetry nll loss for the given lists of question and ctx vectors.
        """
        positive_idx_per_q = positive_idx_per_q or list(range(q_emb.shape[0]))
        scores = q_emb @ p_emb.t() # [N, 2N]
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
        target = torch.range(0, q_emb.size(0)-1).long().to(logits_per_p.device)
        loss2 = F.nll_loss(
            logits_per_p,
            target,
            reduction="mean",
        )

        loss = (loss1 + loss2) / 2

        return loss, correct_predictions_count
