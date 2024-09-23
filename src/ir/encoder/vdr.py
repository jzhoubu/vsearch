from contextlib import nullcontext
from functools import partial
import logging
from typing import List, Union

import torch
from torch import Tensor as T
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig, BatchEncoding, PreTrainedModel

from ..utils.sparsify_utils import build_bow_mask, build_topk_mask, elu1p
from ..utils.visualize_utils import wordcloud_from_dict
from ..training.ddp_utils import get_rank

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)



class VDREncoderConfig(BertConfig):
    """
    Configuration class for VDR Encoder based on BERT.

    Args:
        model_id (str): Base model configuration. Defaults to 'bert-base-uncased'.
        max_len (int): Maximum length of the input sequences. Defaults to 256.
        norm (bool): Whether normalization is to be applied. Defaults to False.
        shift_vocab_num (int): Number to shift in the vocabulary. Defaults to 999 for bert-based-uncased vocab.
    """

    def __init__(
        self,
        model_id='bert-base-uncased',
        max_len=256,
        norm=False,
        shift_vocab_num=999,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_len = max_len
        self.model_id = model_id
        self.norm = norm
        self.shift_vocab_num = shift_vocab_num


class VDREncoder(PreTrainedModel):
    config_class = VDREncoderConfig

    def __init__(self, config: VDREncoderConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.ln = torch.nn.LayerNorm(self.config.hidden_size)
        self.bert_model = AutoModel.from_pretrained(config.model_id, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        self.build_bow_mask = partial(build_bow_mask, vocab_size=config.vocab_size, shift_num=config.shift_vocab_num, norm=config.norm)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
    ) -> T:
        
        outputs = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state_ln = self.ln(last_hidden_state)
        vocab_embs = last_hidden_state_ln @ self.bert_model.embeddings.word_embeddings.weight[self.config.shift_vocab_num:, :].t()
        vocab_embs = elu1p(vocab_embs)
        if self.config.pooling == "max":
            vocab_emb = vocab_embs.max(1)[0]
        elif self.config.pooling == "mean":
            if self.config.pooling_topk:
                vocab_emb = vocab_embs.topk(self.config.pooling_topk, dim=1).values.mean(1)
            else:
                vocab_emb = vocab_emb.mean(1)
        else:
            raise NotImplementedError
        vocab_emb = F.normalize(vocab_emb) if self.config.norm else vocab_emb
        return vocab_emb

    def encode(
        self,
        texts: Union[List[str], str], 
        max_len: int = None, 
    ) -> BatchEncoding:
        max_len = max_len or self.config.max_len
        texts = [texts] if isinstance(texts, str) else texts
        encoding = self.tokenizer.batch_encode_plus(texts, padding=True, truncation=True, max_length=max_len, return_tensors='pt')
        encoding = encoding.to(self.device)
        return encoding

    def embed(
        self, 
        texts: Union[List[str], str], 
        batch_size: int = 128, 
        max_len: int = None, 
        topk: int = None,
        bow: bool = False, 
        activate_lexical: bool = True,
        require_grad: bool = False,
        to_cpu: bool = False,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
        **kwargs
    ) -> T:
        """Embeds texts into lexical representations.

        Args:
            texts (str, List[str]): Text or list of texts to be embedded.
            batch_size (int): Size of batches. Defaults to 128.
            max_len (int): Maximum sequence length.
            topk (int): Number of active dimensions after top-k sparsification. 
                - If topk=0, only activate the dimensions of the presented token;
                - If topk=-1 or None, activate all the dimensions;
                - Otherwise, acitvate only top-k dimension. 
            bow (bool): If True, embeds texts into binary token representations.
            activate_lexical (bool): If True, force to activate token lexical dimension.
            require_grad (bool): If True, keeps gradients for backpropagation. If False, turn model to .eval() for consistent embeddings.
            to_cpu (bool): If True, moves the result to CPU memory.
            convert_to_tensor (bool): If True, returns a Tensor instead of a NumPy array.
            show_progress_bar (bool): If True, displays embedding progress. 

        Returns:
            Tensor: Lexical representations of input texts, with shape [N, V], 
                where N is the number of texts and V is the vocabulary size.
        """

        max_len = max_len or self.config.max_len
        topk = topk or self.config.topk
        texts = [texts] if isinstance(texts, str) else texts
        is_training = self.training

        if not require_grad and is_training:
            self.eval()

        with torch.no_grad() if not require_grad else nullcontext():
            batch_embs = []
            num_text = len(texts)
            iterator = range(0, num_text, batch_size)
            for batch_start in tqdm(iterator) if show_progress_bar else iterator:
                
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: batch_start : {batch_start}")
                batch_texts = texts[batch_start : batch_start + batch_size]
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: encode_start : {batch_start}")
                encoding = self.encode(batch_texts, max_len=max_len)
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: build_bow_start : {batch_start}")
                bow_mask = self.build_bow_mask(encoding.input_ids)
                logger.debug(f"RANK-{get_rank()}, in the vdr.embed loop: forward_start : {batch_start}")

                if bow:
                    batch_emb = bow_mask
                else:
                    batch_emb = self(**encoding)
                    if topk == 0: 
                        # Activating only dimensions corresponding to presented tokens
                        topk_mask = torch.zeros_like(batch_emb)
                    elif topk == None or topk == -1: 
                        # Acitvate all dimensions
                        topk_mask = torch.ones_like(batch_emb)
                    else: 
                        # Acitvate top-k dimensions
                        topk_mask = build_topk_mask(batch_emb, topk)
                    mask = torch.logical_or(bow_mask, topk_mask) if activate_lexical else topk_mask
                    batch_emb *= mask
                batch_embs.append(batch_emb)
            emb = torch.cat(batch_embs, dim=0)
            if not convert_to_tensor:
                emb = emb.cpu().numpy()
            elif to_cpu:
                emb = emb.cpu()

        if is_training and not self.training:
            self.train()
        return emb

    def disentangle(self, text: str, topk: int = 768, visual=False, save_file=None):
        topk_result = self.embed(text).topk(topk)
        topk_token_ids = topk_result.indices.flatten().tolist()
        topk_token_ids = [x + self.config.shift_vocab_num for x in topk_token_ids if x >= self.config.shift_vocab_num]
        topk_values = topk_result.values.flatten().tolist()
        topk_tokens = self.tokenizer.convert_ids_to_tokens(topk_token_ids)
        results = dict(zip(topk_tokens, topk_values))
        if visual:
            wordcloud_from_dict(results, max_words=topk, save_file=save_file)
        return results

    dst = disentangle

