import logging
from typing import Tuple, List
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch import Tensor as T
from transformers import AutoModel, BertConfig, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)

class DPREncoderConfig(BertConfig):
    def __init__(
        self,
        max_seq_len=256,
        pretrained=True,
        model_id='bert-base-uncased',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_seq_len = max_seq_len
        self.pretrained = pretrained
        self.model_id = model_id


class DPREncoder(PreTrainedModel):
    def __init__(self, 
                 config: DPREncoderConfig, 
                 **kwargs
        ):
        super().__init__(config, **kwargs)
        self.config = config
        self.bert_model = AutoModel.from_pretrained(config.model_id, add_pooling_layer=False)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_id)

    def forward(
        self,
        input_ids: T,
        token_type_ids: T,
        attention_mask: T,
    ) -> Tuple[T, ...]:
        
        outputs = self.bert_model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        )
        sequence_output = outputs[0]
        return sequence_output[:, 0, :]

    def encode(self, 
               text: str, 
               max_len: int = 256, 
               training: bool = False,
               **kwargs):
        texts = [text]
        texts_emb = self.batch_encode(texts, batch_size=1, max_len=max_len, training=training)
        text_emb = texts_emb[0]
        return text_emb

    def batch_encode(self, 
                     texts: List[str], 
                     batch_size: int = 128, 
                     max_len: int = 256, 
                     training: bool = False,
                     verbose: bool = False,
                     **kwargs):
        if not training:
            self.eval()

        with torch.no_grad() if not training else nullcontext():
            batch_texts_embs = []
            num_text = len(texts)
            iterator = range(0, num_text, batch_size)
            for batch_start in tqdm(iterator) if verbose else iterator:
                batch_texts = texts[batch_start : batch_start + batch_size]
                batch_encodings = self.tokenizer.batch_encode_plus(
                     batch_texts, 
                     padding="max_length", 
                     truncation=True, 
                     max_length=max_len, 
                     return_tensors='pt'
                    ).to(self.device)
                batch_texts_emb = self(**batch_encodings)
                if not training:
                    batch_texts_emb = batch_texts_emb.cpu()
                batch_texts_embs.append(batch_texts_emb)
            texts_emb = torch.cat(batch_texts_embs, dim=0)

        self.train()
        assert texts_emb.size(0) == len(texts), f"encoded tensor with size: {texts_emb.size} while having {len(texts)} text input"
        return texts_emb

