import logging
from typing import Tuple, List, Union
from contextlib import nullcontext
from tqdm import tqdm
import torch
from torch import Tensor as T
from transformers import AutoModel, BertConfig, AutoTokenizer, BatchEncoding, PreTrainedModel

logger = logging.getLogger(__name__)

class DPREncoderConfig(BertConfig):
    def __init__(
        self,
        max_len=256,
        model_id='bert-base-uncased',
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.max_len = max_len
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
    
    def encode(
        self,
        texts: Union[List[str], str], 
        max_len: int = None, 
    ) -> BatchEncoding:
        max_len = max_len or self.config.max_len
        texts = [texts] if isinstance(texts, str) else texts
        encoding = self.tokenizer.batch_encode_plus(texts, padding="max_length", truncation=True, max_length=max_len, return_tensors='pt')
        encoding = encoding.to(self.device)
        return encoding


    def embed(
        self, 
        texts: Union[List[str], str], 
        batch_size: int = 128, 
        max_len: int = None, 
        require_grad: bool = False,
        to_cpu: bool = False,
        convert_to_tensor: bool = True,
        show_progress_bar: bool = False,
        **kwargs
    ) -> T:

        max_len = max_len or self.config.max_len
        texts = [texts] if isinstance(texts, str) else texts
        is_training = self.training

        if not require_grad:
            self.eval()
            
        with torch.no_grad() if not require_grad else nullcontext():
            batch_embs = []
            num_text = len(texts)
            iterator = range(0, num_text, batch_size)
            for batch_start in tqdm(iterator) if show_progress_bar else iterator:
                batch_texts = texts[batch_start : batch_start + batch_size]
                encoding = self.encode(batch_texts, max_len=max_len)
                batch_emb = self(**encoding)
                batch_embs.append(batch_emb)
            emb = torch.cat(batch_embs, dim=0)
            if not convert_to_tensor:
                emb = emb.cpu().numpy()
            elif to_cpu:
                emb = emb.cpu()
        
        if is_training:
            self.train()

        return emb
