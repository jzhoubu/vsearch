import logging
import json
import torch
import hydra
import collections
from typing import List, Dict
from omegaconf import DictConfig

logger = logging.getLogger()

BiEncoderPassage = collections.namedtuple(
    "BiEncoderPassage", 
    ["text", "title"]
)

BiEncoderSample = collections.namedtuple(
    "BiEncoderSample",
    [
        "query",
        "answers",
        "positive_passages",
        "negative_passages", 
        "hard_negative_passages",
    ],
)

def _normalize(text):
    text = text.replace("’", "'").replace("\n", " ")
    return text

def create_biencoder_passage(d: Dict, normalize=True):
    text = _normalize(d['text']) if normalize else d['text']
    return BiEncoderPassage(text, d.get('title', None))

class BiEncoderDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        file: str, 
        shuffle_positives: bool = False, 
        norm: bool = True, 
        download_link: str = None
    ):
        super().__init__()
        self.file = file
        self.norm = norm
        self.shuffle_positives = shuffle_positives
        self.download_link = download_link
        self.data = []
        
    def load_data(self, require_positive=True, require_hard_negative=True):
        if self.file.endswith("jsonl"):
            data = [json.loads(sample) for sample in open(self.file, "r")]
        else:
            raise NotImplementedError
        self.data = []
        for sample in data:
            if require_positive and len(sample["positive_ctxs"]) == 0:
                continue
            if require_hard_negative and len(sample["hard_negative_ctxs"]) == 0:
                continue

            sample_query = _normalize(sample["question"]) if self.norm else sample["question"]
            if "answer" in sample:
                sample_answer = sample["answer"]
            elif "answers" in sample:
                sample_answer = sample["answers"]
            else:
                sample_answer = [ctx['text'] for ctx in sample["positive_ctxs"]]
            sample_positive_passages = [create_biencoder_passage(x, self.norm) for x in sample["positive_ctxs"]]
            sample_negative_passages = [create_biencoder_passage(x, self.norm) for x in sample["negative_ctxs"]] if "negative_ctxs" in sample else []
            sample_hard_negative_passages = [create_biencoder_passage(x, self.norm) for x in sample["hard_negative_ctxs"]] if "hard_negative_ctxs" in sample else []
            
            biencoder_sample = BiEncoderSample(sample_query, sample_answer, sample_positive_passages, sample_negative_passages, sample_hard_negative_passages)
            self.data.append(biencoder_sample)

        logger.info("Load data size: {}".format(len(self.data)))

    def __getitem__(self, index) -> BiEncoderSample:
        sample = self.data[index]
        return sample
        
    def __len__(self):
        return len(self.data)
    

class BiencoderDatasetsCfg(object):
    def __init__(self, cfg: DictConfig):
        self.train_datasets_names = cfg.train_datasets
        self.shard_id = cfg.local_rank if cfg.local_rank != -1 else 0
        self.distributed_factor = cfg.distributed_world_size or 1
        self.require_positive = cfg.train.require_positive
        self.require_hard_negative = cfg.train.require_hard_negative
        self.local_shards_dataloader = False
        
        logger.info("train_datasets: %s", self.train_datasets_names)
        if self.train_datasets_names:
            print(cfg.data_stores[self.train_datasets_names[0]])
            self.train_datasets = [hydra.utils.instantiate(cfg.data_stores[ds_name]) for ds_name in self.train_datasets_names]
        else:
            self.train_datasets = []

        if cfg.dev_datasets:
            self.dev_datasets_names = cfg.dev_datasets
            logger.info("dev_datasets: %s", self.dev_datasets_names)
            self.dev_datasets = [hydra.utils.instantiate(cfg.data_stores[ds_name]) for ds_name in self.dev_datasets_names]

        self.sampling_rates = cfg.train.train_sampling_rates
