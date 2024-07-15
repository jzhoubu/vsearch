import logging
import torch
import json
import mmap
import glob
import numpy as np
from tqdm import tqdm
from scipy.sparse import load_npz, csr_array, vstack
from typing import Any, Union, List
from torch import Tensor as T
from .base import SearchResults, Index

logger = logging.getLogger(__name__)


def scipy_csr_to_torch_csr(mat: np.array, device="cpu"):
    mat_csr = torch.sparse_csr_tensor(
        torch.from_numpy(mat.indptr),
        torch.from_numpy(mat.indices),
        torch.from_numpy(mat.data),
        size=mat.shape,
    )
    mat_csr = mat_csr.to(device)
    return mat_csr



def calculate_offsets(file_path):
    offsets = []
    with open(file_path, 'r') as f:  # 修改这里，使用只读模式
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start = 0
        while True:
            end = mm.find(b'\n', start)
            if end == -1:
                if start < mm.size():  # 处理可能的最后一行
                    offsets.append((start, mm.size()))
                break
            offsets.append((start, end))
            start = end + 1
        mm.close()
    return offsets

def load_line(file_path, offsets, line_number):
    with open(file_path, 'r') as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        start, end = offsets[line_number]
        line = mm[start:end].decode('utf-8')
        mm.close()
    return json.loads(line)

class BinaryTokenIndex(Index):
    """
    Binary token index.
    """
    def __init__(
            self, 
            index_file: str = None,  
            fp16: bool = True, 
            shift: int = 0, 
            device: str = "cpu", 
            data_file: str = None,
            low_memory: bool = False,
            **kwargs):
        self.data = None
        self.index = None
        self.low_memory = low_memory
        self.init_index(index_file, fp16, shift, device)
        if data_file:
            self.load_data(data_file)

    def init_index(self, files, fp16, shift, device):
        files = sorted(glob.glob(files))
        logger.info(f"***** Loading index ({len(files)} shards in total) *****")
        shards = [load_npz(f)[:, shift:] for f in files]
        if len(shards) > 1:
            vector = vstack(shards)
        else:
            vector = shards[0]
        if fp16:
            vector = vector.astype(np.float16)
        logger.info(f"***** Converting index to torch CSR Tensor *****")
        self.vector = scipy_csr_to_torch_csr(vector, device=device)
        self.device = device

    def to_cpu(self):
        self.vector = self.vector.to("cpu")
        self.device = "cpu"

    def to_cuda(self, device="cuda"):
        logger.info(f"***** Moving vectors store to device: {device} *****")
        self.vector = self.vector.to(device)
        self.device = device

    def load_data(self, file):
        if not self.low_memory:
            self.data = [json.loads(l) for l in open(file, 'r')]
        else:
            self.offsets = calculate_offsets(file)
            self.data_file = file

    def get_sample(self, i):
        if not self.low_memory:
            return self.data[i]
        else:
            sample = load_line(self.data_file, self.offsets, i)
            return sample

    def search(self, q_embs: T, k: int) -> SearchResults:
        q_embs = q_embs.to(self.device).type(self.vector.dtype)
        with torch.no_grad():
            scores = torch.matmul(q_embs, self.vector.t())
        scores_topk = scores.topk(k)
        results = SearchResults(scores_topk.indices, scores_topk.values)        
        return results

    def __len__(self):
        return len(self.data) if self.data else 0

    def __repr__(self):
        return repr(self.vector)

    def __str__(self):
        info = (
            f'Index Type      : {type(self).__name__}\n'
            f'Vector Type     : {self.vector.layout}\n'
            f'Vector Shape    : {self.vector.shape}\n'
            f'Vector Device   : {self.device}\n'
            f'Number of Texts : {len(self.data) if self.data else 0}\n'
        )
        return info

