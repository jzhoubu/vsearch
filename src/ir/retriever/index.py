import logging
import torch
import json
import mmap
import glob
import numpy as np
from enum import Enum
from tqdm import tqdm
from scipy.sparse import load_npz, csr_array, vstack, save_npz
from typing import Any, Union, List, Optional
from torch import Tensor as T
from typing import Any, Protocol, NamedTuple, List

logger = logging.getLogger(__name__)

class SearchResults(NamedTuple):
    ids: List[int]
    scores: List[float]

class IndexType(Enum):
    DENSE = 'dense'
    SPARSE = 'sparse'
    BAG_OF_TOKEN = 'bag_of_token'

class Index():
    index_type = IndexType.DENSE

    def __init__(self, index_file: Optional[str] = None, data_file: Optional[str] = None, fp16: bool = True, device: str = "cpu", low_memory: bool = False):
        self.data = None
        self.vector = None
        self.low_memory = low_memory
        self.device = device
        self.init_index(index_file, fp16)
        self.load_data(data_file)

    def init_index(self, index_path: Optional[str], fp16: bool = True):
        if index_path:
            files = sorted(glob.glob(files))
            logger.info(f"***** Loading {self.index_type.value} Index from {len(files)} files *****")
            shards = [torch.load(file, map_location=torch.device('cpu')) for file in files]
            vector = vstack(shards) if len(shards) > 1 else shards[0]
            self.vector = vector.astype(torch.float16) if fp16 else vector
            self.vector = self.vector.to(self.device)
            logger.info(f"Index initialized on {self.device}.")
            
    def load_data(self, data_file: Optional[str]):
        if data_file:
            if not self.low_memory:
                self.data = [json.loads(l) for l in open(data_file, 'r')]
            else:
                self.offsets = self._calculate_offsets(data_file)
                self.data_file = data_file

    def move_to_device(self, device: str):
        logger.info(f"Moving index to {device}.")
        self.vector = self.vector.to(device)
        self.device = device

    def _calculate_offsets(self, data_file: str):
        offsets = []
        with open(data_file, 'r') as file:
            offset = 0
            for line in file:
                offsets.append(offset)
                offset += len(line.encode('utf-8'))
        return offsets

    def _find_file_and_line_index(self, global_index: int):
        cumulative_lines = 0
        for i, file_offsets in enumerate(self.offsets):
            num_lines = len(file_offsets)
            if global_index < cumulative_lines + num_lines:
                return i, global_index - cumulative_lines
            cumulative_lines += num_lines

    def _load_line(self, file_path: str, offset: int):
        with open(file_path, 'r') as file:
            file.seek(offset)
            return json.loads(file.readline())

    def get_sample(self, index: int):
        if not self.low_memory:
            return self.data[index]
        else:
            file_index, line_index = self._find_file_and_line_index(index)
            return self._load_line(self.data_files[file_index], self.offsets[file_index][line_index])

    def search(self, q_embs: T, k: int) -> SearchResults:
        q_embs = q_embs.to(self.device).type(self.vector.dtype)
        with torch.no_grad():
            scores = torch.matmul(q_embs, self.vector.t())
        scores_topk = scores.topk(k)
        results = SearchResults(scores_topk.indices, scores_topk.values)        
        return results

    def save(self, path):
        """
        Save the dense vector index to a file in .pt format.

        Parameters:
            path (str): The file path where the index will be saved.
        """
        try:
            vector_to_save = self.vector.cpu()
            torch.save(vector_to_save, path)
            logger.info(f"Index successfully saved to {path}")
        except Exception as e:
            logger.error(f"Failed to save index to {path}: {e}")
            raise

    def __len__(self):
        return len(self.data) if self.data else 0

    def __repr__(self):
        return repr(self.vector)

    def __str__(self):
        info = (
            f'Index Type        : {type(self).__name__}\n'
            f'Vector Shape      : {self.vector.shape}\n'
            f'Vector Dtype      : {self.vector.dtype}\n'
            f'Vector Layout     : {self.vector.layout}\n'
            f'Number of Texts   : {len(self.data) if self.data else 0}\n'
            f'Device            : {self.device}\n'
        )
        return info

class SparseIndex(Index):
    index_type = IndexType.SPARSE

    def __init__(
        self,
        index_file: Optional[str] = None,
        data_file: Optional[str] = None,
        fp16: bool = True,
        device: str = "cpu",
        low_memory: bool = False,
        shift: int = 0,
    ):
        self.shift = shift
        super().__init__(index_file, data_file, fp16, device, low_memory)


    def _scipy_csr_to_torch_csr(self, mat: csr_array) -> T:
        """
        Convert a SciPy CSR matrix to a PyTorch CSR tensor.

        Parameters:
            mat (scipy.sparse.csr_array): The SciPy CSR matrix to convert.

        Returns:
            torch.Tensor: The converted PyTorch CSR tensor on the specified device.
        """
        mat_csr = torch.sparse_csr_tensor(
            torch.from_numpy(mat.indptr),
            torch.from_numpy(mat.indices),
            torch.from_numpy(mat.data),
            size=mat.shape,
        )
        mat_csr = mat_csr.to(self.device)
        return mat_csr

    def init_index(self, index_file: Optional[str], fp16: bool):
        """
        Initialize the index by loading from files and converting to a PyTorch CSR tensor.

        Parameters:
            index_file (Optional[str]): Path pattern to the index files to load.
            fp16 (bool): Whether to convert the index to 16-bit floating point precision.
        """
        if index_file:
            files = sorted(glob.glob(index_file))
            logger.info(f"***** Loading {self.index_type.value} Index from {len(files)} files *****")
            shards = [load_npz(f)[:, self.shift:] for f in files]        
            vector = vstack(shards) if len(shards) > 1 else shards[0]
            vector = vector.astype(np.float16) if fp16 else vector
            logger.info(f"***** Converting Sparse index to torch CSR Tensor *****")
            self.vector = self._scipy_csr_to_torch_csr(vector)
            self.vector = self.vector.to(self.device)

    def save(self, path):
        """
        Save the sparse vector index to a file in .npz format.

        Parameters:
            path (str): The file path where the index will be saved.
        """
        try:
            # Extract components from PyTorch CSR tensor
            indptr = self.vector.crow_indices().cpu().numpy()
            indices = self.vector.col_indices().cpu().numpy()
            data = self.vector.values().cpu().numpy()
            shape = self.vector.size()
            # Convert to a SciPy CSR matrix
            vector_scipy_csr = csr_array((data, indices, indptr), shape=shape)
            # Save in .npz format
            save_npz(path, vector_scipy_csr)
            logger.info(f"Index successfully saved to {path}")
        
        except Exception as e:
            logger.error(f"Failed to save index to {path}: {e}")
            raise


class BoTIndex(SparseIndex):
    index_type = IndexType.BAG_OF_TOKEN

    def __init__(
        self,
        index_file: Optional[str] = None,
        data_file: Optional[str] = None,
        fp16: bool = True,
        device: str = "cpu",
        low_memory: bool = False,
        shift: int = 0,
    ):
        self.shift = shift
        super().__init__(index_file, data_file, fp16, device, low_memory)
        