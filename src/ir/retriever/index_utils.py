import logging
import argparse
import numpy as np
import json
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.sparse import save_npz, csr_array, vstack

def get_first_unique_n(iterable, n):
    """Yields (in order) the first N unique elements of iterable. 
    Might yield less if data too short."""
    seen = set()
    for e in iterable:
        if e in seen:
            continue
        seen.add(e)
        yield e
        if len(seen) == n:
            return