"""
This script builds a binary token index of text files. 
The output is a binary sparse matrix, saved in .npz format.

Example Command:
TEXT=/path/to/your/textfile.jsonl
SAVE=/path/to/save/index.npz
python -m inference.build_index.binary_token_index --text_file=$TEXT --save_file=$SAVE --batch_size=32
"""

import logging
import argparse
import numpy as np
import json
import time
import os
from tqdm import tqdm
from transformers import AutoTokenizer
from scipy.sparse import save_npz, csr_array, vstack

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

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
            
def build_bow_store(texts, tokenizer, num_shift=0, batch_size=32, max_len=256, max_token=None):
    N = batch_size
    V = len(tokenizer.vocab)
    batch_bow_emb = np.zeros([N, V])
    all_bow_embs = []
    indexing_time = 0

    for batch_start in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[batch_start: batch_start+batch_size]
        batch_bow_emb[:, :] = 0
        start_time = time.time()
        input_ids = tokenizer(batch_texts, max_length=max_len).input_ids
        for i, row in enumerate(input_ids):
            row = row if not max_token else list(get_first_unique_n(row, max_token))
            batch_bow_emb[i, row] = 1
            # np.add.at(batch_bow_emb[i], row, 1)
            # batch_bow_emb = np.clip(batch_bow_emb, 0, 1)

        if len(batch_texts) < batch_size:
            trimmed_bow_emb = batch_bow_emb[:len(batch_texts), num_shift:]
        else:
            trimmed_bow_emb = batch_bow_emb[:, num_shift:]
        
        indexing_time += time.time() - start_time
        batch_bow_emb_csr = csr_array(trimmed_bow_emb)
        all_bow_embs.append(batch_bow_emb_csr)
    all_bow_emb = vstack(all_bow_embs)
    logger.info("***** Finish Indexing *****")
    logger.info(f"***** Time for indexing (exclude i/o): {int(indexing_time)} s *****")
    return all_bow_emb

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds a binary token index from text files using a tokenizer")
    parser.add_argument('-p', '--text_file', required=True, help='Path to the text file to be indexed.')
    parser.add_argument('-s', '--save_file', required=True, help='Path where the output .npz index file will be saved.')
    parser.add_argument('-id', '--tokenizer_id', default="bert-base-uncased", help='Specifies the tokenizer model ID from Hugging Face model hub. ')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help='Number of texts to process in each batch.')
    parser.add_argument('-l', '--max_len', default=256, type=int, help='Maximum length of token sequences to consider.')
    parser.add_argument('-m', '--max_token', type=int, help='Maximum number of unique tokens to consider per document. If not specified, all tokens are considered.')
    parser.add_argument('-n', '--num_shift', default=0, type=int, help='Used to ignore certain token positions （e.g., unused token）.')

    args = parser.parse_args()
    logger.info(args)

    logger.info(f"***** Load Tokenizer: {args.tokenizer_id} *****")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_id)

    logger.info(f"***** Load Text: {args.text_file} *****")
    texts = [json.loads(l) for l in open(args.text_file, "r")]

    logger.info("***** Start Indexing *****")
    start_time_all = time.time()
    bow_csr_emb = build_bow_store(texts, tokenizer, 
                                  num_shift=args.num_shift, batch_size=args.batch_size, 
                                  max_len=args.max_len, max_token=args.max_token)
    
    logger.info(f"***** Time for indexing (include i/o): {int(time.time()-start_time_all)} s *****")
    save_dir = os.path.dirname(args.save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_npz(args.save_file, bow_csr_emb)
    sparse_rate = 100 * bow_csr_emb.sum() / (bow_csr_emb.shape[0] * bow_csr_emb.shape[1])
    logger.info(f"***** Index save to: {args.save_file} *****")
    logger.info(f"***** Index shape: {bow_csr_emb.shape} *****")
    logger.info(f"***** Index sparsity rate: {sparse_rate:.2f}% *****")
