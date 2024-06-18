"""
This script builds an embedding-based index of sparse disentangled (lexical) retriever. 
The output is a sparse matrix embeddings, saved in .npz format.
The process supports both single and sharded indexing for handling large datasets efficiently.

Example Commands:
- For indexing a small text file into a single index:
    TEXT=/path/to/your/textfile.jsonl
    SAVE=/path/to/save/index.npz
    python -m inference.build_index.sparse_index --text_file=$TEXT --save_file=$SAVE --batch_size=32 --device="cuda"

- For indexing a large text file into multiple sharded indices:
    TEXT=/path/to/your/textfile.jsonl
    NUM_SHARD=8
    SHARD_ID=1
    SAVE=/path/to/save/index${SHARD_ID}.npz
    python -m inference.build_index.sparse_index --text_file=$TEXT --save_file=$SAVE --batch_size=32 --device="cuda" --num_shard=$NUM_SHARD --shard_id=$SHARD_ID
"""

import argparse
import os
import json
import time
import logging
from tqdm import tqdm
from scipy.sparse import save_npz, csr_array, vstack
from src.ir import Retriever

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Builds an embedding-based index from text files using a pre-trained retriever model.")
    parser.add_argument('-p', '--text_file', required=True, help='Path to the text file to be indexed.')
    parser.add_argument('-s', '--save_file', required=True, help='Path where the output .npz index file will be saved.')
    parser.add_argument('-c', '--checkpoint', default='vsearch/vdr-nq', type=str, help='Path to the model checkpoint.')
    parser.add_argument('-bs', '--batch_size', default=32, type=int, help='Number of texts to process in each batch.')
    parser.add_argument('-k', '--topk', type=int, help='Top-k sparsity of sparse lexical retriever (optional, default is the retriever default sparsity).')
    parser.add_argument('-n', '--num_shard', default=1, type=int, help='Total number of shards to divide the text file into.')
    parser.add_argument('-i', '--shard_id', default=1, type=int, help='Specific shard number to process (starts from 0).')
    parser.add_argument('-d', '--device', default="cuda", type=str, help='Device to run the model on (e.g., "cuda" or "cpu").')

    args = parser.parse_args()
    logger.info(args)

    logger.info(f"***** Load Retriever: {args.checkpoint} *****")
    model = Retriever.from_pretrained(args.checkpoint)
    model = model.to(args.device)

    logger.info(f"***** Load Text: {args.text_file} *****")
    logger.info(f"***** Shard {args.shard_id+1}/{args.num_shard} *****")
    texts = [json.loads(l) for l in open(args.text_file, 'r')]
    shard_size = len(texts) // args.num_shard + 1
    text_shard = texts[args.shard_id * shard_size: (args.shard_id+1) * shard_size]

    logger.info("***** Start Indexing *****")
    all_p_csr = []
    indexing_time = 0
    start_time_all = time.time()
    for i in tqdm(range(0, len(text_shard), args.batch_size), desc="Processing Corpus Batches"):
        start_time = time.time()
        batch_texts = text_shard[i:i+args.batch_size]
        batch_p_emb = model.encode_corpus(batch_texts, batch_size=args.batch_size, convert_to_tensor=False)
        indexing_time += time.time() - start_time
        p_csr = csr_array(batch_p_emb)
        all_p_csr.append(p_csr)
    p_emb_csr = vstack(all_p_csr)
    
    logger.info("***** Finish Indexing *****")
    logger.info(f"***** Time for indexing (exclude i/o): {int(indexing_time)} s *****")
    logger.info(f"***** Time for indexing (include i/o): {int(time.time()-start_time_all)} s *****")

    save_dir = os.path.dirname(args.save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 

    save_npz(args.save_file, p_emb_csr)
    sparse_rate = 100 * p_emb_csr.nnz / (p_emb_csr.shape[0] * p_emb_csr.shape[1])
    logger.info(f"***** Index save to: {args.save_file} *****")
    logger.info(f"***** Index matrix shape: {p_emb_csr.shape} *****")
    logger.info(f"***** Index sparsity rate: {sparse_rate:.2f}% *****")
