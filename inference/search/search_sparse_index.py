"""
This script implements parametric search based on a pre-built embedding-based index. 
"""
import json
import torch
import logging
import argparse
import glob
import os
from tqdm import tqdm
from scipy.sparse import load_npz
from src.ir.index.binary_token_index import scipy_csr_to_torch_csr
from src.ir import Retriever
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


def load_queries(query_file):
    ids, queries = [], []
    with open(query_file, 'r') as file:
        for idx, line in enumerate(file):
            sample = json.loads(line)
            if isinstance(sample, str):
                ids.append(idx)
                queries.append(sample)
            elif isinstance(sample, dict):
                ids.append(sample['_id'])
                queries.append(sample['text'])
    return ids, queries

def perform_search(q_embs, index_files, args):
    results = {}
    num_processed = 0
    for shard_id, file_path in enumerate(index_files):
        logger.info(f"***** Loading Index Shard {shard_id+1}/{len(index_files)} *****")
        assert file_path.endswith(".npz")
        shard = load_npz(file_path)

        logger.info(f"***** Searching on Shard {shard_id+1}/{len(index_files)} *****")
        for i in tqdm(range(0, shard.shape[0], args.batch_size_p), desc=f"Processing Shard {shard_id+1}"):
            batch = shard[i: i + args.batch_size_p]
            batch = scipy_csr_to_torch_csr(batch)
            batch = batch.to(args.device)
            batch_results = torch.matmul(q_embs, batch.t())
            topk_values, topk_indices = batch_results.topk(args.topk)
            topk_values = topk_values.tolist()
            topk_indices = topk_indices.tolist()
            update_results(results, ids, num_processed, topk_indices, topk_values)
            num_processed += batch.shape[0]
    return results

def update_results(results, ids, num_processed, topk_indices, topk_values):
    for query_idx in range(len(ids)):
        qid = ids[query_idx]
        results.setdefault(qid, {})
        for idx, score in zip(topk_indices[query_idx], topk_values[query_idx]):
            global_idx = num_processed + idx
            results[qid][global_idx] = score
        results[qid] = dict(sorted(results[qid].items(), key=lambda item: -item[1])[:args.topk])

def get_rows_from_coo(coo_indices, coo_values, num_cols, row_indices):
    # Select the rows that match the given row indices
    mask = torch.isin(coo_indices[0], torch.tensor(row_indices))
    
    selected_indices = coo_indices[:, mask]
    selected_values = coo_values[mask]
    
    # Adjust the row indices to be zero-based for the new tensor
    row_mapping = {old_row: new_row for new_row, old_row in enumerate(row_indices)}
    selected_indices[0] = torch.tensor([row_mapping[row.item()] for row in selected_indices[0]])
    
    # Determine the shape of the new sparse tensor
    new_shape = (len(row_indices), num_cols)
    
    return torch.sparse_coo_tensor(selected_indices, selected_values, new_shape)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Path to the checkpoint directory.')
    parser.add_argument('-q', '--query_file', help='Path to the query jsonl file.')
    parser.add_argument('-p', '--text_file', help='Path to the corpus jsonl file.')
    parser.add_argument('-i', '--index_file', help='Path to embedding-based index files (support glob expression).')
    parser.add_argument('-s', '--save_file', help='Path to the file to save.')    
    parser.add_argument('-bs_q', '--batch_size_q', default=32, type=int, help='Batch size of queries for embedding.')
    parser.add_argument('-bs_p', '--batch_size_p', default=1000, type=int, help='Batch size of passages for exhaustive search.')
    parser.add_argument('-a_q', '--activation_q', default=768, type=int, help='Number of activations on query representations.')
    parser.add_argument('-k', '--topk', default=100, type=int, help='Number of retrieved passages.')
    parser.add_argument('-d', '--device', default="cuda", type=str)
    args = parser.parse_args()

    logger.info(args)

    # Load Retriever
    logger.info(f"***** Loading Model: {args.checkpoint} *****")
    retriever = Retriever.from_pretrained(args.checkpoint)
    retriever = retriever.to(args.device)

    # Load Queries
    logger.info(f"***** Loading Query: {args.query_file} *****")
    ids, queries = load_queries(args.query_file)
    q_embs = retriever.encoder_q.embed(queries, topk=args.activation_q, batch_size=128)
    q_embs = q_embs.to(args.device)

    # Load Index
    index_files = sorted(glob.glob(args.index_file))
    assert all(file.endswith('.npz') or file.endswith('.pt') for file in index_files), "Unsupported file format."
    
    # Search
    results = perform_search(q_embs, index_files, args)
    
    # Save
    if not os.path.exists(os.path.dirname(args.save_file)):
        os.makedirs(os.path.dirname(args.save_file))
    with open(args.save_file, "w") as outfile:
        json.dump(results, outfile)
    logger.info(f"***** Results saved to => {args.save_file} *****")