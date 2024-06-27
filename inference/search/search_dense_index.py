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
from src.ir import Retriever

logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='Path to the checkpoint directory.')
    parser.add_argument('-q', '--query_file', help='Path to the query jsonl file.')
    parser.add_argument('-i', '--index_file', help='Path to embedding-based index files (support glob expression).')
    parser.add_argument('-s', '--save_file', help='Path to the file to save.')    
    parser.add_argument('-bs_q', '--batch_size_q', default=32, type=int, help='Batch size of queries for embedding.')
    parser.add_argument('-bs_p', '--batch_size_p', default=1000, type=int, help='Batch size of passages for exhaustive search.')
    parser.add_argument('-k', '--topk', default=100, type=int, help='Number of retrieved passages.')
    parser.add_argument('-d', '--device', default="cuda", type=str)
    parser.add_argument('--fp16', default=True, type=bool)

    args = parser.parse_args()

    logger.info(args)

    logger.info(f"### Loading Model: {args.checkpoint} ###")
    retriever = Retriever.from_pretrained(args.checkpoint)
    retriever = retriever.to(args.device)

    logger.info(f"### Loading Query: {args.query_file} ###")
    ids = []
    queries = []
    with open(args.query_file, 'r') as f:
        for i, line in enumerate(f):
            sample = json.loads(line)
            if isinstance(sample, str):
                ids.append(i)
                queries.append(sample)
            elif isinstance(sample, dict):
                ids.append(sample['_id'])
                queries.append(sample['text'])
    q_embs = retriever.encoder_q.embed(queries, batch_size=32)
    q_embs = q_embs.to(args.device)
    q_embs = q_embs.half() if args.fp16 else q_embs
    
    # process index
    files = sorted(glob.glob(args.index_file))
    assert all([file.endswith('.pt') for file in files])
    print(f"### Found {len(files)} Indexes. ###")
    for file in files:
        print(f"***** {file} ###")

    # start search
    print(f"### Start Search on {len(files)} Shards ###")    
    results = {}
    num_processed = 0
    for shard_id, file in enumerate(files):
        print(f"### Searching on Shard {shard_id+1}/{len(files)} ###")
        data_shard = torch.load(file, map_location=torch.device('cpu'))
        data_shard = data_shard.half() if args.fp16 else data_shard
        N = data_shard.shape[0]
        for i in tqdm(range(0, N, args.batch_size_p)):
            batch = data_shard[i: i + args.batch_size_p]
            batch = batch.to(args.device)
            batch_results = q_embs @ batch.t()
            topk_values, topk_indices = batch_results.topk(args.topk)
            topk_values = topk_values.tolist()
            topk_indices = topk_indices.tolist()
            
            num_q = q_embs.shape[0]
            num_batch_p = batch.shape[0]
            for i in range(num_q):
                qid = ids[i]
                results[qid] = results.get(qid, {})
                ret_indices = topk_indices[i]
                ret_scores = topk_values[i]
                for ret_id_in_shard, ret_score in zip(ret_indices, ret_scores):
                    ret_id_global = num_processed + ret_id_in_shard
                    results[qid][ret_id_global] = ret_score
                results[qid] = dict(sorted(results[qid].items(), key=lambda kv: -kv[1])[:args.topk])
                
            num_processed += num_batch_p

    print(f"### Save results to => {args.save_file} ###") 
    save_dir = os.path.dirname(args.save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(results, open(args.save_file, "w"))
