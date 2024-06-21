"""
This script implements semi-parametric beta search on a pre-built binary token index. 
"""

import logging
import argparse
import json
import os
import torch
from tqdm import tqdm
from scipy.sparse import load_npz, csr_array
from src.ir import Retriever
from src.ir.index.binary_token_index import BinaryTokenIndex

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Argument Parser for Semi-parametric Beta Search')
    parser.add_argument('-c', '--checkpoint', help='Path to the checkpoint directory.')
    parser.add_argument('-q', '--query_file', help='Path to the query jsonl file.')
    parser.add_argument('-p', '--text_file', help='Path to the corpus jsonl file.')
    parser.add_argument('-i', '--index_file', help='Path to binary csr index file')
    parser.add_argument('-s', '--save_file', help='Path to the file to save (.json file).')    

    parser.add_argument('-bs_q', '--batch_size_q', default=32, type=int, help='Batch size of queries for search.')
    parser.add_argument('-a_q', '--activation_q', default=768, type=int, help='Number of activations on query representations.')
    parser.add_argument('-k', '--topk', default=100, type=int, help='Number of retrieved passages.')
    parser.add_argument('-m', '--num_rerank', default=0, type=int, help='Number of passages to re-rank.')
    parser.add_argument('-d', '--device', default="cuda", type=str, help='Device for execution')

    args = parser.parse_args()
    
    logger.info(args)

    logger.info(f"***** Loading Model: {args.checkpoint} *****")
    retriever = Retriever.from_pretrained(args.checkpoint)
    retriever = retriever.to(args.device)

    logger.info(f"***** Loading Query: {args.query_file} *****")
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

    if args.num_rerank > 0:
        logger.info(f"***** Loading Corpus: {args.text_file} *****")
        data = [json.loads(l) for l in open(args.text_file, 'r')]
    else:
        data = None


    logger.info(f"***** Loading Binary Token Index: {args.index_file} *****")
    index = BinaryTokenIndex(index_file=args.index_file, fp16=True, device=args.device)
    index.data = data
    retriever.index = index
    TENSOR_FLAG = args.device!='cpu'
    logger.info(f"***** Start Beta Search *****") 
    num_ret = max(args.topk, args.num_rerank)
    results = {}
    for i in tqdm(range(0, len(queries), args.batch_size_q)):
        questions_batch = queries[i: i+args.batch_size_q]
        q_emb = retriever.encoder_q.embed(questions_batch, batch_size=32, convert_to_tensor=TENSOR_FLAG)
        batch_results = retriever.retrieve(q_emb, a=768, k=num_ret)
        ret_indices, ret_scores = batch_results.ids, batch_results.scores

        if args.num_rerank > 0:
            ret_indices_to_rerank = ret_indices[:, :args.num_rerank]
            ret_texts_to_rerank = [data[i] for i in ret_indices_to_rerank.flatten().tolist()]
            p_embs_to_rerank = retriever.encoder_p.embed(ret_texts_to_rerank, batch_size=32)
            p_embs_to_rerank = p_embs_to_rerank.view(-1, args.num_rerank, q_emb.shape[-1])
            p_embs_to_rerank = p_embs_to_rerank.to(args.device)
            rerank_results = torch.bmm(p_embs_to_rerank, q_emb.unsqueeze(-1).to(args.device)).squeeze()
            rerank_scores, rerank_indices = rerank_results.topk(args.num_rerank)
            ret_indices[:, :args.num_rerank] = ret_indices[torch.arange(ret_indices.size(0)).unsqueeze(1), rerank_indices.cpu()]
            ret_scores[:, :args.num_rerank] = ret_scores[torch.arange(ret_indices.size(0)).unsqueeze(1), rerank_indices.cpu()]
                                
        for j in range(q_emb.shape[0]):
            qid = ids[i+j]
            results[qid] = dict(zip(ret_indices[j].tolist(), ret_scores[j].tolist()))

    logger.info(f"***** Saving results to => {args.save_file} *****")
    save_dir = os.path.dirname(args.save_file)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(results, open(args.save_file, "w"))
