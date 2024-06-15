import argparse
import json
from tqdm import tqdm
import torch
from src.vdr.modeling.retriever.retriever import Retriever

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--text_file')
    parser.add_argument('-c', '--checkpoint', default='jzhoubu/dpr-nq')
    parser.add_argument('-bs', '--batch_size', default=256, type=int)
    parser.add_argument('-n', '--num_shard', default=1, type=int)
    parser.add_argument('-i', '--shard_id', default=0, type=int)
    parser.add_argument('-d', '--device', default="cuda", type=str)
    parser.add_argument('-s', '--save_dir', default='./')    
    args = parser.parse_args()
    
    print(args)

    print(f"### Load Model: {args.checkpoint} ###")
    dpr = Retriever.from_pretrained(args.checkpoint)
    dpr = dpr.to(args.device)

    print(f"### Load Text: {args.text_file} ###")
    print(f"### Shard {args.shard_id}/{args.num_shard} ###")
    texts = [json.loads(l) for l in open(args.text_file, 'r')]
    shard_size = len(texts) // args.num_shard
    text_shard = texts[args.shard_id * shard_size : (args.shard_id+1) * shard_size]

    p_embs = []
    for i in tqdm(range(0, len(text_shard), 10000)):
        batch_texts = text_shard[i:i+10000]
        batch_p_emb = dpr.encoder_p.embed(batch_texts, batch_size=args.batch_size)
        batch_p_emb = batch_p_emb.cpu()
        p_embs.append(batch_p_emb)
    
    p_emb = torch.cat(p_embs, dim=0)
    print(p_emb.shape)
    str_len = len(str(args.num_shard))
    shard_id_str = str(args.shard_id).zfill(str_len)
    save_file = f"{args.save_dir}/shard{shard_id_str}.pt"
    print(f"### Save torch tensor to: {save_file} ###")
    torch.save(p_emb, save_file)
    