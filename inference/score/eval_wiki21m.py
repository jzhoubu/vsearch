import json
import unicodedata
import logging
import numpy as np
import argparse
from tqdm import tqdm
from src.ir.utils.qa_utils import has_answer
import csv

logging.basicConfig(level = logging.INFO)
logger = logging.getLogger()


def normalize(text):
    """Follow a similar normalization as DPR (Dense Passage Retrieval).
    """
    text = unicodedata.normalize("NFD", text)
    text = text.replace("’", "'").replace("\n", " ")
    return text

def parse_qa_csv_file(location):
    res = []
    with open(location) as ifile:
        reader = csv.reader(ifile, delimiter="\t")
        for row in reader:
            question = row[0]
            answers = eval(row[1])
            res.append((question, answers))
    return res

def jsonKeys2int(x):
    if isinstance(x, dict):
        return {int(k):v for k,v in x.items()}
    return x



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Argument Parser for Wiki21m Evaluation based on Search Results.")
    
    # Required arguments
    parser.add_argument('--result_file', type=str, help="Path to search results json file.", required=True)
    parser.add_argument('--text_file', type=str, help="Path to the corpus jsonl file.", required=True)
    parser.add_argument('--qa_file', type=str, help="Path to the QA csv file.", required=True)
    
    # Optional arguments
    # parser.add_argument('-o', '--output_file', type=str, default=None, help="Output file for evaluation results.")
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help="Batch size for evaluation.")
    parser.add_argument('-k', '--k', type=list, default=[1,5,10,20,100], help="List of k values for top-k accuracy evaluation.")

    args = parser.parse_args()
    print(args)

    # Load Results
    results = json.load(open(args.result_file), object_hook=jsonKeys2int)

    # Load Corpus
    texts = [json.loads(l) for l in open(args.text_file, 'r')]
    logger.info(f"***** Load Corpus with {len(texts)} Passages *****")

    # Load QAS
    qas = parse_qa_csv_file(args.qa_file)
    questions = [x[0] for x in qas]
    questions = [normalize(q) for q in questions]
    answers = [x[1] for x in qas]
    answers = [[normalize(a) for a in x] for x in answers]
    logger.info(f"***** Load {len(questions)} Q-A pairs *****")

    # Eval
    max_k = max(args.k)
    acc = np.zeros([len(questions), max_k])
    for i in tqdm(range(len(questions))):
        ret_ids = [x for (x, score) in results[i].items()][:max_k]
        ret_texts = [texts[i] for i in ret_ids]
        answer = answers[i]
        for rank, text in enumerate(ret_texts):
            if has_answer(answer, text, 'string'):
                acc[i][rank] = 1
    for k in args.k:
        logger.info(f"***** Top-{k} acc: {acc[:, :k].max(-1).mean()*100:.2f}% *****")





