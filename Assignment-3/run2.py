'''
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query 
'''
import gc
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import argparse
import json 
import time
import pandas as pd
from tqdm import tqdm
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import os
from utils import load_model_tokenizer, PromptUtils, get_queries_and_items

# -------------------------
# Do NOT change
# -------------------------
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) 
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query_to_docs_attention(attentions, query_span, doc_spans):
    """
    attentions: tuple(num_layers) of [1, heads, N, N]
    query_span: (start, end)
    doc_spans: list of (start, end)
    """
    doc_scores = torch.zeros(len(doc_spans), device=attentions[0].device)
    
    # TODO 1: implement to get final query to doc attention stored in doc_scores
    sum_at = torch.zeros(attentions[0][0][0].shape, device=attentions[0].device)
    num_heads = 0
    for layer_at in attentions:
        num_heads += layer_at.shape[1]
        sum_at += layer_at[0].sum(dim=0).cpu()

    avg_at = sum_at / num_heads
    for i, (start, end) in enumerate(doc_spans):
        doc_scores[i] = avg_at[query_span[0]:query_span[1], start:end].sum()

    return doc_scores


def analyze_gold_attention(result, save_path="plot2/gold_attention_plot.png"):
    # TODO 2: visualize graph
    """
    input -> result: list of dicts with keys:
                        - gold_position
                        - gold_score
                        - gold_rank
    GOAL: Using the results data, generate a visualization that shows how attention to the gold tool varies with its position in the prompt.

    Requirements:
        - The plot should clearly illustrate whether position affects attention.
        - Save the plot as an image file under folder plot2.
        - You are free to choose how to aggregate and visualize the data.
    """
    os.makedirs("plot2", exist_ok=True)
    pos = np.array([res["gold_position"] for res in result])
    scores = np.array([res["gold_score"] for res in result])
    ranks = np.array([res["gold_rank"] for res in result])

    plt.figure(figsize=(15, 6))
    plt.scatter(pos, scores, c=ranks, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Rank')
    plt.xlabel('Position')
    plt.ylabel('Attention Score')
    plt.title('Attention to Gold Tool by Position')
    plt.savefig(save_path)
    plt.close()

def get_query_span(input_ids, tokenizer, query):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    query_span = (None, None)
    query_tokens = tokenizer(query, add_special_tokens=False).input_ids
    input_ids_list = input_ids.tolist()

    for i in range(len(input_ids_list) - len(query_tokens) + 1):
        if input_ids_list[i:i+len(query_tokens)] == query_tokens:
            query_span = (i, i + len(query_tokens))
            break

    return query_span


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=64)
parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument('--top_heads', type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == '__main__':
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"
    
    tokenizer, model = load_model_tokenizer(model_name=model_name, device=device, dtype=torch.float16)
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(model.config, "head_dim", model.config.hidden_size // model.config.num_attention_heads)
    num_key_value_groups = num_heads//model.config.num_key_value_heads
    softmax_scaling=d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()
 

    print("---- debug print start ----")
    print(f"seed: {args.seed}, dataset: {args.dataset}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    for qix in tqdm(range(len(test_queries))):
        sample =  test_queries[qix]
        qid = sample["qid"]
        question = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        # --------------------
        # Do Not change the shuffling here
        # --------------------
        num_dbs = len(tools)
        shuffled_keys = list(tools.keys())
        random.shuffle(shuffled_keys)

        putils = PromptUtils(
            tokenizer=tokenizer, 
            doc_ids=shuffled_keys, 
            dict_all_docs=tools,
            )
        item_spans = putils.doc_spans
        doc_lengths = putils.doc_lengths
        map_docname_id = putils.dict_doc_name_id
        map_id_docname = {v:k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)
        
        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors = "pt", add_special_tokens = False).to(device)

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------"*5)
            print(prompt)
            print("-------"*5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0]: item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0]: item_spans[-1][1]]))
            print("-------"*5)


        with torch.no_grad():
            attentions = model(**inputs).attentions
            '''
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            '''
        
        query_span = get_query_span(inputs.input_ids, tokenizer, question)

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank- rank of gold tool in doc_scores
        sorted_indices = torch.argsort(doc_scores, descending=True).cpu().tolist()
        gold_rank = sorted_indices.index(gold_tool_id)+1 if gold_tool_id in sorted_indices else None
        # TODO: find gold_score - score of gold tool
        gold_score = doc_scores[gold_tool_id].item() if gold_tool_id in doc_scores else None
        top1_indices = sorted_indices[:1]
        top5_indices = sorted_indices[:5]
        if gold_tool_id in top1_indices:
            recall_at_1_hits += 1
        if gold_tool_id in top5_indices:
            recall_at_5_hits += 1
        results.append({
            "qid": qid,
            "gold_position": gold_tool_id,
            "gold_score": gold_score,
            "gold_rank": gold_rank
        })

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
    print(f"recall@1: {recall_at_1_hits / len(test_queries):.4f}, recall@5: {recall_at_5_hits / len(test_queries):.4f}")
    analyze_gold_attention(results)

    