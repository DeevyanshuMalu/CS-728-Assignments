"""
Part 2: are we lost in the middle?

Goal:
    - visualize the attention from the query to gold document based on the distance between them
    - use attention as a metric to rank documents for a query
"""

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
from code3 import get_toolwise_attention_scores


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
    selected_heads = [
        (l, h) for l in range(len(attentions)) for h in range(attentions[0].shape[1])
    ]

    toolwise_scores = get_toolwise_attention_scores(
        query_span, doc_spans, attentions
    )  # shape: (num_tools, num_layers, num_heads)
    toolwise_scores_selected = toolwise_scores[
        :, [h[0] for h in selected_heads], [h[1] for h in selected_heads]
    ]  # shape: (num_tools, num_selected_heads) Here num_selected_heads = num_layers * num_heads

    doc_scores = toolwise_scores_selected.sum(dim=1)  # shape: (num_tools,)
    return doc_scores


def analyze_gold_attention(
    result,
    save_path="plot2/gold_attention_plot.png",
    data_save_path="plot2/gold_attention_plot_data.csv",
):
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

    # Aggregate across all queries: one row per gold position.
    df = pd.DataFrame(
        {
            "gold_position": [res["gold_position"] for res in result],
            "gold_score": [res["gold_score"] for res in result],
        }
    )
    grouped = (
        df.groupby("gold_position", as_index=False)
        .agg(
            mean_score=("gold_score", "mean"),
            n=("gold_score", "size"),
        )
        .sort_values("gold_position")
    )

    # Persist the aggregated data used to create the plot.
    grouped.to_csv(data_save_path, index=False)

    x = grouped["gold_position"].to_numpy()
    y = grouped["mean_score"].to_numpy()

    plt.figure(figsize=(15, 6))
    plt.plot(x, y, marker="o", linewidth=2, label="Mean attention")
    plt.xlabel("Gold Tool Position in Prompt")
    plt.ylabel("Mean Attention to Gold Tool")
    plt.title("Aggregated Attention to Gold Tool vs Position")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def get_query_span(query, putils, tokenizer):
    # TODO 3: Query span
    """
    Identify the token span corresponding to the query.
    Note: you are free to add/remove args in this function
    """
    prompt_pre = (
        putils.prompt_prefix
        + putils.all_docs_info_string
        + putils.prompt_seperator
        + putils.add_text1
        + putils.prompt_seperator
        + "Query:"
    )
    prompt_pre_len = len(tokenizer(prompt_pre, add_special_tokens=False).input_ids)

    query_tokens = tokenizer(" " + query, add_special_tokens=False).input_ids
    start = prompt_pre_len
    end = start + len(query_tokens)

    return (start, end)


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=64)
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-1B-Instruct")
parser.add_argument("--top_heads", type=int, default=20)
parser.add_argument("--debug", action="store_true", help="Enable debug mode")
args = parser.parse_args()


if __name__ == "__main__":
    seed_all(seed=args.seed)
    model_name = args.model
    device = "cuda:0"

    tokenizer, model = load_model_tokenizer(
        model_name=model_name, device=device, dtype=torch.float16
    )
    num_heads = model.config.num_attention_heads
    num_layers = model.config.num_hidden_layers
    d = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    num_key_value_groups = num_heads // model.config.num_key_value_heads
    softmax_scaling = d**-0.5
    train_queries, test_queries, tools = get_queries_and_items()

    print("---- debug print start ----")
    print(f"seed: {args.seed}, model: {model_name}")
    print("model.config._attn_implementation: ", model.config._attn_implementation)

    dict_head_freq = {}
    df_data = []
    avg_latency = []
    count = 0
    start_time = time.time()
    results = []
    correct_at_1 = 0
    correct_at_5 = 0
    total = 0

    for qix in tqdm(range(len(test_queries))):
        sample = test_queries[qix]
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
        map_id_docname = {v: k for k, v in map_docname_id.items()}
        db_lengths_pt = torch.tensor(doc_lengths, device=device)

        gold_tool_id = map_docname_id[gold_tool_name]

        prompt = putils.create_prompt(query=question)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(
            device
        )

        if args.debug and qix < 5:
            ip_ids = inputs.input_ids[0].cpu()
            print("-------" * 5)
            print(prompt)
            print("-------" * 5)
            print("---- doc1 ----")
            print(tokenizer.decode(ip_ids[item_spans[0][0] : item_spans[0][1]]))
            print("---- lastdoc ----")
            print(tokenizer.decode(ip_ids[item_spans[-1][0] : item_spans[-1][1]]))
            print("-------" * 5)

        with torch.no_grad():
            attentions = model(**inputs).attentions
            """
                attentions - tuple of length = # layers
                attentions[0].shape - [1, h, N, N] : first layer's attention matrix for h heads
            """

        query_span = get_query_span(question, putils, tokenizer)

        doc_scores = query_to_docs_attention(attentions, query_span, item_spans)

        # TODO: find gold_rank - rank of gold tool in doc_scores
        ranked_docs = torch.argsort(doc_scores, descending=True)
        gold_rank = (ranked_docs == gold_tool_id).nonzero(as_tuple=True)[0].item()
        # TODO: find gold_score - score of gold tool
        gold_score = doc_scores[gold_tool_id].item()
        total += 1

        if gold_rank == 0:
            correct_at_1 += 1
        if gold_rank < 5:
            correct_at_5 += 1
        results.append(
            {
                "qid": qid,
                "gold_position": gold_tool_id,
                "gold_score": gold_score,
                "gold_rank": gold_rank,
            }
        )

        # TODO: calucalte recall@1, recall@5 metric and print at end of loop
    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total

    os.makedirs("results/part_2", exist_ok=True)
    f = open(f"results/part_2/results.json", "w")

    json.dump(
        {
            "recall@1": recall_at_1,
            "recall@5": recall_at_5,
        },
        f,
        indent=4,
    )
    f.close()

    print(f"\nRecall@1 (selected heads): {recall_at_1:.4f}")
    print(f"Recall@5 (selected heads): {recall_at_5:.4f}")

    analyze_gold_attention(results)
