"""
Part 1: Classical Retrieval Methods

Goal:
    - Encode queries and tools independently
    - Compute similarity to retrieve top-k tools
    - Evaluate BM25, msmarco-MiniLM, UAE-large-v1
    - Report Recall@1 and Recall@5
"""

import json
import os
import numpy as np
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from utils import get_queries_and_items


def tokenize_text(text):
    return text.lower().split()


def evaluate_bm25(test_queries, tools):
    tool_names = list(tools.keys())
    tool_descriptions = [tools[name] for name in tool_names]

    tokenized_corpus = [tokenize_text(desc) for desc in tool_descriptions]
    bm25 = BM25Okapi(tokenized_corpus)

    correct_at_1 = 0
    correct_at_5 = 0
    total = len(test_queries)

    for sample in tqdm(test_queries, desc="BM25"):
        query = sample["text"]
        gold_tool_name = sample["gold_tool_name"]

        tokenized_query = tokenize_text(query)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = np.argsort(scores)[::-1]
        ranked_tools = [tool_names[i] for i in ranked_indices]

        if ranked_tools[0] == gold_tool_name:
            correct_at_1 += 1
        if gold_tool_name in ranked_tools[:5]:
            correct_at_5 += 1

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    return recall_at_1, recall_at_5


def evaluate_dense(test_queries, tools, model_name):
    tool_names = list(tools.keys())
    tool_texts = [
        f"tool_id: {name}\ntool description: {tools[name]}" for name in tool_names
    ]

    model = SentenceTransformer(model_name)

    tool_embeddings = model.encode(tool_texts, show_progress_bar=True, batch_size=32)
    tool_embeddings = tool_embeddings / np.linalg.norm(
        tool_embeddings, axis=1, keepdims=True
    )

    correct_at_1 = 0
    correct_at_5 = 0
    total = len(test_queries)

    query_texts = [sample["text"] for sample in test_queries]
    query_embeddings = model.encode(
        query_texts, show_progress_bar=True, batch_size=64
    )
    query_embeddings = query_embeddings / np.linalg.norm(
        query_embeddings, axis=1, keepdims=True
    )

    similarities = query_embeddings @ tool_embeddings.T

    for i, sample in enumerate(tqdm(test_queries, desc=model_name.split("/")[-1])):
        gold_tool_name = sample["gold_tool_name"]
        gold_idx = tool_names.index(gold_tool_name)

        ranked_indices = np.argsort(similarities[i])[::-1]

        if ranked_indices[0] == gold_idx:
            correct_at_1 += 1
        if gold_idx in ranked_indices[:5]:
            correct_at_5 += 1

    recall_at_1 = correct_at_1 / total
    recall_at_5 = correct_at_5 / total
    return recall_at_1, recall_at_5


if __name__ == "__main__":
    train_queries, test_queries, tools = get_queries_and_items()

    print(f"Tools: {len(tools)}, Test queries: {len(test_queries)}")

    results = {}

    # BM25
    print("\n--- BM25 ---")
    r1, r5 = evaluate_bm25(test_queries, tools)
    print(f"Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    results["BM25"] = {"recall@1": r1, "recall@5": r5}

    # msmarco-MiniLM
    print("\n--- msmarco-MiniLM ---")
    r1, r5 = evaluate_dense(
        test_queries, tools, "sentence-transformers/msmarco-MiniLM-L-6-v3"
    )
    print(f"Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    results["msmarco-MiniLM"] = {"recall@1": r1, "recall@5": r5}

    # UAE-large-v1
    print("\n--- UAE-large-v1 ---")
    r1, r5 = evaluate_dense(
        test_queries, tools, "WhereIsAI/UAE-Large-V1"
    )
    print(f"Recall@1: {r1:.4f}, Recall@5: {r5:.4f}")
    results["UAE-large-v1"] = {"recall@1": r1, "recall@5": r5}

    # Save results
    os.makedirs("results/part_1", exist_ok=True)
    with open("results/part_1/results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("\n=== Summary ===")
    print(f"{'Method':<20} {'Recall@1':>10} {'Recall@5':>10}")
    print("-" * 42)
    for method, metrics in results.items():
        print(f"{method:<20} {metrics['recall@1']:>10.4f} {metrics['recall@5']:>10.4f}")

    print(f"\nResults saved to results/part_1/results.json")
