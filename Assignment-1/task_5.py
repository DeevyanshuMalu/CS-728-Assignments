import json
import yaml
import numpy as np
import torch
import os
import hashlib
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import preprocess

yaml_file = "specifications.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    specs = yaml.safe_load(f) or {}

ccnews_path = specs.get("ccnews")
conll_vocab_path = specs.get("conll_vocab")
svd_dimensions = specs.get("svd_dimensions")
svd_output_dir = specs.get("svd_output_dir")
tfidf_query_words = specs.get("tfidf_query_words")

if not ccnews_path:
    raise ValueError("ccnews path not found in specifications.yaml")

with open(conll_vocab_path, "r", encoding="utf-8") as f:
    vocab = json.load(f)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

print("Loading CC-News data...")
with open(ccnews_path, "r", encoding="utf-8") as f:
    ccnews_data = json.load(f)

print(f"CC-News entries: {len(ccnews_data)}")

print("Building term-document matrix...")

doc_hashes = {}
doc_count = 0

for word, passages in tqdm(ccnews_data.items(), desc="Counting docs"):
    for doc_idx, passage in passages:
        h = hashlib.md5(passage.encode("utf-8")).hexdigest()
        if h not in doc_hashes:
            doc_hashes[h] = doc_count
            doc_count += 1

num_docs = len(doc_hashes)
print(f"Unique documents: {num_docs}")

Z = lil_matrix((vocab_size, num_docs), dtype=np.float32)

for word, passages in tqdm(ccnews_data.items(), desc="Building matrix"):
    if word not in vocab:
        continue
    word_idx = vocab[word]

    for doc_idx, passage in passages:
        h = hashlib.md5(passage.encode("utf-8")).hexdigest()
        d_idx = doc_hashes[h]
        tokens = preprocess(passage)
        count = tokens.count(word)
        Z[word_idx, d_idx] += count

Z = Z.tocsr()
print(f"Raw matrix shape: {Z.shape}, non-zero: {Z.nnz}")

print("Applying TF-IDF transformation...")
transformer = TfidfTransformer()
Z_tfidf = transformer.fit_transform(Z.T).T
Z_tfidf = Z_tfidf.tocsr()
print(f"TF-IDF matrix shape: {Z_tfidf.shape}, non-zero: {Z_tfidf.nnz}")

os.makedirs(svd_output_dir, exist_ok=True)


def get_neighbors(embeddings, word, vocab, k=5):
    if word not in vocab:
        return []

    word_idx = vocab[word]
    word_vec = embeddings[word_idx].reshape(1, -1)

    sims = cosine_similarity(word_vec, embeddings)[0]

    idx2word = {v: k for k, v in vocab.items()}

    top_idx = np.argsort(sims)[::-1]

    neighbors = []
    for idx in top_idx:
        if idx != word_idx:
            neighbors.append((idx2word[idx], sims[idx]))
            if len(neighbors) >= k:
                break

    return neighbors


for d in svd_dimensions:
    print(f"\n{'='*50}")
    print(f"TF-IDF SVD with d={d}")
    print("=" * 50)

    svd = TruncatedSVD(n_components=d, algorithm="randomized", random_state=42)
    W_tfidf = svd.fit_transform(Z_tfidf)

    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Embeddings shape: {W_tfidf.shape}")

    np.save(f"{svd_output_dir}/svd_tfidf_embeddings_d{d}.npy", W_tfidf)

    torch.save(
        {
            "embeddings": torch.tensor(W_tfidf, dtype=torch.float32),
            "vocab": vocab,
            "dimension": d,
        },
        f"{svd_output_dir}/svd_tfidf_embeddings_d{d}.pt",
    )

    print(f"TF-IDF SVD embeddings saved to {svd_output_dir}/")

    print(f"\nQuality Check 1: Raw SVD vs TF-IDF SVD (top-5 neighbors, d={d})")
    print("-" * 50)

    raw_svd_path = f"{svd_output_dir}/svd_embeddings_d{d}.npy"
    W_raw = np.load(raw_svd_path)

    for word in tfidf_query_words:
        if word not in vocab:
            print(f"\n  '{word}' not in vocab, skipping")
            continue

        raw_neighbors = get_neighbors(W_raw, word, vocab, k=5)
        tfidf_neighbors = get_neighbors(W_tfidf, word, vocab, k=5)

        print(f"\n  {word}:")
        print(f"    {'Raw SVD':<30s} {'TF-IDF SVD':<30s}")
        print(f"    {'-'*28:<30s} {'-'*28:<30s}")
        for i in range(5):
            raw_str = (
                f"{raw_neighbors[i][0]} ({raw_neighbors[i][1]:.3f})"
                if i < len(raw_neighbors)
                else "-"
            )
            tfidf_str = (
                f"{tfidf_neighbors[i][0]} ({tfidf_neighbors[i][1]:.3f})"
                if i < len(tfidf_neighbors)
                else "-"
            )
            print(f"    {i+1}. {raw_str:<28s} {tfidf_str:<28s}")

print(f"\n{'='*50}")
print("Done! TF-IDF SVD embeddings ready for Quality Check 2 (MLP training)")
print(f"Embeddings saved to {svd_output_dir}/")
print("=" * 50)
