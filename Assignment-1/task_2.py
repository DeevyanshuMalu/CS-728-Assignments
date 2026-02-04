import json
import yaml
import numpy as np
import torch
import os
import hashlib
from scipy.sparse import lil_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from utils import preprocess

# Loading parameters
yaml_file = "specifications.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    specs = yaml.safe_load(f) or {}

ccnews_path = specs.get("ccnews")
conll_vocab_path = specs.get("conll_vocab")
svd_dimensions = specs.get("svd_dimensions")
svd_output_dir = specs.get("svd_output_dir")
query_words = specs.get("query_words")

if not ccnews_path:
    raise ValueError("ccnews path not found in specifications.yaml")

# Load vocabulary
with open(conll_vocab_path, 'r', encoding='utf-8') as f:
    vocab = json.load(f)

vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# Load ccnews data
print("Loading CC-News data...")
with open(ccnews_path, 'r', encoding='utf-8') as f:
    ccnews_data = json.load(f)

print(f"CC-News entries: {len(ccnews_data)}")

# Build term-document matrix
# Hash passages to get unique documents
print("Building term-document matrix...")

doc_hashes = {}
doc_count = 0

for word, passages in tqdm(ccnews_data.items(), desc="Counting docs"):
    for doc_idx, passage in passages:
        h = hashlib.md5(passage.encode('utf-8')).hexdigest()
        if h not in doc_hashes:
            doc_hashes[h] = doc_count
            doc_count += 1

num_docs = len(doc_hashes)
print(f"Unique documents: {num_docs}")

# Build sparse matrix
Z = lil_matrix((vocab_size, num_docs), dtype=np.float32)

for word, passages in tqdm(ccnews_data.items(), desc="Building matrix"):
    if word not in vocab:
        continue
    word_idx = vocab[word]

    for doc_idx, passage in passages:
        h = hashlib.md5(passage.encode('utf-8')).hexdigest()
        d_idx = doc_hashes[h]
        tokens = preprocess(passage)
        count = tokens.count(word.lower())
        Z[word_idx, d_idx] += count

Z = Z.tocsr()
print(f"Matrix shape: {Z.shape}, non-zero: {Z.nnz}")

# SVD for different dimensions
dimensions = svd_dimensions

os.makedirs(svd_output_dir, exist_ok=True)

def get_neighbors(embeddings, word, vocab, k=5):
    """Get top-k nearest neighbors for a word"""
    if word not in vocab:
        return []

    word_idx = vocab[word]
    word_vec = embeddings[word_idx].reshape(1, -1)

    sims = cosine_similarity(word_vec, embeddings)[0]

    # Build reverse vocab
    idx2word = {v: k for k, v in vocab.items()}

    # Get top indices (excluding self)
    top_idx = np.argsort(sims)[::-1]

    neighbors = []
    for idx in top_idx:
        if idx != word_idx:
            neighbors.append((idx2word[idx], sims[idx]))
            if len(neighbors) >= k:
                break

    return neighbors

# Train and save for each dimension
for d in dimensions:
    print(f"\n{'='*50}")
    print(f"SVD with d={d}")
    print('='*50)

    svd = TruncatedSVD(n_components=d, algorithm='randomized', random_state=42)
    W = svd.fit_transform(Z)

    print(f"Explained variance: {svd.explained_variance_ratio_.sum():.4f}")
    print(f"Embeddings shape: {W.shape}")

    # Save embeddings
    np.save(f'{svd_output_dir}/svd_embeddings_d{d}.npy', W)

    # Also save in torch format for task_4 compatibility
    torch.save({
        'embeddings': torch.tensor(W, dtype=torch.float32),
        'vocab': vocab,
        'dimension': d
    }, f'{svd_output_dir}/svd_embeddings_d{d}.pt')

    # Print nearest neighbors
    print(f"\nNearest neighbors (d={d}):")
    for word in query_words:
        if word not in vocab:
            print(f"  '{word}' not in vocab")
            continue

        neighbors = get_neighbors(W, word, vocab, k=5)
        print(f"\n  {word}:")
        for i, (n, sim) in enumerate(neighbors, 1):
            print(f"    {i}. {n} ({sim:.3f})")

print("\n" + "="*50)
print(f"Done! Embeddings saved to {svd_output_dir}/")
print("="*50)
