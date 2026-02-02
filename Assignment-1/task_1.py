import json
import ijson
import torch
import numpy as np
import os
import yaml
from collections import defaultdict
from utils import preprocess

# Loss function
def loss_function(cooc_matrix, word_vectors):
    # Compute the loss based on the co-occurrence matrix and the word vectors
    pass

# Loading parameters
yaml_file = "specifications.yaml"
with open(yaml_file, "r", encoding="utf-8") as f:
    specs = yaml.safe_load(f) or {}

ccnews_path = specs.get("ccnews")
embedding_dimension = specs.get("embedding_dimension")
window_size = specs.get("window_size")
x_max = specs.get("x_max")
alpha = specs.get("alpha")
lr = specs.get("lr")
conll_vocab_path = specs.get("conll_vocab")
epochs = specs.get("epochs")
if not ccnews_path:
    raise ValueError("ccnews path not found in specifications.yaml")

# Encodings
word2id={}
cooc_matrix = defaultdict(float)

# Debugging
with open(ccnews_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    vocab = list(data.keys())
    corpus = list(data.values())

for i in vocab:
    word2id[i] = len(word2id)

if os.path.exists(conll_vocab_path):
    with open(conll_vocab_path, 'r', encoding='utf-8') as f:
        conll_vocab = json.load(f)
else:
    with open(conll_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(word2id, f, ensure_ascii=False, indent=2)
    conll_vocab = word2id

# Preprocessing the set
# with open(ccnews_path, 'r', encoding='utf-8') as f:
#     data = json.load(f)
#     for vocab,corpus in data.items():
#         word2id[corpus[0][0]] = vocab
#     for vocab,corpus in data.items():
#         for item in corpus:
#             idx = item[0]
#             passage = item[1]
#             tokens = preprocess(passage)
#             for i in range(max(idx-window_size,0),min(idx+window_size,len(tokens))):
#                 context_word = tokens[i]
#                 if context_word != tokens[idx]:
#                     cooc_matrix[word2id[tokens[idx]], word2id[context_word]] += 1
    # for item in corpus:
    #     idx, passage = item[0]
    #     print("passage:", len(passage))
    #     tokens = preprocess(passage)
    #     print("tokens:", len(tokens))
    #     print("idx:", idx)
    #     for i in range(max(idx-window_size,0),min(idx+window_size,len(tokens))):
    #         context_word = tokens[i]
    #         if context_word != tokens[idx]:
    #             cooc_matrix[word2int[tokens[idx]], word2int[context_word]] += 1

# Training
# rows, cols, vals = zip(*[(i, j, x) for (i, j), x in cooc_matrix.items()])
# rows = np.array(rows, dtype=np.int32)
# cols = np.array(cols, dtype=np.int32)
# vals = np.array(vals, dtype=np.float32)

# print("rows:", rows[:5])
# print("cols:", cols[:5])
# print("vals:", vals[:5])