import json
import ijson
import torch
import numpy as np
import os
import yaml
from collections import defaultdict
from utils import preprocess, save_matrix, load_matrix
from tqdm import tqdm
from collections import Counter
from glove_training import train_glove, get_embeddings, plot_loss

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
cooc_path = specs.get("cooc_path")
if not ccnews_path:
    raise ValueError("ccnews path not found in specifications.yaml")

# Encodings
conll_vocab={}
cooc_matrix = defaultdict(float)

# Debugging
with open(ccnews_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
    vocab = list(data.keys())
    corpus = list(data.values())

for i in vocab:
    conll_vocab[i] = len(conll_vocab)

if os.path.exists(conll_vocab_path):
    with open(conll_vocab_path, 'r', encoding='utf-8') as f:
        conll_vocab = json.load(f)
else:
    with open(conll_vocab_path, 'w', encoding='utf-8') as f:
        json.dump(conll_vocab, f, ensure_ascii=False, indent=2)

# Preprocessing the set
if cooc_path and os.path.exists(cooc_path):
    cooc_matrix = load_matrix(cooc_path)
    rows = cooc_matrix.row
    cols = cooc_matrix.col
    vals = cooc_matrix.data
else:
    for target, values in tqdm(data.items(), desc="Processing"):
        if target not in conll_vocab:
            raise Exception("Target not found in vocab")
        target_id = conll_vocab[target]
        target = target.lower()
        for idx, passage in values:
            tokens = preprocess(passage)
            if target not in tokens:
                print("Target:", target)
                print("Passage:", passage)
                raise Exception("Target not found in tokens")
            
            target_idx = tokens.index(target)
            left = max(target_idx - window_size, 0)
            right = min(target_idx + window_size + 1, len(tokens))
            context_tokens = tokens[left:target_idx] + tokens[target_idx+1:right]
            context_ids = [conll_vocab[w] for w in context_tokens if w in conll_vocab]
            for context_id in context_ids:
                cooc_matrix[target_id, context_id] += 1

    # Saving Sparse Matrix
    rows, cols, vals = zip(*[(i, j, x) for (i, j), x in cooc_matrix.items()])
    rows = np.array(rows, dtype=np.int32)
    cols = np.array(cols, dtype=np.int32)
    vals = np.array(vals, dtype=np.float32)
    save_matrix(rows, cols, vals, len(conll_vocab),cooc_path)

# Training
model = train_glove(
    rows=rows,
    cols=cols,
    vals=vals,
    vocab_size=len(conll_vocab),
    embedding_dim=embedding_dimension,
    epochs=epochs,
    batch_size=512,
    learning_rate=lr,
    x_max=x_max,
    alpha=alpha,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

plot_loss(model.epoch_losses, embedding_dimension, lr, 512, x_max, alpha, save_path=f'loss_plot_{embedding_dimension}_{window_size}_{lr}.png')
# Get final embeddings
embeddings = get_embeddings(model)

# Save embeddings
np.save(f"glove_embeddings_{embedding_dimension}_{window_size}_{lr}.npy", embeddings)
np.save('vocab.npy', vocab)

print(f"Training complete! Embeddings shape: {embeddings.shape}")

# Save the model
torch.save(model.state_dict(), f'glove_model_{embedding_dimension}_{window_size}_{lr}.pth')