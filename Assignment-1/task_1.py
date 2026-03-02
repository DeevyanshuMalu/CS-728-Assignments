import json
import ijson
import torch
import numpy as np
import os
import yaml
from collections import defaultdict
from utils import (
    preprocess,
    save_matrix,
    load_matrix,
    get_neighbors,
    train_glove,
    get_embeddings,
    plot_loss,
    plot_latency,
)
from tqdm import tqdm
from collections import Counter

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
qpath = specs.get("qpath")
epath = specs.get("epath")
ppath = specs.get("ppath")
query_words = specs.get("query_words", [])

if not ccnews_path:
    raise ValueError("ccnews path not found in specifications.yaml")

conll_vocab = {}
cooc_matrix = defaultdict(float)
avg_latency = []

with open(ccnews_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    vocab = list(data.keys())
    corpus = list(data.values())

for i in vocab:
    conll_vocab[i] = len(conll_vocab)

if os.path.exists(conll_vocab_path):
    with open(conll_vocab_path, "r", encoding="utf-8") as f:
        conll_vocab = json.load(f)
else:
    with open(conll_vocab_path, "w", encoding="utf-8") as f:
        json.dump(conll_vocab, f, ensure_ascii=False, indent=2)

for w in window_size:
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
            for idx, passage in values:
                tokens = preprocess(passage)
                if target not in tokens:
                    print("Target:", target)
                    print("Passage:", passage)
                    raise Exception("Target not found in tokens")
                for target_idx in range(len(tokens)):
                    if tokens[target_idx] == target:
                        left = max(target_idx - w, 0)
                        right = min(target_idx + w + 1, len(tokens))
                        context_tokens = (
                            tokens[left:target_idx] + tokens[target_idx + 1 : right]
                        )
                        context_ids = [
                            conll_vocab[w] for w in context_tokens if w in conll_vocab
                        ]
                        for context_id in context_ids:
                            cooc_matrix[target_id, context_id] += 1

        rows, cols, vals = zip(*[(i, j, x) for (i, j), x in cooc_matrix.items()])
        rows = np.array(rows, dtype=np.int32)
        cols = np.array(cols, dtype=np.int32)
        vals = np.array(vals, dtype=np.float32)
        save_matrix(rows, cols, vals, len(conll_vocab), cooc_path)

    for e in embedding_dimension:
        for r in lr:
            model = train_glove(
                rows=rows,
                cols=cols,
                vals=vals,
                vocab_size=len(conll_vocab),
                embedding_dim=e,
                epochs=epochs,
                batch_size=512,
                learning_rate=r,
                x_max=x_max,
                alpha=alpha,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )

            plot_loss(
                model.epoch_losses,
                e,
                r,
                512,
                x_max,
                alpha,
                save_path=ppath + f"loss_plot_{e}_{w}_{r}.png",
            )
            avg_latency.append(np.mean(model.epoch_latencies))
            embeddings = get_embeddings(model)

            np.save(epath + f"glove_embeddings_{e}_{w}_{r}.npy", embeddings)
            np.save("vocab.npy", vocab)

            print(f"Training complete! Embeddings shape: {embeddings.shape}")

            torch.save(model.state_dict(), f"glove_model_{e}_{w}_{r}.pth")
            with open(
                qpath + f"query_embeddings_{e}_{w}_{r}.txt", "w", encoding="utf-8"
            ) as f:
                for q in query_words:
                    if q in conll_vocab:
                        q_idx = conll_vocab[q]
                        q_emb = embeddings[q_idx]
                        f.write(f"{q}: {q_emb.tolist()}\n")
                    else:
                        f.write(f"{q}: Not found in vocab\n")
    plot_latency(
        avg_latency,
        embedding_dimension,
        lr[0],
        save_path=ppath + f"latency_plot_{w}.png",
    )

chosen_embeddings = np.load(epath + f"glove_embeddings_100_3_0.1.npy")
with open("glove_embeddings/query_neighbors.txt", "w", encoding="utf-8") as f:
    for w in query_words:
        neighbors = get_neighbors(chosen_embeddings, w, conll_vocab, k=5)
        f.write(f"{w}: {neighbors}\n")
