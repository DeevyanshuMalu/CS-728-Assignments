import json
import torch
import re
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse import load_npz

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def word2features(sent, i):
    word = sent[i]

    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[:3]": word[:3],
        "word[:2]": word[:2],
        "word[:1]": word[:1],
        "word[-3:]": word[-3:],
        "word[-2:]": word[-2:],
        "word[-1:]": word[-1:],
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:word.isdigit()": word1.isdigit(),
            }
        )
    else:
        features["BOS"] = True

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:word.isdigit()": word1.isdigit(),
            }
        )
    else:
        features["EOS"] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(sent_tags):
    return [str(tag) for tag in sent_tags]


def prepare_embeddings_with_unk(vocab, embeddings):
    """
    Ensures <UNK> is in vocab and adds a mean embedding for it if not already present.
    """
    if "<UNK>" not in vocab:
        # Add <UNK> to vocab with a new index
        unk_idx = len(vocab)
        vocab["<UNK>"] = unk_idx

        # Calculate mean embedding
        mean_emb = embeddings.mean(dim=0, keepdim=True)

        # Append mean embedding to the matrix
        embeddings = torch.cat([embeddings, mean_emb], dim=0)
    else:
        print("<UNK> already in vocab")

    return vocab, embeddings


class NERDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocab, label_map):
        self.data = load_jsonl(data_path)
        self.vocab = vocab
        self.label_map = label_map

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        tags = item["tags"]

        token_ids = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
        return torch.tensor(token_ids), torch.tensor(tags)


def collate_fn(batch):
    tokens, tags = zip(*batch)
    return torch.cat(tokens), torch.cat(tags)

# def chunk_text(text, tokenizer, max_length=512):
#     tokens = tokenizer.encode(text)
#     chunks = []
    
#     for i in range(0, len(tokens), max_length):
#         chunk = tokens[i:i + max_length]
#         chunks.append(tokenizer.decode(chunk))
    
#     return chunks

# Process each chunk separately
def preprocess(text):
    tokens = re.split(r'[\s\n]+', text)
    tokens = [t for t in tokens if t]
    return tokens


def cosine_sim(vec1, vec2):
    import numpy as np
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def get_neighbors(embeddings, word, vocab, k=5):
    """Get top-k similar words using cosine similarity"""
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if hasattr(embeddings, 'numpy'):
        embeddings = embeddings.numpy()

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

def save_matrix(rows, cols, vals, vocab_size,path):
    # Create sparse matrix
    cooccurrence_matrix = coo_matrix((vals, (rows, cols)), shape=(vocab_size, vocab_size))
    save_npz(path, cooccurrence_matrix)

# Load
def load_matrix(path):
    loaded_matrix = load_npz(path)
    return loaded_matrix