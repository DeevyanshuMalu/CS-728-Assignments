import json
import torch
import re
from scipy.sparse import coo_matrix, save_npz
from scipy.sparse import load_npz
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time

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


# Glove training
class GloVeDataset(Dataset):
    def __init__(self, rows, cols, vals):
        self.rows = torch.LongTensor(rows)
        self.cols = torch.LongTensor(cols)
        self.vals = torch.FloatTensor(vals)
        

    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.vals[idx]

class GloVeModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(GloVeModel, self).__init__()
        self.epoch_losses = []
        self.epoch_latencies = []
        self.w_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.w_context = nn.Embedding(vocab_size, embedding_dim)
        
        # Bias terms
        self.w_biases = nn.Embedding(vocab_size, 1)
        self.w_context_biases = nn.Embedding(vocab_size, 1)
        
        # Initialize embeddings with uniform distribution
        self.w_embeddings.weight.data.uniform_(-0.5, 0.5)
        self.w_context.weight.data.uniform_(-0.5, 0.5)
        self.w_biases.weight.data.zero_()
        self.w_context_biases.weight.data.zero_()
        
    def forward(self, target_word, context_word):
        # Get embeddings
        w_embed = self.w_embeddings(target_word)
        w_ctx = self.w_context(context_word)
        
        # Get biases
        w_bias = self.w_biases(target_word).squeeze()
        w_ctx_bias = self.w_context_biases(context_word).squeeze()
        
        # Compute dot product + biases
        prediction = torch.sum(w_embed * w_ctx, dim=1) + w_bias + w_ctx_bias
        
        return prediction

def weighting_function(x, x_max=100, alpha=0.75):
    """GloVe weighting function"""
    weights = torch.where(x < x_max, (x / x_max) ** alpha, torch.ones_like(x))
    return weights

def glove_loss(predictions, targets, weights):
    """GloVe loss function"""
    loss = weights * (predictions - torch.log(targets)) ** 2
    return loss.mean()

def train_glove(rows, cols, vals, vocab_size, embedding_dim=100, 
                epochs=30, batch_size=512, learning_rate=0.05, 
                x_max=100, alpha=0.75, device='cuda' if torch.cuda.is_available() else 'cpu'):
    
    # Create dataset and dataloader
    dataset = GloVeDataset(rows, cols, vals)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    # Initialize model
    model = GloVeModel(vocab_size, embedding_dim).to(device)
    optimizer = optim.Adagrad(model.parameters(), lr=learning_rate)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        start_time = time.time()
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for target_ids, context_ids, cooc_counts in pbar:
            target_ids = target_ids.to(device)
            context_ids = context_ids.to(device)
            cooc_counts = cooc_counts.to(device)
            predictions = model(target_ids, context_ids)
            weights = weighting_function(cooc_counts, x_max, alpha)
            loss = glove_loss(predictions, cooc_counts, weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            model.epoch_latencies.append(time.time() - start_time)
        avg_loss = total_loss / len(dataloader)
        model.epoch_losses.append(avg_loss)
        print(f'Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}')
    
    return model

# After training, get final embeddings
def get_embeddings(model):
    """Get final word embeddings (average of word and context embeddings)"""
    model.eval()
    with torch.no_grad():
        word_vecs = model.w_embeddings.weight.cpu().numpy()
        context_vecs = model.w_context.weight.cpu().numpy()
        embeddings = (word_vecs + context_vecs) 
    
    return embeddings

def plot_loss(epoch_losses, embedding_dim, learning_rate, batch_size, x_max, alpha, save_path='loss_plot.png'):
    """Plot loss over epochs with hyperparameters"""
    
    plt.figure(figsize=(12, 7))
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Loss', fontsize=12, fontweight='bold')
    plt.title('GloVe Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    hyperparams_text = (
        f'Hyperparameters:\n'
        f'Embedding Dim: {embedding_dim}\n'
        f'Learning Rate: {learning_rate}\n'
        f'Batch Size: {batch_size}\n'
        f'x_max: {x_max}\n'
        f'alpha: {alpha}\n'
        f'Final Loss: {epoch_losses[-1]:.4f}'
    )
    plt.text(0.02, 0.98, hyperparams_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_path}")

def plot_latency(latency, embedding_dims, learning_rate, save_path='latency_plot.png'):
    """
    Plot Latency/epoch vs embedding dimension
   """
    plt.figure(figsize=(12, 7))
    plt.plot(embedding_dims, latency, marker='o', linewidth=2, markersize=8, 
             color='steelblue', label=f'LR: {learning_rate}')
    
    plt.xlabel('Embedding Dimension', fontsize=12, fontweight='bold')
    plt.ylabel('Latency/epoch (seconds)', fontsize=12, fontweight='bold')
    plt.title('Latency/epoch vs Embedding Dimension', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=10, loc='best')
    stats_text = (
        f'Statistics:\n'
        f'Min Avg Latency: {min(latency):.2f}s\n'
        f'Max Avg Latency: {max(latency):.2f}s\n'
        f'Dims Tested: {len(embedding_dims)}\n'
        f'LR: {learning_rate}'
    )
    plt.text(0.02, 0.98, stats_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    # plt.show()
    print(f"Latency/epoch plot saved to {save_path}")