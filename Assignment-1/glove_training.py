import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

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
        # Word embeddings (main and context)
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
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for target_ids, context_ids, cooc_counts in pbar:
            target_ids = target_ids.to(device)
            context_ids = context_ids.to(device)
            cooc_counts = cooc_counts.to(device)
            
            # Forward pass
            predictions = model(target_ids, context_ids)
            
            # Compute weights
            weights = weighting_function(cooc_counts, x_max, alpha)
            
            # Compute loss
            loss = glove_loss(predictions, cooc_counts, weights)
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
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
        
        # Average word and context vectors (standard practice)
        embeddings = (word_vecs + context_vecs) / 2
    
    return embeddings

def plot_loss(epoch_losses, embedding_dim, learning_rate, batch_size, x_max, alpha, save_path='loss_plot.png'):
    """Plot loss over epochs with hyperparameters"""
    
    plt.figure(figsize=(12, 7))
    
    # Plot loss
    epochs = range(1, len(epoch_losses) + 1)
    plt.plot(epochs, epoch_losses, 'b-', linewidth=2, marker='o', markersize=4)
    
    # Styling
    plt.xlabel('Epoch', fontsize=12, fontweight='bold')
    plt.ylabel('Average Loss', fontsize=12, fontweight='bold')
    plt.title('GloVe Training Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Add hyperparameters as text box
    hyperparams_text = (
        f'Hyperparameters:\n'
        f'Embedding Dim: {embedding_dim}\n'
        f'Learning Rate: {learning_rate}\n'
        f'Batch Size: {batch_size}\n'
        f'x_max: {x_max}\n'
        f'alpha: {alpha}\n'
        f'Final Loss: {epoch_losses[-1]:.4f}'
    )
    
    # Position text box
    plt.text(0.02, 0.98, hyperparams_text,
             transform=plt.gca().transAxes,
             fontsize=10,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved to {save_path}")
