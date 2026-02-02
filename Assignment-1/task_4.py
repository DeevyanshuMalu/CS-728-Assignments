import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse
import os
from sklearn.metrics import classification_report
from utils import NERDataset, collate_fn, prepare_embeddings_with_unk


# Load data
with open("conll_data/vocab.json", "r") as f:
    vocab_dict = json.load(f)

with open("conll_data/label.json", "r") as f:
    label_dict = json.load(f)

num_classes = len(label_dict)


class NERMLP(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_dim,
        output_dim,
        pretrained_embeddings=None,
    ):
        super(NERMLP, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        # x shape: (total_tokens_in_batch)
        embeds = self.embedding(x)  # (total_tokens_in_batch, embedding_dim)
        logits = self.mlp(embeds)
        return logits


def get_model_path(args):
    os.makedirs("models", exist_ok=True)
    filename = (
        f"mlp_{args.embed_type}_d{args.embed_dim}_h{args.hidden_dim}_"
        f"lr{args.learning_rate}_e{args.epochs}_b{args.batch_size}.pth"
    )
    return os.path.join("models", filename)


def train(args):
    global vocab_dict
    temp_emb = torch.randn(len(vocab_dict), args.embed_dim)
    vocab_dict, temp_emb = prepare_embeddings_with_unk(vocab_dict, temp_emb)

    # Load dataset
    train_dataset = NERDataset("conll_data/train.jsonl", vocab_dict, label_dict)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    # Initialize model and prepare embeddings

    model = NERMLP(
        len(vocab_dict),
        args.embed_dim,
        args.hidden_dim,
        num_classes,
        pretrained_embeddings=temp_emb,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    for epoch in range(args.epochs):
        total_loss = 0
        for i, (tokens, tags) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}"
                )

        print(
            f"Epoch {epoch+1} completed. Average Loss: {total_loss/len(train_loader):.4f}"
        )

    # Save model
    model_path = get_model_path(args)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")


def test(args):
    # Make sure vocab has <UNK> if it was added during training
    global vocab_dict
    if "<UNK>" not in vocab_dict:
        vocab_dict["<UNK>"] = len(vocab_dict)

    # Load dataset
    test_dataset = NERDataset("conll_data/test.jsonl", vocab_dict, label_dict)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    # Initialize model
    model = NERMLP(len(vocab_dict), args.embed_dim, args.hidden_dim, num_classes)
    model_path = get_model_path(args)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found! Please train the model first.")
        return

    model.eval()
    all_preds = []
    all_tags = []

    with torch.no_grad():
        for tokens, tags in test_loader:
            outputs = model(tokens)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_tags.extend(tags.cpu().tolist())

    # Get label names for report
    target_names = [None] * len(label_dict)
    for name, idx in label_dict.items():
        target_names[idx] = name

    print("Performance Evaluation:")
    print(
        classification_report(all_tags, all_preds, target_names=target_names, digits=3)
    )

    print("\nPerformance Evaluation (excluding 'O'):")
    labels_no_o = [idx for name, idx in label_dict.items() if name != "O"]
    target_names_no_o = [target_names[i] for i in labels_no_o]
    print(
        classification_report(
            all_tags,
            all_preds,
            labels=labels_no_o,
            target_names=target_names_no_o,
            digits=3,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NER using MLP")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test"],
        default="train",
        help="Mode: train or test",
    )
    parser.add_argument(
        "--embed_type",
        type=str,
        choices=["glove", "svd"],
        default="glove",
        help="Embedding type",
    )
    parser.add_argument(
        "--embed_dim", type=int, default=100, help="Embedding dimension"
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=128, help="Hidden layer dimension"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
