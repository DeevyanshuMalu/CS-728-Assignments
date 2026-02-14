import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import argparse
import os
from sklearn.metrics import classification_report
from utils import NERDataset, collate_fn, prepare_embeddings_with_unk
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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
            # self.embedding.weight.requires_grad = False

        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        embeds = self.embedding(x)
        logits = self.mlp(embeds)
        return logits


def get_model_path(args):
    os.makedirs("models", exist_ok=True)
    filename = (
        f"mlp_{args.embed_type}_d{args.embed_dim}_h{args.hidden_dim}_"
        f"lr{args.learning_rate}_e{args.epochs}_b{args.batch_size}.pth"
    )
    return os.path.join("models", filename)


def train(args, embeddings):
    global vocab_dict
    vocab_dict, embeddings = prepare_embeddings_with_unk(vocab_dict, embeddings)

    train_dataset = NERDataset("conll_data/train.jsonl", vocab_dict, label_dict)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
    )

    val_dataset = NERDataset("conll_data/valid.jsonl", vocab_dict, label_dict)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = NERMLP(
        len(vocab_dict),
        args.embed_dim,
        args.hidden_dim,
        num_classes,
        pretrained_embeddings=embeddings,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for i, (tokens, tags) in enumerate(train_loader):
            tokens, tags = tokens.to(device), tags.to(device)
            optimizer.zero_grad()
            outputs = model(tokens)
            loss = criterion(outputs, tags)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{args.epochs}], Step [{i+1}/{len(train_loader)}], Train Loss: {loss.item():.4f}"
                )

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for tokens, tags in val_loader:
                tokens, tags = tokens.to(device), tags.to(device)
                outputs = model(tokens)
                loss = criterion(outputs, tags)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(
            f"Epoch {epoch+1} completed. Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}"
        )

    model_path = get_model_path(args)
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, args.epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training and Validation Loss ({args.embed_type})")
    plt.legend()
    plt.grid(True)

    plot_path = model_path.replace(".pth", ".png")
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")
    plt.close()


def test(args):
    global vocab_dict
    if "<UNK>" not in vocab_dict:
        vocab_dict["<UNK>"] = len(vocab_dict)

    test_dataset = NERDataset("conll_data/test.jsonl", vocab_dict, label_dict)
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = NERMLP(len(vocab_dict), args.embed_dim, args.hidden_dim, num_classes).to(
        device
    )
    model_path = get_model_path(args)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Model loaded from {model_path}")
    else:
        print(f"Model file {model_path} not found! Please train the model first.")
        return

    model.eval()
    all_preds = []
    all_tags = []

    with torch.no_grad():
        for tokens, tags in test_loader:
            tokens, tags = tokens.to(device), tags.to(device)
            outputs = model(tokens)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().tolist())
            all_tags.extend(tags.cpu().tolist())

    target_names = [None] * len(label_dict)
    for name, idx in label_dict.items():
        target_names[idx] = name

    report1 = classification_report(
        all_tags, all_preds, target_names=target_names, digits=3
    )
    print("Performance Evaluation:")
    print(report1)

    labels_no_o = [idx for name, idx in label_dict.items() if name != "O"]
    target_names_no_o = [target_names[i] for i in labels_no_o]
    report2 = classification_report(
        all_tags,
        all_preds,
        labels=labels_no_o,
        target_names=target_names_no_o,
        digits=3,
    )
    print("\nPerformance Evaluation (excluding 'O'):")
    print(report2)

    report_path = model_path.replace(".pth", ".txt")
    with open(report_path, "w") as f:
        f.write("Performance Evaluation:\n")
        f.write(report1)
        f.write("\n\nPerformance Evaluation (excluding 'O'):\n")
        f.write(report2)
    print(f"Classification reports saved to {report_path}")


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
        choices=["glove", "svd", "svd_tfidf"],
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

    if args.embed_type == "svd":
        embeddings = np.load(f"embeddings/svd_embeddings_d{args.embed_dim}.npy")
        embeddings = normalize(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings).to(device)
        assert args.embed_dim == embeddings.shape[1]
    elif args.embed_type == "svd_tfidf":
        embeddings = np.load(f"embeddings/svd_tfidf_embeddings_d{args.embed_dim}.npy")
        embeddings = normalize(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings).to(device)
        assert args.embed_dim == embeddings.shape[1]
    elif args.embed_type == "glove":
        embeddings = np.load(
            f"glove_embeddings/embeddings/glove_embeddings_{args.embed_dim}_3_0.1.npy"
        )
        embeddings = normalize(embeddings, axis=1)
        embeddings = torch.from_numpy(embeddings).to(device)
        assert args.embed_dim == embeddings.shape[1]

    if args.mode == "train":
        train(args, embeddings)
    elif args.mode == "test":
        test(args)
