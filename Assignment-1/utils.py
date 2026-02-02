import json
import torch
import re

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

def preprocess(text):
    text = text.lower()
    tokens = re.findall(r"\b[a-z0-9]+\b", text)
    return tokens
