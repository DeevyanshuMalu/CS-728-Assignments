import json


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
