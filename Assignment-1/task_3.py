import argparse
import os
import joblib
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from utils import load_jsonl, sent2features, sent2labels
from collections import Counter


def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))


def print_top_features(args):

    crf = joblib.load(args.model_path)

    print("\nTop positive features:")
    print_state_features(Counter(crf.state_features_).most_common(20))

    print("\nTop negative features:")
    print_state_features(Counter(crf.state_features_).most_common()[-20:])


def train(args):
    print(f"Loading training data from {args.train_data}...")
    train_data = load_jsonl(args.train_data)

    print("Extracting features...")
    X_train = [sent2features(s["tokens"]) for s in train_data]
    y_train = [sent2labels(s["tags"]) for s in train_data]

    print("Training CRF model...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
        verbose=True,
    )
    crf.fit(X_train, y_train)

    print(f"Saving model to {args.model_path}...")
    joblib.dump(crf, args.model_path)
    print("Training complete.")


def test(args):
    if not os.path.exists(args.model_path):
        print(f"Model file {args.model_path} not found!")
        return

    print(f"Loading model from {args.model_path}...")
    crf = joblib.load(args.model_path)

    print(f"Loading test data from {args.test_data}...")
    test_data = load_jsonl(args.test_data)

    X_test = [sent2features(s["tokens"]) for s in test_data]
    y_test = [sent2labels(s["tags"]) for s in test_data]

    print("Predicting...")
    y_pred = crf.predict(X_test)

    labels = list(crf.classes_)
    # Remove '0' (O) if you want to see performance on named entities specifically,
    # but usually we show all.

    print("Performance Evaluation:")
    # We use flatten to get a flat list of labels for standard classification report
    # or use sklearn_crfsuite.metrics.flat_classification_report
    report = metrics.flat_classification_report(y_test, y_pred, labels=labels, digits=3)
    print(report)

    print("\nPerformance Evaluation (excluding 'O'):")
    labels_no_o = [l for l in labels if l != "0"]
    report_no_o = metrics.flat_classification_report(
        y_test, y_pred, labels=labels_no_o, digits=3
    )
    print(report_no_o)

    # Save reports to file
    report_path = args.model_path.replace(".joblib", ".txt")
    with open(report_path, "w") as f:
        f.write("Performance Evaluation:\n")
        f.write(report)
        f.write("\n\nPerformance Evaluation (excluding 'O'):\n")
        f.write(report_no_o)
    print(f"Classification reports saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(description="NER using CRF")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "test", "top_features"],
        default="train",
        help="Mode: train or test",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="crf_model.joblib",
        help="Path to save/load the model",
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="conll_data/train.jsonl",
        help="Path to training data",
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="conll_data/test.jsonl",
        help="Path to test data",
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "test":
        test(args)
    elif args.mode == "top_features":
        print_top_features(args)


if __name__ == "__main__":
    main()
