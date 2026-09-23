"""Train the query-complexity router.

Examples:
  # quick CPU model on the included seed labels
  python train_router.py --backend logreg

  # DistilBERT on Adaptive-RAG's labeled data (+ the seed labels), best run on a GPU / Colab
  python train_router.py --backend distilbert --data path/to/adaptive_rag_train.json router_data/seed_labels.jsonl

Input files may be .json (a list, or {"data": [...]}) or .jsonl. Each row needs a question field
(question/query/input/text) and a label field (answer/label/complexity/mode) with values like
A/B/C (Adaptive-RAG), 0/1/2, or direct/single/multi.
"""
import argparse
import json
import os
import random
from collections import Counter

import config
from router import MODES, normalize_label

QUESTION_KEYS = ("question", "query", "input", "text")
LABEL_KEYS = ("answer", "label", "complexity", "mode")


def read_rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        data = json.load(f)
    return data if isinstance(data, list) else data.get("data", [])


def load_examples(paths: list[str]) -> list[tuple[str, str]]:
    examples = []
    for path in paths:
        kept = 0
        for row in read_rows(path):
            question = next((row[k] for k in QUESTION_KEYS if k in row), None)
            label = next((row[k] for k in LABEL_KEYS if k in row), None)
            if isinstance(label, list):
                label = label[0] if label else None
            mode = normalize_label(label) if label is not None else None
            if isinstance(question, str) and question.strip() and mode:
                examples.append((question.strip(), mode))
                kept += 1
        print(f"{path}: kept {kept} labeled questions")
    return examples


def split(examples, val_fraction=0.15, seed=42):
    from sklearn.model_selection import train_test_split

    questions, labels = zip(*examples)
    counts = Counter(labels)
    stratify = labels if min(counts.values()) >= 2 else None
    return train_test_split(questions, labels, test_size=val_fraction, random_state=seed, stratify=stratify)


def report(y_true, y_pred):
    from sklearn.metrics import accuracy_score, classification_report

    print(f"\nValidation accuracy: {accuracy_score(y_true, y_pred):.3f}")
    print(classification_report(y_true, y_pred, labels=list(MODES), zero_division=0))


def train_logreg(x_train, y_train, x_val, y_val, out_dir):
    import joblib
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    embedder = SentenceTransformer(config.EMBED_MODEL)
    embed = lambda xs: embedder.encode(list(xs), normalize_embeddings=True, show_progress_bar=False)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(embed(x_train), y_train)
    report(y_val, clf.predict(embed(x_val)))

    joblib.dump(clf, os.path.join(out_dir, "logreg.joblib"))
    return {"backend": "logreg", "embed_model": config.EMBED_MODEL}


def train_distilbert(x_train, y_train, x_val, y_val, out_dir, epochs=3, batch_size=32, lr=3e-5):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    base = "distilbert-base-uncased"
    label2id = {m: i for i, m in enumerate(MODES)}
    tokenizer = AutoTokenizer.from_pretrained(base)
    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(MODES), id2label=dict(enumerate(MODES)), label2id=label2id
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    encode = lambda qs: tokenizer(list(qs), padding=True, truncation=True, max_length=64, return_tensors="pt").to(device)

    train = list(zip(x_train, y_train))
    for epoch in range(epochs):
        model.train()
        random.shuffle(train)
        total = 0.0
        for i in range(0, len(train), batch_size):
            qs, ys = zip(*train[i : i + batch_size])
            labels = torch.tensor([label2id[y] for y in ys], device=device)
            loss = model(**encode(qs), labels=labels).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item() * len(qs)
        print(f"epoch {epoch + 1}/{epochs}  loss {total / len(train):.4f}")

    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(x_val), batch_size):
            logits = model(**encode(x_val[i : i + batch_size])).logits
            preds += [MODES[j] for j in logits.argmax(-1).tolist()]
    report(y_val, preds)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {"backend": "distilbert", "base_model": base}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", nargs="+", default=["router_data/seed_labels.jsonl"])
    parser.add_argument("--backend", choices=["logreg", "distilbert"], default="logreg")
    parser.add_argument("--out", default=config.ROUTER_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    random.seed(42)
    examples = load_examples(args.data)
    if len(examples) < 10:
        raise SystemExit("Need at least 10 labeled questions; check your --data files.")
    print(f"Label counts: {dict(Counter(m for _, m in examples))}")

    x_train, x_val, y_train, y_val = split(examples)
    os.makedirs(args.out, exist_ok=True)
    if args.backend == "logreg":
        meta = train_logreg(x_train, y_train, x_val, y_val, args.out)
    else:
        meta = train_distilbert(list(x_train), list(y_train), list(x_val), list(y_val), args.out, epochs=args.epochs)

    meta.update(train_size=len(x_train), val_size=len(x_val), data=args.data)
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved router to {args.out}/")


if __name__ == "__main__":
    main()
