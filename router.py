"""Query-complexity router: predicts which answering mode a question needs.

Backends (chosen automatically from router_model/meta.json):
  logreg      sentence embeddings + logistic regression   (fast, trains on CPU in seconds)
  distilbert  fine-tuned DistilBERT classifier             (stronger, train on a GPU/Colab)
  heuristic   keyword rules, used when no trained model exists
"""
import json
import os
import re

import config

MODES = ("direct", "single", "multi")

# Adaptive-RAG labels questions A (no retrieval), B (single-step), C (multi-step).
_LABEL_MAP = {
    "a": "direct", "b": "single", "c": "multi",
    "0": "direct", "1": "single", "2": "multi",
    "direct": "direct", "no_retrieval": "direct", "zero": "direct", "none": "direct",
    "single": "single", "single_step": "single", "one": "single",
    "multi": "multi", "multi_step": "multi", "iterative": "multi",
}

_MULTI_CUES = re.compile(
    r"\b(and (which|what|how|who|when|where|why)|compare|comparison|difference between|"
    r"differ|both|versus|vs\.?|which one|whereas)\b",
    re.IGNORECASE,
)
_DIRECT_START = re.compile(r"^\s*(what is|what are|what's|define|who is)\b", re.IGNORECASE)


def normalize_label(label) -> str | None:
    return _LABEL_MAP.get(str(label).strip().lower())


def heuristic_route(question: str) -> str:
    words = question.split()
    if _MULTI_CUES.search(question) or len(words) > 20:
        return "multi"
    if re.search(r"\bstand for\b", question, re.IGNORECASE):
        return "direct"
    if _DIRECT_START.match(question) and len(words) <= 6:
        return "direct"
    return "single"


class Router:
    def __init__(self, model_dir: str = config.ROUTER_DIR):
        self.kind = "heuristic"
        meta_path = os.path.join(model_dir, "meta.json")
        if not os.path.exists(meta_path):
            return
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        self.kind = meta["backend"]

        if self.kind == "logreg":
            import joblib
            from sentence_transformers import SentenceTransformer

            self.clf = joblib.load(os.path.join(model_dir, "logreg.joblib"))
            self.embedder = SentenceTransformer(meta["embed_model"])
        elif self.kind == "distilbert":
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.eval()
        else:
            raise ValueError(f"unknown router backend: {self.kind}")

    def predict(self, question: str) -> tuple[str, dict]:
        """Return (mode, {mode: probability})."""
        if self.kind == "heuristic":
            mode = heuristic_route(question)
            return mode, {m: float(m == mode) for m in MODES}

        if self.kind == "logreg":
            vector = self.embedder.encode([question], normalize_embeddings=True)
            probs = dict(zip(self.clf.classes_, self.clf.predict_proba(vector)[0]))
        else:
            import torch

            inputs = self.tokenizer(question, return_tensors="pt", truncation=True, max_length=64)
            with torch.no_grad():
                p = torch.softmax(self.model(**inputs).logits[0], dim=-1).tolist()
            probs = {self.model.config.id2label[i]: p[i] for i in range(len(p))}

        probs = {m: float(probs.get(m, 0.0)) for m in MODES}
        return max(probs, key=probs.get), probs
