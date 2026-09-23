"""Dense retrieval over the FAISS index built by build_index.py."""
import json
import os

import config


class Retriever:
    def __init__(self, index_dir: str = config.INDEX_DIR, embedder=None):
        import faiss
        from sentence_transformers import SentenceTransformer

        index_path = os.path.join(index_dir, "chunks.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"No index at {index_path}. Run: python build_index.py")
        self.index = faiss.read_index(index_path)
        with open(os.path.join(index_dir, "chunks.json"), encoding="utf-8") as f:
            self.chunks = json.load(f)
        self.embedder = embedder or SentenceTransformer(config.EMBED_MODEL)

    def search(self, query: str, k: int = config.TOP_K) -> list[dict]:
        vector = self.embedder.encode([query], normalize_embeddings=True).astype("float32")
        scores, ids = self.index.search(vector, k)
        return [
            {**self.chunks[i], "score": float(s)}
            for s, i in zip(scores[0], ids[0])
            if i != -1
        ]
