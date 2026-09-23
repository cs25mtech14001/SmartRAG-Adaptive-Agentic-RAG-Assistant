"""Build the search index.

1. Download the Wikipedia articles listed in data/topics.txt (cached in data/articles.json)
2. Clean and split them into overlapping word chunks
3. Embed chunks with a sentence-transformer and store them in a FAISS index

Usage:  python build_index.py            # uses the article cache if present
        python build_index.py --refresh  # re-download articles
"""
import argparse
import json
import os
import re
import time

import config

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "SmartRAG/1.0 (educational RAG project)"}

_TRAILING_SECTIONS = re.compile(
    r"\n=+\s*(See also|References|External links|Notes|Further reading|Bibliography|Sources)\s*=+",
    re.IGNORECASE,
)
_HEADING = re.compile(r"^=+\s*(.*?)\s*=+\s*$", re.MULTILINE)


def clean_article(text: str) -> str:
    """Drop reference-style trailing sections and turn '== Heading ==' lines into plain text."""
    match = _TRAILING_SECTIONS.search(text)
    if match:
        text = text[: match.start()]
    text = _HEADING.sub(lambda m: f"{m.group(1)}.", text)
    return re.sub(r"\n{2,}", "\n", text).strip()


def chunk_text(text: str, size: int = config.CHUNK_WORDS, overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into chunks of `size` words, each overlapping the previous by `overlap` words."""
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, max(len(words) - overlap, 1), step):
        piece = words[start : start + size]
        if piece:
            chunks.append(" ".join(piece))
    return chunks


def fetch_article(title: str) -> tuple[str, str]:
    import requests

    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": 1,
        "redirects": 1,
        "titles": title,
        "format": "json",
    }
    response = requests.get(WIKI_API, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    page = next(iter(response.json()["query"]["pages"].values()))
    return page.get("title", title), page.get("extract", "")


def load_articles(refresh: bool) -> dict[str, str]:
    if os.path.exists(config.ARTICLES_CACHE) and not refresh:
        with open(config.ARTICLES_CACHE, encoding="utf-8") as f:
            return json.load(f)

    with open(config.TOPICS_FILE, encoding="utf-8") as f:
        titles = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    articles = {}
    for title in titles:
        try:
            real_title, text = fetch_article(title)
        except Exception as exc:  # keep going if one article fails
            print(f"  ! skipped {title}: {exc}")
            continue
        if text:
            articles[real_title] = text
            print(f"  fetched {real_title} ({len(text.split())} words)")
        else:
            print(f"  ! no text for {title}")
        time.sleep(0.2)  # be polite to Wikipedia

    os.makedirs(os.path.dirname(config.ARTICLES_CACHE), exist_ok=True)
    with open(config.ARTICLES_CACHE, "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False)
    return articles


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="re-download articles instead of using the cache")
    args = parser.parse_args()

    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer

    print("Loading articles...")
    articles = load_articles(args.refresh)

    chunks = []
    for title, text in articles.items():
        for piece in chunk_text(clean_article(text)):
            chunks.append({"id": len(chunks), "title": title, "text": piece})
    print(f"{len(articles)} articles -> {len(chunks)} chunks")

    print(f"Embedding with {config.EMBED_MODEL}...")
    embedder = SentenceTransformer(config.EMBED_MODEL)
    vectors = embedder.encode(
        [f"{c['title']}. {c['text']}" for c in chunks],
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])  # inner product on unit vectors = cosine similarity
    index.add(vectors)

    os.makedirs(config.INDEX_DIR, exist_ok=True)
    faiss.write_index(index, os.path.join(config.INDEX_DIR, "chunks.faiss"))
    with open(os.path.join(config.INDEX_DIR, "chunks.json"), "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False)
    with open(os.path.join(config.INDEX_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({"embed_model": config.EMBED_MODEL, "chunks": len(chunks), "articles": len(articles)}, f, indent=2)
    print(f"Saved index to {config.INDEX_DIR}/")


if __name__ == "__main__":
    main()
