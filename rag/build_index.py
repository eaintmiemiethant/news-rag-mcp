import json
import numpy as np
from pathlib import Path

import faiss
from sentence_transformers import SentenceTransformer

from rag.config import (
    RAG_CORPUS_PATH,
    RAG_INDEX_PATH,
    RAG_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
)

def load_corpus(path: Path):
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            docs.append(json.loads(line))
    return docs

def main():
    print(f"Loading corpus from {RAG_CORPUS_PATH} ...")
    docs = load_corpus(RAG_CORPUS_PATH)
    print(f"Loaded {len(docs)} docs")

    texts = [(doc.get("title", "") + "\n" + doc.get("text", "")).strip() for doc in docs]

    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    print("Encoding corpus...")
    embeddings = embedder.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    # Normalize for cosine similarity
    embeddings = embeddings.astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
    embeddings = embeddings / norms

    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(embeddings)

    RAG_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(RAG_INDEX_PATH))
    print(f"Saved FAISS index to {RAG_INDEX_PATH}")

    # Save docs as metadata in same order
    with open(RAG_METADATA_PATH, "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(json.dumps(doc) + "\n")
    print(f"Saved metadata to {RAG_METADATA_PATH}")

if __name__ == "__main__":
    main()
