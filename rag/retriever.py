import json
from pathlib import Path
from typing import List, Dict

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from rag.config import (
    RAG_INDEX_PATH,
    RAG_METADATA_PATH,
    EMBEDDING_MODEL_NAME,
    TOP_K_RETRIEVAL,
)

class NewsRetriever:
    def __init__(self):
        self.index = faiss.read_index(str(RAG_INDEX_PATH))
        self.docs = []
        with open(RAG_METADATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                self.docs.append(json.loads(line))

        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def retrieve(self, article_text: str, top_k: int = TOP_K_RETRIEVAL) -> List[Dict]:
        query_emb = self.embedder.encode([article_text], convert_to_numpy=True)
        query_emb = query_emb.astype("float32")
        norms = np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-12
        query_emb = query_emb / norms

        distances, indices = self.index.search(query_emb, top_k)
        indices = indices[0].tolist()

        return [self.docs[i] for i in indices]
