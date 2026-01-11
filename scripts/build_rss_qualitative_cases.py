import json
from pathlib import Path

import pandas as pd

from tqdm import tqdm

# Project root
ROOT = Path(__file__).resolve().parents[1]

RAG_CORPUS_PATH = ROOT / "data" / "processed" / "rag_corpus.jsonl"
OUT_PATH = ROOT / "data" / "processed" / "rss_qualitative_cases.csv"

# How many RSS examples to generate (for thesis case studies)
N_EXAMPLES = 30  # 20–50 is a good range


# Make rag.* importable
import sys
sys.path.insert(0, str(ROOT))

from rag.summarizer import NewsSummariser


def load_rss_docs(max_examples: int):
    docs = []
    with open(RAG_CORPUS_PATH, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()
            if not title or not text:
                continue

            article_text = f"{title}. {text}"
            docs.append(
                {
                    "article_id": obj.get("id"),
                    "title": title,
                    "article_text": article_text,
                }
            )
            if len(docs) >= max_examples:
                break
    return docs


def main():
    if not RAG_CORPUS_PATH.exists():
        raise FileNotFoundError(f"RAG corpus not found at {RAG_CORPUS_PATH}")

    print(f"Loading up to {N_EXAMPLES} RSS docs from: {RAG_CORPUS_PATH}")
    docs = load_rss_docs(N_EXAMPLES)
    print(f"Loaded {len(docs)} docs for qualitative analysis.")

    summariser = NewsSummariser()

    rows = []

    for doc in tqdm(docs, desc="Generating RSS qualitative summaries"):
        article_text = doc["article_text"]

        vanilla = summariser.summarize_vanilla(article_text)
        # Use debug=True to see context in console, but we also want the titles in CSV
        retrieved_docs = summariser.retriever.retrieve(article_text, top_k=5)
        retrieved_titles = [d.get("title", "") for d in retrieved_docs]

        rag = summariser.summarize_rag(article_text)

        rows.append(
            {
                "article_id": doc["article_id"],
                "title": doc["title"],
                "article_text": article_text,
                "vanilla_summary": vanilla,
                "rag_summary": rag,
                "retrieved_titles": " || ".join(retrieved_titles),
            }
        )

    df = pd.DataFrame(rows)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved RSS qualitative cases to: {OUT_PATH}")
    print("Columns:", df.columns.tolist())
    print("Sample row:")
    print(df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
