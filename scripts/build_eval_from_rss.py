import json
from pathlib import Path

import pandas as pd

# Project root
ROOT = Path(__file__).resolve().parents[1]

# Input: RAG corpus we already built from rss-clean
RAG_CORPUS_PATH = ROOT / "data" / "processed" / "rag_corpus.jsonl"

# Output: evaluation CSV
OUT_PATH = ROOT / "data" / "processed" / "news_eval.csv"

# How many examples to include in eval (None = use all)
N_EVAL = 200 


def main():
    if not RAG_CORPUS_PATH.exists():
        raise FileNotFoundError(f"RAG corpus not found at {RAG_CORPUS_PATH}. "
                                f"Make sure you've run build_rag_corpus_from_rss first.")

    print(f"Loading RAG corpus from: {RAG_CORPUS_PATH}")
    docs = []

    with open(RAG_CORPUS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            doc_id = obj.get("id")
            title = (obj.get("title") or "").strip()
            text = (obj.get("text") or "").strip()

            # Require at least a title and some text
            if not title or not text:
                continue

            # Article text for the model = title + summary (a bit richer than just summary)
            article_text = f"{title}. {text}"

            # Gold summary for evaluation = title (headline-style)
            gold_summary = title

            docs.append(
                {
                    "article_id": doc_id,
                    "article_text": article_text,
                    "gold_summary": gold_summary,
                }
            )

    print(f"Total valid docs in corpus: {len(docs)}")

    if N_EVAL is not None and len(docs) > N_EVAL:
        docs = docs[:N_EVAL]
        print(f"Using first {N_EVAL} examples for evaluation.")

    df = pd.DataFrame(docs)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved evaluation dataset to: {OUT_PATH}")
    print("Columns:", df.columns.tolist())
    print("\nSample row:")
    print(df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
