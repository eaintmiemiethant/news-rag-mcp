import os
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Optional: silence tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Paths
HERE = Path(__file__).resolve()
PROJECT_ROOT = HERE.parents[1]  # .../news-rag-mcp/

DATA_PATH = PROJECT_ROOT / "data" / "processed" / "news_eval.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "news_eval_with_summaries.csv"
METRICS_PATH = PROJECT_ROOT / "data" / "processed" / "news_eval_rouge.json"

ARTICLE_COL = "article_text"
GOLD_COL = "gold_summary"
ID_COL = "article_id"  # already present from build_eval_from_rss

# Limit number of examples for quick runs (None = all)
MAX_EXAMPLES = None  # e.g. set to 50 for quick testing

from rag.summarizer import NewsSummariser


def main():
    # ---- Load evaluation data ----
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Evaluation file not found at {DATA_PATH}. "
            f"Create it first (e.g. using scripts/build_eval_from_rss.py)."
        )

    print(f"Loading evaluation data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    # Basic checks
    for col in [ARTICLE_COL, GOLD_COL]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' in {DATA_PATH}. "
                f"Found columns: {df.columns.tolist()}"
            )

    if MAX_EXAMPLES is not None:
        df = df.head(MAX_EXAMPLES)
        print(f"Using first {MAX_EXAMPLES} examples for this run.")

    print(f"Total examples to evaluate: {len(df)}")

    # ---- Load summariser (BART + RAG) ----
    print("Loading NewsSummariser (this may take a moment the first time)...")
    summariser = NewsSummariser()

    vanilla_summaries = []
    rag_summaries = []

    # ---- Generate summaries ----
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating summaries"):
        article_text = str(row[ARTICLE_COL])

        vanilla = summariser.summarize_vanilla(article_text)
        rag = summariser.summarize_rag(article_text)

        vanilla_summaries.append(vanilla)
        rag_summaries.append(rag)

    df["vanilla_summary"] = vanilla_summaries
    df["rag_summary"] = rag_summaries

    # ---- Save full result table ----
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved detailed results to: {OUTPUT_PATH}")

    # ---- Compute ROUGE scores ----
    try:
        import evaluate
    except ImportError:
        print("\nTo compute ROUGE, install 'evaluate' and 'rouge-score':")
        print("    pip install evaluate rouge-score")
        return

    rouge = evaluate.load("rouge")

    gold_list = df[GOLD_COL].astype(str).tolist()
    vanilla_list = df["vanilla_summary"].astype(str).tolist()
    rag_list = df["rag_summary"].astype(str).tolist()

    print("\nComputing ROUGE for VANILLA vs GOLD...")
    rouge_vanilla = rouge.compute(
        predictions=vanilla_list,
        references=gold_list,
        use_stemmer=True,
    )
    print("VANILLA ROUGE:", rouge_vanilla)

    print("\nComputing ROUGE for RAG vs GOLD...")
    rouge_rag = rouge.compute(
        predictions=rag_list,
        references=gold_list,
        use_stemmer=True,
    )
    print("RAG ROUGE:", rouge_rag)

    # ---- Save metrics to JSON ----
    import json

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "vanilla": rouge_vanilla,
                "rag": rouge_rag,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\nSaved ROUGE metrics to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
