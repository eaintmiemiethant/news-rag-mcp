import json
from pathlib import Path
from transformers import pipeline

# ==== Zero-shot LABELS ====
LABELS = [
    "hate speech",
    "misinformation",
    "natural disaster",
    "political unrest",
    "health crisis"
]

MODEL_NAME = "facebook/bart-large-mnli"   # zero-shot NLI model

# ==== RSS FILE ====
INGEST_DATE = "2025-11-08"
RSS_FILE = Path(
    "/Users/wailyanpyae/Desktop/news-rag-mcp/data/raw/rss-clean"
) / f"ingest_date={INGEST_DATE}" / "rss_1762606742.jsonl"
# ==== OUTPUT ====
OUT_FILE = Path(f"/Users/wailyanpyae/Desktop/news-rag-mcp/data/processed/rss-zero-shot/pred_{INGEST_DATE}.jsonl")
OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

def load_rss(path):
    items = []
    with path.open() as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items

def make_text(item):
    title = item.get("title", "") or ""
    summary = item.get("summary", "") or ""
    return f"{title}. {summary}"

def main():
    print("Loading zero-shot classifier...")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

    print(f"Loading RSS from {RSS_FILE}")
    items = load_rss(RSS_FILE)

    print("Running zero-shot predictions...")
    with OUT_FILE.open("w") as fout:
        for it in items:
            text = make_text(it)
            result = classifier(
                text,
                LABELS,
                multi_label=True
            )

            out = {**it}
            for label, score in zip(result["labels"], result["scores"]):
                out[f"zero_{label.replace(' ', '_')}"] = score

            fout.write(json.dumps(out, ensure_ascii=False) + "\n")

    print(f"Saved results to {OUT_FILE}")

if __name__ == "__main__":
    main()
