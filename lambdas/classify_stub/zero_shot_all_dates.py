import json
from pathlib import Path
from transformers import pipeline

# ===========================
# ZERO-SHOT SETTINGS
# ===========================
LABELS = [
    "hate speech",
    "misinformation",
    "natural disaster",
    "political unrest",
    "health crisis",
]

MODEL_NAME = "facebook/bart-large-mnli"

# ===========================
# PATHS
# ===========================
BASE_INPUT_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp/data/raw/rss-clean")
BASE_OUTPUT_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp/data/processed/rss-zero-shot")


def load_jsonl(path: Path):
    """Load a JSONL file."""
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
    return items


def make_text(item):
    title = item.get("title", "") or ""
    summary = item.get("summary", "") or ""
    return f"{title}. {summary}"


def classify_item(text, classifier):
    result = classifier(text, LABELS, multi_label=True)
    return {label.replace(" ", "_"): score for label, score in zip(result["labels"], result["scores"])}


def process_folder(folder: Path, classifier):
    print(f"\n📁 Processing folder: {folder.name}")

    # 1. Find RSS file inside folder
    jsonl_files = list(folder.glob("rss_*.jsonl"))
    if not jsonl_files:
        print(f"⚠️ No RSS file found inside {folder}")
        return

    rss_file = jsonl_files[0]  # pick the only one
    print(f"   Found RSS file: {rss_file.name}")

    # 2. Load items
    items = load_jsonl(rss_file)
    print(f"   Loaded {len(items)} RSS records.")

    # 3. Output folder
    out_dir = BASE_OUTPUT_DIR / folder.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "rss_classified.jsonl"

    print(f"   Output → {out_file}")

    # 4. Classify
    with out_file.open("w", encoding="utf-8") as fout:
        for item in items:
            text = make_text(item)
            scores = classify_item(text, classifier)

            # merge scores into item
            for k, v in scores.items():
                item[f"zero_{k}"] = v

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"   ✅ Completed {folder.name}")


def main():
    print("🚀 Loading zero-shot model...")
    classifier = pipeline("zero-shot-classification", model=MODEL_NAME)

    print(f"🔍 Looking for ingest_date folders in: {BASE_INPUT_DIR}")
    folders = sorted([f for f in BASE_INPUT_DIR.iterdir() if f.is_dir() and "ingest_date=" in f.name])

    if not folders:
        print("❌ No ingest_date folders found!")
        return

    print(f"📦 Found {len(folders)} ingest_date folders:")
    for f in folders:
        print("   -", f.name)

    # Process each ingest folder one by one
    for folder in folders:
        process_folder(folder, classifier)

    print("\n🎉 ALL DONE!")

if __name__ == "__main__":
    main()
