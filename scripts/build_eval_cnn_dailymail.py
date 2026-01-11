from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data" / "processed" / "news_eval_cnn.csv"

N_EVAL = 200  # 50/100/200 are all fine


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "You need the 'datasets' library. Install it with:\n"
            "    pip install datasets"
        )

    print("Loading CNN/DailyMail dataset (test split)...")
    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")

    print(f"Total test examples available: {len(ds)}")
    n = min(N_EVAL, len(ds))
    print(f"Using first {n} examples.")

    articles = ds["article"][:n]
    highlights = ds["highlights"][:n]

    df = pd.DataFrame({
        "article_id": list(range(1, n + 1)),
        "article_text": articles,
        "gold_summary": highlights,
    })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    print(f"Saved CNN/DailyMail eval dataset to: {OUT_PATH}")
    print("Columns:", df.columns.tolist())
    print("\nSample row:")
    print(df.head(1).to_string(index=False))


if __name__ == "__main__":
    main()
