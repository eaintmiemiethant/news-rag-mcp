import json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ===================== CONFIG =====================

BASE_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp")

RSS_GLOB = BASE_DIR / "data/processed/rss-zero-shot/ingest_date=*/rss_classified.jsonl"

FINETUNED_MODEL_DIR = BASE_DIR / "models/distilbert-multilabel-unified"

LABELS = [
    "hate_speech",
    "misinformation",
    "natural_disaster",
    "political_unrest",
    "health_crisis",
]

NUM_SAMPLES = 300   # keep it fast


# ===================== HELPERS =====================

def load_rss_df():
    files = sorted(glob(str(RSS_GLOB)))
    if not files:
        raise FileNotFoundError(f"No rss_classified.jsonl files found for pattern: {RSS_GLOB}")

    records = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records.append(rec)

    df = pd.DataFrame(records)

    # Build text: title + summary
    title = df.get("title", "")
    summary = df.get("summary", "")
    df["text"] = (title.fillna("") + ". " + summary.fillna("")).str.strip()

    # Create binary labels from zero_* scores
    for lbl in LABELS:
        score_col = f"zero_{lbl}"
        if score_col not in df.columns:
            raise KeyError(f"Expected column '{score_col}' in rss_classified.jsonl but not found.")
        df[lbl] = (df[score_col] >= 0.5).astype(int)

    df = df.dropna(subset=["text"])

    # sample a subset for speed
    # if len(df) > NUM_SAMPLES:
    #     df = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)

    return df


def compute_metrics(y_true, y_pred, prefix=""):
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_label = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        f"{prefix}precision_micro": precision_micro,
        f"{prefix}recall_micro": recall_micro,
        f"{prefix}f1_micro": f1_micro,
        f"{prefix}precision_macro": precision_macro,
        f"{prefix}recall_macro": recall_macro,
        f"{prefix}f1_macro": f1_macro,
    }

    for i, name in enumerate(LABELS):
        metrics[f"{prefix}f1_{name}"] = per_label[2][i]

    return metrics


# ===================== MAIN =====================

def main():
    print("📂 Loading RSS classified data...")
    df = load_rss_df()
    print(f"  Using {len(df)} RSS items")

    texts = df["text"].astype(str).tolist()
    y_true = df[LABELS].astype(int).values  # pseudo ground truth from zero_* scores

    device = "mps" if torch.backends.mps.is_available() else -1
    print(f"🖥  Using device: {device}")

    # ---------- FINETUNED DISTILBERT ----------
    print("\n🧠 Running fine-tuned DistilBERT on RSS sample...")
    tok = AutoTokenizer.from_pretrained(FINETUNED_MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_DIR)

    ft_pipe = pipeline(
        "text-classification",
        model=model,
        tokenizer=tok,
        device=device,
        top_k=None,
        function_to_apply="sigmoid",
    )

    ft_preds = []
    for text in texts:
        out = ft_pipe(text)
        # For multilabel with HF pipeline → [[{label, score}, ...]]
        items = out[0] if isinstance(out, list) else out

        scores = [0.0] * len(LABELS)
        for item in items:
            lbl = item["label"]
            if lbl.startswith("LABEL_"):
                idx = int(lbl.replace("LABEL_", ""))
            else:
                idx = LABELS.index(lbl)
            scores[idx] = float(item["score"])

        ft_preds.append([1 if s >= 0.5 else 0 for s in scores])

    ft_preds = np.array(ft_preds)
    ft_metrics = compute_metrics(y_true, ft_preds, prefix="ft_")

    print("\n===== FINETUNED DISTILBERT ON RSS (vs zero_* labels) =====")
    for k, v in ft_metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
