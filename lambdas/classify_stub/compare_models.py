import numpy as np
import pandas as pd
from pathlib import Path
from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

# ======== PATHS ========
BASE_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp")
DATA_FILE = BASE_DIR / "data/processed/unified_labels.csv"
FINETUNED_DIR = BASE_DIR / "models/distilbert-multilabel-unified"

LABEL_NAMES = [
    "hate_speech",
    "misinformation",
    "natural_disaster",
    "political_unrest",
    "health_crisis",
]

# ======== HELPERS ========

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
        f"{prefix}_precision_micro": precision_micro,
        f"{prefix}_recall_micro": recall_micro,
        f"{prefix}_f1_micro": f1_micro,
        f"{prefix}_precision_macro": precision_macro,
        f"{prefix}_recall_macro": recall_macro,
        f"{prefix}_f1_macro": f1_macro,
    }

    for i, name in enumerate(LABEL_NAMES):
        metrics[f"{prefix}_f1_{name}"] = per_label[2][i]

    return metrics


# ======== MAIN ========

def main():
    print("📂 Loading dataset...")
    df = pd.read_csv(DATA_FILE)

    texts = df["text"].astype(str).tolist()
    y_true = df[LABEL_NAMES].values

    # ==============================
    # 1. ZERO-SHOT PREDICTIONS
    # ==============================
    print("🔍 Running Zero-shot BART-MNLI...")
    zs_model = pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device="mps:0"
    )

    zs_preds = []
    for t in texts:
        out = zs_model(
            t,
            candidate_labels=LABEL_NAMES,
            multi_label=True
        )
        labels = out["labels"]
        scores = out["scores"]

        # Convert to binary with threshold=0.5
        binary = [1 if scores[labels.index(lbl)] >= 0.5 else 0 for lbl in LABEL_NAMES]
        zs_preds.append(binary)

    zs_preds = np.array(zs_preds)

    zs_metrics = compute_metrics(y_true, zs_preds, prefix="zs")
    print("\n===== ZERO SHOT PERFORMANCE =====")
    for k, v in zs_metrics.items():
        print(f"{k}: {v:.4f}")

    # ==============================
    # 2. FINETUNED MODEL PREDICTIONS
    # ==============================
    print("\n🧠 Loading fine-tuned DistilBERT...")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_DIR)
    clf = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        return_all_scores=True,
        device="mps:0"
    )

    tuned_preds = []
    for t in texts:
        out = clf(t)
        scores = [o["score"] for o in out]  # sigmoid outputs
        binary = [1 if s >= 0.5 else 0 for s in scores]
        tuned_preds.append(binary)

    tuned_preds = np.array(tuned_preds)

    tuned_metrics = compute_metrics(y_true, tuned_preds, prefix="tuned")
    print("\n===== FINETUNED MODEL PERFORMANCE =====")
    for k, v in tuned_metrics.items():
        print(f"{k}: {v:.4f}")

    # ==============================
    # 3. COMPARISON TABLE
    # ==============================
    print("\n===== COMPARISON =====")
    for name in ["f1_micro", "f1_macro"] + [f"f1_{x}" for x in LABEL_NAMES]:
        zs_val = zs_metrics[f"zs_{name}"]
        tuned_val = tuned_metrics[f"tuned_{name}"]
        diff = tuned_val - zs_val

        print(f"{name:20s}  ZS={zs_val:.4f}   FT={tuned_val:.4f}   Δ={diff:+.4f}")


if __name__ == "__main__":
    main()
