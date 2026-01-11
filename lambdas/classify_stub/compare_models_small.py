import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import precision_recall_fscore_support

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

##########################################
# CONFIG
##########################################

BASE_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp")

# SMALL TEST SET (CHANGE THIS!)
TEST_FILE = BASE_DIR / "data/processed/unified_labels.csv"

NUM_SAMPLES = 300    # <<--- fast!

LABELS = [
    "hate_speech",
    "misinformation",
    "natural_disaster",
    "political_unrest",
    "health_crisis",
]

FINETUNED_MODEL = BASE_DIR / "models/distilbert-multilabel-unified"


##########################################
# LOAD SMALL TEST SET
##########################################

df = pd.read_csv(TEST_FILE)

df = df.sample(NUM_SAMPLES, random_state=42).reset_index(drop=True)
texts = df["text"].fillna("").astype(str).tolist()
y_true = df[LABELS].values

print(f"Loaded {len(df)} samples for comparison.")


##########################################
# ZERO-SHOT MODEL: BART MNLI
##########################################

print("Loading zero-shot classifier (BART)...")

zero_shot = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli",
    device="mps" if torch.backends.mps.is_available() else -1
)

zs_preds = []
for text in texts:
    out = zero_shot(
        text,
        candidate_labels=LABELS,
        multi_label=True
    )
    # Convert scores to binary using threshold 0.5
    zs_preds.append([1 if s >= 0.5 else 0 for s in out["scores"]])

zs_preds = torch.tensor(zs_preds).numpy()


##########################################
# FINETUNED DISTILBERT
##########################################

print("Loading fine-tuned model...")

tok = AutoTokenizer.from_pretrained(FINETUNED_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL)

finetune_pipe = pipeline(
    "text-classification",
    model=model,
    tokenizer=tok,
    device="mps" if torch.backends.mps.is_available() else -1,
    top_k=None,
    function_to_apply="sigmoid",
)

ft_preds = []
for text in texts:
    out = finetune_pipe(text)

    # unwrap nested list
    items = out[0] if isinstance(out, list) else out

    scores = [0] * len(LABELS)

    for item in items:
        lbl = item["label"]

        # Handle "LABEL_0" type labels
        if lbl.startswith("LABEL_"):
            idx = int(lbl.replace("LABEL_", ""))
        else:
            idx = LABELS.index(lbl)

        scores[idx] = item["score"]

    ft_preds.append([1 if s >= 0.5 else 0 for s in scores])

ft_preds = torch.tensor(ft_preds).numpy()


##########################################
# METRICS
##########################################

def compute_metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return p, r, f

zs_p, zs_r, zs_f = compute_metrics(y_true, zs_preds)
ft_p, ft_r, ft_f = compute_metrics(y_true, ft_preds)

print("\n==================== RESULTS ====================")
print(f"Zero-shot BART (facebook/bart-large-mnli)")
print(f"  Precision: {zs_p:.4f}")
print(f"  Recall:    {zs_r:.4f}")
print(f"  F1:        {zs_f:.4f}")

print("\nFine-tuned DistilBERT")
print(f"  Precision: {ft_p:.4f}")
print(f"  Recall:    {ft_r:.4f}")
print(f"  F1:        {ft_f:.4f}")

print("\nDONE.")
