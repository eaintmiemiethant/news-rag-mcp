import numpy as np
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import precision_recall_fscore_support
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)

# ========= CONFIG =========

BASE_DIR = Path("/Users/wailyanpyae/Desktop/news-rag-mcp")
DATA_FILE = BASE_DIR / "data/processed/unified_labels.csv"

MODEL_NAME = "distilbert-base-uncased"  # later: try bert-base-uncased, roberta-base

LABEL_NAMES = [
    "hate_speech",
    "misinformation",
    "natural_disaster",
    "political_unrest",
    "health_crisis",
]

OUTPUT_DIR = BASE_DIR / "models/distilbert-multilabel-unified"


# ========= DATA LOADING =========

def load_and_split():
    ds = load_dataset("csv", data_files=str(DATA_FILE))["train"]

    # 80/10/10 split
    ds_train_val = ds.train_test_split(test_size=0.2, seed=42)
    ds_train = ds_train_val["train"]
    ds_temp = ds_train_val["test"]
    ds_val_test = ds_temp.train_test_split(test_size=0.5, seed=42)
    ds_val = ds_val_test["train"]
    ds_test = ds_val_test["test"]
    return ds_train, ds_val, ds_test


# ========= PREPROCESSING =========

def tokenize_datasets(train_ds, val_ds, test_ds):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    cols_to_keep = ["text"] + LABEL_NAMES

    def trim(ds):
        drop = [c for c in ds.column_names if c not in cols_to_keep]
        return ds.remove_columns(drop) if drop else ds

    # keep only text + label columns
    train_ds = trim(train_ds)
    val_ds = trim(val_ds)
    test_ds = trim(test_ds)

    def preprocess(batch):
        # batch["text"] is a list of arbitrary Python objects (str, float, None, etc.)
        raw_texts = batch["text"]

        # 1) Safely convert everything to string
        texts = []
        for t in raw_texts:
            if t is None:
                texts.append("")  # allow empty, tokenizer can handle ""
            else:
                texts.append(str(t))

        # 2) Tokenize
        enc = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=256,
        )

        # 3) Build label matrix as floats for BCEWithLogitsLoss
        n = len(texts)
        labels = []
        for i in range(n):
            labels.append([
                float(batch["hate_speech"][i]),
                float(batch["misinformation"][i]),
                float(batch["natural_disaster"][i]),
                float(batch["political_unrest"][i]),
                float(batch["health_crisis"][i]),
            ])

        enc["labels"] = labels
        return enc

    # batched=True so preprocess gets dict-of-lists
    train_enc = train_ds.map(preprocess, batched=True)
    val_enc = val_ds.map(preprocess, batched=True)
    test_enc = test_ds.map(preprocess, batched=True)

    return tokenizer, train_enc, val_enc, test_enc


# ========= METRICS =========

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = 1 / (1 + np.exp(-logits))  # sigmoid

    # labels came in as floats; binarize them to 0/1 for metrics
    labels_bin = (labels >= 0.5).astype(int)
    y_pred = (probs >= 0.5).astype(int)

    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        labels_bin, y_pred, average="micro", zero_division=0
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels_bin, y_pred, average="macro", zero_division=0
    )
    per_label = precision_recall_fscore_support(
        labels_bin, y_pred, average=None, zero_division=0
    )

    metrics = {
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
    }

    for i, name in enumerate(LABEL_NAMES):
        metrics[f"precision_{name}"] = per_label[0][i]
        metrics[f"recall_{name}"] = per_label[1][i]
        metrics[f"f1_{name}"] = per_label[2][i]

    return metrics


# ========= MAIN =========

def main():
    print("📂 Loading unified dataset...")
    train_ds, val_ds, test_ds = load_and_split()
    print(f"  Train size: {len(train_ds)}")
    print(f"  Val size:   {len(val_ds)}")
    print(f"  Test size:  {len(test_ds)}")

    print("🔠 Tokenizing...")
    tokenizer, train_enc, val_enc, test_enc = tokenize_datasets(train_ds, val_ds, test_ds)

    print("🧠 Initializing model...")
    num_labels = len(LABEL_NAMES)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        problem_type="multi_label_classification",
    )

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        eval_strategy="epoch",          # for transformers 4.57.1
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,  # a bit smaller for MPS/mac
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_dir=str(OUTPUT_DIR / "logs"),
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_enc,
        eval_dataset=val_enc,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print("🚀 Training...")
    trainer.train()

    print("📏 Evaluating on test set...")
    metrics = trainer.evaluate(test_enc)
    print("\n===== TEST METRICS =====")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\n💾 Saving final model...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("\n✅ Done.")


if __name__ == "__main__":
    main()
