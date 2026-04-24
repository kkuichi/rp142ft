import os, json, torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)



# Nastavenia


set_seed(42)

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

# Hyperparametre
MODEL_NAME = "google/electra-base-discriminator"
OUTPUT_DIR =  "electra_best_model"
MAX_LEN = 64
EPOCHS = 4
LR = 2e-5
TRAIN_BS = 16 #
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Dataset

def load_jsonl_text(path):
    if not os.path.exists(path):
        print(f"Súbor neexistuje: {path}")
        return pd.DataFrame()
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({"text": obj.get("text", ""), "label": int(obj.get("label", 0))})
    return pd.DataFrame(rows)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=64):
        self.encodings = tokenizer(df["text"].astype(str).tolist(), truncation=True, padding=True, max_length=max_length)
        self.labels = df["label"].tolist()
    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)


# 2) weighted trainer

class WeightedTrainer(Trainer):
    # Vlastný Trainer s váženou CrossEntropyLoss
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = torch.tensor([1.0, 1.6], device=device, dtype=logits.dtype)
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    probs = torch.nn.functional.softmax(torch.tensor(p.predictions), dim=-1)[:, 1].numpy()
    return {
        "accuracy": accuracy_score(p.label_ids, preds),
        "f1": f1_score(p.label_ids, preds, zero_division=0),
        "auroc": roc_auc_score(p.label_ids, probs)
    }


# 3) Tréning

if __name__ == "__main__":
    train_df = load_jsonl_text(TRAIN_PATH)
    val_df   = load_jsonl_text(VAL_PATH)

    # Inicializácia tokenizeru a modelu
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

    # Vytvorenie datasetov
    train_ds = TextDataset(train_df, tokenizer, MAX_LEN)
    val_ds   = TextDataset(val_df, tokenizer, MAX_LEN)

    # Nastavenie tréningových argumentov
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auroc",
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        per_device_train_batch_size=TRAIN_BS,
        fp16=True,
        report_to="none"
    )

    # Inicializácia traineru
    trainer = WeightedTrainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    print("tréning ELECTRA")
    trainer.train()

    # Ukladanie
    model.save_pretrained("electra_final_model")
    print(f" Model uložený ")