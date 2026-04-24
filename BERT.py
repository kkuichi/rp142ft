import json, torch
import numpy as np
import pandas as pd
from torch import nn
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding, set_seed
)


# Nastavenia

set_seed(42)


TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

MODEL_NAME = "bert-large-uncased"
OUTPUT_DIR = "bert_results"

MAX_LEN = 128
EPOCHS = 4
TRAIN_BS = 8
LR = 1e-5

# Voľba zariadenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1) Načítanie dát

def load_jsonl_safe(path):

    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            data.append(json.loads(line))

    df = pd.DataFrame(data)
    if 'text' not in df.columns:
        print(f"CHYBA: Súbor {path} neobsahuje stĺpec 'text'!")
        return pd.DataFrame()

    return df

# 2) Dataset a Trainer

class HatefulTextDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length=128):
        texts = df["text"].fillna("").astype(str).tolist()

        # Tokenizácia textov bez dynamického paddingu
        self.encodings = tokenizer(texts, truncation=True, padding=False, max_length=max_length)
        self.labels = df["label"].tolist()

    def __getitem__(self, idx):
        # Vytvorenie jednej vzorky datasetu
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    def __len__(self): return len(self.labels)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # Váhy tried – vyššia váha pre pozitívnu triedu
        weights = torch.tensor([1.0, 1.6], device=device, dtype=logits.dtype)
        # Výpočet loss funkcie
        loss_fct = nn.CrossEntropyLoss(weight=weights)
        return (loss_fct(logits, labels), outputs) if return_outputs else loss_fct(logits, labels)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    # Pravdepodobnosť pozitívnej triedy
    probs = torch.nn.functional.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, probs)
    }


# 3) Tréning

if __name__ == "__main__":
    print(f"baseline modelu: {MODEL_NAME}")

    # Načítanie tréningových, validačných a testovacích dát
    train_df = load_jsonl_safe(TRAIN_PATH)
    val_df   = load_jsonl_safe(VAL_PATH)
    test_df  = load_jsonl_safe(TEST_PATH)

    # Kontrola dát
    if train_df.empty or val_df.empty:
        print("Dáta sú prázdne.")
    else:
        # Načítanie tokenizeru predtrénovaného modelu
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        # Vytvorenie datasetov pre tréning, validáciu a testovanie
        train_ds = HatefulTextDataset(train_df, tokenizer, MAX_LEN)
        val_ds   = HatefulTextDataset(val_df, tokenizer, MAX_LEN)
        test_ds  = HatefulTextDataset(test_df, tokenizer, MAX_LEN)

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)

        # Nastavenie tréningových argumentov
        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="auroc",
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=TRAIN_BS,
            learning_rate=LR,
            fp16=True,
            logging_steps=50,
            report_to="none"
        )

        # Inicializácia traineru
        trainer = WeightedTrainer(
            model=model, args=training_args, train_dataset=train_ds,
            eval_dataset=val_ds, data_collator=DataCollatorWithPadding(tokenizer),
            compute_metrics=compute_metrics
        )

        trainer.train()

        # Ukladanie
        print("\nUkladanie modelu BERT")
        torch.save(model.state_dict(), "bert_large_baseline.pt")
