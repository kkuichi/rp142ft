import os, json, torch, numpy as np, pandas as pd
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, set_seed
)



# Nastavenia

set_seed(42)

IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

MODEL_NAME = "cardiffnlp/twitter-roberta-base-hate"

OUTPUT_DIR =  "roberta_hate_model_AUC"
BEST_MODEL_SAVE_DIR = "roberta_final_AUC"

# Voľba zariadenia
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Načítanie dát
def load_text_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            rows.append({
                "text": obj.get("text", ""),
                "label": int(obj.get("label", 0))
            })
    return pd.DataFrame(rows)

class MemeTextDataset(torch.utils.data.Dataset):
    # Dataset pre textovú klasifikáciu
    def __init__(self, df, tokenizer):
        self.encodings = tokenizer(
            df["text"].tolist(),
            truncation=True,
            padding=True,
            max_length=64
        )
        self.labels = df["label"].tolist()

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# 2) Weighted Trainer

class WeightedTrainer(Trainer):
    # Vlastný Trainer s váženou stratovou funkciou
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        weights = torch.tensor([1.0, 1.8], device=logits.device)
        loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss


# 3) Metriky
def compute_metrics(eval_pred):
    # Výpočet metrík počas validácie
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)


    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    f1 = f1_score(labels, preds, zero_division=0)

    try:
        auroc = roc_auc_score(labels, probs)
    except Exception:
        auroc = float("nan")

    return {
        "f1": f1,
        "auroc": auroc
    }


# 4) Tréningový cyklus

# Načítanie tréningových a validačných dát
train_df = load_text_data(TRAIN_PATH)
val_df = load_text_data(VAL_PATH)

# Inicializácia tokenizeru a modelu
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
).to(device)

# Nastavenie tréningových argumentov
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_auroc",
    greater_is_better=True,
    num_train_epochs=10,
    learning_rate=1e-5,
    per_device_train_batch_size=16,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

# Inicializácia traineru
trainer = WeightedTrainer(
    model=model,
    args=args,
    train_dataset=MemeTextDataset(train_df, tokenizer),
    eval_dataset=MemeTextDataset(val_df, tokenizer),
    compute_metrics=compute_metrics
)

print(f"Tréning RoBERTa")
trainer.train()

trainer.save_model(BEST_MODEL_SAVE_DIR)
tokenizer.save_pretrained(BEST_MODEL_SAVE_DIR)

print(f"Tréning dokončený a model uložený do: {BEST_MODEL_SAVE_DIR}")