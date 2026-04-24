import os, json
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import f1_score, accuracy_score, classification_report, roc_auc_score
from transformers import ViTImageProcessor, ViTForImageClassification, Trainer, TrainingArguments, set_seed



# Nastavenia
set_seed(42)


IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

MODEL_NAME = "google/vit-base-patch16-224"
OUTPUT_DIR =  "vit_best_model"
BATCH_SIZE = 16
EPOCHS = 10
LR = 2e-5

device = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Dataset

class ViTDataset(torch.utils.data.Dataset):
    def __init__(self, df, processor):
        self.paths = df["img_path"].tolist()
        self.labels = df["label"].tolist()
        self.processor = processor

    def __len__(self): return len(self.labels)

    def __getitem__(self, idx):
        try: img = Image.open(self.paths[idx]).convert("RGB")
        except: img = Image.new('RGB', (224, 224), (0, 0, 0))
        inputs = self.processor(images=img, return_tensors="pt")
        return {"pixel_values": inputs["pixel_values"].squeeze(0), "labels": torch.tensor(self.labels[idx])}

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            img_name = os.path.basename(obj.get("img", ""))
            rows.append({"label": int(obj["label"]), "img_path": os.path.join(IMG_ROOT, img_name)})
    return pd.DataFrame(rows)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy()
    # threshold experimentálne optimalizovaný
    preds = (probs >= 0.35).astype(int)

    return {
        "f1": f1_score(labels, preds, zero_division=0),
        "acc": accuracy_score(labels, preds),
        "auc": roc_auc_score(labels, probs)
    }

# 2) Weighted trainer

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        weights = torch.tensor([1.0, 1.5], device=model.device)
        loss_fct = nn.CrossEntropyLoss(weight=weights)

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


# 3) Hlavný blok

if __name__ == "__main__":
    train_df = load_jsonl(TRAIN_PATH)
    val_df = load_jsonl(VAL_PATH)

    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    model = ViTForImageClassification.from_pretrained(MODEL_NAME, num_labels=2, ignore_mismatched_sizes=True)

    # Zmrazenie základu
    for param in model.vit.parameters():
        param.requires_grad = False

    # odmrazenie posledných 2 vrstiev Transformera
    for param in model.vit.encoder.layer[-2:].parameters():
        param.requires_grad = True

    # Odmrazenie aj normovacej vrstvy
    for param in model.vit.layernorm.parameters():
        param.requires_grad = True


    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="auc",
        greater_is_better=True,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        remove_unused_columns=False,
        fp16=True if device == "cuda" else False,
        report_to="none"
    )

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=ViTDataset(train_df, processor),
        eval_dataset=ViTDataset(val_df, processor),
        compute_metrics=compute_metrics
    )

    print("Tréning ViT")
    trainer.train()

    trainer.save_model(OUTPUT_DIR)
    processor.save_pretrained(OUTPUT_DIR)
    print(f"Tréning dokončený")