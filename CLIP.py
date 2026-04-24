import os, json, time
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from transformers import (
    CLIPModel,
    CLIPProcessor,
    get_linear_schedule_with_warmup,
    set_seed
)


# Nastavenia
set_seed(42)
IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"


BEST_AUC_SAVE = "clip_best_auc.pt"
MODEL_NAME = "openai/clip-vit-base-patch32"

# Hyperparametre
MAX_LEN = 64
EPOCHS = 8
TRAIN_BS = 32
NUM_WORKERS = 2

LR_CLIP = 1e-6
LR_HEAD = 1e-4
WEIGHT_DECAY = 0.01

# Voľba zariadenia
device = "cuda" if torch.cuda.is_available() else "cpu"


# 1) Dataset s Augmentáciou
class HatefulMemesCLIPDataset(Dataset):
    def __init__(self, df, processor, is_train=False, max_len=64):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.max_len = max_len
        self.is_train = is_train

        # Augmentácia pre tréningové dáta
        if self.is_train:
            self.aug_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.RandomRotation(degrees=5)
            ])
        else:
            self.aug_transform = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = str(row["text"])
        label = int(row["label"])
        img_path = row["img_path"]

        try:
            image = Image.open(img_path).convert("RGB")
        except:
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.aug_transform:
            image = self.aug_transform(image)

        # Spoločné spracovanie textu a obrázka pomocou CLIP processor-a
        enc = self.processor(
            text=[text],
            images=[image],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len
        )

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values": enc["pixel_values"].squeeze(0),
            "labels": torch.tensor(label, dtype=torch.long)
        }


# 2) Advanced Fusion Model

class AdvancedCLIPClassifier(nn.Module):
    def __init__(self, input_dim=2048):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, text_feats, image_feats):
        # Fúzia textových a obrazových príznakov
        fused_feats = torch.cat([
            text_feats,
            image_feats,
            text_feats * image_feats,
            torch.abs(text_feats - image_feats)
        ], dim=1)
        return self.classifier(fused_feats)


# 3) Focal Loss

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction="none")
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


# 4) Evaluate Funkcia

@torch.no_grad()
def evaluate(clip_model, classifier, loader, criterion):
    clip_model.eval()
    classifier.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        att_mask = batch["attention_mask"].to(device)
        pixels = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = clip_model(
            input_ids=input_ids,
            pixel_values=pixels,
            attention_mask=att_mask,
            return_dict=True
        )

        # Získanie embeddingov pre text a obrázok
        text_feats = outputs.text_embeds
        img_feats = outputs.image_embeds

        # L2 normalizácia embeddingov
        text_feats = text_feats / text_feats.norm(p=2, dim=-1, keepdim=True)
        img_feats = img_feats / img_feats.norm(p=2, dim=-1, keepdim=True)

        # Výstup klasifikačnej hlavy
        logits = classifier(text_feats, img_feats)
        loss = criterion(logits, labels)

        # Spojenie výstupov zo všetkých batchov
        total_loss += loss.item() * labels.size(0)
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits).numpy()
    y_true = torch.cat(all_labels).numpy()
    y_pred = np.argmax(logits, axis=1)

    y_prob = torch.softmax(torch.tensor(logits), dim=1).numpy()[:, 1]

    # Výpočet metrík
    auroc = roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    cm = confusion_matrix(y_true, y_pred)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "auroc": auroc,
        "cm": cm,
        "report": classification_report(y_true, y_pred, digits=4, zero_division=0)
    }


# 5) Trénovanie modelu

if __name__ == "__main__":


    def load_df(path):
        data = []
        if not os.path.exists(path):
            print(f" Súbor nenájdený: {path}")
            return pd.DataFrame()
        with open(path, "r") as f:
            for line in f:
                obj = json.loads(line)
                obj["img_path"] = os.path.join(IMG_ROOT, obj["img"].split("/")[-1])
                data.append(obj)
        return pd.DataFrame(data)

    # Načítanie dát
    train_df = load_df(TRAIN_PATH)
    val_df = load_df(VAL_PATH)
    test_df = load_df(TEST_PATH)

    # Inicializácia CLIP processora a datasetov
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    train_ds = HatefulMemesCLIPDataset(train_df, processor, is_train=True, max_len=MAX_LEN)
    val_ds = HatefulMemesCLIPDataset(val_df, processor, is_train=False, max_len=MAX_LEN)
    test_ds = HatefulMemesCLIPDataset(test_df, processor, is_train=False, max_len=MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=TRAIN_BS, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_BS)
    test_loader = DataLoader(test_ds, batch_size=TRAIN_BS)

    # Inicializácia modelov
    clip_model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    classifier = AdvancedCLIPClassifier(input_dim=2048).to(device)

    # Čiastočné odmrazenie
    for param in clip_model.parameters():
        param.requires_grad = False

    # Odomknutie poslednej vrstvy textovej a obrazovej vetvy
    for param in clip_model.vision_model.encoder.layers[-1].parameters():
        param.requires_grad = True
    for param in clip_model.text_model.encoder.layers[-1].parameters():
        param.requires_grad = True

    # Stratová funkcia a Optimizer
    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    class_weights = torch.tensor([1.0, float(neg / pos)], device=device)
    criterion = FocalLoss(weight=class_weights)

    optimizer = torch.optim.AdamW([
        {"params": clip_model.parameters(), "lr": LR_CLIP},
        {"params": classifier.parameters(), "lr": LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    # Mixed precision tréning
    scaler = torch.amp.GradScaler("cuda")

    best_auroc = -1.0

    # tréningový cyklus
    for epoch in range(1, EPOCHS + 1):
        clip_model.train()
        classifier.train()

        pbar = tqdm(train_loader, desc=f"Epocha {epoch}")
        for batch in pbar:
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                outputs = clip_model(
                    input_ids=batch["input_ids"].to(device),
                    pixel_values=batch["pixel_values"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    return_dict=True
                )

                # Získanie embeddingov
                t_feat = outputs.text_embeds
                i_feat = outputs.image_embeds

                # Normalizácia embeddingov
                t_feat = t_feat / t_feat.norm(p=2, dim=-1, keepdim=True)
                i_feat = i_feat / i_feat.norm(p=2, dim=-1, keepdim=True)

                # Klasifikácia
                logits = classifier(t_feat, i_feat)
                loss = criterion(logits, batch["labels"].to(device))

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Validácia na konci epochy
        val_m = evaluate(clip_model, classifier, val_loader, criterion)
        print(
            f"\n Epoch {epoch} | "
            f"Val F1: {val_m['f1']:.4f} | "
            f"Recall: {val_m['recall']:.4f} | "
            f"AUROC: {val_m['auroc']:.4f}"
        )

        # ukladanie podľa AUROC
        if val_m["auroc"] > best_auroc:
            best_auroc = val_m["auroc"]
            torch.save(classifier.state_dict(), BEST_AUC_SAVE)
            print(f" Najlepší model uložený : {BEST_AUC_SAVE}")

    # Test
    print("\n Test")
    if os.path.exists(BEST_AUC_SAVE):
        classifier.load_state_dict(torch.load(BEST_AUC_SAVE))
        test_m = evaluate(clip_model, classifier, test_loader, criterion)

        print("\n" + "="*40)
        print("         FINÁLNY REPORT (TEST)         ")
        print("="*40)
        print(test_m["report"])
        print(f"Finálne AUROC: {test_m['auroc']:.4f}")

        cm = test_m["cm"]
        print("\nMatica :")
        print(cm)
        print("="*40)
    else:
        print("Test nie je možný.")