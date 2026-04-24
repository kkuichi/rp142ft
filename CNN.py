import os, json, torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


# Nastavenia

IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

# Hyperparametre
SEED = 42
IMG_SIZE = 224
EPOCHS = 10
BATCH_SIZE = 32
LR = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# 1) SimpleCNN

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

class ImgOnlyDataset(Dataset):
    def __init__(self, df, transform):
        self.paths = df["img_path"].tolist()
        self.labels = df["label"].astype(int).tolist()
        self.transform = transform
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        try:
            img = Image.open(self.paths[idx]).convert("RGB")
        except:
            img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), (0, 0, 0))
        return self.transform(img), int(self.labels[idx])


# Vyhodnotenie

@torch.no_grad()
def evaluate(model, loader):
    # Vyhodnotenie modelu na validačných dátach
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())

    all_preds = (np.array(all_probs) >= 0.5).astype(int)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return auc, f1, acc


# 2) Tréningový cyklus

if __name__ == '__main__':
    def load_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                rows.append({"label": int(obj["label"]), "img_path": os.path.join(IMG_ROOT, os.path.basename(obj["img"]))})
        return pd.DataFrame(rows)

    train_df = load_jsonl(TRAIN_PATH)
    val_df = load_jsonl(VAL_PATH)

    # Augmentácia pre tréningové dáta
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Transformácie pre validačné dáta
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # DataLoadery
    train_loader = DataLoader(ImgOnlyDataset(train_df, train_transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImgOnlyDataset(val_df, val_transform), batch_size=BATCH_SIZE, shuffle=False)

    # Inicializácia modelu
    model = SimpleCNN().to(device)

    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    class_weights = torch.tensor([1.0, float(neg / pos if pos > 0 else 1.0)], device=device)

    # Stratová funkcia a optimalizátor
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    best_auc = -1.0
    best_path = "cnn_image_only_best.pt"

    # Tréning
    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epocha {epoch}/{EPOCHS}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Vyhodnotenie po epoche
        auc, f1, acc = evaluate(model, val_loader)
        print(f"\n[VAL] AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f"Uložený najlepší model (AUC: {auc:.4f})")

    print(f"\nTréning dokončený, najlepšie AUC: {best_auc:.4f}")