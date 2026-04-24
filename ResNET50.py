import os, json, torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50, ResNet50_Weights
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
LR = 1e-5

# Voľba zariadenia
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(SEED)
np.random.seed(SEED)


# 1) Model a Dataset
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        # Načítanie predtrénovaných váh modelu ResNet50
        weights = ResNet50_Weights.DEFAULT
        self.resnet = resnet50(weights=weights)
        in_features = self.resnet.fc.in_features

        # Nahradenie pôvodnej klasifikačnej vrstvy vlastnou hlavou
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

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
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x)

        # Pravdepodobnosti pozitívnej triedy
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(y.numpy())


    all_preds = (np.array(all_probs) >= 0.5).astype(int)

    # Výpočet metrík
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return auc, f1, acc


# 2) Trénovanie

if __name__ == '__main__':

    def load_jsonl_resnet(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                img_name = os.path.basename(obj.get("img", ""))
                rows.append({
                    "label": int(obj.get("label", 0)),
                    "img_path": os.path.join(IMG_ROOT, img_name)
                })
        return pd.DataFrame(rows)


    # Načítanie tréningových a validačných dát
    train_df = load_jsonl_resnet(TRAIN_PATH)
    val_df = load_jsonl_resnet(VAL_PATH)

    # Definícia transformácií pre obrázky
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    # Vytvorenie DataLoaderov
    train_loader = DataLoader(ImgOnlyDataset(train_df, transform), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ImgOnlyDataset(val_df, transform), batch_size=BATCH_SIZE, shuffle=False)

    # Inicializácia modelu
    model = ResNetClassifier().to(device)

    pos = train_df["label"].sum()
    neg = len(train_df) - pos
    class_weights = torch.tensor([1.0, float(neg / pos if pos > 0 else 1.0)], device=device)

    # Stratová funkcia a optimalizátor
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_auc = -1.0
    best_path = "resnet50_best.pt"

    for epoch in range(1, EPOCHS + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Získavanie metrík
        auc, f1, acc = evaluate(model, val_loader)
        print(f"\n [VAL] AUC: {auc:.4f} | F1: {f1:.4f} | Acc: {acc:.4f}")

        # Ukladanie
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
            print(f" Uložený najlepší model (AUC: {auc:.4f})")

    print(f"\n Tréning dokončený, najlepšie AUC: {best_auc:.4f}")