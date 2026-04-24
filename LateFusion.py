import os, json, torch, gc
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import resnet50

from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    accuracy_score, confusion_matrix, classification_report
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, set_seed,
    ViTImageProcessor, ViTForImageClassification
)


# Nastavenia


set_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"

TEXT_CONFIG = {
    "BERT": {
        "path": "bert_results/checkpoint-4252",
        "hub": "bert-base-uncased"
    },
    "ELECTRA": {
        "path": "electra_final_model",
        "hub": "google/electra-base-discriminator"
    },
    "RoBERTa": {
        "path": "roberta_hate_model_AUC/checkpoint-5320",
        "hub": "cardiffnlp/twitter-roberta-base-hate"
    }
}

IMG_CONFIG = {
    "ResNet50": "Best_models/resnet50_best.pt",
    "SimpleCNN": "Best_models/cnn_image_only_best.pt",
    "ViT": "vit_best_model"
}


# 1) Pomocné funckie

def min_max_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-10)

def find_best_threshold(y_true, y_prob):
    """
    Vyhľadá najlepší klasifikačný prah na validačných dátach
    na základe maximálneho F1-score.
    """

    best_f1, best_thr = -1, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr

def metric_bundle(y_true, y_prob, thr):
    # Vypočíta základné metriky klasifikácie pri zadanom prahu.
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    auc = roc_auc_score(y_true, y_prob)
    mcc = matthews_corrcoef(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    recall = tp / (tp + fn + 1e-10)

    return {
        "Accuracy": acc,
        "F1-Score": f1,
        "AUROC": auc,
        "MCC": mcc,
        "Recall": recall,
        "CM": cm
    }

def evaluate_probs(y_true, y_prob, thr, name, modality):
    # Vyhodnotí unimodálny model pri zadanom prahu
    m = metric_bundle(y_true, y_prob, thr)

    print(f"\n {modality} {name}")
    print(f"Thr: {thr:.3f} | AUROC: {m['AUROC']:.4f} | F1: {m['F1-Score']:.4f} | Recall: {m['Recall']:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(m["CM"])

    return {
        "Kombinácia": name,
        "Modalita": modality,
        "Best Alpha": np.nan,
        "Best Thr": thr,
        "F1-Score": m["F1-Score"],
        "AUROC": m["AUROC"],
        "MCC": m["MCC"],
        "Accuracy": m["Accuracy"],
        "Recall": m["Recall"],
        "TN": m["CM"][0, 0],
        "FP": m["CM"][0, 1],
        "FN": m["CM"][1, 0],
        "TP": m["CM"][1, 1]
    }

def alpha_fusion(text_val, img_val, text_test, img_test, y_val, y_test, name):
    # Realizuje neskorú fúziu
    best_a, best_auc = 0, -1

    # Hľadanie optimálnej váhy alpha v intervale <0,1>
    for a in np.linspace(0, 1, 101):
        p = a * text_val + (1 - a) * img_val
        auc = roc_auc_score(y_val, p)
        if auc > best_auc:
            best_auc = auc
            best_a = a

    # Výpočet validačných pravdepodobností pre najlepšiu alpha
    p_val = best_a * text_val + (1 - best_a) * img_val
    best_thr = find_best_threshold(y_val, p_val)

    # Výpočet pravdepodobností na testovacej množine
    p_test = best_a * text_test + (1 - best_a) * img_test
    m = metric_bundle(y_test, p_test, best_thr)

    print(f"\nFUSION {name}")
    print(f"Alpha: {best_a:.2f} | Thr: {best_thr:.3f} | AUROC: {m['AUROC']:.4f} | F1: {m['F1-Score']:.4f} | Recall: {m['Recall']:.4f}")
    print("Confusion matrix [[TN FP],[FN TP]]:")
    print(m["CM"])

    return {
        "Kombinácia": name,
        "Modalita": "Fusion",
        "Best Alpha": best_a,
        "Best Thr": best_thr,
        "F1-Score": m["F1-Score"],
        "AUROC": m["AUROC"],
        "MCC": m["MCC"],
        "Accuracy": m["Accuracy"],
        "Recall": m["Recall"],
        "TN": m["CM"][0, 0],
        "FP": m["CM"][0, 1],
        "FN": m["CM"][1, 0],
        "TP": m["CM"][1, 1]
    }


# 2) Modely

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
        return self.classifier(self.features(x))

class ResNet50Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = resnet50(weights=None)
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.resnet.fc.in_features, 2)
        )

    def forward(self, x):
        return self.resnet(x)

# 3) Dáta

def load_jsonl(path):
    # Načítanie JSONL súboru
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            data.append({
                "text": obj["text"],
                "label": int(obj["label"]),
                "img_path": os.path.join(IMG_ROOT, os.path.basename(obj["img"]))
            })
    return pd.DataFrame(data)

# Načítanie validačných a testovacích dát
val_df = load_jsonl(VAL_PATH)
test_df = load_jsonl(TEST_PATH)

y_val = val_df["label"].values
y_test = test_df["label"].values

# 4) Obrazové predikcie

img_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

img_preds_val, img_preds_test = {}, {}

for name, path in IMG_CONFIG.items():
    print(f"\n Spracovanie obrazového modelu: {name}")

    # Vetva pre Vision Transformer
    if name == "ViT":
        processor = ViTImageProcessor.from_pretrained(path)
        model = ViTForImageClassification.from_pretrained(path).to(device)
        model.eval()

        def get_vit_probs(df):
            # získa pravdepodobnosť pozitívnej triedy pre ViT model
            probs = []
            for img_path in tqdm(df["img_path"].tolist(), desc=name):
                img = Image.open(img_path).convert("RGB")
                inputs = processor(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    p = torch.softmax(model(**inputs).logits, dim=1)[:, 1].item()
                probs.append(p)
            return np.array(probs)

        img_preds_val[name] = get_vit_probs(val_df)
        img_preds_test[name] = get_vit_probs(test_df)

        del model, processor
        gc.collect()
        torch.cuda.empty_cache()
    # Vetva pre CNN / ResNet50
    else:
        model = SimpleCNN().to(device) if name == "SimpleCNN" else ResNet50Wrapper().to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        def get_img_probs(df):
            class ImgDS(Dataset):
                def __init__(self, paths):
                    self.paths = list(paths)

                def __len__(self):
                    return len(self.paths)

                def __getitem__(self, idx):
                    return img_tfms(Image.open(self.paths[idx]).convert("RGB"))

            loader = DataLoader(ImgDS(df["img_path"].tolist()), batch_size=32, shuffle=False)
            probs = []
            for x in tqdm(loader, desc=name):
                with torch.no_grad():
                    p = torch.softmax(model(x.to(device)), dim=1)[:, 1]
                probs.extend(p.cpu().numpy())
            return np.array(probs)

        img_preds_val[name] = get_img_probs(val_df)
        img_preds_test[name] = get_img_probs(test_df)
        # Uvoľnenie pamäte
        del model
        gc.collect()
        torch.cuda.empty_cache()

    p = img_preds_test[name]
    print(f"{name}: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}, std={p.std():.4f}")


# 5) Textové predikcie

text_preds_val, text_preds_test = {}, {}

for name, cfg in TEXT_CONFIG.items():
    print(f"\nSpracovanie textového modelu: {name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg["hub"])
    model = AutoModelForSequenceClassification.from_pretrained(cfg["path"]).to(device)
    model.eval()

    def get_text_probs(df):
        # Získa pravdepodobnosti pozitívnej triedy pre textový model
        probs = []
        for text in tqdm(df["text"].tolist(), desc=name):
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=64
            ).to(device)
            with torch.no_grad():
                p = torch.softmax(model(**inputs).logits, dim=1)[:, 1]
            probs.append(p.item())
        return np.array(probs)

    text_preds_val[name] = get_text_probs(val_df)
    text_preds_test[name] = get_text_probs(test_df)

    # Uvoľnenie pamäte
    del model
    gc.collect()
    torch.cuda.empty_cache()

    p = text_preds_test[name]
    print(f"{name}: min={p.min():.4f}, max={p.max():.4f}, mean={p.mean():.4f}, std={p.std():.4f}")


# 6) Fúzia

results = []


print("Unimodálne")
print("="*60)

# Vyhodnotenie textových modelov
for name in text_preds_val:
    val_probs = min_max_norm(text_preds_val[name])
    test_probs = min_max_norm(text_preds_test[name])

    thr = find_best_threshold(y_val, val_probs)
    results.append(evaluate_probs(y_test, test_probs, thr, name, "Text"))

# Vyhodnotenie obrazových modelov
for name in img_preds_val:
    val_probs = min_max_norm(img_preds_val[name])
    test_probs = min_max_norm(img_preds_test[name])

    thr = find_best_threshold(y_val, val_probs)
    results.append(evaluate_probs(y_test, test_probs, thr, name, "Image"))


print("Fúzia")
print("="*60)

# Vyhodnotenie všetkých kombinácií textového a obrazového modelu
for t_name in text_preds_val:
    for i_name in img_preds_val:
        t_v = min_max_norm(text_preds_val[t_name])
        i_v = min_max_norm(img_preds_val[i_name])
        t_t = min_max_norm(text_preds_test[t_name])
        i_t = min_max_norm(img_preds_test[i_name])

        results.append(
            alpha_fusion(
                t_v, i_v, t_t, i_t,
                y_val, y_test,
                f"{t_name}+{i_name}"
            )
        )



# 7) VÝSLEDKY
df_res = pd.DataFrame(results).sort_values(by="AUROC", ascending=False)

csv_path = "late_fusion_unimodal_results.csv"
df_res.to_csv(csv_path, index=False)


print("Poradie všetkých modelov")
print("="*60)
print(df_res.to_string(index=False))


print("Unimodálne")
print("="*60)
print(df_res[df_res["Modalita"].isin(["Text", "Image"])].to_string(index=False))


print("Fúzia")
print("="*60)
print(df_res[df_res["Modalita"] == "Fusion"].to_string(index=False))

best_text = df_res[df_res["Modalita"] == "Text"].sort_values("AUROC", ascending=False).iloc[0]
best_image = df_res[df_res["Modalita"] == "Image"].sort_values("AUROC", ascending=False).iloc[0]
best_fusion = df_res[df_res["Modalita"] == "Fusion"].sort_values("AUROC", ascending=False).iloc[0]

print("\n" + "="*60)
print("MODELY")
print("="*60)
print("\nBest TEXT:")
print(best_text.to_string())
print("\nBest IMAGE:")
print(best_image.to_string())
print("\nBest FUSION:")
print(best_fusion.to_string())

