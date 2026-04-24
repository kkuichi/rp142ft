import os
import json

import random

from tqdm.auto import tqdm

from pathlib import Path
from typing import List, Dict, Any
import numpy as np
from PIL import Image
from sklearn.utils.class_weight import compute_class_weight

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import functional as TVF

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from transformers import (
    BertTokenizerFast,
    VisualBertConfig,
    VisualBertModel,
    get_linear_schedule_with_warmup,
)


#Nastavenia

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


IMG_ROOT = "img"

TRAIN_PATH = "data_HF/train.jsonl"
VAL_PATH = "data_HF/dev_merged.jsonl"
TEST_PATH = "data_HF/test_merged.jsonl"



BEST_MODEL_PATH = "visualbert_best_merged_AUC.pt"

# Hyperparametre
MODEL_NAME = "uclanlp/visualbert-vqa-coco-pre"
MAX_LEN = 64
MAX_REGIONS = 36
BATCH_SIZE = 16
EPOCHS = 5
LR = 2e-5
WEIGHT_DECAY = 0.01
NUM_WORKERS = 0

CACHE_DIR = "visualbert_region_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
# Voľba zariadenia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
use_amp = torch.cuda.is_available()



# 1) Dataset

def load_jsonl(path: str, require_label: bool = True) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Kontrola povinných polí
            if "img" not in obj or "text" not in obj:
                continue

            if require_label and "label" not in obj:
                continue

            img_name = os.path.basename(obj["img"])
            img_path = os.path.join(IMG_ROOT, img_name)

            if not os.path.exists(img_path):
                continue

            row = {
                "id": obj.get("id", img_name),
                "img": img_name,
                "img_path": img_path,
                "text": obj["text"],
            }
            if "label" in obj:
                row["label"] = int(obj["label"])

            rows.append(row)
    return rows

# Načítanie tréningových, validačných a testovacích dát
train_rows = load_jsonl(TRAIN_PATH, require_label=True)
val_rows   = load_jsonl(VAL_PATH, require_label=True)
test_rows  = load_jsonl(TEST_PATH, require_label=True)



# 2) Faster R-CNN

class FasterRCNNRegionExtractor:
    def __init__(self, device: str = "cpu", max_regions: int = 36):
        self.device = device
        self.max_regions = max_regions

        # Načítanie predtrénovaného detektora
        weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.detector = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(
            weights=weights
        ).to(device)
        self.detector.eval()

        self.feature_dim = 1024

    @torch.no_grad()
    def extract(self, pil_image: Image.Image):
        image = pil_image.convert("RGB")
        image_tensor = TVF.to_tensor(image).to(self.device)

        # Predspracovanie obrázka pre Faster R-CNN
        images, _ = self.detector.transform([image_tensor], None)

        # Extrakcia feature máp
        features = self.detector.backbone(images.tensors)
        proposals, _ = self.detector.rpn(images, features, None)

        proposal_boxes = proposals[0]
        if proposal_boxes.numel() == 0:
            return self._empty_features()

        # Obmedzenie počtu regiónov
        proposal_boxes = proposal_boxes[: self.max_regions]

        # ROI pooling a extrakcia regiónových embeddingov
        roi_features = self.detector.roi_heads.box_roi_pool(
            features,
            [proposal_boxes],
            images.image_sizes,
        )
        roi_features = self.detector.roi_heads.box_head(roi_features)

        num_regions = roi_features.size(0)
        if num_regions == 0:
            return self._empty_features()

        # Predpripravené nulové tenzory pevnej veľkosti
        visual_embeds = torch.zeros(
            self.max_regions, self.feature_dim, dtype=torch.float32, device=self.device
        )
        visual_attention_mask = torch.zeros(
            self.max_regions, dtype=torch.long, device=self.device
        )

        # Vyplnenie skutočne nájdených regiónov
        visual_embeds[:num_regions] = roi_features[:num_regions]
        visual_attention_mask[:num_regions] = 1

        return visual_embeds.cpu(), visual_attention_mask.cpu()

    def _empty_features(self):
        visual_embeds = torch.zeros(
            self.max_regions, self.feature_dim, dtype=torch.float32
        )
        visual_attention_mask = torch.zeros(
            self.max_regions, dtype=torch.long
        )
        return visual_embeds, visual_attention_mask

# Inicializácia extraktora regiónových príznakov
region_extractor = FasterRCNNRegionExtractor(device=DEVICE, max_regions=MAX_REGIONS)


# 3) cache region feature

def get_cache_path(img_name: str) -> str:
    stem = Path(img_name).stem
    return os.path.join(CACHE_DIR, f"{stem}_regions.pt")


def get_or_create_region_features(img_path: str, img_name: str):
    cache_path = get_cache_path(img_name)

    if os.path.exists(cache_path):
        item = torch.load(cache_path, map_location="cpu")
        return item["visual_embeds"], item["visual_attention_mask"]

    image = Image.open(img_path).convert("RGB")
    visual_embeds, visual_attention_mask = region_extractor.extract(image)

    torch.save(
        {
            "visual_embeds": visual_embeds,
            "visual_attention_mask": visual_attention_mask,
        },
        cache_path,
    )
    return visual_embeds, visual_attention_mask



# 4) Dataset

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")


class VisualBertHatefulDataset(Dataset):
    # Dataset pre multimodálnu klasifikáciu
    def __init__(self, rows: List[Dict[str, Any]], max_len: int = 64):
        self.rows = rows
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int):
        item = self.rows[idx]

        enc = tokenizer(
            item["text"],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        # Načítanie alebo vytvorenie regiónových príznakov
        visual_embeds, visual_attention_mask = get_or_create_region_features(
            item["img_path"], item["img"]
        )

        out = {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "token_type_ids": enc["token_type_ids"].squeeze(0),
            "visual_embeds": visual_embeds,
            "visual_attention_mask": visual_attention_mask,
            "label": torch.tensor(item["label"], dtype=torch.long),
            "id": item["id"],
        }
        return out

# Vytvorenie datasetov
train_ds = VisualBertHatefulDataset(train_rows, MAX_LEN)
val_ds   = VisualBertHatefulDataset(val_rows, MAX_LEN)
test_ds  = VisualBertHatefulDataset(test_rows, MAX_LEN)

# Vytvorenie DataLoaderov
train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
)



# 5) Model

class VisualBertForBinaryClassification(nn.Module):
    def __init__(self, model_name: str = MODEL_NAME, visual_dim: int = 1024, num_labels: int = 2, weights=None):
        super().__init__()

        # Načítanie konfigurácie modelu
        config = VisualBertConfig.from_pretrained(model_name)
        expected_visual_embedding_dim = 2048
        config.visual_embedding_dim = expected_visual_embedding_dim
        config.num_labels = num_labels

        # Načítanie predtrénovaného VisualBERT modelu
        self.visualbert = VisualBertModel.from_pretrained(
            model_name,
            config=config,
            ignore_mismatched_sizes=True,
        )

        if visual_dim != expected_visual_embedding_dim:
            self.visual_input_projector = nn.Linear(visual_dim, expected_visual_embedding_dim)
        else:
            self.visual_input_projector = None

        # Stratová funkcia s váhami tried
        if weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Klasifikačná hlava
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        visual_embeds,
        visual_attention_mask,
        labels=None,
    ):
        if self.visual_input_projector:
            visual_embeds = self.visual_input_projector(visual_embeds)

        # Visual token type ids pre obrazové regióny
        visual_token_type_ids = torch.ones_like(visual_attention_mask)

        # Forward pass cez VisualBERT
        outputs = self.visualbert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            visual_token_type_ids=visual_token_type_ids,
            return_dict=True,
        )

        pooled_output = outputs.pooler_output

        # Klasifikačné logity
        logits = self.classifier(self.dropout(pooled_output))

        # Loss funckia
        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return loss, logits


# 6) Váhy

train_labels = [x["label"] for x in train_rows]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1]),
    y=train_labels
)
class_weights = torch.tensor(class_weights, dtype=torch.float32)




# 7) Metriky

def compute_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    try:
        auroc = roc_auc_score(y_true, y_prob)
    except Exception:
        auroc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auroc": auroc,
    }


@torch.no_grad()
def evaluate(model, loader, split_name="VAL"):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []
    total_loss = 0.0
    total_items = 0

    eval_bar = tqdm(loader, desc=f"{split_name} [EVAL]", leave=False)

    for batch in eval_bar:
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        visual_embeds = batch["visual_embeds"].to(DEVICE)
        visual_attention_mask = batch["visual_attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        # Forward pass modelu
        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            labels=labels,
        )

        # Pravdepodobnosti pozitívnej triedy
        probs = torch.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
        preds = (probs >= 0.45).astype(int).tolist()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_items += bs

        all_labels.extend(labels.cpu().tolist())
        all_preds.extend(preds)
        all_probs.extend(probs.tolist())

        eval_bar.set_postfix(loss=f"{total_loss / max(total_items, 1):.4f}")

    metrics = compute_metrics(all_labels, all_preds, all_probs)
    avg_loss = total_loss / max(total_items, 1)

    print(f"\n=== VisualBERT {split_name} ===")
    print(f"loss:      {avg_loss:.4f}")
    print(f"accuracy:  {metrics['accuracy']:.4f}")
    print(f"precision: {metrics['precision']:.4f}")
    print(f"recall:    {metrics['recall']:.4f}")
    print(f"f1:        {metrics['f1']:.4f}")
    print(f"auroc:     {metrics['auroc']:.4f}")

    print("\nConfusion matrix [[TN FP],[FN TP]]:")
    print(confusion_matrix(all_labels, all_preds))

    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, digits=4, zero_division=0))

    return avg_loss, metrics



# 8) Tréningový cyklus

# Inicializácia modelu
model = VisualBertForBinaryClassification(
    model_name=MODEL_NAME,
    visual_dim=1024,
    weights=class_weights.to(DEVICE)
).to(DEVICE)

# Optimalizátor
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

num_training_steps = EPOCHS * len(train_loader)
num_warmup_steps = int(0.1 * num_training_steps)

# Scheduler pre postupnú zmenu learning rate
scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)


best_auroc = -1.0

# Hlavný tréningový cyklus
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0

    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]", leave=True)

    for step, batch in enumerate(train_bar, start=1):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        token_type_ids = batch["token_type_ids"].to(DEVICE)
        visual_embeds = batch["visual_embeds"].to(DEVICE)
        visual_attention_mask = batch["visual_attention_mask"].to(DEVICE)
        labels = batch["label"].to(DEVICE)

        optimizer.zero_grad()

        loss, logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            visual_embeds=visual_embeds,
            visual_attention_mask=visual_attention_mask,
            labels=labels,
        )

        loss.backward()

        # Orezanie gradientov
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()

        train_bar.set_postfix(
            loss=f"{running_loss/step:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}"
        )

    # Vyhodnotenie na validačnej množine po každej epoche
    _, val_metrics = evaluate(model, val_loader, split_name="VAL")


    if val_metrics["auroc"] > best_auroc:
        best_auroc = val_metrics["auroc"]
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print(f"Najlepší model podľa AUROC uložený : {BEST_MODEL_PATH}")




# 9) Test

print("\nFinálny test")
model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location=DEVICE))
evaluate(model, test_loader, split_name="TEST")