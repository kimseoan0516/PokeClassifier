"""
PokeClassifier — Kaggle Notebook Version
=========================================
Dataset: lantian773030/pokemonclassification
Data path: /kaggle/input/pokemonclassification/PokemonData

Usage: paste each cell block into a Kaggle notebook cell.
"""

# ── Cell 1: Install & Imports ──────────────────────────────────────────────
import json
import time
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np

print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_DIR = "/kaggle/input/pokemonclassification/PokemonData"
OUTPUT_DIR = Path("/kaggle/working/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Cell 2: Config ─────────────────────────────────────────────────────────
@dataclass
class ExperimentConfig:
    name: str
    backbone: str
    pretrained: bool
    fine_tune_mode: str   # scratch | frozen | partial | full
    num_epochs: int = 15
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_classes: int = 150
    image_size: int = 224

EXPERIMENTS = [
    ExperimentConfig("exp1_resnet18_scratch",  "resnet18",        False, "scratch", lr=1e-3),
    ExperimentConfig("exp2_resnet18_frozen",   "resnet18",        True,  "frozen",  lr=1e-3),
    ExperimentConfig("exp3_resnet50_partial",  "resnet50",        True,  "partial", lr=5e-4),
    ExperimentConfig("exp4_efficientnet_full", "efficientnet_b0", True,  "full",    lr=1e-4),
]


# ── Cell 3: Dataset ─────────────────────────────────────────────────────────
def get_transforms(image_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

def get_dataloaders(cfg, seed=42):
    base = datasets.ImageFolder(DATA_DIR, transform=get_transforms(cfg.image_size, False))
    class_names = base.classes
    n = len(base)
    g = torch.Generator().manual_seed(seed)
    n_test = int(n * 0.15)
    n_val  = int(n * 0.15)
    n_train = n - n_val - n_test
    train_ds, val_ds, test_ds = random_split(base, [n_train, n_val, n_test], generator=g)

    train_ds.dataset = datasets.ImageFolder(DATA_DIR, transform=get_transforms(cfg.image_size, True))
    eval_ds = datasets.ImageFolder(DATA_DIR, transform=get_transforms(cfg.image_size, False))
    val_ds.dataset = eval_ds
    test_ds.dataset = eval_ds

    kw = dict(batch_size=cfg.batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds,   shuffle=False, **kw),
            DataLoader(test_ds,  shuffle=False, **kw),
            class_names)


# ── Cell 4: Model ───────────────────────────────────────────────────────────
def build_model(cfg):
    w = "DEFAULT" if cfg.pretrained else None
    if cfg.backbone == "resnet18":
        m = models.resnet18(weights=w)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        _freeze(m, cfg.fine_tune_mode, "fc")
    elif cfg.backbone == "resnet50":
        m = models.resnet50(weights=w)
        m.fc = nn.Linear(m.fc.in_features, cfg.num_classes)
        _freeze(m, cfg.fine_tune_mode, "fc")
    elif cfg.backbone == "efficientnet_b0":
        m = models.efficientnet_b0(weights=w)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, cfg.num_classes)
        _freeze(m, cfg.fine_tune_mode, "classifier")
    return m

def _freeze(model, mode, head):
    if mode == "frozen":
        for n, p in model.named_parameters():
            if not n.startswith(head):
                p.requires_grad = False
    elif mode == "partial":
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if any(n.startswith(x) for x in ["layer3","layer4","features.6","features.7","features.8",head,"classifier"]):
                p.requires_grad = True


# ── Cell 5: Train loop ──────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, train=True):
    model.train() if train else model.eval()
    loss_sum, correct, total = 0, 0, 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for imgs, labels in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            if train:
                optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * imgs.size(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += imgs.size(0)
    return loss_sum / total, correct / total

def train_experiment(cfg):
    torch.manual_seed(42)
    print(f"\n{'='*60}\n{cfg.name}\nbackbone={cfg.backbone} pretrained={cfg.pretrained} mode={cfg.fine_tune_mode}\n{'='*60}")

    exp_dir = OUTPUT_DIR / cfg.name
    exp_dir.mkdir(exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(cfg)
    cfg.num_classes = len(class_names)

    model = build_model(cfg).to(DEVICE)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}
    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, optimizer, train=True)
        vl, va = run_epoch(model, val_loader,   criterion, optimizer, train=False)
        scheduler.step()
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl);   history["val_acc"].append(va)
        print(f"Epoch {epoch:2d}/{cfg.num_epochs} | "
              f"Train {tl:.4f}/{ta:.4f} | Val {vl:.4f}/{va:.4f} | {time.time()-t0:.1f}s")
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), exp_dir / "best_model.pth")

    model.load_state_dict(torch.load(exp_dir / "best_model.pth"))
    test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, train=False)
    print(f"\n>>> Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    results = {
        "config": {"name":cfg.name,"backbone":cfg.backbone,
                   "pretrained":cfg.pretrained,"fine_tune_mode":cfg.fine_tune_mode},
        "history": history,
        "best_val_acc": best_val_acc,
        "test_loss": test_loss,
        "test_acc": test_acc,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    with open(exp_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    return results, model, test_loader, class_names


# ── Cell 6: Run all experiments ─────────────────────────────────────────────
all_results = []
for cfg in EXPERIMENTS:
    res, model, test_loader, class_names = train_experiment(cfg)
    all_results.append(res)


# ── Cell 7: Learning curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for i, res in enumerate(all_results):
    h = res["history"]
    ep = range(1, len(h["train_loss"]) + 1)
    name = res["config"]["name"].replace("_", "\n")

    axes[0][i].plot(ep, h["train_loss"], label="Train")
    axes[0][i].plot(ep, h["val_loss"],   label="Val")
    axes[0][i].set_title(f"{name}\nTest Acc: {res['test_acc']:.3f}")
    axes[0][i].set_ylabel("Loss"); axes[0][i].legend(); axes[0][i].grid(alpha=0.3)

    axes[1][i].plot(ep, h["train_acc"], label="Train")
    axes[1][i].plot(ep, h["val_acc"],   label="Val")
    axes[1][i].set_ylabel("Accuracy"); axes[1][i].legend(); axes[1][i].grid(alpha=0.3)

axes[0][0].set_ylabel("Loss"); axes[1][0].set_ylabel("Accuracy")
plt.suptitle("Learning Curves — PokeClassifier", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "learning_curves.png", dpi=150)
plt.show()
print("Saved: learning_curves.png")


# ── Cell 8: Comparison bar chart ────────────────────────────────────────────
names     = [r["config"]["name"].replace("_","\n") for r in all_results]
val_accs  = [r["best_val_acc"] for r in all_results]
test_accs = [r["test_acc"]     for r in all_results]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(12, 5))
b1 = ax.bar(x - 0.2, val_accs,  0.35, label="Best Val Acc",  color="steelblue")
b2 = ax.bar(x + 0.2, test_accs, 0.35, label="Test Acc",      color="coral")
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("Accuracy"); ax.set_ylim(0, 1)
ax.set_title("Experiment Comparison"); ax.legend(); ax.grid(axis="y", alpha=0.3)
for rect, v in [(b, vv) for b, vals in [(b1, val_accs),(b2, test_accs)] for b, vv in zip(b, vals)]:
    ax.text(rect.get_x() + rect.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "comparison.png", dpi=150)
plt.show()
print("Saved: comparison.png")
