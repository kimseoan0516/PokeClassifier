import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from config import EXPERIMENTS, ExperimentConfig
from dataset import get_dataloaders
from models import build_model, count_trainable_params


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def train_experiment(cfg: ExperimentConfig, data_dir: str, output_dir: str):
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Experiment: {cfg.name}")
    print(f"Backbone: {cfg.backbone} | Pretrained: {cfg.pretrained} | Mode: {cfg.fine_tune_mode}")
    print(f"Device: {device}")
    print(f"{'='*60}")

    exp_dir = Path(output_dir) / cfg.name
    exp_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader, class_names = get_dataloaders(
        data_dir,
        image_size=cfg.image_size,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )
    cfg.num_classes = len(class_names)

    model = build_model(cfg).to(device)
    print(f"Trainable parameters: {count_trainable_params(model):,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.num_epochs)

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{cfg.num_epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), exp_dir / "best_model.pth")

    # Load best model and evaluate on test set
    model.load_state_dict(torch.load(exp_dir / "best_model.pth", map_location=device))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}")

    # Save history + test result
    results = {
        "config": {
            "name": cfg.name,
            "backbone": cfg.backbone,
            "pretrained": cfg.pretrained,
            "fine_tune_mode": cfg.fine_tune_mode,
            "num_epochs": cfg.num_epochs,
            "lr": cfg.lr,
        },
        "history": history,
        "test_loss": test_loss,
        "test_acc": test_acc,
        "best_val_acc": best_val_acc,
    }
    with open(exp_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save class names
    with open(exp_dir / "class_names.json", "w") as f:
        json.dump(class_names, f)

    print(f"Results saved to {exp_dir}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Train a Pokemon classifier")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to dataset folder (one subfolder per class)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save experiment results")
    parser.add_argument("--exp", type=str, default=None,
                        help="Run a specific experiment by name (default: all)")
    args = parser.parse_args()

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in EXPERIMENTS if e.name == args.exp]
        if not exps:
            print(f"Unknown experiment: {args.exp}. Available: {[e.name for e in EXPERIMENTS]}")
            return

    for cfg in exps:
        train_experiment(cfg, args.data_dir, args.output_dir)

    print("\nAll experiments completed.")


if __name__ == "__main__":
    main()
