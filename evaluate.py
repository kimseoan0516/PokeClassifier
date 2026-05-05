"""
Evaluate a trained model and compute per-class precision, recall, F1.
Generates a classification report and confusion matrix heatmap.
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from config import EXPERIMENTS
from dataset import get_dataloaders
from models import build_model


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    for images, labels in loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())
    return np.array(all_labels), np.array(all_preds)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path, max_classes=30):
    """Plot confusion matrix (only top-N classes for readability)."""
    cm = confusion_matrix(y_true, y_pred)
    if len(class_names) > max_classes:
        # Show only the most-confused classes
        per_class_err = 1 - cm.diagonal() / (cm.sum(axis=1) + 1e-8)
        top_idx = np.argsort(per_class_err)[-max_classes:][::-1]
        cm = cm[np.ix_(top_idx, top_idx)]
        class_names = [class_names[i] for i in top_idx]

    fig, ax = plt.subplots(figsize=(max(10, len(class_names) // 2),
                                    max(8, len(class_names) // 2)))
    sns.heatmap(cm, annot=len(class_names) <= 30, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (most-confused classes)")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_learning_curves(results_dir: Path):
    """Plot training/validation loss and accuracy curves for all experiments."""
    exp_dirs = [d for d in results_dir.iterdir() if d.is_dir() and (d / "results.json").exists()]
    if not exp_dirs:
        print("No experiment results found.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, exp_dir in enumerate(sorted(exp_dirs)):
        if i >= 4:
            break
        with open(exp_dir / "results.json") as f:
            data = json.load(f)
        cfg = data["config"]
        hist = data["history"]
        epochs = range(1, len(hist["train_loss"]) + 1)
        label = cfg["name"].replace("exp", "Exp").replace("_", " ")
        axes[i].plot(epochs, hist["train_loss"], label="Train Loss")
        axes[i].plot(epochs, hist["val_loss"], label="Val Loss")
        axes[i].set_title(f"{label}\nTest Acc: {data['test_acc']:.3f}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Loss")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_dir / "learning_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Learning curves saved to {save_path}")

    # Accuracy curves
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for i, exp_dir in enumerate(sorted(exp_dirs)):
        if i >= 4:
            break
        with open(exp_dir / "results.json") as f:
            data = json.load(f)
        cfg = data["config"]
        hist = data["history"]
        epochs = range(1, len(hist["train_acc"]) + 1)
        axes[i].plot(epochs, hist["train_acc"], label="Train Acc")
        axes[i].plot(epochs, hist["val_acc"], label="Val Acc")
        axes[i].set_title(f"{cfg['name'].replace('_', ' ')}\nTest Acc: {data['test_acc']:.3f}")
        axes[i].set_xlabel("Epoch")
        axes[i].set_ylabel("Accuracy")
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = results_dir / "accuracy_curves.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Accuracy curves saved to {save_path}")


def compare_experiments(results_dir: Path):
    """Generate a comparison bar chart of all experiments."""
    exp_dirs = sorted([d for d in results_dir.iterdir()
                       if d.is_dir() and (d / "results.json").exists()])
    names, test_accs, best_val_accs = [], [], []
    for exp_dir in exp_dirs:
        with open(exp_dir / "results.json") as f:
            data = json.load(f)
        names.append(data["config"]["name"].replace("exp", "Exp ").replace("_", "\n"))
        test_accs.append(data["test_acc"])
        best_val_accs.append(data["best_val_acc"])

    x = np.arange(len(names))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - width / 2, best_val_accs, width, label="Best Val Acc", color="steelblue")
    ax.bar(x + width / 2, test_accs, width, label="Test Acc", color="coral")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("Accuracy")
    ax.set_title("Experiment Comparison: Validation vs Test Accuracy")
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    for xi, (v, t) in enumerate(zip(best_val_accs, test_accs)):
        ax.text(xi - width / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
        ax.text(xi + width / 2, t + 0.01, f"{t:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    save_path = results_dir / "comparison.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Comparison chart saved to {save_path}")


def evaluate_experiment(cfg, data_dir: str, results_dir: Path, device):
    exp_dir = results_dir / cfg.name
    model_path = exp_dir / "best_model.pth"
    if not model_path.exists():
        print(f"Model not found: {model_path}. Run training first.")
        return

    with open(exp_dir / "class_names.json") as f:
        class_names = json.load(f)
    cfg.num_classes = len(class_names)

    _, _, test_loader, _ = get_dataloaders(data_dir, image_size=cfg.image_size,
                                           batch_size=cfg.batch_size,
                                           num_workers=cfg.num_workers, seed=cfg.seed)
    model = build_model(cfg).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))

    y_true, y_pred = predict_all(model, test_loader, device)
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True, zero_division=0)
    report_str = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(f"\n--- {cfg.name} ---")
    print(report_str)

    with open(exp_dir / "classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    with open(exp_dir / "classification_report.txt", "w") as f:
        f.write(report_str)

    plot_confusion_matrix(y_true, y_pred, class_names, exp_dir / "confusion_matrix.png")
    print(f"Confusion matrix saved to {exp_dir}/confusion_matrix.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--exp", type=str, default=None)
    parser.add_argument("--curves_only", action="store_true",
                        help="Only generate learning curves and comparison chart")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results_dir = Path(args.results_dir)

    plot_learning_curves(results_dir)
    compare_experiments(results_dir)

    if args.curves_only:
        return

    exps = EXPERIMENTS
    if args.exp:
        exps = [e for e in EXPERIMENTS if e.name == args.exp]

    for cfg in exps:
        evaluate_experiment(cfg, args.data_dir, results_dir, device)


if __name__ == "__main__":
    main()
