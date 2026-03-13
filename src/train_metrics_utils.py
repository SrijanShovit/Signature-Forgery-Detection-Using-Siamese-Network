import torch
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def compute_best_threshold(model, val_loader, device="mps"):
    """
    Compute optimal threshold (margin) on validation set using ROC curve.
    Returns: best_threshold, fpr, tpr, thresholds, auc_score
    """
    model.eval()
    all_distances = []
    all_labels = []

    with torch.no_grad():
        for x1, x2, y in val_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()
            preds, distances = model.calc_preds(x1, x2, threshold=None)
            all_distances.append(distances.cpu())
            all_labels.append(y.cpu())

    all_distances = torch.cat(all_distances).numpy()
    all_labels = torch.cat(all_labels).numpy()

    fpr, tpr, thresholds = roc_curve(all_labels, -all_distances)  # negative distances: closer = positive
    auc_score = auc(fpr, tpr)

    # Best threshold = maximize TPR - FPR (you can also compute EER)
    best_idx = np.argmax(tpr - fpr)
    best_threshold = thresholds[best_idx]

    return best_threshold, fpr, tpr, thresholds, auc_score


def plot_roc(fpr, tpr, auc_score, save_path=None):
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_distance_distribution(distances, labels, save_path=None):
    plt.figure(figsize=(6, 4))
    plt.hist(distances[labels == 1], bins=50, alpha=0.6, label="Genuine")
    plt.hist(distances[labels == 0], bins=50, alpha=0.6, label="Forged")
    plt.xlabel("Embedding Distance")
    plt.ylabel("Count")
    plt.title("Distance Distribution")
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def call_plot_distribution(val_loader,model,device="mps"):
    model = model.to(device)
    model.eval()
    all_distances = []
    all_labels = []
    for x1, x2, y in val_loader:
        _, distances = model.calc_preds(x1.to(device), x2.to(device))
        all_distances.append(distances.cpu())
        all_labels.append(y)
    all_distances = torch.cat(all_distances)
    all_labels = torch.cat(all_labels)
    plot_distance_distribution(all_distances.numpy(), all_labels.numpy())


def plot_train_stats(metrics):
    epochs = metrics['epoch']

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Val Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")

    plt.subplot(1,2,2)
    plt.plot(epochs, metrics['train_acc'], label='Train Acc')
    plt.plot(epochs, metrics['val_acc'], label='Val Acc')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")

    plt.show()


def load_lightning_metrics(logger_csv_path: str):
    csv_path = Path(logger_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"{logger_csv_path} not found")

    df = pd.read_csv(csv_path)

    # Lightning CSV logs metrics at each step; we want **per epoch summary**
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    grouped = df.groupby('epoch')

    for epoch, group in grouped:
        metrics['epoch'].append(epoch)
        # train metrics: pick last step in epoch
        train_steps = group[group['train_loss'].notna()]
        metrics['train_loss'].append(train_steps['train_loss'].values[-1])
        metrics['train_acc'].append(train_steps['train_acc'].values[-1])

        # val metrics: pick last step in epoch
        val_steps = group[group['val_loss'].notna()]
        if len(val_steps) > 0:
            metrics['val_loss'].append(val_steps['val_loss'].values[-1])
            metrics['val_acc'].append(val_steps['val_acc'].values[-1])
        else:
            metrics['val_loss'].append(None)
            metrics['val_acc'].append(None)

    return metrics


def plot_triplet_training_metrics(csv_path: str):
    df = pd.read_csv(csv_path)

    # separate train / val rows
    train_df = df[df["train_loss"].notna()].copy()
    val_df = df[df["val_loss"].notna()].copy()
    lr_df = df[df["lr"].notna()].copy()

    # group by epoch (Lightning logs multiple rows per epoch)
    train_df = train_df.groupby("epoch").mean(numeric_only=True).reset_index()
    val_df = val_df.groupby("epoch").mean(numeric_only=True).reset_index()
    lr_df = lr_df.groupby("epoch").mean(numeric_only=True).reset_index()

    # ---- Loss plot ----
    plt.figure()
    plt.plot(train_df["epoch"], train_df["train_loss"], marker="o", label="Train Loss")
    plt.plot(val_df["epoch"], val_df["val_loss"], marker="o", label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Triplet Training Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Triplet violation plot ----
    plt.figure()
    plt.plot(train_df["epoch"], train_df["train_triplet_violation"], marker="o", label="Train Violation")
    plt.plot(val_df["epoch"], val_df["val_triplet_violation"], marker="o", label="Validation Violation")
    plt.xlabel("Epoch")
    plt.ylabel("Triplet Violation Rate")
    plt.title("Triplet Violation Rate During Training")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---- Learning rate plot ----
    plt.figure()
    plt.plot(lr_df["epoch"], lr_df["lr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.grid(True)
    plt.show()