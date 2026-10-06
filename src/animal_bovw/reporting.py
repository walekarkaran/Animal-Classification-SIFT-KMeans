"""Metrics, result tables and figures."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .features import FeatureParams, SiftExtractor, preprocess


def evaluate(y_true, y_pred) -> dict:
    """Accuracy and macro-averaged precision / recall / F1 (arguments in sklearn's order: true, pred)."""
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return {"accuracy": accuracy_score(y_true, y_pred), "macro_precision": precision,
            "macro_recall": recall, "macro_f1": f1}


def summary_table(rows: list) -> pd.DataFrame:
    """rows: dicts with vocab_size, model, train metrics and test metrics."""
    return pd.DataFrame(rows).sort_values("test_accuracy", ascending=False).reset_index(drop=True)


def text_report(y_true, y_pred, class_names) -> str:
    return classification_report(y_true, y_pred, labels=class_names, digits=4, zero_division=0)


def plot_model_comparison(summary: pd.DataFrame, chance: float, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 0.6 * len(summary) + 1.5))
    models = summary["model"][::-1]
    ax.barh(models, summary["test_accuracy"][::-1], color="#4C72B0", label="test accuracy")
    ax.scatter(summary["train_accuracy"][::-1], models, color="#DD8452", zorder=3, label="train accuracy")
    ax.axvline(chance, color="gray", ls="--", label=f"chance ({chance:.2f})")
    for y, acc in enumerate(summary["test_accuracy"][::-1]):
        ax.text(acc - 0.01, y, f"{acc:.3f}", va="center", ha="right", fontsize=8, color="white")
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Accuracy")
    ax.set_title("Classifier comparison on visual-word histograms")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, title: str, path: Path) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(class_names)), class_names, rotation=40, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="row-normalised")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_pipeline_example(image_path: Path, params: FeatureParams, path: Path) -> None:
    """Original image, the preprocessed input to SIFT, and the detected keypoints."""
    image = cv2.imread(str(image_path))
    resized = cv2.resize(image, (params.image_width, params.image_height))
    processed = preprocess(image, params)
    keypoints, _ = SiftExtractor(params).keypoints_and_descriptors(image)
    with_kp = cv2.drawKeypoints(resized, keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    fig, axes = plt.subplots(1, 3, figsize=(11, 6), constrained_layout=True)
    panels = [(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), "Resized input"),
              (processed, "Canny edges" if params.use_canny else "Grayscale"),
              (cv2.cvtColor(with_kp, cv2.COLOR_BGR2RGB), f"{len(keypoints)} SIFT keypoints")]
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.savefig(path, dpi=150)
    plt.close(fig)
