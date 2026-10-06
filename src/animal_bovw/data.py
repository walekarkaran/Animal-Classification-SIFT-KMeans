"""Dataset discovery: one sub-folder per class, with optional separate train/ and test/ splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Split:
    paths: list
    labels: list   # class names


@dataclass
class Dataset:
    train: Split
    test: Split
    class_names: list


def list_images(split_dir: Path) -> Split:
    """Images directly inside each class folder (nested folders such as `Label/` are ignored)."""
    paths, labels = [], []
    for class_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
        for image in sorted(class_dir.iterdir()):
            if image.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(image)
                labels.append(class_dir.name)
    if not paths:
        raise FileNotFoundError(f"No images found in class folders under {split_dir}")
    return Split(paths, labels)


def load_dataset(root: str | Path, test_size: float = 0.2, seed: int = 42) -> Dataset:
    """Use root/train and root/test when present; otherwise make a stratified split of root/."""
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    if (root / "train").is_dir() and (root / "test").is_dir():
        train, test = list_images(root / "train"), list_images(root / "test")
    else:
        full = list_images(root / "train" if (root / "train").is_dir() else root)
        tr_p, te_p, tr_y, te_y = train_test_split(full.paths, full.labels, test_size=test_size,
                                                  stratify=full.labels, random_state=seed)
        train, test = Split(tr_p, tr_y), Split(te_p, te_y)

    unseen = set(test.labels) - set(train.labels)
    if unseen:
        raise ValueError(f"Test classes missing from training data: {sorted(unseen)}")
    return Dataset(train, test, sorted(set(train.labels)))
