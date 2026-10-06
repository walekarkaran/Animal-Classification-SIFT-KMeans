#!/usr/bin/env python
"""Classify images with a model saved by run_benchmark.py.

Example:
    python scripts/predict.py --model outputs/run-.../best_model.joblib --images photo1.jpg photo2.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

from animal_bovw.model import AnimalClassifier


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--images", required=True, nargs="+", type=Path)
    args = p.parse_args()

    model = AnimalClassifier.load(args.model)
    print(f"Model: {model.name}")
    for path, label in zip(args.images, model.predict(args.images)):
        print(f"{path}: {label}")


if __name__ == "__main__":
    main()
