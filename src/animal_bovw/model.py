"""End-to-end model: image path -> SIFT -> visual-word histogram -> classifier -> class name."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from .features import FeatureParams, SiftExtractor
from .vocabulary import BagOfVisualWords


@dataclass
class AnimalClassifier:
    """Bundles everything needed to classify a new image with the exact training-time pipeline."""

    params: FeatureParams
    vocabulary: BagOfVisualWords
    classifier: object          # fitted scikit-learn estimator over histograms
    class_names: list
    name: str = ""

    def encode(self, paths) -> np.ndarray:
        descriptors = SiftExtractor(self.params).descriptors_from_paths(paths)
        return self.vocabulary.transform(descriptors)

    def predict(self, paths) -> list:
        return list(self.classifier.predict(self.encode(paths)))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> AnimalClassifier:
        return joblib.load(path)
