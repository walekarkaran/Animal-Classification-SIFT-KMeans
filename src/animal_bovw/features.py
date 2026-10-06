"""Image preprocessing and SIFT descriptor extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm

SIFT_DIM = 128


@dataclass(frozen=True)
class FeatureParams:
    image_width: int = 280
    image_height: int = 430
    use_canny: bool = True
    canny_low: int = 30
    canny_high: int = 100
    max_keypoints: int = 0     # 0 = unlimited


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def preprocess(image: np.ndarray, params: FeatureParams) -> np.ndarray:
    """Resize, convert to grayscale and (optionally) take the Canny edge map.

    Every image — training or test, any class — goes through exactly the same steps.
    """
    image = cv2.resize(image, (params.image_width, params.image_height))
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if params.use_canny:
        return cv2.Canny(gray, params.canny_low, params.canny_high)
    return gray


class SiftExtractor:
    def __init__(self, params: FeatureParams | None = None):
        self.params = params or FeatureParams()
        self._sift = cv2.SIFT_create(nfeatures=self.params.max_keypoints)

    def keypoints_and_descriptors(self, image: np.ndarray):
        keypoints, descriptors = self._sift.detectAndCompute(preprocess(image, self.params), None)
        if descriptors is None:   # blank image / no keypoints found
            descriptors = np.empty((0, SIFT_DIM), dtype=np.float32)
        return keypoints, descriptors

    def descriptors(self, image: np.ndarray) -> np.ndarray:
        """(n_keypoints, 128) float32 SIFT descriptors."""
        return self.keypoints_and_descriptors(image)[1]

    def descriptors_from_paths(self, paths, desc: str = "SIFT") -> list:
        return [self.descriptors(read_image(p)) for p in tqdm(paths, desc=desc, leave=False)]
