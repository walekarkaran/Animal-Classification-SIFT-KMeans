"""Experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class Config:
    # data
    data_root: str = "data/animals"     # contains train/<Class>/*.jpg and optionally test/<Class>/*.jpg
    test_size: float = 0.2               # used only when there is no test/ folder

    # features
    image_width: int = 280               # images are resized to width x height before SIFT
    image_height: int = 430
    use_canny: bool = True               # detect SIFT on the Canny edge map instead of the grayscale image
    canny_low: int = 30
    canny_high: int = 100
    max_keypoints: int = 0               # 0 = keep every SIFT keypoint

    # visual vocabulary
    vocab_size: int = 17                 # number of K-Means clusters (visual words)
    vocab_sample: int = 200_000          # descriptors sampled from the training set to fit K-Means

    # experiment
    seed: int = 42
    output_dir: str = "outputs"

    def to_dict(self) -> dict:
        return asdict(self)
