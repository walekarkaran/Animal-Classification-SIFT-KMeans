"""Bag of Visual Words: a K-Means codebook over SIFT descriptors and per-image word histograms."""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import MiniBatchKMeans


class BagOfVisualWords(BaseEstimator, TransformerMixin):
    """Learn one shared visual vocabulary and encode each image as a normalised word histogram.

    `fit` takes a list of per-image descriptor arrays from the *training* images only;
    labels are never used, so the encoding cannot leak class information.
    """

    def __init__(self, vocab_size: int = 17, sample_size: int = 200_000, random_state: int = 42):
        self.vocab_size = vocab_size
        self.sample_size = sample_size
        self.random_state = random_state

    def fit(self, descriptor_list, y=None):
        descriptors = np.vstack([d for d in descriptor_list if len(d)])
        if len(descriptors) < self.vocab_size:
            raise ValueError(f"Need at least {self.vocab_size} descriptors, got {len(descriptors)}")
        rng = np.random.default_rng(self.random_state)
        if len(descriptors) > self.sample_size:
            descriptors = descriptors[rng.choice(len(descriptors), self.sample_size, replace=False)]
        self.kmeans_ = MiniBatchKMeans(n_clusters=self.vocab_size, batch_size=4096, n_init=3,
                                       random_state=self.random_state).fit(descriptors.astype(np.float32))
        return self

    def transform(self, descriptor_list) -> np.ndarray:
        """(n_images, vocab_size) histograms, L1-normalised so images with many or few keypoints compare."""
        histograms = np.zeros((len(descriptor_list), self.vocab_size), dtype=np.float32)
        for i, descriptors in enumerate(descriptor_list):
            if len(descriptors):
                words = self.kmeans_.predict(descriptors.astype(np.float32))
                counts = np.bincount(words, minlength=self.vocab_size)
                histograms[i] = counts / counts.sum()
        return histograms
