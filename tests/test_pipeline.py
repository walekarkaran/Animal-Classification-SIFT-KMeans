"""Fast tests on small synthetic images — no dataset or network access needed."""

import cv2
import numpy as np
import pytest

from animal_bovw.classifiers import build_classifiers
from animal_bovw.data import load_dataset
from animal_bovw.features import SIFT_DIM, FeatureParams, SiftExtractor, preprocess
from animal_bovw.model import AnimalClassifier
from animal_bovw.vocabulary import BagOfVisualWords

PARAMS = FeatureParams(image_width=160, image_height=160)
CLASSES = ["circles", "lines", "squares"]


def synthetic_image(kind: str, rng: np.random.Generator) -> np.ndarray:
    image = np.full((200, 200, 3), 255, dtype=np.uint8)
    for _ in range(int(rng.integers(6, 10))):
        x, y = (int(v) for v in rng.integers(20, 180, size=2))
        s = int(rng.integers(8, 20))
        if kind == "circles":
            cv2.circle(image, (x, y), s, (0, 0, 0), 2)
        elif kind == "squares":
            cv2.rectangle(image, (x - s, y - s), (x + s, y + s), (0, 0, 0), 2)
        else:
            cv2.line(image, (x - s, y - s), (x + s, y + 2 * s), (0, 0, 0), 2)
    return image


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("animals")
    rng = np.random.default_rng(0)
    for split, n in [("train", 25), ("test", 10)]:
        for cls in CLASSES:
            folder = root / split / cls
            folder.mkdir(parents=True)
            for i in range(n):
                cv2.imwrite(str(folder / f"{i}.png"), synthetic_image(cls, rng))
        (root / split / CLASSES[0] / "Label").mkdir()   # nested annotation folders must be ignored
    return root


def test_load_dataset_uses_train_and_test_folders(dataset_dir):
    data = load_dataset(dataset_dir)
    assert data.class_names == CLASSES
    assert len(data.train.paths) == 75 and len(data.test.paths) == 30


def test_preprocess_and_sift_shapes():
    image = synthetic_image("squares", np.random.default_rng(1))
    edges = preprocess(image, PARAMS)
    assert edges.shape == (160, 160) and set(np.unique(edges)) <= {0, 255}
    descriptors = SiftExtractor(PARAMS).descriptors(image)
    assert descriptors.ndim == 2 and descriptors.shape[1] == SIFT_DIM and len(descriptors) > 0


def test_blank_image_gives_empty_descriptors_and_zero_histogram():
    blank = np.full((200, 200, 3), 255, dtype=np.uint8)
    descriptors = SiftExtractor(PARAMS).descriptors(blank)
    assert descriptors.shape == (0, SIFT_DIM)
    rng = np.random.default_rng(0)
    vocab = BagOfVisualWords(vocab_size=5).fit([rng.random((50, SIFT_DIM)).astype(np.float32)])
    assert np.all(vocab.transform([descriptors]) == 0)


def test_histograms_are_normalised_and_use_every_bin():
    rng = np.random.default_rng(0)
    descriptors = [rng.random((int(rng.integers(20, 60)), SIFT_DIM)).astype(np.float32) for _ in range(30)]
    vocab = BagOfVisualWords(vocab_size=17).fit(descriptors)
    hist = vocab.transform(descriptors)
    assert hist.shape == (30, 17)
    assert np.allclose(hist.sum(axis=1), 1.0)
    assert (hist.sum(axis=0) > 0).sum() == 17   # no permanently empty word (the original bin bug)


def test_end_to_end_model_learns_and_roundtrips(dataset_dir, tmp_path):
    data = load_dataset(dataset_dir)
    extractor = SiftExtractor(PARAMS)
    train_desc = extractor.descriptors_from_paths(data.train.paths)
    vocab = BagOfVisualWords(vocab_size=20).fit(train_desc)
    clf = build_classifiers()["SVM (RBF)"].fit(vocab.transform(train_desc), data.train.labels)

    model = AnimalClassifier(PARAMS, vocab, clf, data.class_names, "test")
    model.save(tmp_path / "model.joblib")
    loaded = AnimalClassifier.load(tmp_path / "model.joblib")
    predictions = loaded.predict(data.test.paths)
    accuracy = np.mean(np.array(predictions) == np.array(data.test.labels))
    assert accuracy > 0.8
