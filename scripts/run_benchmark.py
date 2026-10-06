#!/usr/bin/env python
"""Train and evaluate the SIFT + Bag-of-Visual-Words pipeline with several classifiers.

SIFT descriptors are extracted once; for each vocabulary size a shared K-Means codebook is fitted on
the training images only, every image is encoded as a word histogram, and each classifier is trained
on the training histograms and scored on the held-out test set.

Example:
    python scripts/run_benchmark.py --data-root data/animals --vocab-sizes 17 50 100 200
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

from sklearn.base import clone

from animal_bovw.classifiers import build_classifiers
from animal_bovw.config import Config
from animal_bovw.data import load_dataset
from animal_bovw.features import FeatureParams, SiftExtractor
from animal_bovw.model import AnimalClassifier
from animal_bovw.reporting import (
    evaluate,
    plot_confusion_matrix,
    plot_model_comparison,
    plot_pipeline_example,
    summary_table,
    text_report,
)
from animal_bovw.vocabulary import BagOfVisualWords


def parse_args() -> tuple[Config, list]:
    d = Config()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=d.data_root)
    p.add_argument("--vocab-sizes", nargs="+", type=int, default=[d.vocab_size],
                   help="one or more K-Means vocabulary sizes to compare")
    p.add_argument("--vocab-sample", type=int, default=d.vocab_sample)
    p.add_argument("--no-canny", dest="use_canny", action="store_false",
                   help="run SIFT on the grayscale image instead of the Canny edge map")
    p.add_argument("--canny-low", type=int, default=d.canny_low)
    p.add_argument("--canny-high", type=int, default=d.canny_high)
    p.add_argument("--max-keypoints", type=int, default=d.max_keypoints)
    p.add_argument("--test-size", type=float, default=d.test_size)
    p.add_argument("--seed", type=int, default=d.seed)
    p.add_argument("--output-dir", default=d.output_dir)
    args = vars(p.parse_args())
    vocab_sizes = args.pop("vocab_sizes")
    return Config(**args, vocab_size=vocab_sizes[0]), vocab_sizes


def main() -> None:
    cfg, vocab_sizes = parse_args()
    run_dir = Path(cfg.output_dir) / time.strftime("run-%Y%m%d-%H%M%S")
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "benchmark.log")])
    log = logging.getLogger("animal_bovw")
    (run_dir / "config.json").write_text(json.dumps({**cfg.to_dict(), "vocab_sizes": vocab_sizes}, indent=2))

    data = load_dataset(cfg.data_root, cfg.test_size, cfg.seed)
    log.info("Classes (%d): %s", len(data.class_names), ", ".join(data.class_names))
    log.info("Train images: %d | Test images: %d", len(data.train.paths), len(data.test.paths))

    params = FeatureParams(cfg.image_width, cfg.image_height, cfg.use_canny, cfg.canny_low,
                           cfg.canny_high, cfg.max_keypoints)
    extractor = SiftExtractor(params)
    train_desc = extractor.descriptors_from_paths(data.train.paths, "SIFT (train)")
    test_desc = extractor.descriptors_from_paths(data.test.paths, "SIFT (test)")
    n_kp = [len(d) for d in train_desc]
    log.info("SIFT keypoints per training image: mean %.0f, min %d, max %d",
             sum(n_kp) / len(n_kp), min(n_kp), max(n_kp))
    plot_pipeline_example(data.test.paths[0], params, run_dir / "figures" / "pipeline_example.png")

    rows, best = [], None
    for k in vocab_sizes:
        vocab = BagOfVisualWords(k, cfg.vocab_sample, cfg.seed).fit(train_desc)   # training images only
        x_train, x_test = vocab.transform(train_desc), vocab.transform(test_desc)
        for name, template in build_classifiers(cfg.seed).items():
            clf = clone(template).fit(x_train, data.train.labels)
            train_m = evaluate(data.train.labels, clf.predict(x_train))
            y_pred = clf.predict(x_test)
            test_m = evaluate(data.test.labels, y_pred)
            rows.append({"vocab_size": k, "model": name, "train_accuracy": train_m["accuracy"],
                         **{f"test_{m}": v for m, v in test_m.items()}})
            log.info("k=%-4d %-20s train_acc=%.4f  test_acc=%.4f  test_macro_f1=%.4f",
                     k, name, train_m["accuracy"], test_m["accuracy"], test_m["macro_f1"])
            if best is None or test_m["accuracy"] > best[0]:
                best = (test_m["accuracy"], AnimalClassifier(params, vocab, clf, data.class_names,
                                                             f"{name} (k={k})"), y_pred)

    summary = summary_table(rows)
    summary.to_csv(run_dir / "summary.csv", index=False)
    chance = max(data.test.labels.count(c) for c in data.class_names) / len(data.test.labels)
    for k in vocab_sizes:
        plot_model_comparison(summary[summary.vocab_size == k].reset_index(drop=True), chance,
                              run_dir / "figures" / f"model_comparison_k{k}.png")

    _, model, y_pred = best
    model.save(run_dir / "best_model.joblib")
    report = text_report(data.test.labels, y_pred, data.class_names)
    (run_dir / "best_classification_report.txt").write_text(f"{model.name}\n\n{report}")
    plot_confusion_matrix(data.test.labels, y_pred, data.class_names, f"Confusion matrix — {model.name}",
                          run_dir / "figures" / "best_confusion_matrix.png")

    log.info("\n%s", summary.to_string(index=False, float_format="%.4f"))
    log.info("Majority-class baseline accuracy: %.4f", chance)
    log.info("Best: %s — test accuracy %.4f\n%s", model.name, best[0], report)
    log.info("Outputs written to %s", run_dir)


if __name__ == "__main__":
    main()
