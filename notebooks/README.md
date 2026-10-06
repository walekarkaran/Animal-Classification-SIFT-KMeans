# Notebooks

`animal_classification_sift_kmeans.ipynb` is the original exploratory notebook from 2024, kept with its
outputs for reference. It uses hard-coded Windows paths and is not meant to be re-run.

**Its reported accuracies (≈98%) are not valid estimates.** Each class was encoded with its own
K-Means vocabulary, which leaks the label into the features, and the feature table contained duplicated
rows. The main [README](../README.md#about-the-original-notebook) explains this in full. The maintained,
corrected implementation is the `animal_bovw` package in [`../src/`](../src/), run through
[`../scripts/run_benchmark.py`](../scripts/run_benchmark.py).
