"""Classifiers trained on the visual-word histograms."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def build_classifiers(seed: int = 42) -> dict[str, Pipeline]:
    """The four models from the original notebook plus an RBF-kernel SVM.

    Histogram features are standardised for the linear / kernel models; trees do not need it.
    """
    return {
        "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000,
                                                                                  random_state=seed)),
        "Decision Tree": make_pipeline(DecisionTreeClassifier(max_depth=9, random_state=seed)),
        "Random Forest": make_pipeline(RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=seed)),
        "SVM (linear)": make_pipeline(StandardScaler(), SVC(kernel="linear", random_state=seed)),
        "SVM (RBF)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", random_state=seed)),
    }
