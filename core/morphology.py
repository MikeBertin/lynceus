"""Morphology label schema and metric helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Three visual-morphology classes from real human Galaxy Zoo: CANDELS votes
# (Simmons et al. 2017), the canonical top-level split. Cross-matched to real
# JWST NIRCam cutouts. Labels and order are fixed here so the data layer,
# training, export and the browser demo agree. (The synthetic Sersic generator
# renders these same three classes for tests / the no-download fallback.)
CLASSES = ("featured", "smooth", "merger")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Human-friendly labels + one-line descriptions for the demo UI.
CLASS_LABELS = {
    "featured": "Featured / disk",
    "smooth": "Smooth",
    "merger": "Merger",
}
CLASS_BLURB = {
    "featured": "Disk-like: structure, clumps or spiral arms (Galaxy Zoo: featured).",
    "smooth": "Smooth and rounded: elliptical / early-type, no features.",
    "merger": "Merging or tidally disturbed: two bodies or tidal debris.",
}


def report(y_true, y_pred) -> dict:
    """Per-class precision/recall/F1 as a plain dict (json-serialisable)."""
    return classification_report(
        y_true, y_pred,
        labels=list(range(len(CLASSES))),
        target_names=list(CLASSES),
        output_dict=True,
        zero_division=0,
    )


def confusion(y_true, y_pred) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=list(range(len(CLASSES))))


def majority_baseline_accuracy(y_true) -> float:
    """Accuracy of always predicting the most common class."""
    y = np.asarray(y_true)
    if y.size == 0:
        return 0.0
    counts = np.bincount(y, minlength=len(CLASSES))
    return float(counts.max() / y.size)
