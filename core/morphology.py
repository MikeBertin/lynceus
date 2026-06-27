"""Morphology label schema and metric helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Single dominant-class scheme for the M1 on-ramp. The real CEERS VISUAL
# catalogue (Kartaltepe+ 2023) uses multi-flag classifications; collapsing to a
# dominant class keeps the on-ramp simple and the demo legible. The labels and
# their order are fixed here so training, export and the browser demo agree.
CLASSES = ("disk", "spheroid", "irregular", "point_source", "merger")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Human-friendly labels + one-line descriptions for the demo UI.
CLASS_LABELS = {
    "disk": "Disk",
    "spheroid": "Spheroid",
    "irregular": "Irregular",
    "point_source": "Point source",
    "merger": "Merger",
}
CLASS_BLURB = {
    "disk": "Rotation-supported, exponential light profile (Sersic n~1).",
    "spheroid": "Centrally concentrated bulge/elliptical (Sersic n~4).",
    "irregular": "Clumpy, asymmetric — common in the early universe.",
    "point_source": "Unresolved: a star, quasar or compact source.",
    "merger": "Two interacting components / tidal disturbance.",
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
