"""Morphology label schema and metric helpers."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

# Three structural classes derived from single-Sersic profile fits (DJA /
# van der Wel et al. 2025) of real CEERS NIRCam galaxies: the classic
# early/late dichotomy plus a compact/unresolved class. Labels and order are
# fixed here so the data layer, training, export and the browser demo agree.
# (The synthetic Sersic generator renders these same three classes.)
CLASSES = ("disk", "spheroid", "compact")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASSES)}

# Human-friendly labels + one-line descriptions for the demo UI.
CLASS_LABELS = {
    "disk": "Disk",
    "spheroid": "Spheroid",
    "compact": "Compact",
}
CLASS_BLURB = {
    "disk": "Late-type, rotation-supported — low Sersic index (n < 1.2).",
    "spheroid": "Early-type bulge/elliptical — high Sersic index (n > 2.5).",
    "compact": "Barely resolved (R_eff < 0.09'') — point-source-like; where the Little Red Dots hide.",
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
