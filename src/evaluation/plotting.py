"""Shared repetition-level plotting helpers."""

from __future__ import annotations

import numpy as np

from src.evaluation.metrics import mean_confidence_interval


def interval(values):
    values = np.asarray(values, float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("interval requires finite replicate values")
    mean, _, low, high = mean_confidence_interval(values)
    return mean, low, high


def points(axis, values, x, color, label=None):
    values = np.asarray(values, float)
    mean, low, high = interval(values)
    axis.scatter(np.full(len(values), x), values, s=18, alpha=0.45, color=color)
    if np.isfinite(low):
        axis.errorbar(
            x,
            mean,
            yerr=[[mean - low], [high - mean]],
            fmt="D",
            color=color,
            capsize=3,
            markersize=5,
            label=label,
        )
    else:
        axis.plot(x, mean, "D", color=color, markersize=5, label=label)
