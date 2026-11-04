"""Observation-time helpers shared by estimators and correctness checks."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def observed_survival_indicator(
    raw: Mapping[str, Any], horizon_time: int
) -> np.ndarray:
    """Whether the observed record establishes survival through ``horizon_time``.

    Censoring exactly at the horizon still establishes survival through that horizon;
    an event at the horizon does not.
    """
    if horizon_time < 0:
        raise ValueError("horizon_time cannot be negative")
    return (
        (raw["observed_time"] > horizon_time)
        | ((raw["observed_time"] == horizon_time) & (raw["delta"] == 0))
    ).astype(np.float32)


def observed_survival_next(raw: Mapping[str, Any], decision_time: int) -> np.ndarray:
    """Observed survival through the end of one decision interval."""
    if decision_time < 0:
        raise ValueError("decision_time cannot be negative")
    return observed_survival_indicator(raw, decision_time + 1)
