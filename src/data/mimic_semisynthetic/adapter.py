"""Dimension-general views reusing the unchanged survival tensor conventions."""

from __future__ import annotations

import numpy as np

from src.data.survival_sim.dataset import (
    SurvivalSyntheticDataset,
    SurvivalSyntheticDatasetCollection,
)


def identity_scaling(raw):
    """Scaling that leaves inputs untouched.

    Source histories are already standardized with nuisance patients only, in
    ``HistoryPreprocessor``, so the dataset must not estimate a second normalization.
    """
    return {
        "static_mean": np.zeros(raw["baseline"].shape[1]),
        "static_std": np.ones(raw["baseline"].shape[1]),
        "covariate_mean": np.zeros(raw["states"].shape[2]),
        "covariate_std": np.ones(raw["states"].shape[2]),
    }


def history_dataset(raw, task="survival"):
    """One split view carrying the identity scaling."""
    return SurvivalSyntheticDataset(raw, task, identity_scaling(raw))


class HistoryCollection(SurvivalSyntheticDatasetCollection):
    """Adds explicit, disjoint three-way subject partitions to the shared collection."""

    def _initialize(self, raw_splits, *, task, scaling_params=None):
        if set(raw_splits) != {"train", "val", "test"}:
            raise ValueError("explicit three-way split required; no split fallback")
        ids = [
            set(raw_splits[s]["patient_id"].tolist()) for s in ("train", "val", "test")
        ]
        if any(
            len(ids[i]) != len(raw_splits[s]["patient_id"])
            for i, s in enumerate(("train", "val", "test"))
        ):
            raise ValueError("duplicate subject within a partition")
        if any(not v for v in ids) or any(
            ids[i] & ids[j] for i in range(3) for j in range(i)
        ):
            raise ValueError("empty or overlapping subject partitions")
        super()._initialize(
            raw_splits,
            task=task,
            scaling_params=scaling_params or identity_scaling(raw_splits["train"]),
        )


def landmark_prediction_view(dataset, landmark):
    """Physically remove all post-landmark inputs before final model prediction."""
    raw = {key: value.copy() for key, value in dataset.raw.items()}
    if not 0 <= landmark < raw["treatments"].shape[1]:
        raise ValueError("landmark outside record")
    raw["states"][:, landmark + 1 :] = np.nan
    raw["state_observed"][:, landmark + 1 :] = 0
    raw["treatments"][:, landmark:] = np.nan
    raw["treatment_observed"][:, landmark:] = 0
    raw["at_risk"][:, landmark + 1 :] = 0
    raw["event_free_next"][:, landmark:] = np.nan
    end = raw["treatments"].shape[1]
    raw["event_time"][:] = end + 1
    raw["censoring_time"][:] = end + 1
    raw["observed_time"][:] = end
    raw["delta"][:] = 0
    return SurvivalSyntheticDataset(raw, "survival", dataset.scaling_params)
