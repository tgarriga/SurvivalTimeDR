"""Real pre-entry histories initializing the shared synthetic survival process."""

import json
from pathlib import Path

import numpy as np

from src.data.mimic_semisynthetic.checkpoints import sha256
from src.data.mimic_semisynthetic.histories import (
    FEATURES,
    SPLITS,
    STATIC,
    HistoryPreprocessor,
    subject_partition,
)
from src.data.survival_sim.dataset import SurvivalSyntheticDataset
from src.data.survival_sim.dgp import SurvivalSimConfig, simulate_split
from src.models.transformer_training import patient_train_validation_indices

ENTRY = 7
CHANNELS = ("heart_rate", "mean_bp", "respiratory_rate")
VALIDATION_SEED = 20260917


def inner_rows(ids):
    return patient_train_validation_indices(
        list(ids), 0.15, VALIDATION_SEED, patient_ids=ids
    )


def prepare_arrays(values, baseline, ids, roles):
    """Fit all initialization transforms on nuisance inner-fit prefixes only."""
    ids, roles = np.asarray(ids), np.asarray(roles)
    values, baseline = np.asarray(values), np.asarray(baseline)
    if values.shape[0] != len(ids) or values.shape[1] < ENTRY + 1:
        raise ValueError("insufficient or misaligned real histories")
    if len(np.unique(ids)) != len(ids) or set(roles) != set(SPLITS):
        raise ValueError("patient roles must be complete and IDs unique")
    if not np.array_equal(roles, [subject_partition(int(i)) for i in ids]):
        raise ValueError("patient split identity mismatch")
    prefix = values[:, : ENTRY + 1].copy()
    nu = np.flatnonzero(roles == "nuisance")
    fit, validation = inner_rows(ids[nu])
    fit = nu[fit]
    validation = nu[validation]
    prep = HistoryPreprocessor.fit(prefix[fit], baseline[fit])
    x, w = prep.transform(prefix, baseline)
    columns = [FEATURES.index(c) for c in CHANNELS]
    history = x[:, :, columns].astype(np.float32)
    # The third static channel is a declared observed-history summary.
    respiratory_mean = history[:, :, 2].mean(axis=1).astype(float)
    centre = float(respiratory_mean[fit].mean())
    scale = max(float(respiratory_mean[fit].std()), 1e-6)
    static = np.column_stack(
        [w[:, 0], baseline[:, 1], np.clip((respiratory_mean - centre) / scale, -5, 5)]
    ).astype(np.float32)
    if not np.isin(static[:, 1], [0, 1]).all():
        raise ValueError("recorded sex must remain binary")
    arrays = dict(
        history=history,
        baseline=static,
        patient_id=ids.astype(np.int64),
        roles=roles,
        preprocessing_fit_ids=ids[fit],
        preprocessing_validation_ids=ids[validation],
    )
    metadata = dict(
        entry=ENTRY,
        prefix_steps=ENTRY + 1,
        grid_hours=3,
        channels=list(CHANNELS),
        static_channels=[
            "standardized_age",
            "female_0_1",
            "standardized_prefix_mean_respiratory_score",
        ],
        preprocessor=prep.as_dict(),
        respiratory_mean_centre=centre,
        respiratory_mean_scale=scale,
        transformation_fit="nuisance inner-training patients; pre-24-hour measurements only",
        initial_treatment_history="artificial all-zero history; not observed clinical treatment",
        subsequent_states="manuscript structural equations, not a fitted physiological model",
    )
    return arrays, metadata


def prepare_archive(path, expected_sha256=None):
    path = Path(path)
    if expected_sha256 is None:
        raise ValueError(
            "an explicit archive SHA-256 is required; the unified initialization command computes it"
        )
    expected = expected_sha256
    if sha256(path) != expected:
        raise ValueError("source archive identity mismatch")
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive["manifest"].item()))
        if (
            tuple(meta["features"]) != FEATURES
            or tuple(meta["static_features"]) != STATIC
        ):
            raise ValueError("source feature order mismatch")
        if meta["schema"] not in (2, 3) or (
            meta["landmark_hours"],
            meta["grid_hours"],
        ) != (24, 3):
            raise ValueError("source time contract mismatch")
        result = prepare_arrays(
            archive["values"],
            archive["baseline"],
            archive["subject_id"],
            archive["split"],
        )
    result[1].update(source_schema=meta["schema"], source_archive_sha256=expected)
    if sha256(path) != expected:
        raise ValueError("source changed during preparation")
    return result


def prefix_only(raw):
    """Keep full grid/time scaling while removing every future test observation."""
    result = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in raw.items()}
    end = result["treatments"].shape[1]
    result["states"][:, ENTRY + 1 :] = np.nan
    result["state_observed"][:, ENTRY + 1 :] = 0
    for key in (
        "treatments",
        "event_free_next",
        "event_next",
        "censoring_continue",
        "true_treatment_probability",
        "true_event_hazard",
        "true_censoring_probability",
        "theta_true",
    ):
        result[key][:, ENTRY:] = np.nan
    result["treatment_observed"][:, ENTRY:] = 0
    result["censoring_label_observed"][:, ENTRY:] = 0
    result["at_risk"][:, ENTRY + 1 :] = 0
    result["event_time"][:] = end + 1
    result["censoring_time"][:] = end + 1
    result["observed_time"][:] = end
    result["delta"][:] = 0
    return result


def make_splits(data, replicate, simulation=None, *, study_seed=2027):
    cfg = simulation or SurvivalSimConfig()
    seed = study_seed + 100003 * replicate
    splits = {}
    for index, (name, role) in enumerate(zip(("train", "val", "test"), SPLITS)):
        selected = data["roles"] == role
        splits[name] = simulate_split(
            int(selected.sum()),
            seed + (11, 20011, 40009)[index],
            cfg,
            initial_history=data["history"][selected],
            initial_baseline=data["baseline"][selected],
            patient_ids=data["patient_id"][selected],
        )
    # Model scaling is separate from the real-history initialization transform.
    fit, _ = inner_rows(splits["train"]["patient_id"])
    small = {
        k: (
            v[fit].copy()
            if isinstance(v, np.ndarray)
            and v.ndim
            and len(v) == len(splits["train"]["patient_id"])
            else v
        )
        for k, v in splits["train"].items()
    }
    scaling = SurvivalSyntheticDataset(small).get_scaling_params()
    splits["test"] = prefix_only(splits["test"])
    return splits, scaling


def validate_prepared(data):
    """Check the patient and preprocessing contract before reusing an archive."""
    ids, roles = np.asarray(data["patient_id"]), np.asarray(data["roles"])
    if ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer):
        raise ValueError("prepared patient IDs must be a one-dimensional integer array")
    if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
        raise ValueError("prepared patient IDs must be globally unique and nonnegative")
    if roles.shape != ids.shape or set(roles) != set(SPLITS):
        raise ValueError("prepared patient roles must be aligned and complete")
    if not np.array_equal(roles, [subject_partition(int(i)) for i in ids]):
        raise ValueError("prepared patient split identity mismatch")
    history, baseline = np.asarray(data["history"]), np.asarray(data["baseline"])
    if history.shape != (len(ids), ENTRY + 1, 3) or baseline.shape != (len(ids), 3):
        raise ValueError("prepared history or baseline dimensions mismatch")
    if not np.isfinite(history).all() or not np.isfinite(baseline).all():
        raise ValueError("prepared arrays must be finite")
    if not np.isin(baseline[:, 1], [0, 1]).all():
        raise ValueError("prepared sex channel must be binary")
    nuisance_ids = ids[roles == "nuisance"]
    fit, validation = inner_rows(nuisance_ids)
    if not np.array_equal(data["preprocessing_fit_ids"], nuisance_ids[fit]):
        raise ValueError("prepared preprocessing fit IDs mismatch")
    if not np.array_equal(
        data["preprocessing_validation_ids"], nuisance_ids[validation]
    ):
        raise ValueError("prepared preprocessing validation IDs mismatch")
