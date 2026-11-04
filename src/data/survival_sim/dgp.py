"""Factual simulator for the canonical long-memory survival process."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np

from . import equations as eq


@dataclass(frozen=True)
class SurvivalSimConfig:
    """Configuration for the canonical non-Markovian synthetic survival process."""

    max_seq_length: int = 32
    overlap_strength: float = 0.75
    heterogeneity_strength: float = 3.0
    feedback_strength: float = 0.8
    censoring_intercept: float = 5.2
    event_intercept: float = -4.0
    treatment_effect_scale: float = 1.0
    propensity_clip: float = 0.08
    censoring_enabled: bool = True

    def __post_init__(self):
        if self.max_seq_length < 1:
            raise ValueError("max_seq_length must be positive")
        if not 0.0 <= self.propensity_clip < 0.5:
            raise ValueError("propensity_clip must be in [0, 0.5)")
        numeric = (
            self.overlap_strength,
            self.heterogeneity_strength,
            self.feedback_strength,
            self.censoring_intercept,
            self.event_intercept,
            self.treatment_effect_scale,
            self.propensity_clip,
        )
        if not np.isfinite(numeric).all():
            raise ValueError("simulation parameters must be finite")


def simulate_split(
    num_patients: int,
    seed: int,
    config: SurvivalSimConfig | dict,
    *,
    initial_history=None,
    initial_baseline=None,
    patient_ids=None,
) -> dict:
    """Simulate one factual patient split with long-memory treatment-confounder feedback."""
    cfg = (
        config if isinstance(config, SurvivalSimConfig) else SurvivalSimConfig(**config)
    )
    if int(num_patients) != num_patients or num_patients < 1:
        raise ValueError("num_patients must be a positive integer")
    start_time = 0
    if initial_history is not None:
        history = np.asarray(initial_history, dtype=np.float32)
        supplied_baseline = np.asarray(initial_baseline, dtype=np.float32)
        supplied_ids = np.asarray(patient_ids)
        if (
            history.ndim != 3
            or history.shape[0] != num_patients
            or history.shape[2] != 3
        ):
            raise ValueError(
                "initial history must have shape (patients, prefix length, 3)"
            )
        start_time = history.shape[1] - 1
        if not 0 <= start_time < cfg.max_seq_length:
            raise ValueError("initial history must end before final decision")
        if (
            supplied_baseline.shape != (num_patients, 3)
            or not np.isfinite(supplied_baseline).all()
        ):
            raise ValueError("finite three-channel baseline required")
        if (
            not np.isfinite(history).all()
            or not np.isin(supplied_baseline[:, 1], [0, 1]).all()
        ):
            raise ValueError(
                "finite history and binary second baseline channel required"
            )
        if supplied_ids.shape != (num_patients,) or not np.issubdtype(
            supplied_ids.dtype, np.integer
        ):
            raise ValueError("integer patient IDs required")
        if len(np.unique(supplied_ids)) != num_patients or np.any(supplied_ids < 0):
            raise ValueError("unique nonnegative patient IDs required")
    elif initial_baseline is not None or patient_ids is not None:
        raise ValueError("baseline and IDs require an initial history")
    rng = np.random.default_rng(seed)
    n, max_time = int(num_patients), int(cfg.max_seq_length)

    baseline = np.column_stack(
        [rng.normal(size=n), rng.binomial(1, 0.5, size=n), rng.normal(size=n)]
    ).astype(np.float32)
    states = np.full((n, max_time + 1, 3), np.nan, dtype=np.float32)
    treatments = np.full((n, max_time), np.nan, dtype=np.float32)
    at_risk = np.zeros((n, max_time), dtype=np.float32)
    state_observed = np.zeros((n, max_time + 1), dtype=np.float32)
    treatment_observed = np.zeros((n, max_time), dtype=np.float32)
    event_free_next = np.full((n, max_time), np.nan, dtype=np.float32)
    event_next = np.full((n, max_time), np.nan, dtype=np.float32)
    censoring_continue = np.full((n, max_time), np.nan, dtype=np.float32)
    censoring_label_observed = np.zeros((n, max_time), dtype=np.float32)
    true_treatment_probability = np.full((n, max_time), np.nan, dtype=np.float32)
    true_event_hazard = np.full((n, max_time), np.nan, dtype=np.float32)
    true_censoring_probability = np.full((n, max_time), np.nan, dtype=np.float32)
    theta_true = np.full((n, max_time), np.nan, dtype=np.float32)
    event_time = np.full(n, max_time + 1, dtype=np.int32)
    censoring_time = np.full(n, max_time + 1, dtype=np.int32)
    observed_time = np.full(n, max_time, dtype=np.int32)
    delta = np.zeros(n, dtype=np.int32)

    states[:, 0, 0] = (
        0.55 * baseline[:, 0] + 0.35 * baseline[:, 1] + rng.normal(0, 0.45, n)
    )
    states[:, 0, 1] = (
        0.45 * baseline[:, 2] - 0.20 * baseline[:, 0] + rng.normal(0, 0.40, n)
    )
    states[:, 0, 2] = rng.normal(0, 0.15, n)
    state_observed[:, 0] = 1.0
    if initial_history is not None:
        baseline = supplied_baseline.copy()
        states[:, : start_time + 1] = history
        state_observed[:, : start_time + 1] = 1.0
        # Artificial untreated initialization, not observed clinical treatment.
        treatments[:, :start_time] = 0.0
        treatment_observed[:, :start_time] = 1.0
    active = np.ones(n, dtype=bool)

    for k in range(start_time, max_time):
        rows = np.flatnonzero(active)
        if len(rows) == 0:
            break
        at_risk[rows, k] = 1.0
        treatment_observed[rows, k] = 1.0

        terms = eq.history_terms(states, treatments, rows, k)
        p = eq.propensity(cfg, baseline[rows], terms, k)
        true_treatment_probability[rows, k] = p
        a = rng.binomial(1, p).astype(float)
        treatments[rows, k] = a

        hazard = eq.hazard(cfg, baseline[rows], terms, k, a)
        true_event_hazard[rows, k] = hazard
        theta_true[rows, k] = cfg.treatment_effect_scale * eq.effect_modifier(
            cfg, baseline[rows], terms
        )
        event = rng.binomial(1, hazard).astype(bool)
        event_next[rows, k] = event.astype(float)
        event_free_next[rows, k] = 1.0 - event.astype(float)

        event_rows = rows[event]
        if len(event_rows):
            event_time[event_rows] = k + 1
            observed_time[event_rows] = k + 1
            delta[event_rows] = 1
            active[event_rows] = False

        survive_rows = rows[~event]
        if len(survive_rows) == 0:
            continue
        survive_terms = eq.history_terms(states, treatments, survive_rows, k)
        next_state = eq.transition(
            cfg,
            baseline[survive_rows],
            survive_terms,
            k,
            treatments[survive_rows, k].astype(float),
            rng.normal(0, 0.28, len(survive_rows)),
            rng.normal(0, 0.22, len(survive_rows)),
            rng.normal(0, 0.08, len(survive_rows)),
        )
        states[survive_rows, k + 1] = next_state
        state_observed[survive_rows, k + 1] = 1.0

        successor_terms = eq.history_terms(states, treatments, survive_rows, k + 1)
        if cfg.censoring_enabled:
            g = eq.censoring_continuation(cfg, successor_terms, k + 1)
            remain = rng.binomial(1, g).astype(bool)
        else:
            g = np.ones(len(survive_rows), dtype=float)
            remain = np.ones(len(survive_rows), dtype=bool)
        true_censoring_probability[survive_rows, k] = g
        censoring_label_observed[survive_rows, k] = 1.0
        censoring_continue[survive_rows, k] = remain.astype(float)

        censored_rows = survive_rows[~remain]
        if len(censored_rows):
            censoring_time[censored_rows] = k + 1
            observed_time[censored_rows] = k + 1
            active[censored_rows] = False

        continuing_rows = survive_rows[remain]
        observed_time[continuing_rows] = k + 1

    return {
        "baseline": baseline,
        "states": states,
        "treatments": treatments,
        "at_risk": at_risk,
        "state_observed": state_observed,
        "treatment_observed": treatment_observed,
        "event_free_next": event_free_next,
        "event_next": event_next,
        "censoring_continue": censoring_continue,
        "censoring_label_observed": censoring_label_observed,
        "true_treatment_probability": true_treatment_probability,
        "true_event_hazard": true_event_hazard,
        "true_censoring_probability": true_censoring_probability,
        "theta_true": theta_true,
        "event_time": event_time,
        "censoring_time": censoring_time,
        "observed_time": observed_time,
        "delta": delta,
        "patient_id": (
            np.arange(n, dtype=np.int32)
            if initial_history is None
            else supplied_ids.astype(np.int64, copy=True)
        ),
        "config": cfg,
        "config_json": json.dumps(asdict(cfg)),
    }
