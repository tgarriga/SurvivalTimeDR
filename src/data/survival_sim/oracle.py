"""Paired Monte Carlo truth for the canonical long-memory survival process."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import equations as eq
from .dgp import SurvivalSimConfig


def _resolve_config(raw, config) -> SurvivalSimConfig:
    cfg = config if config is not None else raw.get("config")
    if isinstance(cfg, np.ndarray) and cfg.shape == ():
        cfg = cfg.item()
    if cfg is None and raw.get("config_json") is not None:
        import json

        encoded = raw["config_json"]
        if isinstance(encoded, np.ndarray) and encoded.shape == ():
            encoded = encoded.item()
        cfg = json.loads(str(encoded))
    if cfg is None:
        cfg = SurvivalSimConfig(max_seq_length=raw["treatments"].shape[1])
    return cfg if isinstance(cfg, SurvivalSimConfig) else SurvivalSimConfig(**dict(cfg))


def _initial_buffers(raw, ids, landmark: int, n_rollouts: int):
    """Eight-step left-padded factual history at the intervention landmark."""
    ids = np.asarray(ids, dtype=int)
    state_indices = np.maximum(np.arange(landmark - 7, landmark + 1), 0)
    state = np.asarray(raw["states"][ids[:, None], state_indices[None, :]], dtype=float)
    state_buffer = np.broadcast_to(
        state[:, None, :, :], (len(ids), n_rollouts, 8, 3)
    ).copy()

    treatment_buffer_base = np.zeros((len(ids), 8), dtype=float)
    available = min(landmark, 8)
    if available:
        treatment_buffer_base[:, -available:] = np.nan_to_num(
            raw["treatments"][ids, landmark - available : landmark], nan=0.0
        )
    treatment_buffer = np.broadcast_to(
        treatment_buffer_base[:, None, :], (len(ids), n_rollouts, 8)
    ).copy()
    cumulative = np.nansum(raw["treatments"][ids, :landmark], axis=1)
    cumulative_sum = np.broadcast_to(cumulative[:, None], (len(ids), n_rollouts)).copy()
    w = np.broadcast_to(
        raw["baseline"][ids, None, :].astype(float), (len(ids), n_rollouts, 3)
    ).copy()
    return w, state_buffer, treatment_buffer, cumulative_sum


def _paired_draws(
    patient_ids,
    seed: int,
    n_rollouts: int,
    max_horizon: int,
):
    """Create deterministic per-patient draws, independent of batching."""
    per_patient = []
    shape = (n_rollouts, max_horizon)
    for patient_id in patient_ids:
        sequence = np.random.SeedSequence([int(seed), int(patient_id)])
        rng = np.random.default_rng(sequence)
        per_patient.append(
            (
                rng.random(shape),
                rng.random(shape),
                rng.normal(0, 0.28, shape),
                rng.normal(0, 0.22, shape),
                rng.normal(0, 0.08, shape),
            )
        )
    return tuple(
        np.stack([draws[index] for draws in per_patient], axis=0) for index in range(5)
    )


def _rollout_batch(raw, ids, landmark, regime, horizon_list, n_rollouts, cfg, draws):
    w, state_buffer, treatment_buffer, cumulative_sum = _initial_buffers(
        raw, ids, landmark, n_rollouts
    )
    u_treatment, u_event, eps_s, eps_r, eps_b = draws
    alive = np.ones((len(ids), n_rollouts), dtype=bool)
    snapshots: dict[int, np.ndarray] = {}
    max_horizon = max(horizon_list)

    for step in range(max_horizon):
        k = landmark + step
        terms = eq.buffer_terms(state_buffer, treatment_buffer, cumulative_sum, k)
        if step < len(regime):
            a = np.full(alive.shape, float(regime[step]))
        else:
            p = eq.propensity(cfg, w, terms, k)
            a = (u_treatment[..., step] < p).astype(float)
        hazard = eq.hazard(cfg, w, terms, k, a)
        alive &= u_event[..., step] >= hazard
        if step + 1 in horizon_list:
            snapshots[step + 1] = alive.copy()

        if step == max_horizon - 1:
            continue
        next_state = eq.transition(
            cfg,
            w,
            terms,
            k,
            a,
            eps_s[..., step],
            eps_r[..., step],
            eps_b[..., step],
        )
        next_state = np.where(alive[..., None], next_state, state_buffer[..., -1, :])
        state_buffer[..., :-1, :] = state_buffer[..., 1:, :]
        state_buffer[..., -1, :] = next_state
        treatment_buffer[..., :-1] = treatment_buffer[..., 1:]
        treatment_buffer[..., -1] = a
        cumulative_sum += a
    return snapshots


def oracle_pair(
    raw,
    landmark_time,
    horizons,
    regime_a,
    regime_b,
    n_rollouts: int = 1000,
    seed: int = 123,
    config: SurvivalSimConfig | None = None,
    batch_size: int = 64,
) -> pd.DataFrame:
    """Paired counterfactual survival truth under two finite treatment sequences."""
    cfg = _resolve_config(raw, config)
    regime_a = tuple(int(a) for a in regime_a)
    regime_b = tuple(int(a) for a in regime_b)
    if any(action not in (0, 1) for action in regime_a + regime_b):
        raise ValueError("regimes must contain binary treatment values")
    if len(regime_a) != len(regime_b):
        raise ValueError("regime lengths must agree")
    scalar_horizon = np.isscalar(horizons)
    horizon_list = (
        (int(horizons),) if scalar_horizon else tuple(int(h) for h in horizons)
    )
    if not horizon_list or min(horizon_list) < 1:
        raise ValueError("horizons must be positive")
    if len(set(horizon_list)) != len(horizon_list):
        raise ValueError("horizons must be unique")
    if landmark_time < 0 or landmark_time >= raw["treatments"].shape[1]:
        raise ValueError("landmark is outside available follow-up")
    if len(regime_a) > max(horizon_list):
        raise ValueError("regime length cannot exceed the largest horizon")
    if landmark_time + max(horizon_list) > raw["treatments"].shape[1]:
        raise ValueError("landmark plus horizon exceeds available follow-up")
    if n_rollouts < 2:
        raise ValueError("n_rollouts must be at least two")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    ids = np.flatnonzero(raw["at_risk"][:, landmark_time] > 0)
    if len(ids) == 0:
        raise ValueError("no patients remain at risk at the requested landmark")
    records: list[dict] = []
    for start in range(0, len(ids), batch_size):
        batch = ids[start : start + batch_size]
        draws = _paired_draws(
            raw["patient_id"][batch],
            seed,
            n_rollouts,
            max(horizon_list),
        )
        surv_a = _rollout_batch(
            raw, batch, landmark_time, regime_a, horizon_list, n_rollouts, cfg, draws
        )
        surv_b = _rollout_batch(
            raw, batch, landmark_time, regime_b, horizon_list, n_rollouts, cfg, draws
        )
        for horizon in horizon_list:
            for position, patient in enumerate(batch):
                a_draws = surv_a[horizon][position].astype(float)
                b_draws = surv_b[horizon][position].astype(float)
                difference = a_draws - b_draws
                record = {
                    "patient_id": int(raw["patient_id"][patient]),
                    "horizon": horizon,
                    "survival_a_true": float(a_draws.mean()),
                    "survival_b_true": float(b_draws.mean()),
                    "effect_true": float(difference.mean()),
                    "mcse_a": float(a_draws.std(ddof=1) / np.sqrt(n_rollouts)),
                    "mcse_b": float(b_draws.std(ddof=1) / np.sqrt(n_rollouts)),
                    "mcse_effect": float(difference.std(ddof=1) / np.sqrt(n_rollouts)),
                }
                if scalar_horizon:
                    record.pop("horizon")
                records.append(record)
    truth = pd.DataFrame(records)
    order = ["patient_id"] if scalar_horizon else ["patient_id", "horizon"]
    return truth.sort_values(order).reset_index(drop=True)
