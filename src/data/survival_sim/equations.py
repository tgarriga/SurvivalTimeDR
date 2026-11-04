"""Canonical structural equations for the long-memory survival simulation.

The simulator, analytic one-step truth and counterfactual oracle all call this module.
The observed process is deliberately not first-order Markov in ``X_k``: treatment,
event risk and state transitions depend on lagged states, rolling summaries and prior
treatment exposure. Learners are not given these summaries directly; a sequence model
must recover the relevant information from the observed history.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from scipy.special import expit


class HistoryTerms(NamedTuple):
    x: np.ndarray
    lag1: np.ndarray
    lag2: np.ndarray
    mean4: np.ndarray
    mean8: np.ndarray
    trend3: np.ndarray
    prev_a: np.ndarray
    recent4: np.ndarray
    cumulative_a: np.ndarray


def history_terms(states, treatments, rows, k: int) -> HistoryTerms:
    """Observed-history summaries at calendar time ``k`` for selected patients."""
    idx = np.asarray(rows, dtype=int)
    x = np.asarray(states[idx, k], dtype=float)
    lag1 = np.asarray(states[idx, max(0, k - 1)], dtype=float)
    lag2 = np.asarray(states[idx, max(0, k - 2)], dtype=float)
    mean4 = np.nanmean(
        np.asarray(states[idx, max(0, k - 3) : k + 1], dtype=float), axis=1
    )
    mean8 = np.nanmean(
        np.asarray(states[idx, max(0, k - 7) : k + 1], dtype=float), axis=1
    )
    trend3 = x - np.asarray(states[idx, max(0, k - 3)], dtype=float)
    if k == 0:
        zero = np.zeros(len(idx), dtype=float)
        prev_a = recent4 = cumulative_a = zero
    else:
        observed = np.asarray(treatments[idx, :k], dtype=float)
        prev_a = np.nan_to_num(observed[:, -1], nan=0.0)
        recent4 = np.nanmean(observed[:, max(0, k - 4) : k], axis=1)
        cumulative_a = np.nanmean(observed, axis=1)
    return HistoryTerms(
        x, lag1, lag2, mean4, mean8, trend3, prev_a, recent4, cumulative_a
    )


def buffer_terms(
    state_buffer, treatment_buffer, cumulative_sum, k: int
) -> HistoryTerms:
    """Same summaries as :func:`history_terms`, from an eight-step rollout buffer."""
    x = state_buffer[..., -1, :]
    lag1 = state_buffer[..., -(min(k, 1) + 1), :]
    lag2 = state_buffer[..., -(min(k, 2) + 1), :]
    mean4 = state_buffer[..., -min(k + 1, 4) :, :].mean(axis=-2)
    mean8 = state_buffer[..., -min(k + 1, 8) :, :].mean(axis=-2)
    trend3 = x - state_buffer[..., -(min(k, 3) + 1), :]
    if k == 0:
        zero = np.zeros(x.shape[:-1], dtype=float)
        prev_a = recent4 = zero
    else:
        prev_a = treatment_buffer[..., -1]
        recent4 = treatment_buffer[..., -min(k, 4) :].mean(axis=-1)
    cumulative_a = cumulative_sum / max(k, 1)
    return HistoryTerms(
        x, lag1, lag2, mean4, mean8, trend3, prev_a, recent4, cumulative_a
    )


def propensity_logit(cfg, w, terms: HistoryTerms, k: int):
    x, _, _, mean4, mean8, trend3, prev_a, recent4, cumulative_a = terms
    return (
        -0.10
        + 0.62 * x[..., 0]
        - 0.25 * x[..., 1]
        - 0.12 * x[..., 2]
        + 0.28 * mean4[..., 0]
        - 0.12 * mean8[..., 1]
        + 0.18 * trend3[..., 0]
        + 0.28 * w[..., 1]
        + 0.45 * prev_a
        + 0.22 * recent4
        + 0.10 * cumulative_a
        + 0.12 * np.sin(k / 4.0)
    )


def propensity(cfg, w, terms: HistoryTerms, k: int):
    """P(A_k=1 | H_k), including the configured overlap scaling and clip."""
    eta = propensity_logit(cfg, w, terms, k)
    return np.clip(
        expit(cfg.overlap_strength * eta),
        cfg.propensity_clip,
        1.0 - cfg.propensity_clip,
    )


def effect_modifier(cfg, w, terms: HistoryTerms):
    x, _, _, mean4, _, trend3, _, recent4, _ = terms
    return cfg.heterogeneity_strength * (
        0.48
        + 0.16 * w[..., 1]
        + 0.18 * np.tanh(x[..., 0])
        + 0.14 * np.tanh(mean4[..., 0])
        - 0.10 * x[..., 2]
        - 0.08 * recent4
        + 0.06 * np.tanh(trend3[..., 0])
    )


def treatment_hazard_term(cfg, a, prev_a, theta):
    return cfg.treatment_effect_scale * (-a * theta - 0.08 * prev_a + 0.10 * a * prev_a)


def hazard(cfg, w, terms: HistoryTerms, k: int, a):
    """P(event in [k,k+1) | H_k, A_k=a)."""
    x, _, _, mean4, mean8, trend3, prev_a, _, _ = terms
    theta = effect_modifier(cfg, w, terms)
    return expit(
        cfg.event_intercept
        + 0.035 * k
        + 0.58 * x[..., 0]
        - 0.30 * x[..., 1]
        + 0.16 * x[..., 2]
        + 0.22 * mean4[..., 0]
        + 0.10 * mean8[..., 2]
        + 0.12 * trend3[..., 0]
        + 0.12 * w[..., 0]
        + treatment_hazard_term(cfg, a, prev_a, theta)
    )


def transition(cfg, w, terms: HistoryTerms, k: int, a, eps_s, eps_r, eps_b):
    """Draw-free transition equation for X_{k+1}; noise is supplied by the caller."""
    x, lag1, lag2, mean4, mean8, _, prev_a, recent4, _ = terms
    scale = cfg.treatment_effect_scale * cfg.feedback_strength
    s_next = (
        0.50 * x[..., 0]
        + 0.18 * lag1[..., 0]
        - 0.10 * x[..., 1]
        + 0.12 * mean4[..., 0]
        + 0.10 * w[..., 0]
        + 0.06 * np.sin(k / 3.5)
        + scale
        * (
            -a * (0.34 + 0.12 * w[..., 1] + 0.10 * np.tanh(mean4[..., 0]))
            + 0.05 * prev_a
        )
        + eps_s
    )
    r_next = (
        0.56 * x[..., 1]
        + 0.16 * lag2[..., 1]
        - 0.10 * x[..., 0]
        - 0.07 * mean4[..., 0]
        + 0.10 * w[..., 2]
        - 0.05 * x[..., 2]
        - scale * 0.08 * a
        + eps_r
    )
    b_next = (
        0.74 * x[..., 2]
        + scale * (0.65 * a + 0.15 * recent4)
        + 0.04 * np.tanh(mean8[..., 0])
        + eps_b
    )
    return np.stack([s_next, r_next, b_next], axis=-1)


def censoring_continuation(cfg, successor_terms: HistoryTerms, j: int):
    """P(C>j | H_j,T>j,C>j-1), evaluated on the successor history."""
    x, _, _, mean4, _, _, prev_a, _, _ = successor_terms
    return expit(
        cfg.censoring_intercept
        - 0.28 * x[..., 0]
        + 0.16 * x[..., 1]
        - 0.10 * x[..., 2]
        - 0.12 * mean4[..., 0]
        - 0.08 * prev_a
        - 0.020 * j
    )


def one_step_survival(cfg, w, terms: HistoryTerms, k: int, a):
    return 1.0 - hazard(cfg, w, terms, k, a)
