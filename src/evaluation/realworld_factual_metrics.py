from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from src.data.mimic_realworld.extract import ENTRY
from src.protocols import (
    file_sha256,
    load_protocol,
    protocol_snapshot,
    runtime_versions,
)

PROTOCOL = load_protocol("realworld")
PROBABILITY_FLOOR = PROTOCOL["evaluation"]["probability_floor"]
WEIGHT_CAP = PROTOCOL["evaluation"]["weight_cap"]
BOOTSTRAP_DRAWS = PROTOCOL["evaluation"]["bootstrap_draws"]
BOOTSTRAP_SEED = PROTOCOL["evaluation"]["bootstrap_seed"]
METHODS = ("pi", "ra", "iptw_ipcw", "dr", "tv_survcaus")
if PROTOCOL["evaluation"]["folds"] != 2:
    raise ValueError("the preserved evaluation uses exactly two hash-defined folds")
if PROTOCOL["evaluation"]["regime"] != [1, 1]:
    raise ValueError(
        "the paper's factual evaluation scores only continued treatment (1,1)"
    )


def validate_prediction_runs(runs, test_ids):
    required = {"patient_id", "scenario", "cap", "seed", *METHODS}
    if not required.issubset(runs.columns):
        raise ValueError(
            f"missing prediction columns: {sorted(required.difference(runs.columns))}"
        )
    if (
        runs.empty
        or runs[list(required)].isna().any().any()
        or not np.isfinite(runs[list(METHODS)]).all().all()
    ):
        raise ValueError("missing or nonfinite clinical predictions")
    if len(np.unique(test_ids)) != len(test_ids):
        raise ValueError("test patient IDs are not unique")
    if not runs["cap"].eq(PROTOCOL["weight_cap"]).all():
        raise ValueError("clinical predictions must use the configured training cap")
    expected = {
        (scenario["name"], seed)
        for scenario in PROTOCOL["scenarios"]
        for seed in range(PROTOCOL["study"]["n_replicates"])
    }
    groups = runs.groupby(["scenario", "seed"], sort=False)
    if set(groups.groups) != expected:
        raise ValueError(
            "clinical predictions require both paper horizons, the paper cap, and all five fitting seeds"
        )
    for _, frame in groups:
        if frame["patient_id"].duplicated().any() or set(frame["patient_id"]) != set(
            test_ids
        ):
            raise ValueError(
                "predictions must cover each test patient exactly once per horizon/seed"
            )


def followers(raw, regime, landmark=ENTRY):
    follow = np.ones(len(raw["patient_id"]), dtype=bool)
    for step, action in enumerate(regime):
        k = landmark + step
        at_risk = raw["at_risk"][:, k] > 0
        died_before = (raw["delta"] == 1) & (raw["event_time"] <= k)
        given = raw["treatments"][:, k]
        ok = np.where(
            at_risk, given == action, np.where(died_before, True, action == 0)
        )
        follow &= ok
    return follow


def evaluation_fold(patient_ids):
    return np.array(
        [
            hashlib.sha256(f"eval-fold:{int(p)}".encode()).digest()[0] % 2
            for p in patient_ids
        ]
    )


def decision_features(raw, k):
    states = np.nan_to_num(raw["states"][:, k, :])
    previous = (
        np.nan_to_num(raw["treatments"][:, k - 1]) if k > 0 else np.zeros(len(states))
    )
    in_icu = (raw["at_risk"][:, k] > 0).astype(float)
    return np.column_stack([states, previous, in_icu, raw["baseline"]])


def treatment_probabilities(raw, regime, landmark=ENTRY):
    from sklearn.linear_model import LogisticRegression

    fold = evaluation_fold(raw["patient_id"])
    probabilities = np.full((len(fold), len(regime)), np.nan)
    for step in range(len(regime)):
        k = landmark + step
        alive = ~((raw["delta"] == 1) & (raw["event_time"] <= k))
        target = np.where(
            raw["at_risk"][:, k] > 0, np.nan_to_num(raw["treatments"][:, k]), 0.0
        )
        x = decision_features(raw, k)
        for held_out in (0, 1):
            train = alive & (fold != held_out)
            model = LogisticRegression(max_iter=2000).fit(x[train], target[train])
            rows = alive & (fold == held_out)
            probabilities[rows, step] = model.predict_proba(x[rows])[:, 1]
    return probabilities


def follower_weights(raw, regime, probabilities, landmark=ENTRY):
    weight = np.ones(len(raw["patient_id"]))
    for step, action in enumerate(regime):
        k = landmark + step
        alive = ~((raw["delta"] == 1) & (raw["event_time"] <= k))
        p = probabilities[:, step] if action == 1 else 1.0 - probabilities[:, step]
        p = np.maximum(np.where(alive, p, 1.0), PROBABILITY_FLOOR)
        weight = np.where(alive, weight / p, weight)
    return np.minimum(weight, WEIGHT_CAP)


def factual_scores(prediction, survived, known, follow, weight):
    p = np.clip(np.asarray(prediction, float), 0.0, 1.0)
    sel = follow & known
    w, y, q = weight[sel], survived[sel], p[sel]
    brier = float(np.sum(w * (y - q) ** 2) / np.sum(w))
    observed = float(np.sum(w * y) / np.sum(w))
    constant = float(np.sum(w * (y - observed) ** 2) / np.sum(w))
    return dict(
        weighted_brier=brier,
        brier_skill=1.0 - brier / constant,
        calibration_in_the_large=observed - float(p[known].mean()),
        unweighted_brier=float(np.mean((y - q) ** 2)),
        weighted_observed_survival=observed,
        mean_prediction=float(p[known].mean()),
        n_followers=int(sel.sum()),
        n_follower_deaths=int((y == 0).sum()),
        ess=float(w.sum() ** 2 / np.square(w).sum()),
    )


def paired_bootstrap(
    a, b, survived, known, follow, weight, n_boot=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED
):
    rng = np.random.default_rng(seed)
    n = len(a)
    pa, pb = np.clip(a, 0, 1), np.clip(b, 0, 1)
    point = (
        factual_scores(pa, survived, known, follow, weight)["weighted_brier"]
        - factual_scores(pb, survived, known, follow, weight)["weighted_brier"]
    )
    draws = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        sel = follow[i] & known[i]
        w, y = weight[i][sel], survived[i][sel]
        da = np.sum(w * (y - pa[i][sel]) ** 2) / np.sum(w)
        db = np.sum(w * (y - pb[i][sel]) ** 2) / np.sum(w)
        draws.append(da - db)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(point), float(low), float(high)


def evaluate(runs, test_raw, survived_by_tau, known_by_tau, regime=(1, 1)):
    validate_prediction_runs(runs, test_raw["patient_id"])
    follow = followers(test_raw, regime)
    probabilities = treatment_probabilities(test_raw, regime)
    weight = follower_weights(test_raw, regime, probabilities)
    index = pd.Index(test_raw["patient_id"])
    per_seed, paired = [], []
    cap = runs["cap"].iloc[0]
    for scenario, group in runs.groupby("scenario"):
        tau = int(scenario.split("tau")[1])
        y, known = survived_by_tau[tau], known_by_tau[tau]
        if not np.any(follow & known):
            raise ValueError(
                f"no known adherent outcomes at tau={tau}; weighted Brier is undefined"
            )
        for seed, frame in group.groupby("seed"):
            order = index.get_indexer(frame.patient_id)
            for method in METHODS:
                pred = np.full(len(index), np.nan)
                pred[order] = frame[method].to_numpy()
                per_seed.append(
                    dict(
                        scenario=scenario,
                        cap=cap,
                        seed=seed,
                        method=method,
                        **factual_scores(pred, y, known, follow, weight),
                    )
                )
        mean = group.groupby("patient_id")[list(METHODS)].mean().reindex(index)
        for method in METHODS:
            if method == "dr":
                continue
            point, low, high = paired_bootstrap(
                mean["dr"].to_numpy(), mean[method].to_numpy(), y, known, follow, weight
            )
            paired.append(
                dict(
                    scenario=scenario,
                    cap=cap,
                    contrast=f"dr minus {method}",
                    brier_difference=point,
                    ci_low=low,
                    ci_high=high,
                )
            )
    per_seed = pd.DataFrame(per_seed)
    summary = (
        per_seed.groupby(["scenario", "method"])
        .agg(
            weighted_brier=("weighted_brier", "mean"),
            seed_sd=("weighted_brier", "std"),
            brier_skill=("brier_skill", "mean"),
            calibration_in_the_large=("calibration_in_the_large", "mean"),
            unweighted_brier=("unweighted_brier", "mean"),
            n_followers=("n_followers", "first"),
            n_follower_deaths=("n_follower_deaths", "first"),
            ess=("ess", "first"),
        )
        .reset_index()
    )
    summary.insert(1, "cap", cap)
    summary["seed_t_halfwidth"] = (
        stats.t.ppf(0.975, N_SEEDS_MINUS_ONE)
        * summary.seed_sd
        / np.sqrt(N_SEEDS_MINUS_ONE + 1)
    )
    weights = dict(
        follow=int(follow.sum()),
        weight_mean=float(weight[follow].mean()),
        weight_max=float(weight[follow].max()),
        capped=int((weight[follow] >= WEIGHT_CAP).sum()),
    )
    weights["discharged_followers"] = discharged_follower_counts(
        test_raw, survived_by_tau, known_by_tau, follow, weight
    )
    return per_seed, summary, pd.DataFrame(paired), weights


def discharged_follower_counts(test_raw, survived_by_tau, known_by_tau, follow, weight):
    result = {}
    for tau in (8, 16):
        known = known_by_tau[tau].astype(bool)
        survived = survived_by_tau[tau]
        sel = follow & known
        left_alive = (test_raw["delta"] == 0) & (
            test_raw["censoring_time"] <= ENTRY + tau
        )
        group = sel & left_alive
        result[f"tau{tau}"] = dict(
            followers_known=int(sel.sum()),
            left_icu_alive_before_horizon=int(group.sum()),
            of_them_alive_at_horizon=int((survived[group] == 1).sum()),
            of_them_dead_at_horizon=int((survived[group] == 0).sum()),
            weight_share=float(weight[group].sum() / weight[sel].sum()),
        )
    return result


N_SEEDS_MINUS_ONE = PROTOCOL["study"]["n_replicates"] - 1


def main():
    from src.data.mimic_history_cate import inner_rows
    from src.data.mimic_realworld.tensors import make_splits
    from src.evaluation.mimic_realworld_factual import (
        N_SEEDS,
        SPLIT_SEEDS,
        load_trajectories,
        split_roles,
    )

    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--runs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument(
        "--split-seed", type=int, choices=SPLIT_SEEDS, default=SPLIT_SEEDS[0]
    )
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    data, roles, evaluation = load_trajectories(a.data)
    stored_roles = roles
    roles = split_roles(data["patient_id"], a.split_seed)
    if a.split_seed == SPLIT_SEEDS[0] and not np.array_equal(roles, stored_roles):
        raise ValueError("original patient hash differs from stored trajectory roles")
    if len(data["patient_id"]) != PROTOCOL["dataset"]["expected_patients"] or (
        a.split_seed == SPLIT_SEEDS[0]
        and int((roles == "test").sum())
        != PROTOCOL["dataset"]["expected_test_patients"]
    ):
        raise ValueError("clinical scoring cohort does not match the manuscript")
    splits, _ = make_splits(data, roles, inner_rows)
    test = roles == "test"
    survived = {tau: evaluation[f"survived_tau{tau}"][test] for tau in (8, 16)}
    known = {tau: evaluation[f"known_tau{tau}"][test].astype(bool) for tau in (8, 16)}
    frames = []
    trajectory_sha = file_sha256(a.data)
    for seed in range(N_SEEDS):
        completion = Path(a.runs) / f"seed_{seed}" / "complete.json"
        prediction = completion.with_name("predictions.csv")
        done = json.loads(completion.read_text())
        if done["status"] != "complete" or done["smoke"] or done["seed"] != seed:
            raise ValueError(f"seed {seed} is not a complete reporting run")
        if done["split_seed"] != a.split_seed:
            raise ValueError(f"seed {seed} uses a different patient hash")
        if done["trajectories_sha256"] != trajectory_sha:
            raise ValueError(
                f"seed {seed} was fitted from a different input"
            )
        if (
            done["configuration"]["protocol"] != PROTOCOL
            or done["configuration"]["training"] != load_protocol("training")
        ):
            raise ValueError(f"seed {seed} used different protocol/training settings")
        if done["predictions_sha256"] != file_sha256(prediction):
            raise ValueError(f"seed {seed} predictions changed after completion")
        frame = pd.read_csv(prediction)
        if not frame["seed"].eq(seed).all():
            raise ValueError(
                "prediction seed labels differ from their completion record"
            )
        frames.append(frame)
    per_seed, summary, paired, weights = evaluate(
        pd.concat(frames, ignore_index=True),
        splits["test"],
        survived,
        known,
        tuple(PROTOCOL["evaluation"]["regime"]),
    )
    per_seed.to_csv(out / "per_seed.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    paired.to_csv(out / "paired.csv", index=False)
    (out / "weights.json").write_text(json.dumps(weights, indent=2))
    (out / "complete.json").write_text(
        json.dumps(
            dict(
                status="complete",
                replicates=N_SEEDS,
                split_seed=a.split_seed,
                trajectories_sha256=trajectory_sha,
                configuration=protocol_snapshot("realworld"),
                environment=runtime_versions(),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(summary.round(4).to_string(index=False))
        print(paired.round(4).to_string(index=False))
    print(json.dumps(weights))


if __name__ == "__main__":
    main()
