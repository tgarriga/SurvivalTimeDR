"""Controlled censoring/support follow-up using the shared survival learners."""

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.mimic_history_cate import ENTRY, make_splits, validate_prepared
from src.data.mimic_semisynthetic.checkpoints import sha256, write_json
from src.data.survival_sim import equations as eq
from src.data.survival_sim.dgp import SurvivalSimConfig, simulate_split
from src.evaluation.experiments import StudyScenario, run_study_replicate
from src.evaluation.metrics import summarize_study
from src.evaluation.mimic_history_cate_study import (
    Recorder,
    configure,
)
from src.evaluation.runtime import peak_rss_kib
from src.evaluation.weight_diagnostics import (
    _dr_weight_clip_flags,
    _FittedProbabilityView,
    _iptw_regime_signal,
    _match_clip_record,
)
from src.models.transformer_training import (
    TransformerTrainer,
    patient_train_validation_indices,
)
from src.protocols import load_protocol, synthetic_config

PROTOCOL = load_protocol("stress")
STUDY_SEED = PROTOCOL["study"]["seed"]
CALIBRATION_SEED = PROTOCOL["calibration_seed"]
SCENARIOS = tuple(StudyScenario(**scenario) for scenario in PROTOCOL["scenarios"])
CELLS = PROTOCOL["conditions"]
if PROTOCOL["calibration_patients"] != 1024:
    raise ValueError("the paper uses 1024 calibration patients")


def incidence_from_continuations(continuation, alive):
    """Expected observed dropout before endpoint, conditional on uncensored paths."""
    g, alive = np.asarray(continuation, float), np.asarray(alive, bool)
    if g.shape != alive.shape or g.ndim != 2 or not len(g):
        raise ValueError("invalid path arrays")
    if not np.isfinite(g[alive]).all() or np.any((g[alive] < 0) | (g[alive] > 1)):
        raise ValueError("invalid continuation probability")
    if np.any(alive[:, 1:] & ~alive[:, :-1]):
        raise ValueError("survival cannot restart")
    remaining = np.ones(len(g))
    total = np.zeros(len(g))
    for k in range(g.shape[1]):
        selected = alive[:, k]
        remaining[~selected] = 0.0
        total[selected] += remaining[selected] * (1 - g[selected, k])
        remaining[selected] *= g[selected, k]
    return float(total.mean())


def expected_missing(raw, cfg, horizon=8):
    if np.any(raw["censoring_time"] <= cfg.max_seq_length):
        raise ValueError("uncensored development paths required")
    successors = range(ENTRY + 1, ENTRY + horizon)
    alive = raw["state_observed"][:, ENTRY + 1 : ENTRY + horizon] > 0
    g = np.full(alive.shape, np.nan)
    for column, j in enumerate(successors):
        ids = np.flatnonzero(alive[:, column])
        if len(ids):
            g[ids, column] = eq.censoring_continuation(
                cfg, eq.history_terms(raw["states"], raw["treatments"], ids, j), j
            )
    return incidence_from_continuations(g, alive)


def calibrate_intercept(raw, cfg, target):
    low, high = -5.0, 12.0
    flo = expected_missing(raw, replace(cfg, censoring_intercept=low))
    fhi = expected_missing(raw, replace(cfg, censoring_intercept=high))
    if not fhi < target < flo:
        raise ValueError("target outside fixed attainable bracket")
    trace = []
    for iteration in range(32):
        middle = (low + high) / 2
        rate = expected_missing(raw, replace(cfg, censoring_intercept=middle))
        trace.append(dict(iteration=iteration, intercept=middle, expected_missing=rate))
        if abs(rate - target) < 1e-7:
            return middle, rate, trace
        if rate > target:
            low = middle
        else:
            high = middle
    raise ValueError("fixed calibration tolerance not reached")


def copy_prepared(source, destination, expected_sha256=None):
    if expected_sha256 is None:
        raise ValueError(
            "an explicit prepared-input SHA-256 is required; the unified initialization command computes it"
        )
    expected = expected_sha256
    if sha256(source) != expected:
        raise ValueError("prepared cohort identity mismatch")
    shutil.copyfile(source, destination)
    if sha256(destination) != expected or sha256(source) != expected:
        raise ValueError("prepared archive changed during copy")


def initialize(prepared, root, expected_prepared_sha256=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    copy_prepared(prepared, root / "prepared.npz", expected_prepared_sha256)
    with np.load(root / "prepared.npz", allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    validate_prepared(data)
    fit_ids = data["preprocessing_fit_ids"]
    rng = np.random.default_rng(CALIBRATION_SEED)
    chosen = np.sort(rng.choice(fit_ids, 1024, replace=False))
    selected = np.isin(data["patient_id"], chosen)
    assert int(selected.sum()) == len(chosen)
    np.save(root / "calibration_ids.npy", chosen)
    cells = []
    development = {}
    traces = []
    for cell in CELLS:
        cfg = replace(
            SurvivalSimConfig(**PROTOCOL["simulation"]),
            overlap_strength=cell["overlap_strength"],
            event_intercept=cell["event_intercept"],
        )
        key = (cfg.overlap_strength, cfg.event_intercept)
        if key not in development:
            seed = CALIBRATION_SEED + 17321 * (len(development) + 1)
            raw = simulate_split(
                1024,
                seed,
                replace(cfg, censoring_enabled=False),
                initial_history=data["history"][selected],
                initial_baseline=data["baseline"][selected],
                patient_ids=data["patient_id"][selected],
            )
            path = root / f"development_{len(development)}.npz"
            np.savez_compressed(path, **{k: v for k, v in raw.items() if k != "config"})
            development[key] = (raw, seed, str(path.name))
        raw, seed, path = development[key]
        if cell["target"]:
            intercept, rate, trace = calibrate_intercept(raw, cfg, cell["target"])
            for row in trace:
                traces.append(dict(cell=cell["name"], **row))
            cfg = replace(cfg, censoring_intercept=intercept)
        else:
            cfg = replace(cfg, censoring_enabled=False)
            rate = 0.0
        cells.append(
            dict(
                **cell,
                simulation=asdict(cfg),
                development_seed=seed,
                development_path=path,
                expected_development_missing=rate,
            )
        )
    pd.DataFrame(traces).to_csv(root / "calibration_trace.csv", index=False)
    manifest = dict(
        schema=1,
        cells=cells,
        study_seed=STUDY_SEED,
        n_replicates=PROTOCOL["study"]["n_replicates"],
        cap=PROTOCOL["weight_cap"],
        epochs=load_protocol("training")["nuisance_epochs"],
        prepared_sha256=sha256(root / "prepared.npz"),
        calibration_seed=CALIBRATION_SEED,
        calibration_target="observed censoring strictly before tau8 endpoint and before death",
        scenarios=[asdict(s) for s in SCENARIOS],
        reference="same real prefixes and transforms as completed history-CATE study; new simulation seeds",
        training_config=asdict(
            synthetic_config(
                "stress",
                "cuda",
                dict(
                    n_nuisance=int((data["roles"] == "nuisance").sum()),
                    n_second_stage=int((data["roles"] == "second_stage").sum()),
                    n_test=int((data["roles"] == "test").sum()),
                ),
            )
        ),
    )
    write_json(root / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "status": "initialized",
                "cells": [
                    {
                        k: c[k]
                        for k in ("name", "target", "expected_development_missing")
                    }
                    for c in cells
                ],
            }
        )
    )


def inputs(root):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if sha256(root / "prepared.npz") != manifest["prepared_sha256"]:
        raise ValueError("prepared input changed")
    with np.load(root / "prepared.npz", allow_pickle=False) as f:
        data = {k: f[k].copy() for k in f.files}
    return root, manifest, data


def validate_targets(dataset, cfg, targets, mask, label):
    targets, mask = np.asarray(targets), np.asarray(mask)
    if (
        targets.shape != mask.shape
        or len(targets) != len(dataset)
        or not np.isfinite(mask).all()
        or np.any(mask < 0)
    ):
        raise ValueError("invalid target/mask")
    fit, val = patient_train_validation_indices(
        dataset, cfg.validation_fraction, cfg.validation_seed
    )
    record = {"label": label}
    for name, rows in [("fit", fit), ("validation", val)]:
        if name == "validation" and cfg.validation_fraction == 0:
            continue
        selected = mask[rows] > 0
        if not selected.any() or not np.isfinite(targets[rows][selected]).all():
            raise ValueError("empty/nonfinite " + name + " targets: " + label)
        record[name + "_patients"] = len(rows)
        record[name + "_targets"] = int(selected.sum())
        record[name + "_mask_mass"] = float(mask[rows].sum())
    return record


class GuardTrainer(TransformerTrainer):
    def __init__(self, cfg, role, records):
        super().__init__(cfg)
        self.role = role
        self.records = records
        self.model_index = 0

    def fit(self, model, dataset, task, **kwargs):
        record = validate_targets(
            dataset,
            self.config,
            dataset.data["outputs"],
            dataset.data["target_mask"],
            self.role + "/" + task,
        )
        self.records.append(
            dict(
                **record,
                tau=self.config.tau,
                regime=list(self.config.treatment_sequence),
                model_index=self.model_index,
            )
        )
        self.model_index += 1
        return super().fit(model, dataset, task, **kwargs)


def factual_rates(raw, horizon):
    end = ENTRY + horizon
    censor = (raw["censoring_time"] < end) & (raw["censoring_time"] < raw["event_time"])
    event = (raw["delta"] > 0) & (raw["event_time"] <= end)
    survived = raw["state_observed"][:, end] > 0
    if not np.all(censor.astype(int) + event.astype(int) + survived.astype(int) == 1):
        raise ValueError("endpoint observation categories inconsistent")
    return dict(
        n=len(censor),
        endpoint_missing_fraction=float(censor.mean()),
        observed_event_fraction=float(event.mean()),
        observed_survival_fraction=float(survived.mean()),
        censored_exactly_at_endpoint_fraction=float(
            (raw["censoring_time"] == end).mean()
        ),
    )


class StressRecorder(Recorder):
    def _record(self, phase, **kw):
        super()._record(phase, **kw)
        if phase != "shared":
            return
        cfg, col = kw["config"], kw["collection"]
        dataset = col.val_f
        view = _FittedProbabilityView(kw["probability"], dataset, prediction_cache={})
        fit, val = patient_train_validation_indices(
            dataset, cfg.validation_fraction, cfg.validation_seed
        )
        records = []
        for scenario in cfg.scenarios:
            for landmark in range(
                ENTRY, cfg.simulation.max_seq_length - scenario.tau + 1
            ):
                for arm, regime in [
                    ("one", scenario.regime_one),
                    ("zero", scenario.regime_zero),
                ]:
                    signal = _iptw_regime_signal(
                        dataset.raw,
                        regime,
                        view,
                        landmark,
                        landmark + scenario.tau,
                        cfg.probability_clip,
                        cfg.cumulative_weight_clip,
                    )
                    flags = _dr_weight_clip_flags(
                        dataset.raw,
                        regime,
                        view,
                        landmark,
                        landmark + scenario.tau,
                        cfg.probability_clip,
                        cfg.cumulative_weight_clip,
                    )
                    for split, rows in [
                        ("all", np.arange(len(dataset))),
                        ("fit", fit),
                        ("validation", val),
                    ]:
                        sub = {k: np.asarray(v)[rows] for k, v in signal.items()}
                        eligible = sub["eligible"]
                        w = sub["ipw_weight"][eligible]
                        positive = w[w > 0]
                        records.append(
                            dict(
                                scenario=scenario.name,
                                landmark=landmark,
                                arm=arm,
                                split=split,
                                n_eligible=int(eligible.sum()),
                                n_positive_ipw=len(positive),
                                ipw_ess=(
                                    float(w.sum() ** 2 / np.square(w).sum())
                                    if len(positive)
                                    else 0.0
                                ),
                                **_match_clip_record(sub, flags[rows]),
                            )
                        )
        pd.DataFrame(records).to_csv(self.root.parent / "support.csv", index=False)


def audit_history(history, epochs):
    if not np.array_equal(history.epoch, np.arange(1, len(history) + 1)):
        raise ValueError("noncontiguous history")
    if not np.isfinite(history[["train_loss", "validation_loss"]]).all().all():
        raise ValueError("nonfinite losses")
    best = np.inf
    selected = 0
    bad = 0
    stop = None
    for row in history.itertuples():
        if row.validation_loss < best - 1e-4:
            best = row.validation_loss
            selected = row.epoch
            bad = 0
        else:
            bad += 1
        if row.epoch >= min(10, epochs) and bad >= 5:
            stop = row.epoch
            break
    if (
        len(history) != (stop or epochs)
        or not history.best_epoch.eq(selected).all()
        or not history.stopped_epoch.eq(len(history)).all()
    ):
        raise ValueError("stopping/restored epoch mismatch")
    return dict(
        selected_epoch=int(selected),
        stopped_epoch=len(history),
        hit_ceiling=len(history) == epochs,
    )


def case(root, index):
    configure()
    root, manifest, data = inputs(root)
    if not 0 <= index < 35:
        raise ValueError("case index outside protocol")
    cell_index, rep = divmod(index, 5)
    cell = manifest["cells"][cell_index]
    dest = root / "cases" / f'{cell["name"]}_r{rep}'
    dest.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    simulation = SurvivalSimConfig(**cell["simulation"])
    splits, scaling = make_splits(data, rep, simulation, study_seed=STUDY_SEED)
    test = data["roles"] == "test"
    full_test = simulate_split(
        int(test.sum()),
        STUDY_SEED + 100003 * rep + 40009,
        simulation,
        initial_history=data["history"][test],
        initial_baseline=data["baseline"][test],
        patient_ids=data["patient_id"][test],
    )
    for key in ("patient_id", "baseline"):
        np.testing.assert_array_equal(full_test[key], splits["test"][key])
    assert np.isnan(splits["test"]["states"][:, ENTRY + 1 :]).all()
    rates = []
    for cohort, raw in {**splits, "test_factual_diagnostic": full_test}.items():
        if cohort != "test":
            for scenario in SCENARIOS:
                rates.append(
                    dict(
                        cohort=cohort,
                        scenario=scenario.name,
                        **factual_rates(raw, scenario.tau),
                    )
                )
        np.savez_compressed(
            dest / f"{cohort}.npz", **{k: v for k, v in raw.items() if k != "config"}
        )
    pd.DataFrame(rates).to_csv(dest / "observed_rates.csv", index=False)
    write_json(dest / "scaling.json", {k: v.tolist() for k, v in scaling.items()})
    cfg = synthetic_config(
        "stress",
        "cuda",
        dict(
            n_nuisance=len(splits["train"]["patient_id"]),
            n_second_stage=len(splits["val"]["patient_id"]),
            n_test=len(splits["test"]["patient_id"]),
        ),
        simulation=simulation,
    )
    base = asdict(cfg)
    base["simulation"] = manifest["training_config"]["simulation"]
    if json.loads(json.dumps(base)) != manifest["training_config"]:
        raise ValueError("training configuration changed")
    write_json(dest / "config.json", asdict(cfg))
    records = []

    def validate(ds, c, t, m, label):
        records.append(validate_targets(ds, c, t, m, label))

    recorder = StressRecorder(
        dest / "checkpoints",
        dict(
            prepared_sha256=manifest["prepared_sha256"],
            cell=cell["name"],
            replicate=rep,
            smoke=False,
        ),
    )
    metrics, predictions, diagnostics = run_study_replicate(
        cfg,
        rep,
        raw_splits=splits,
        scaling_params=scaling,
        training_start=ENTRY,
        artifact_callback=recorder,
        nuisance_trainer_factory=lambda c, role: GuardTrainer(c, role, records),
        target_validator=validate,
    )
    assert recorder.count == len(records)
    if not np.isfinite(predictions.select_dtypes(include=[np.number]).to_numpy()).all():
        raise ValueError("nonfinite reporting predictions")
    for scenario in SCENARIOS:
        pred = predictions.query("scenario == @scenario.name")
        assert len(pred) == cfg.n_test
        for row in metrics.query("scenario == @scenario.name").itertuples():
            err = pred[row.method + "_effect"].to_numpy() - pred.effect_true.to_numpy()
            np.testing.assert_allclose(
                [row.cate_rmse, row.cate_mae, row.ate_bias],
                [np.sqrt(np.mean(err**2)), np.mean(np.abs(err)), err.mean()],
                rtol=1e-12,
                atol=1e-12,
            )
    metrics.assign(cap=cfg.cumulative_weight_clip).to_csv(
        dest / "metrics.csv", index=False
    )
    predictions.assign(cap=cfg.cumulative_weight_clip).to_csv(
        dest / "predictions.csv", index=False
    )
    diagnostics.assign(cap=cfg.cumulative_weight_clip).to_csv(
        dest / "diagnostics.csv", index=False
    )
    pd.DataFrame(records).to_csv(dest / "target_support.csv", index=False)
    histories = [
        dict(
            component=str(p.relative_to(dest)),
            **audit_history(pd.read_csv(p), cfg.nuisance_epochs),
        )
        for p in sorted((dest / "checkpoints").rglob("*.history.csv"))
    ]
    assert len(histories) == recorder.count
    pd.DataFrame(histories).to_csv(dest / "fit_audit.csv", index=False)
    write_json(
        dest / "complete.json",
        dict(
            status="complete",
            smoke=False,
            cell=cell["name"],
            replicate=rep,
            component_fits=recorder.count,
            metrics_checked=True,
            epochs_checked=True,
            elapsed_seconds=time.monotonic() - started,
            device=torch.cuda.get_device_name(),
            torch=str(torch.__version__),
            max_rss_kib=peak_rss_kib(),
            peak_allocated_gpu_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_gpu_bytes=torch.cuda.max_memory_reserved(),
        ),
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "cell": cell["name"],
                "replicate": rep,
                "smoke": False,
            }
        )
    )


def report(root):
    root, manifest, _ = inputs(root)
    out = root / "report"
    out.mkdir(exist_ok=False)
    metrics = []
    predictions = []
    diagnostics = []
    rates = []
    support = []
    histories = []
    for cell in manifest["cells"]:
        for rep in range(5):
            case = root / "cases" / f'{cell["name"]}_r{rep}'
            completion = case / "complete.json"
            done = json.loads(completion.read_text())
            assert (
                done["status"] == "complete"
                and not done["smoke"]
            )
            assert (
                done["metrics_checked"]
                and done["epochs_checked"]
            )
            context = dict(cell=cell["name"], replicate=rep)
            for name, frames in [
                ("metrics", metrics),
                ("predictions", predictions),
                ("diagnostics", diagnostics),
                ("observed_rates", rates),
                ("support", support),
                ("fit_audit", histories),
            ]:
                frames.append(pd.read_csv(case / f"{name}.csv").assign(**context))
    metrics = pd.concat(metrics)
    pred = pd.concat(predictions)
    rates = pd.concat(rates)
    support = pd.concat(support)
    histories = pd.concat(histories)
    assert len(metrics) == 350 and len(histories) == 1575
    assert not metrics.duplicated(["cell", "scenario", "method", "replicate"]).any()
    assert (
        metrics.groupby(["cell", "scenario", "method"]).replicate.nunique().eq(5).all()
    )
    for rep in range(5):
        for strength in ("standard", "concentrated"):
            ref = pred.loc[(pred.cell == f"{strength}_c0") & (pred.replicate == rep)]
            for level in (20, 40):
                other = pred.loc[
                    (pred.cell == f"{strength}_c{level}") & (pred.replicate == rep)
                ]
                for scenario in SCENARIOS:
                    a = ref.query("scenario == @scenario.name").sort_values(
                        "patient_id"
                    )
                    b = other.query("scenario == @scenario.name").sort_values(
                        "patient_id"
                    )
                    np.testing.assert_array_equal(
                        a[["patient_id", "effect_true"]].to_numpy(),
                        b[["patient_id", "effect_true"]].to_numpy(),
                    )
        a = pred.query(
            'cell == "standard_c0" and scenario == "L2_tau2" and replicate == @rep'
        ).sort_values("patient_id")
        b = pred.query(
            'cell == "concentrated_c0" and scenario == "L2_tau2" and replicate == @rep'
        ).sort_values("patient_id")
        np.testing.assert_array_equal(
            a[["patient_id", "effect_true"]].to_numpy(),
            b[["patient_id", "effect_true"]].to_numpy(),
        )
    summary = pd.concat(
        [
            summarize_study(metrics.loc[metrics.cell == c["name"]]).assign(
                cell=c["name"]
            )
            for c in manifest["cells"]
        ]
    )
    from src.evaluation.metrics import mean_confidence_interval

    paired = []
    for (cell, scenario), group in metrics.groupby(["cell", "scenario"]):
        pivot = group.pivot(index="replicate", columns="method", values="cate_rmse")
        for comparator in ("gcomp", "ra", "iptw_ipcw", "tv_survcaus"):
            mean, se, low, high = mean_confidence_interval(pivot.dr - pivot[comparator])
            paired.append(
                dict(
                    cell=cell,
                    scenario=scenario,
                    contrast="dr minus " + comparator,
                    mean=mean,
                    se=se,
                    ci_low=low,
                    ci_high=high,
                )
            )
    calibration = []
    for (cell, scenario, rep), group in pred.groupby(["cell", "scenario", "replicate"]):
        y = group.effect_true.to_numpy()
        yc = y - y.mean()
        sy = float(np.sqrt(np.mean(yc**2)))
        for method in ("gcomp", "ra", "iptw_ipcw", "dr", "tv_survcaus"):
            p = group[method + "_effect"].to_numpy()
            pc = p - p.mean()
            sp = float(np.sqrt(np.mean(pc**2)))
            constant = bool(np.all(p == p[0]))
            calibration.append(
                dict(
                    cell=cell,
                    scenario=scenario,
                    replicate=rep,
                    method=method,
                    rmse=float(np.sqrt(np.mean((p - y) ** 2))),
                    bias=float(np.mean(p - y)),
                    truth_sd=sy,
                    prediction_sd=0.0 if constant else sp,
                    correlation=(
                        float(np.mean(pc * yc) / (sp * sy))
                        if not constant and sp > 0 and sy > 0
                        else np.nan
                    ),
                )
            )
    tables = dict(
        calibration=pd.DataFrame(calibration),
        metrics=metrics,
        summary=summary,
        predictions=pred,
        diagnostics=pd.concat(diagnostics, ignore_index=True),
        observed_rates=rates,
        support=support,
        fit_audit=histories,
        paired=pd.DataFrame(paired),
    )
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    pred.groupby(["cell", "scenario", "replicate"]).agg(
        mean_mcse=("mcse_effect", "mean"),
        mean_true_effect=("effect_true", "mean"),
        true_effect_sd=("effect_true", "std"),
    ).to_csv(out / "oracle_summary.csv")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from src.evaluation.plotting import points

    methods = ("gcomp", "ra", "iptw_ipcw", "dr", "tv_survcaus")
    labels = ("g-computation", "RA", "IPTW/IPCW", "DR", "TV-SurvCaus")
    colors = ("#4477AA", "#66BBBB", "#EE7733", "#228833", "#AA77AA")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for row, strength in enumerate(("standard", "concentrated")):
        for col, target in enumerate((0, 20, 40)):
            cell = f"{strength}_c{target}"
            ax = axes[row, col]
            for mi, (method, label, color) in enumerate(zip(methods, labels, colors)):
                for si, scenario in enumerate(SCENARIOS):
                    values = metrics.loc[
                        (metrics.cell == cell)
                        & (metrics.method == method)
                        & (metrics.scenario == scenario.name),
                        "cate_rmse",
                    ]
                    points(
                        ax,
                        values,
                        si + (mi - 2) * 0.13,
                        color,
                        label if si == 0 else None,
                    )
            ax.set(
                title=f"{strength}; target missing {target}% at tau8",
                xticks=[0, 1],
                xticklabels=["L2_tau2", "L3_tau8"],
                ylabel="CATE RMSE",
            )
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "censoring_support.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, cell in zip(axes, ("standard_c20", "higher_event_c20")):
        for mi, (method, label, color) in enumerate(zip(methods, labels, colors)):
            for si, scenario in enumerate(SCENARIOS):
                values = metrics.loc[
                    (metrics.cell == cell)
                    & (metrics.method == method)
                    & (metrics.scenario == scenario.name),
                    "cate_rmse",
                ]
                points(
                    ax, values, si + (mi - 2) * 0.13, color, label if si == 0 else None
                )
        ax.set(
            title=cell,
            xticks=[0, 1],
            xticklabels=["L2_tau2", "L3_tau8"],
            ylabel="CATE RMSE",
        )
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "higher_event.png", dpi=180)
    plt.close(fig)
    write_json(
        out / "complete.json",
        dict(
            status="complete",
            cases=35,
            component_fits=len(histories),
            oracle_invariance_checks="censor-only all horizons; overlap-only fully prescribed short horizon",
        ),
    )
    print(json.dumps({"status": "complete", "report": str(out)}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["init", "case", "report"])
    parser.add_argument("--root", required=True)
    parser.add_argument("--prepared")
    parser.add_argument("--expected-prepared-sha256")
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    if args.command == "init":
        initialize(args.prepared, args.root, args.expected_prepared_sha256)
    elif args.command == "report":
        report(args.root)
    else:
        case(args.root, args.index)


if __name__ == "__main__":
    main()
