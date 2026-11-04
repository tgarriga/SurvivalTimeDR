"""Atomic, versioned model checkpoints with validation on load."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def save_model(path, model, constructor, identity):
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    if any(
        v.is_floating_point() and not bool(torch.isfinite(v).all())
        for v in model.state_dict().values()
    ):
        raise ValueError("refusing a nonfinite model checkpoint")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": 1,
        "class": type(model).__name__,
        "constructor": constructor,
        "identity": identity,
        "state_dict": {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        },
    }
    # Metadata must remain plain values accepted by weights_only loading.
    json.dumps({k: v for k, v in payload.items() if k != "state_dict"}, allow_nan=False)
    tmp = path.with_suffix(".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    return str(path)


def load_model(path, expected_identity, device="cpu"):
    from src.models.survival_ct_nuisance import (
        SurvivalCTNuisanceModel,
        SurvivalSequenceRegressor,
    )
    from src.models.tv_survcaus import TreatmentNumeratorModel, TVSurvCausModel

    constructors = {
        c.__name__: c
        for c in (
            SurvivalCTNuisanceModel,
            SurvivalSequenceRegressor,
            TreatmentNumeratorModel,
            TVSurvCausModel,
        )
    }
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != 1 or payload["identity"] != expected_identity:
        raise ValueError("checkpoint run/feature identity mismatch")
    if payload["class"] not in constructors:
        raise ValueError("unsupported checkpoint class")
    model = constructors[payload["class"]](**payload["constructor"])
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.to(device).eval()
