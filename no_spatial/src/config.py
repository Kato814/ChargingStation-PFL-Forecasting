from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class FoldConfig:
    fold: str
    test: tuple[int, ...]
    validation: tuple[int, ...]
    ridge_alpha: float
    type_weight: float
    history_days: int
    regularization: float


def load_config(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_folds(config: dict) -> tuple[FoldConfig, ...]:
    pairs = [tuple(int(v) for v in pair) for pair in config["target_pairs"]]
    params = config["fold_parameters"]
    folds = []
    for idx, test_pair in enumerate(pairs):
        name = f"fold_{idx + 1}"
        validation_pair = pairs[(idx + 1) % len(pairs)]
        p = params[name]
        folds.append(FoldConfig(fold=name,test=test_pair,validation=validation_pair,ridge_alpha=float(p["ridge_alpha"]),type_weight=float(p["type_weight"]),history_days=int(p["history_days"]),regularization=float(p["regularization"])))
    return tuple(folds)
