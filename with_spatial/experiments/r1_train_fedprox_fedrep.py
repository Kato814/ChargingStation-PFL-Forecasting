from __future__ import annotations

import argparse
import copy
import hashlib
import json
import pickle
import platform
import random
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn

from baselines.protocols import NEURAL_BASELINE_SEEDS, RESIDUAL_MLP_PROTOCOL
from baselines.reference import load_locked_comparison_predictions
from src.config import build_folds, load_config
from src.evaluation import (
    aggregate_system_metrics,
    paired_daily_inference,
    station_wape_table,
    type_aggregate_wape_table,
)

TRAIN_START, TRAIN_END = pd.Timestamp("2022-09-02"), pd.Timestamp("2022-12-15")
VAL_START, VAL_END = pd.Timestamp("2022-12-16"), pd.Timestamp("2023-01-15")
TEST_START, TEST_END = pd.Timestamp("2023-01-16"), pd.Timestamp("2023-02-28")

FEDPROX_MU_GRID = (1e-3, 1e-2, 1e-1, 1.0)
FEDAVG_CALIBRATION_ATOL = 0.003

# FedRep: shared body + local 96-output head.
# The benchmark keeps one representation epoch per round to match the locked
# FedAvg/FedProx local-update depth, while FedRep receives its algorithm-specific
# extra local head optimization before each representation update.
SHARED_PREFIXES = ("fc1.", "ln1.", "fc2.")
PERSONAL_PREFIXES = ("out.",)
FEDREP_HEAD_EPOCHS = 10
FEDREP_REPRESENTATION_EPOCHS = 1
FEDREP_TARGET_HEAD_EPOCHS = 10
FEDREP_TARGET_HEAD_INIT = "initial_seed_head"

FORMAL_METHODS = ("fedavg", "fedprox", "fedrep")


class ResidualMLP(nn.Module):
    def __init__(self):
        super().__init__()
        p = RESIDUAL_MLP_PROTOCOL
        self.fc1 = nn.Linear(p.input_dim, p.hidden_dim_1)
        self.ln1 = nn.LayerNorm(p.hidden_dim_1)
        self.fc2 = nn.Linear(p.hidden_dim_1, p.hidden_dim_2)
        self.out = nn.Linear(p.hidden_dim_2, p.output_dim)
        self.drop = nn.Dropout(p.dropout)

    def forward(self, x):
        x = self.drop(torch.relu(self.ln1(self.fc1(x))))
        x = self.drop(torch.relu(self.fc2(x)))
        return self.out(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def phase_seed(seed: int, round_i: int, station_id: int, phase_code: int, epoch_i: int) -> int:
    # Stable integer composition; avoids Python hash randomization.
    return int(
        seed * 100_000_000
        + (round_i + 1) * 100_000
        + int(station_id) * 10
        + phase_code * 1_000
        + epoch_i
    )


def reseed_torch(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def mask(client, start, end):
    dates = pd.to_datetime(client["forecast_date"]).normalize()
    return np.asarray((dates >= start) & (dates <= end), dtype=bool)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        return p.stdout.strip()
    except Exception:
        return None


def load_clients():
    path = ROOT / "data/clients_with_spatial.pkl"
    with path.open("rb") as f:
        payload = pickle.load(f)
    clients = {int(k): v for k, v in payload["clients"].items()}
    validate_clients(clients)
    return clients


def validate_clients(clients) -> None:
    p = RESIDUAL_MLP_PROTOCOL
    if len(clients) != 40:
        raise AssertionError(f"expected 40 clients, got {len(clients)}")
    for sid, client in sorted(clients.items()):
        x = np.asarray(client["x"])
        y = np.asarray(client["y_residual"])
        if x.ndim != 2 or x.shape[1] != p.input_dim:
            raise AssertionError(f"station {sid} input shape {x.shape} != (*,{p.input_dim})")
        if y.ndim != 2 or y.shape[1] != p.output_dim:
            raise AssertionError(f"station {sid} residual shape {y.shape} != (*,{p.output_dim})")
        if len(x) != len(y):
            raise AssertionError(f"station {sid} x/y length mismatch")


def make_scaler(clients, train_ids):
    x = np.vstack(
        [
            np.asarray(clients[s]["x"], float)[mask(clients[s], TRAIN_START, TRAIN_END)]
            for s in train_ids
        ]
    )
    mean, scale = x.mean(0), x.std(0)
    scale[scale < 1e-8] = 1.0
    return mean, scale


def transform(x, mean, scale):
    return ((np.asarray(x, float) - mean) / scale).astype(np.float32)


def client_train(client, mean, scale):
    m = mask(client, TRAIN_START, TRAIN_END)
    return (
        transform(np.asarray(client["x"])[m], mean, scale),
        np.asarray(client["y_residual"], np.float32)[m],
    )


def weighted_average(states, counts, keys=None):
    keys = list(keys or states[0].keys())
    total = float(sum(counts))
    return {
        k: sum(
            state[k].detach().float() * (n / total)
            for state, n in zip(states, counts)
        )
        for k in keys
    }


def set_trainable_prefixes(model: nn.Module, prefixes: tuple[str, ...] | None) -> list[nn.Parameter]:
    trainable = []
    for name, param in model.named_parameters():
        enabled = prefixes is None or name.startswith(prefixes)
        param.requires_grad = enabled
        if enabled:
            trainable.append(param)
    if not trainable:
        raise AssertionError(f"no trainable parameters for prefixes={prefixes}")
    return trainable


def train_epochs(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    *,
    seed: int,
    round_i: int,
    station_id: int,
    phase_code: int,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    trainable_prefixes: tuple[str, ...] | None = None,
    grad_clip: float | None = None,
    mu: float = 0.0,
    global_params: dict[str, torch.Tensor] | None = None,
    frozen_representation_eval: bool = False,
) -> None:
    p = RESIDUAL_MLP_PROTOCOL
    trainable = set_trainable_prefixes(model, trainable_prefixes)
    optimizer = torch.optim.AdamW(
        trainable,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(y).to(device)
    if frozen_representation_eval:
        # Parameters remain differentiable for the head while dropout in the
        # frozen representation is disabled. LayerNorm has no running state.
        model.eval()
    else:
        model.train()

    for epoch_i in range(int(epochs)):
        local_seed = phase_seed(seed, round_i, station_id, phase_code, epoch_i)
        reseed_torch(local_seed)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(local_seed)
        order = torch.randperm(len(x), generator=generator).cpu().numpy()

        for start in range(0, len(order), p.shared_batch_size):
            idx = torch.as_tensor(
                order[start : start + p.shared_batch_size],
                dtype=torch.long,
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean((model(xt[idx]) - yt[idx]) ** 2)
            if mu > 0:
                if global_params is None:
                    raise AssertionError("FedProx requires global_params")
                prox = sum(
                    torch.sum((param - global_params[name]) ** 2)
                    for name, param in model.named_parameters()
                )
                loss = loss + 0.5 * mu * prox
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(trainable, grad_clip)
            optimizer.step()


def local_epoch(
    model,
    x,
    y,
    seed,
    round_i,
    station_id,
    device,
    mu=0.0,
    global_params=None,
):
    p = RESIDUAL_MLP_PROTOCOL
    train_epochs(
        model,
        x,
        y,
        seed=seed,
        round_i=round_i,
        station_id=station_id,
        phase_code=1,
        device=device,
        epochs=1,
        learning_rate=p.shared_learning_rate,
        weight_decay=p.shared_weight_decay,
        trainable_prefixes=None,
        mu=mu,
        global_params=global_params,
    )


def train_fedavg_prox(clients, train_ids, seed, device, mu=0.0, rounds=None):
    set_seed(seed)
    p = RESIDUAL_MLP_PROTOCOL
    mean, scale = make_scaler(clients, train_ids)
    cached = {s: client_train(clients[s], mean, scale) for s in train_ids}
    global_model = ResidualMLP().to(device)

    for round_i in range(int(rounds or p.shared_rounds)):
        global_params = {
            name: value.detach().clone()
            for name, value in global_model.named_parameters()
        }
        states, counts = [], []
        for sid in train_ids:
            local = copy.deepcopy(global_model).to(device)
            x, y = cached[sid]
            local_epoch(
                local,
                x,
                y,
                seed,
                round_i,
                sid,
                device,
                mu,
                global_params,
            )
            states.append(
                {k: v.detach().clone() for k, v in local.state_dict().items()}
            )
            counts.append(len(x))
        global_model.load_state_dict(weighted_average(states, counts))

    return global_model, mean, scale


def select_state(state: dict[str, torch.Tensor], prefixes: tuple[str, ...], *, cpu: bool = False):
    out = {}
    for key, value in state.items():
        if key.startswith(prefixes):
            tensor = value.detach().clone()
            if cpu:
                tensor = tensor.cpu()
            out[key] = tensor
    return out


def train_fedrep(clients, train_ids, seed, device, rounds=None):
    """
    FedRep adaptation for this benchmark:
      shared representation = fc1 + ln1 + fc2
      personal head = out (128 -> 96)
      each round/client: 10 head epochs, then 1 representation epoch
      server aggregates representation only.
    """
    set_seed(seed)
    p = RESIDUAL_MLP_PROTOCOL
    mean, scale = make_scaler(clients, train_ids)
    cached = {s: client_train(clients[s], mean, scale) for s in train_ids}

    initial_model = ResidualMLP().to(device)
    initial_state = initial_model.state_dict()
    shared = select_state(initial_state, SHARED_PREFIXES)
    initial_head = select_state(initial_state, PERSONAL_PREFIXES, cpu=True)
    heads = {
        sid: select_state(initial_state, PERSONAL_PREFIXES)
        for sid in train_ids
    }

    for round_i in range(int(rounds or p.shared_rounds)):
        representation_updates, counts = [], []

        for sid in train_ids:
            local = ResidualMLP().to(device)
            state = local.state_dict()
            state.update(shared)
            state.update(heads[sid])
            local.load_state_dict(state)
            x, y = cached[sid]

            # Client-head update: representation fixed.
            train_epochs(
                local,
                x,
                y,
                seed=seed,
                round_i=round_i,
                station_id=sid,
                phase_code=2,
                device=device,
                epochs=FEDREP_HEAD_EPOCHS,
                learning_rate=p.shared_learning_rate,
                weight_decay=p.shared_weight_decay,
                trainable_prefixes=PERSONAL_PREFIXES,
                frozen_representation_eval=True,
            )

            # Representation update: optimized head fixed.
            train_epochs(
                local,
                x,
                y,
                seed=seed,
                round_i=round_i,
                station_id=sid,
                phase_code=3,
                device=device,
                epochs=FEDREP_REPRESENTATION_EPOCHS,
                learning_rate=p.shared_learning_rate,
                weight_decay=p.shared_weight_decay,
                trainable_prefixes=SHARED_PREFIXES,
            )

            local_state = local.state_dict()
            representation_updates.append(
                select_state(local_state, SHARED_PREFIXES)
            )
            heads[sid] = select_state(local_state, PERSONAL_PREFIXES)
            counts.append(len(x))

        shared = weighted_average(
            representation_updates,
            counts,
            shared.keys(),
        )

    return shared, initial_head, mean, scale


def to_long(client, dates, y, anchor, pred, fold, seed):
    rows = []
    for day_i, date in enumerate(dates):
        for slot in range(96):
            rows.append(
                {
                    "fold": fold,
                    "seed": seed,
                    "station_id": int(client["station_id"]),
                    "type": str(client["station_type"]),
                    "prediction_date": pd.Timestamp(date),
                    "slot": slot,
                    "y_true": float(y[day_i, slot]),
                    "previous_day_baseline": float(anchor[day_i, slot]),
                    "prediction": float(pred[day_i, slot]),
                }
            )
    return pd.DataFrame(rows)


def predict_direct(model, mean, scale, client, start, end, fold, seed, device):
    m = mask(client, start, end)
    dates = pd.to_datetime(np.asarray(client["forecast_date"])[m]).normalize()
    x = torch.from_numpy(
        transform(np.asarray(client["x"])[m], mean, scale)
    ).to(device)
    model.eval()
    with torch.no_grad():
        residual = model(x).cpu().numpy()
    anchor = np.asarray(client["baseline"], float)[m]
    y = np.asarray(client["y_true"], float)[m]
    return to_long(
        client,
        dates,
        y,
        anchor,
        np.maximum(0, anchor + residual),
        fold,
        seed,
    )


def predict_fedrep(
    shared,
    target_head_init,
    mean,
    scale,
    client,
    start,
    end,
    fold,
    seed,
    device,
):
    """
    New-client FedRep deployment:
      - shared representation remains fixed;
      - a fresh target head starts from the seed's common initial head;
      - head is fit on the rolling 30-day pre-d history only;
      - current day d and all future labels are excluded.
    """
    p = RESIDUAL_MLP_PROTOCOL
    dates = pd.to_datetime(client["forecast_date"]).normalize()
    x = transform(client["x"], mean, scale)
    residual = np.asarray(client["y_residual"], np.float32)
    anchor = np.asarray(client["baseline"], float)
    y = np.asarray(client["y_true"], float)
    rows = []

    for i, date in enumerate(dates):
        if date < start or date > end:
            continue

        history = np.where(
            np.asarray(
                (dates < date)
                & (dates >= date - pd.Timedelta(days=p.target_history_days))
            )
        )[0]
        if len(history) == 0:
            raise AssertionError("FedRep target history missing")

        model = ResidualMLP().to(device)
        state = model.state_dict()
        state.update(shared)
        state.update({k: v.to(device) for k, v in target_head_init.items()})
        model.load_state_dict(state)

        # Use the benchmark's target optimizer settings, but FedRep's
        # algorithm-specific 10-epoch head-only adaptation schedule.
        train_epochs(
            model,
            x[history],
            residual[history],
            seed=seed,
            round_i=10_000 + i,
            station_id=int(client["station_id"]),
            phase_code=4,
            device=device,
            epochs=FEDREP_TARGET_HEAD_EPOCHS,
            learning_rate=p.target_learning_rate,
            weight_decay=p.target_weight_decay,
            trainable_prefixes=PERSONAL_PREFIXES,
            grad_clip=p.target_grad_clip,
            frozen_representation_eval=True,
        )

        model.eval()
        with torch.no_grad():
            r = (
                model(torch.from_numpy(x[i : i + 1]).to(device))
                .cpu()
                .numpy()[0]
            )
        rows.append(
            to_long(
                client,
                [date],
                y[i : i + 1],
                anchor[i : i + 1],
                np.maximum(0, anchor[i : i + 1] + r[None, :]),
                fold,
                seed,
            )
        )

    return pd.concat(rows, ignore_index=True)


def pointwise_mean(frames):
    if not frames:
        raise AssertionError("no frames to average")
    keys = [
        "fold",
        "station_id",
        "type",
        "prediction_date",
        "slot",
        "y_true",
        "previous_day_baseline",
    ]
    out = frames[0].drop(columns="seed").rename(
        columns={"prediction": "p0"}
    )
    for i, frame in enumerate(frames[1:], 1):
        out = out.merge(
            frame.drop(columns="seed").rename(
                columns={"prediction": f"p{i}"}
            ),
            on=keys,
            validate="one_to_one",
        )
    cols = [f"p{i}" for i in range(len(frames))]
    out["prediction"] = out[cols].mean(axis=1)
    return out[keys + ["prediction"]]


def evaluate(name, frame, outdir):
    metrics = aggregate_system_metrics(frame, "prediction")
    baseline_station = station_wape_table(
        frame, "previous_day_baseline"
    ).rename(columns={"WAPE": "baseline_WAPE"})
    method_station = station_wape_table(
        frame, "prediction"
    ).rename(columns={"WAPE": "method_WAPE"})
    station = baseline_station.merge(
        method_station,
        on=["station_id", "type"],
    )
    station["relative_change"] = (
        station.method_WAPE / station.baseline_WAPE - 1
    )

    types = type_aggregate_wape_table(
        frame,
        "prediction",
        "previous_day_baseline",
    )

    locked = load_locked_comparison_predictions(
        ROOT / "results_reference/oof_predictions_source.csv.gz"
    )
    keys = [
        "fold",
        "station_id",
        "type",
        "prediction_date",
        "slot",
        "y_true",
        "previous_day_baseline",
    ]
    comparison = locked[keys + ["proposed_prediction"]].merge(
        frame[keys + ["prediction"]],
        on=keys,
        validate="one_to_one",
    )
    daily = paired_daily_inference(
        comparison,
        proposed_column="proposed_prediction",
        baseline_column="prediction",
    )

    summary = {
        "method": name,
        "record_count": int(len(frame)),
        "WAPE": float(metrics["WAPE"]),
        "nRMSE": float(metrics["nRMSE"]),
        "pooled_WAPE": float(metrics["pooled_WAPE"]),
        "improved_stations": int((station.relative_change < 0).sum()),
        "max_station_relative_degradation": float(
            max(0, station.relative_change.max())
        ),
        "paired_daily_proposed_minus_method": daily,
    }

    frame.to_csv(
        outdir / f"{name}_oof.csv.gz",
        index=False,
        compression="gzip",
    )
    station.to_csv(outdir / f"{name}_station.csv", index=False)
    types.to_csv(outdir / f"{name}_type.csv", index=False)
    (outdir / f"{name}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def fold_split(clients, fold):
    test = sorted(fold.test)
    validation = sorted(fold.validation)
    train = sorted(set(clients) - set(test) - set(validation))
    assert len(train) == 36 and len(validation) == 2 and len(test) == 2
    return train, validation, test


def seed_metric_rows(method: str, frames_by_seed: dict[int, list[pd.DataFrame]]):
    rows = []
    for seed in sorted(frames_by_seed):
        frame = pd.concat(frames_by_seed[seed], ignore_index=True)
        metrics = aggregate_system_metrics(frame, "prediction")
        rows.append(
            {
                "method": method,
                "seed": int(seed),
                "record_count": int(len(frame)),
                "WAPE": float(metrics["WAPE"]),
                "nRMSE": float(metrics["nRMSE"]),
                "pooled_WAPE": float(metrics["pooled_WAPE"]),
            }
        )
    return rows


def formal_run(folds, seeds, rounds, smoke: bool) -> bool:
    p = RESIDUAL_MLP_PROTOCOL
    return (
        not smoke
        and len(folds) == 5
        and list(seeds) == list(NEURAL_BASELINE_SEEDS)
        and int(rounds or p.shared_rounds) == int(p.shared_rounds)
    )


def device_from_arg(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return device


def require_existing_calibration(
    outdir: Path,
    folds,
    seeds,
    rounds,
) -> None:
    path = outdir / "fedavg_calibration.json"
    if not path.exists():
        raise RuntimeError(
            "Formal FedProx/FedRep run requires a prior formal FedAvg "
            "calibration PASS, but fedavg_calibration.json is missing."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_folds = [fold.fold for fold in folds]
    expected_seeds = [int(seed) for seed in seeds]
    expected_rounds = int(rounds or RESIDUAL_MLP_PROTOCOL.shared_rounds)
    protocol_matches = (
        payload.get("formal_protocol") is True
        and payload.get("folds") == expected_folds
        and payload.get("seeds") == expected_seeds
        and int(payload.get("rounds", -1)) == expected_rounds
        and float(payload.get("acceptance_atol", -1.0))
        == FEDAVG_CALIBRATION_ATOL
    )
    if payload.get("status") != "PASS" or not protocol_matches:
        raise RuntimeError(
            "Formal FedProx/FedRep run blocked: the existing FedAvg "
            "calibration is missing, not PASS, or does not match the "
            "locked folds/seeds/rounds/tolerance protocol."
        )


def run_fedavg(
    clients,
    folds,
    seeds,
    device,
    rounds,
    outdir,
    *,
    formal,
):
    per_fold = []
    by_seed = defaultdict(list)

    for fold in folds:
        train, _, test = fold_split(clients, fold)
        seed_frames = []
        for seed in seeds:
            model, mean, scale = train_fedavg_prox(
                clients,
                train,
                seed,
                device,
                0.0,
                rounds,
            )
            frame = pd.concat(
                [
                    predict_direct(
                        model,
                        mean,
                        scale,
                        clients[sid],
                        TEST_START,
                        TEST_END,
                        fold.fold,
                        seed,
                        device,
                    )
                    for sid in test
                ],
                ignore_index=True,
            )
            seed_frames.append(frame)
            by_seed[seed].append(frame)
            print(
                f"[fedavg] {fold.fold} seed={seed}",
                flush=True,
            )
        per_fold.append(pointwise_mean(seed_frames))

    averaged = pd.concat(per_fold, ignore_index=True)
    pd.DataFrame(seed_metric_rows("fedavg_calibration", by_seed)).to_csv(
        outdir / "fedavg_calibration_seed_metrics.csv",
        index=False,
    )

    locked = load_locked_comparison_predictions(
        ROOT / "results_reference/oof_predictions_source.csv.gz"
    )
    locked = locked[locked.fold.isin([f.fold for f in folds])]
    new_wape = aggregate_system_metrics(averaged, "prediction")["WAPE"]
    locked_wape = aggregate_system_metrics(
        locked,
        "fedavg_direct_kw",
    )["WAPE"]
    difference = float(abs(new_wape - locked_wape))

    calibration = {
        "new_WAPE": float(new_wape),
        "locked_WAPE": float(locked_wape),
        "absolute_difference": difference,
        "acceptance_atol": FEDAVG_CALIBRATION_ATOL,
        "formal_protocol": bool(formal),
        "folds": [f.fold for f in folds],
        "seeds": [int(s) for s in seeds],
        "rounds": int(rounds or RESIDUAL_MLP_PROTOCOL.shared_rounds),
        "status": (
            ("PASS" if difference <= FEDAVG_CALIBRATION_ATOL else "REVIEW")
            if formal
            else "DEVELOPMENT_ONLY"
        ),
    }
    (outdir / "fedavg_calibration.json").write_text(
        json.dumps(calibration, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(calibration, ensure_ascii=False, indent=2), flush=True)

    summary = evaluate("fedavg_calibration", averaged, outdir) if formal else None
    return calibration, summary


def run_fedprox(
    clients,
    folds,
    seeds,
    device,
    rounds,
    outdir,
    *,
    formal,
):
    selected = {}
    final_folds = []
    grid_rows = []
    by_seed = defaultdict(list)

    for fold in folds:
        train, validation, test = fold_split(clients, fold)
        cache = {}
        candidates = []

        for mu in FEDPROX_MU_GRID:
            validation_frames = []
            for seed in seeds:
                model, mean, scale = train_fedavg_prox(
                    clients,
                    train,
                    seed,
                    device,
                    mu,
                    rounds,
                )
                cache[(mu, seed)] = (model, mean, scale)
                validation_frame = pd.concat(
                    [
                        predict_direct(
                            model,
                            mean,
                            scale,
                            clients[sid],
                            VAL_START,
                            VAL_END,
                            fold.fold,
                            seed,
                            device,
                        )
                        for sid in validation
                    ],
                    ignore_index=True,
                )
                validation_frames.append(validation_frame)
                print(
                    f"[fedprox-val] {fold.fold} mu={mu} seed={seed}",
                    flush=True,
                )

            validation_mean = pointwise_mean(validation_frames)
            metric = aggregate_system_metrics(
                validation_mean,
                "prediction",
            )
            row = {
                "fold": fold.fold,
                "mu": float(mu),
                "validation_WAPE": float(metric["WAPE"]),
                "validation_nRMSE": float(metric["nRMSE"]),
            }
            grid_rows.append(row)
            candidates.append(row)

        best = min(
            candidates,
            key=lambda row: (row["validation_WAPE"], row["mu"]),
        )
        selected[fold.fold] = best["mu"]

        test_frames = []
        for seed in seeds:
            model, mean, scale = cache[(best["mu"], seed)]
            frame = pd.concat(
                [
                    predict_direct(
                        model,
                        mean,
                        scale,
                        clients[sid],
                        TEST_START,
                        TEST_END,
                        fold.fold,
                        seed,
                        device,
                    )
                    for sid in test
                ],
                ignore_index=True,
            )
            test_frames.append(frame)
            by_seed[seed].append(frame)

        final_folds.append(pointwise_mean(test_frames))
        print(
            f"[fedprox] {fold.fold} selected mu={best['mu']}",
            flush=True,
        )

    pd.DataFrame(grid_rows).to_csv(
        outdir / "fedprox_validation_grid.csv",
        index=False,
    )
    (outdir / "fedprox_selected_mu.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(seed_metric_rows("fedprox", by_seed)).to_csv(
        outdir / "fedprox_seed_metrics.csv",
        index=False,
    )

    averaged = pd.concat(final_folds, ignore_index=True)
    return evaluate("fedprox", averaged, outdir) if formal else None


def run_fedrep(
    clients,
    folds,
    seeds,
    device,
    rounds,
    outdir,
    *,
    formal,
):
    final_folds = []
    validation_rows = []
    by_seed = defaultdict(list)

    for fold in folds:
        train, validation, test = fold_split(clients, fold)
        seed_test_frames = []

        for seed in seeds:
            shared, target_head_init, mean, scale = train_fedrep(
                clients,
                train,
                seed,
                device,
                rounds,
            )

            validation_frame = pd.concat(
                [
                    predict_fedrep(
                        shared,
                        target_head_init,
                        mean,
                        scale,
                        clients[sid],
                        VAL_START,
                        VAL_END,
                        fold.fold,
                        seed,
                        device,
                    )
                    for sid in validation
                ],
                ignore_index=True,
            )
            validation_metric = aggregate_system_metrics(
                validation_frame,
                "prediction",
            )
            validation_rows.append(
                {
                    "fold": fold.fold,
                    "seed": int(seed),
                    "validation_WAPE": float(validation_metric["WAPE"]),
                    "validation_nRMSE": float(validation_metric["nRMSE"]),
                }
            )

            test_frame = pd.concat(
                [
                    predict_fedrep(
                        shared,
                        target_head_init,
                        mean,
                        scale,
                        clients[sid],
                        TEST_START,
                        TEST_END,
                        fold.fold,
                        seed,
                        device,
                    )
                    for sid in test
                ],
                ignore_index=True,
            )
            seed_test_frames.append(test_frame)
            by_seed[seed].append(test_frame)
            print(
                f"[fedrep] {fold.fold} seed={seed}",
                flush=True,
            )

        final_folds.append(pointwise_mean(seed_test_frames))

    pd.DataFrame(validation_rows).to_csv(
        outdir / "fedrep_validation.csv",
        index=False,
    )
    pd.DataFrame(seed_metric_rows("fedrep", by_seed)).to_csv(
        outdir / "fedrep_seed_metrics.csv",
        index=False,
    )

    averaged = pd.concat(final_folds, ignore_index=True)
    return evaluate("fedrep", averaged, outdir) if formal else None


def main():
    parser = argparse.ArgumentParser()
    default_output_dir = ROOT / "results_revision_r1/neural_baselines"
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=list(FORMAL_METHODS),
        default=list(FORMAL_METHODS),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--folds", nargs="*")
    parser.add_argument(
        "--seeds",
        nargs="*",
        type=int,
        default=list(NEURAL_BASELINE_SEEDS),
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
    )
    args = parser.parse_args()

    cfg = load_config(ROOT / "paper_config.json")
    folds = build_folds(cfg)
    clients = load_clients()
    seeds = list(args.seeds)
    rounds = args.rounds

    if args.folds:
        requested = set(args.folds)
        folds = [fold for fold in folds if fold.fold in requested]
        if len(folds) != len(requested):
            known = {fold.fold for fold in build_folds(cfg)}
            raise ValueError(
                f"unknown fold requested; requested={sorted(requested)}, "
                f"known={sorted(known)}"
            )

    if args.smoke:
        folds = folds[:1]
        seeds = seeds[:1]
        rounds = 2
        print("SMOKE MODE: not paper evidence", flush=True)

    if args.smoke and args.output_dir == default_output_dir:
        args.output_dir = ROOT / "results_revision_r1/neural_baselines_smoke"

    device = device_from_arg(args.device)
    outdir = args.output_dir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    formal = formal_run(folds, seeds, rounds, args.smoke)
    cache_path = ROOT / "data/clients_with_spatial.pkl"
    locked_oof_path = ROOT / "results_reference/oof_predictions_source.csv.gz"

    metadata = {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        "git_commit": git_commit(),
        "clients_cache_sha256": sha256(cache_path),
        "locked_oof_sha256": sha256(locked_oof_path),
        "feature_dimension": int(RESIDUAL_MLP_PROTOCOL.input_dim),
        "output_dimension": int(RESIDUAL_MLP_PROTOCOL.output_dim),
        "folds": [fold.fold for fold in folds],
        "seeds": [int(seed) for seed in seeds],
        "rounds": int(rounds or RESIDUAL_MLP_PROTOCOL.shared_rounds),
        "methods": list(args.methods),
        "fedprox_mu_grid": list(FEDPROX_MU_GRID),
        "fedavg_calibration_atol": FEDAVG_CALIBRATION_ATOL,
        "fedrep": {
            "shared_prefixes": list(SHARED_PREFIXES),
            "personal_prefixes": list(PERSONAL_PREFIXES),
            "head_epochs_per_round": FEDREP_HEAD_EPOCHS,
            "representation_epochs_per_round": FEDREP_REPRESENTATION_EPOCHS,
            "target_head_epochs": FEDREP_TARGET_HEAD_EPOCHS,
            "target_history_days": int(
                RESIDUAL_MLP_PROTOCOL.target_history_days
            ),
            "target_head_initialization": FEDREP_TARGET_HEAD_INIT,
            "target_representation_frozen": True,
        },
        "formal_protocol": formal,
        "smoke": args.smoke,
    }
    metadata_text = json.dumps(metadata, ensure_ascii=False, indent=2)
    (outdir / "run_metadata.json").write_text(
        metadata_text,
        encoding="utf-8",
    )
    method_tag = "_".join(args.methods)
    (outdir / f"run_metadata_{method_tag}.json").write_text(
        metadata_text,
        encoding="utf-8",
    )
    print(metadata_text, flush=True)

    summaries = []
    requested_new = any(
        method in args.methods for method in ("fedprox", "fedrep")
    )

    if "fedavg" in args.methods:
        calibration, summary = run_fedavg(
            clients,
            folds,
            seeds,
            device,
            rounds,
            outdir,
            formal=formal,
        )
        if summary is not None:
            summaries.append(summary)
        if formal and requested_new and calibration["status"] != "PASS":
            raise RuntimeError(
                "FedAvg calibration is not PASS; formal FedProx/FedRep "
                "training is blocked until the baseline implementation is reviewed."
            )
    elif formal and requested_new:
        require_existing_calibration(outdir, folds, seeds, rounds)

    if "fedprox" in args.methods:
        summary = run_fedprox(
            clients,
            folds,
            seeds,
            device,
            rounds,
            outdir,
            formal=formal,
        )
        if summary is not None:
            summaries.append(summary)

    if "fedrep" in args.methods:
        summary = run_fedrep(
            clients,
            folds,
            seeds,
            device,
            rounds,
            outdir,
            formal=formal,
        )
        if summary is not None:
            summaries.append(summary)

    if summaries:
        table = pd.DataFrame(summaries)
        table.to_csv(
            outdir / "new_baseline_summary.csv",
            index=False,
        )
        print(table.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
