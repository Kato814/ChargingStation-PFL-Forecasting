from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class NegativeTransferConfig:
    history_days: int = 10
    min_history_days: int = 3
    relative_gain_margin: float = -0.02
    regret_budget: float = 0.03
    regret_warmup_days: int = 7


def recent_performance_eligibility(
    y_true: np.ndarray,
    baseline: np.ndarray,
    group_candidate: np.ndarray,
    config: NegativeTransferConfig,
) -> np.ndarray:
    """Equation (5), first condition: recent performance screen."""
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(group_candidate, dtype=np.float64)
    if y.shape != anchor.shape or y.shape != candidate.shape or y.ndim != 3:
        raise ValueError("arrays must share [days, stations, slots] shape")
    eligible = np.zeros((y.shape[0], y.shape[1]), dtype=np.int8)
    for day_index in range(y.shape[0]):
        start = max(0, day_index - config.history_days)
        if day_index - start < config.min_history_days:
            continue
        baseline_loss = np.abs(y[start:day_index] - anchor[start:day_index]).sum(axis=(0, 2))
        candidate_loss = np.abs(y[start:day_index] - candidate[start:day_index]).sum(axis=(0, 2))
        gain = (baseline_loss - candidate_loss) / np.maximum(baseline_loss, 1e-9)
        eligible[day_index] = (gain >= config.relative_gain_margin).astype(np.int8)
    return eligible


def cumulative_loss_guard(
    y_true: np.ndarray,
    baseline: np.ndarray,
    group_candidate: np.ndarray,
    recent_eligibility: np.ndarray,
    config: NegativeTransferConfig,
) -> np.ndarray:
    """Equation (5), second condition: cumulative excess-loss budget."""
    y = np.asarray(y_true, dtype=np.float64)
    anchor = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(group_candidate, dtype=np.float64)
    eligibility = np.asarray(recent_eligibility, dtype=np.int8)
    if eligibility.shape != y.shape[:2]:
        raise ValueError("eligibility must have [days, stations] shape")

    guarded = np.zeros_like(eligibility)
    for station_index in range(y.shape[1]):
        cumulative_baseline_loss = 0.0
        cumulative_candidate_loss = 0.0
        for day_index in range(y.shape[0]):
            budget_ok = (
                day_index >= config.regret_warmup_days
                and cumulative_candidate_loss - cumulative_baseline_loss
                <= config.regret_budget * max(cumulative_baseline_loss, 1e-9)
            )
            guarded[day_index, station_index] = int(bool(eligibility[day_index, station_index]) and budget_ok)
            cumulative_baseline_loss += float(np.abs(y[day_index, station_index] - anchor[day_index, station_index]).sum())
            cumulative_candidate_loss += float(np.abs(y[day_index, station_index] - candidate[day_index, station_index]).sum())
    return guarded


def fallback_and_redistribute(
    baseline: np.ndarray,
    group_candidate: np.ndarray,
    eligibility: np.ndarray,
) -> np.ndarray:
    """Equation (6): baseline fallback and equal redistribution among eligible stations."""
    anchor = np.asarray(baseline, dtype=np.float64)
    candidate = np.asarray(group_candidate, dtype=np.float64)
    enabled = np.asarray(eligibility, dtype=bool)
    if anchor.shape != candidate.shape or anchor.ndim != 3:
        raise ValueError("prediction arrays must have [days, stations, slots] shape")
    if enabled.shape != anchor.shape[:2]:
        raise ValueError("eligibility must have [days, stations] shape")

    correction = candidate - anchor
    output = anchor.copy()
    for day_index in range(anchor.shape[0]):
        mask = enabled[day_index]
        if not mask.any():
            continue
        for slot in range(anchor.shape[2]):
            total_correction = float(correction[day_index, :, slot].sum())
            selected = correction[day_index, mask, slot].copy()
            selected += (total_correction - float(selected.sum())) / int(mask.sum())
            output[day_index, mask, slot] = np.maximum(0.0, anchor[day_index, mask, slot] + selected)
    return output
