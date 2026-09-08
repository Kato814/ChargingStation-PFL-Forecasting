from __future__ import annotations

from dataclasses import dataclass


NEURAL_BASELINE_SEEDS = (20260612, 20260613, 20260614)


@dataclass(frozen=True, slots=True)
class ResidualMLPProtocol:
    input_dim: int = 539
    hidden_dim_1: int = 128
    hidden_dim_2: int = 128
    output_dim: int = 96
    dropout: float = 0.08
    layer_norm_after_first_hidden: bool = True
    shared_rounds: int = 30
    shared_local_epochs: int = 1
    shared_batch_size: int = 64
    shared_learning_rate: float = 8.0e-4
    shared_weight_decay: float = 1.0e-4
    target_history_days: int = 30
    target_inner_steps: int = 3
    target_learning_rate: float = 5.0e-4
    target_batch_size: int = 64
    target_weight_decay: float = 1.0e-4
    target_grad_clip: float = 5.0


@dataclass(frozen=True, slots=True)
class PFedMeSharedProtocol:
    rounds: int = 30
    local_epochs: int = 1
    batch_size: int = 64
    personal_inner_steps: int = 3
    personal_lambda: float = 1.0
    personal_learning_rate: float = 8.0e-4
    reference_learning_rate: float = 0.1
    server_beta: float = 1.0
    weight_decay: float = 1.0e-4
    grad_clip: float = 5.0


RESIDUAL_MLP_PROTOCOL = ResidualMLPProtocol()
PFEDME_SHARED_PROTOCOL = PFedMeSharedProtocol()


@dataclass(frozen=True, slots=True)
class LocalRidgeProtocol:
    target_history_days: int = 60
    input_dim: int = 539
    output_dim: int = 96


LOCAL_RIDGE_PROTOCOL = LocalRidgeProtocol()
