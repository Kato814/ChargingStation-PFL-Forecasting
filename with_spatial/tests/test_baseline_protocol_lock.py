from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from baselines.protocols import LOCAL_RIDGE_PROTOCOL, NEURAL_BASELINE_SEEDS, PFEDME_SHARED_PROTOCOL, RESIDUAL_MLP_PROTOCOL


def main() -> None:
    mlp = RESIDUAL_MLP_PROTOCOL
    assert (mlp.input_dim, mlp.hidden_dim_1, mlp.hidden_dim_2, mlp.output_dim) == (539, 128, 128, 96)
    assert mlp.dropout == 0.08
    assert mlp.shared_rounds == 30
    assert mlp.shared_local_epochs == 1
    assert mlp.target_history_days == 30
    assert mlp.target_inner_steps == 3
    assert mlp.target_learning_rate == 5.0e-4
    assert NEURAL_BASELINE_SEEDS == (20260612, 20260613, 20260614)
    pf = PFEDME_SHARED_PROTOCOL
    assert pf.rounds == 30 and pf.personal_inner_steps == 3 and pf.personal_lambda == 1.0
    assert LOCAL_RIDGE_PROTOCOL.target_history_days == 60
    print("PASS: neural baseline provenance protocol locked")


if __name__ == "__main__":
    main()
