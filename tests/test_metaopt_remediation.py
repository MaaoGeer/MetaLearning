"""CPU-level regression tests for MetaOpt remediation work."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from src.data.dataset import IntrusionDataset
from src.data.task_sampler import MetaTask
from src.evaluation.adaptation_speed import compute_speed
from src.evaluation.metrics import compute_metrics
from src.evaluation.prediction_artifact import write_prediction_trajectories
from src.evaluation.task_manifest import (
    assert_manifest_split_isolation,
    manifest_reuse_statistics,
    read_task_manifest,
    write_task_manifest,
)
from src.meta_learning.functional import functional_forward
from src.meta_learning.inner_loop import InnerLoop
from src.meta_learning.outer_loop import OuterLoop, normalized_step_weights
from src.meta_optimizer.dummy import DummyMetaOptimizer
from src.meta_optimizer.handcrafted import HandcraftedOptimizer
from src.meta_optimizer.lstm_optimizer import LSTMOptimizer
from src.models import LSTMClassifier


def _task(seed: int = 7, feat: int = 4, seq: int = 5) -> MetaTask:
    generator = torch.Generator().manual_seed(seed)
    return MetaTask(
        support_x=torch.randn(6, seq, feat, generator=generator),
        support_y=torch.tensor([0, 0, 0, 1, 1, 1]),
        query_x=torch.randn(8, seq, feat, generator=generator),
        query_y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        global_classes=[0, 1],
        shot=3,
    )


def _run_functional_optimizer(model, initial_state, task, optimizer, steps=20):
    params = {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in initial_state.items()
    }
    state = optimizer.init_state(params)
    rows = []
    for step in range(steps + 1):
        query_logits = functional_forward(model, params, task.query_x)
        metrics = compute_metrics(query_logits.detach(), task.query_y, num_classes=2)
        rows.append({
            "params": {name: value.detach().clone() for name, value in params.items()},
            "logits": query_logits.detach().clone(),
            "probs": torch.softmax(query_logits.detach(), dim=1),
            "query_loss": F.cross_entropy(query_logits, task.query_y).detach(),
            "metrics": metrics.as_dict(),
            "updates": {},
        })
        if step == steps:
            break
        support_logits = functional_forward(model, params, task.support_x)
        support_loss = F.cross_entropy(support_logits, task.support_y)
        grads = torch.autograd.grad(support_loss, list(params.values()))
        updates, state = optimizer.step(dict(zip(params, grads)), state)
        rows[-1]["support_loss"] = support_loss.detach()
        rows[-1]["updates"] = {
            name: update.detach().clone() for name, update in updates.items()
        }
        params = {
            name: (params[name] + updates[name]).detach().clone().requires_grad_(True)
            for name in params
        }
        state = optimizer.detach_state(state)
    return rows


def test_dummy_meta_optimizer_matches_sgd_for_twenty_steps():
    torch.manual_seed(11)
    model = LSTMClassifier(
        feature_dim=4, n_classes=2, hidden_size=6, num_layers=1
    )
    initial = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("classifier.")
    }
    task = _task()
    lr = 0.17
    dummy_rows = _run_functional_optimizer(
        model, initial, task, DummyMetaOptimizer(lr=lr), steps=20
    )
    sgd_rows = _run_functional_optimizer(
        model, initial, task, HandcraftedOptimizer(kind="sgd", lr=lr), steps=20
    )
    maximum = 0.0
    metric_names = (
        "accuracy", "macro_f1", "roc_auc", "pr_auc",
        "brier_score", "ece",
    )
    for dummy, sgd in zip(dummy_rows, sgd_rows):
        for name in initial:
            maximum = max(
                maximum,
                float((dummy["params"][name] - sgd["params"][name]).abs().max()),
            )
        maximum = max(
            maximum, float((dummy["logits"] - sgd["logits"]).abs().max())
        )
        maximum = max(
            maximum, float((dummy["probs"] - sgd["probs"]).abs().max())
        )
        maximum = max(
            maximum, float((dummy["query_loss"] - sgd["query_loss"]).abs())
        )
        for name in dummy["updates"]:
            maximum = max(
                maximum,
                float((dummy["updates"][name] - sgd["updates"][name]).abs().max()),
            )
        for metric in metric_names:
            maximum = max(
                maximum,
                abs(float(dummy["metrics"][metric]) - float(sgd["metrics"][metric])),
            )
    print(f"dummy_sgd_max_abs_diff={maximum:.12g}")
    assert maximum < 1e-6


def test_second_order_meta_gradients_are_finite_and_nonzero_per_layer():
    torch.manual_seed(13)
    model = LSTMClassifier(feature_dim=4, n_classes=2, hidden_size=6)
    meta_opt = LSTMOptimizer(hidden_size=7, num_layers=2)
    inner = InnerLoop(
        model, meta_opt, inner_steps=3, tbptt_steps=0, first_order=False
    )
    result = OuterLoop(model, inner).run_meta_batch([_task(seed=17)])
    result.meta_loss.backward()
    norms = {}
    for name, parameter in meta_opt.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        norm = float(torch.linalg.vector_norm(parameter.grad))
        assert norm > 0.0, name
        norms[name] = norm
    assert any(name.startswith("cells.0") for name in norms)
    assert any(name.startswith("cells.1") for name in norms)
    assert any(name.startswith("output.") for name in norms)
    # Log-sign preprocessing is parameter-free; cells.0.weight_ih is its first
    # trainable consumer and therefore verifies the input path.
    assert norms["cells.0.weight_ih"] > 0
    print("meta_gradient_norms=" + json.dumps(norms, sort_keys=True))


def test_speed_counts_step_zero_and_keeps_deprecated_legacy_value():
    speed = compute_speed(
        [0.82, 0.84, 0.86],
        [0.8],
        max_steps=2,
        trajectory_includes_step_zero=True,
    )
    assert speed.speeds[0.8] == 0
    assert speed.speeds_deprecated_excluding_step0[0.8] == 1


def test_multistep_weights_are_normalized_and_early_heavy():
    uniform = normalized_step_weights([1, 2, 5, 10, 20], "uniform")
    early = normalized_step_weights(
        [1, 2, 5, 10, 20], "early_heavy", early_heavy_power=0.5
    )
    custom = normalized_step_weights(
        [1, 2, 5], "custom", custom_weights=[4, 2, 1]
    )
    assert sum(uniform.values()) == pytest.approx(1.0)
    assert sum(early.values()) == pytest.approx(1.0)
    assert early[1] > early[2] > early[5] > early[10] > early[20]
    assert custom == pytest.approx({1: 4 / 7, 2: 2 / 7, 5: 1 / 7})


def test_random_horizon_is_seed_deterministic_and_multistep_backpropagates():
    torch.manual_seed(19)
    model_a = LSTMClassifier(feature_dim=4, n_classes=2, hidden_size=6)
    model_b = LSTMClassifier(feature_dim=4, n_classes=2, hidden_size=6)
    model_b.load_state_dict(model_a.state_dict())
    opt_a = LSTMOptimizer(hidden_size=6, num_layers=1)
    opt_b = LSTMOptimizer(hidden_size=6, num_layers=1)
    opt_b.load_state_dict(opt_a.state_dict())
    config = {"enabled": True, "min_steps": 1, "max_steps": 3}
    objective = {
        "mode": "multi_step",
        "supervised_steps": [1, 2, 3],
        "weighting": "early_heavy",
        "include_sampled_horizon": True,
    }
    outer_a = OuterLoop(
        model_a, InnerLoop(model_a, opt_a, inner_steps=3, first_order=False),
        query_objective=objective, random_horizon=config, seed=23,
    )
    outer_b = OuterLoop(
        model_b, InnerLoop(model_b, opt_b, inner_steps=3, first_order=False),
        query_objective=objective, random_horizon=config, seed=23,
    )
    sequence_a = [outer_a.sample_horizon() for _ in range(10)]
    sequence_b = [outer_b.sample_horizon() for _ in range(10)]
    assert sequence_a == sequence_b
    # Fresh identical objects verify deterministic sampling inside run_meta_batch.
    outer_a = OuterLoop(
        model_a, InnerLoop(model_a, opt_a, inner_steps=3, first_order=False),
        query_objective=objective, random_horizon=config, seed=29,
    )
    result = outer_a.run_meta_batch([_task(seed=31), _task(seed=37)])
    assert result.query_loss_by_step
    assert sum(result.weighted_contribution_by_step.values()) == pytest.approx(
        float(result.meta_loss.detach()), rel=1e-5
    )
    result.meta_loss.backward()
    assert sum(
        float(parameter.grad.abs().sum())
        for parameter in opt_a.parameters()
        if parameter.grad is not None
    ) > 0


def test_residual_mode_initializes_near_sgd_and_disabled_is_exact_sgd():
    gradient = {"classifier.weight": torch.randn(2, 5)}
    params = {"classifier.weight": torch.zeros(2, 5)}
    anchored = LSTMOptimizer(
        hidden_size=5,
        num_layers=1,
        update_mode="sgd_residual",
        anchor_lr=0.2,
        residual_zero_init=True,
        gate_init=0.1,
        update_norm_clip=None,
    )
    update, _ = anchored.step(gradient, anchored.init_state(params))
    assert torch.allclose(
        update["classifier.weight"], -0.2 * gradient["classifier.weight"],
        atol=1e-7, rtol=0,
    )
    disabled = LSTMOptimizer(
        hidden_size=5,
        num_layers=1,
        update_mode="sgd_residual",
        anchor_lr=0.2,
        residual_enabled=False,
        residual_zero_init=False,
        trust_region_factor=0.1,
        update_norm_clip=1e-6,
    )
    update, _ = disabled.step(gradient, disabled.init_state(params))
    assert torch.equal(
        update["classifier.weight"], -0.2 * gradient["classifier.weight"]
    )


def _manifest_dataset() -> IntrusionDataset:
    features = np.arange(12 * 2 * 2, dtype=np.float32).reshape(12, 2, 2)
    labels = np.array([0] * 6 + [1] * 6, dtype=np.int64)
    row_ids = np.arange(12 * 2, dtype=np.int64).reshape(12, 2)
    return IntrusionDataset(
        features, labels, sequence_length=2,
        row_ids=row_ids,
        segment_id=np.arange(12, dtype=np.int64),
        order_start=np.arange(12, dtype=np.float64),
        order_end=np.arange(12, dtype=np.float64),
    )


def _manifest_task(dataset: IntrusionDataset, ids=(0, 6, 1, 7)) -> MetaTask:
    s0, s1, q0, q1 = ids
    return MetaTask(
        dataset.features[[s0, s1]], torch.tensor([0, 1]),
        dataset.features[[q0, q1]], torch.tensor([0, 1]),
        [0, 1], [s0, s1], [q0, q1], shot=1,
    )


def test_manifest_reproducibility_reuse_and_split_isolation(tmp_path):
    dataset = _manifest_dataset()
    val_path = tmp_path / "val.json"
    test_path = tmp_path / "test.json"
    common = {
        "base_checkpoint_path": "artifact.pt",
        "base_checkpoint_sha256": "a" * 64,
        "base_initialization_sha256": "b" * 64,
        "dataset": dataset,
    }
    write_task_manifest(
        val_path, [_manifest_task(dataset)],
        protocol={"shot": 1, "q_query": 1, "split": "val",
                  "task_seed": 1, "attack": "ddos"},
        **common,
    )
    write_task_manifest(
        test_path, [_manifest_task(dataset, (2, 8, 3, 9))],
        protocol={"shot": 1, "q_query": 1, "split": "test",
                  "task_seed": 2, "attack": "ddos"},
        **common,
    )
    val = read_task_manifest(val_path)
    test = read_task_manifest(test_path)
    assert_manifest_split_isolation(val, test)
    stats = manifest_reuse_statistics(val)
    assert stats["task_count"] == 1
    assert stats["unique_task_hashes"] == 1
    assert stats["raw_disjoint_task_count_greedy"] == 1
    with pytest.raises(ValueError, match="share"):
        assert_manifest_split_isolation(val, val | {
            "protocol": {**val["protocol"], "split": "test"}
        })


def test_prediction_artifact_schema_round_trip(tmp_path):
    logits = [torch.randn(4, 2) for _ in range(3)]
    destination = write_prediction_trajectories(
        tmp_path / "predictions.npz",
        [{
            "experiment": "smoke",
            "shot": 1,
            "method": "SGD",
            "split": "test_fixed_horizon",
            "task_id": 0,
            "step_logits": logits,
            "labels": torch.tensor([0, 0, 1, 1]),
        }],
    )
    arrays = np.load(destination)
    schema = json.loads(
        destination.with_suffix(".npz.schema.json").read_text(encoding="utf-8")
    )
    assert arrays["logits"].shape == (1, 3, 4, 2)
    assert arrays["labels"].shape == (1, 4)
    assert schema["step_axis_includes_zero"] is True
    assert "roc_auc" in schema["metrics_reconstructable"]

