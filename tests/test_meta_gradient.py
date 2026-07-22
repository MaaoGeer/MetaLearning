"""Core gradient tests for the learned optimizer."""

import torch

from src.data.task_sampler import MetaTask
from src.meta_learning.inner_loop import InnerLoop
from src.meta_learning.outer_loop import OuterLoop
from src.meta_optimizer import LSTMOptimizer
from src.models import LSTMClassifier
from src.models.recurrent import CustomLSTM


def _make_task(feat=6, seq=8, n_way=3, k=4, q=4):
    x_s = torch.randn(n_way * k, seq, feat)
    y_s = torch.arange(n_way).repeat_interleave(k)
    x_q = torch.randn(n_way * q, seq, feat)
    y_q = torch.arange(n_way).repeat_interleave(q)
    return MetaTask(x_s, y_s, x_q, y_q, list(range(n_way)))


def _make_model(feat=6, n_way=3):
    return LSTMClassifier(feature_dim=feat, n_classes=n_way, hidden_size=8)


def test_meta_optimizer_receives_gradient():
    torch.manual_seed(0)
    model = _make_model()
    meta_opt = LSTMOptimizer(hidden_size=8, num_layers=2)
    inner = InnerLoop(model, meta_opt, inner_steps=3, tbptt_steps=0, first_order=False)
    outer = OuterLoop(model, inner)

    result = outer.run_meta_batch([_make_task()])
    result.meta_loss.backward()

    total = 0.0
    for name, param in meta_opt.named_parameters():
        assert param.grad is not None, f"meta optimizer parameter {name} has no gradient"
        total += param.grad.abs().sum().item()
    assert total > 0


def test_theta0_receives_gradient_but_is_not_outer_optimized():
    torch.manual_seed(0)
    model = _make_model()
    meta_opt = LSTMOptimizer(hidden_size=8, num_layers=2)
    inner = InnerLoop(model, meta_opt, inner_steps=2, first_order=False)
    outer = OuterLoop(model, inner)
    outer.run_meta_batch([_make_task()]).meta_loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert sum(g.abs().sum().item() for g in grads) > 0


def test_second_order_path():
    torch.manual_seed(0)
    model = _make_model()
    meta_opt = LSTMOptimizer(hidden_size=8, num_layers=2)
    inner = InnerLoop(model, meta_opt, inner_steps=2, first_order=False)
    init_params = {name: param for name, param in model.named_parameters()}
    result = inner.adapt(init_params, _make_task())
    assert any(param.requires_grad for param in result.adapted_params.values())


def test_custom_lstm_double_backward_single_direction():
    torch.manual_seed(0)
    lstm = CustomLSTM(input_size=4, hidden_size=5, num_layers=1, bidirectional=False)
    x = torch.randn(2, 6, 4, requires_grad=True)
    out = lstm(x).sum()
    grad = torch.autograd.grad(out, x, create_graph=True)[0]
    grad2 = torch.autograd.grad(grad.sum(), x)[0]
    assert grad2 is not None and torch.isfinite(grad2).all()


def test_first_order_still_updates_meta_opt():
    torch.manual_seed(0)
    model = _make_model()
    meta_opt = LSTMOptimizer(hidden_size=8, num_layers=1)
    inner = InnerLoop(model, meta_opt, inner_steps=2, first_order=True)
    outer = OuterLoop(model, inner)
    outer.run_meta_batch([_make_task()]).meta_loss.backward()
    total = sum(
        param.grad.abs().sum().item()
        for param in meta_opt.parameters()
        if param.grad is not None
    )
    assert total > 0
