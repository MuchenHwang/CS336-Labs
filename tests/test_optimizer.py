import numpy
import torch

from cs336.optimizer import (
    Muon,
    zeropower_via_newtonschulz5,
)
from .adapters import get_muon_cls, run_get_lr_cosine_schedule


def test_muon_adapter():
    assert get_muon_cls() is Muon

    parameter = torch.nn.Parameter(torch.ones(2, 2))
    parameter.grad = torch.ones_like(parameter)
    optimizer = Muon(
        [parameter],
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    optimizer.step()
    assert torch.isfinite(parameter).all()


def test_newton_schulz_orthogonalizes_matrix():
    torch.manual_seed(42)
    gradient = torch.randn(8, 5)
    update = zeropower_via_newtonschulz5(gradient)

    assert update.shape == gradient.shape
    assert update.dtype == gradient.dtype
    assert torch.isfinite(update).all()

    original_singular_values = torch.linalg.svdvals(
        gradient / gradient.norm()
    )
    update_singular_values = torch.linalg.svdvals(update)
    original_condition = (
        original_singular_values.max() / original_singular_values.min()
    )
    update_condition = (
        update_singular_values.max() / update_singular_values.min()
    )
    assert update_condition < original_condition * 0.5


def test_muon_uses_orthogonalized_matrix_update():
    torch.manual_seed(42)
    parameter = torch.nn.Parameter(torch.zeros(6, 4))
    parameter.grad = torch.randn_like(parameter)
    learning_rate = 0.05
    optimizer = Muon(
        [parameter],
        lr=learning_rate,
        weight_decay=0.0,
    )

    optimizer.step()

    gradient = parameter.grad
    momentum = 0.1 * gradient
    nesterov_update = gradient.lerp(momentum, 0.9)
    expected_update = zeropower_via_newtonschulz5(nesterov_update)
    expected_update *= max(
        1.0,
        parameter.size(0) / parameter.size(1),
    ) ** 0.5
    torch.testing.assert_close(
        parameter,
        -learning_rate * expected_update,
    )


def test_muon_fallback_for_vector_parameters():
    torch.manual_seed(42)
    actual = torch.nn.Parameter(torch.tensor([1.0, -2.0, 0.5]))
    initial = actual.detach().clone()
    learning_rate = 3e-4
    optimizer = Muon(
        [actual],
        lr=learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0,
    )
    gradient = torch.tensor([2.0, -3.0, 0.0])
    actual.grad = gradient

    optimizer.step()

    expected = initial - learning_rate * gradient.sign()
    torch.testing.assert_close(actual, expected)


def test_get_lr_cosine_schedule():
    max_learning_rate = 1
    min_learning_rate = 1 * 0.1
    warmup_iters = 7
    cosine_cycle_iters = 21

    expected_lrs = [
        0,
        0.14285714285714285,
        0.2857142857142857,
        0.42857142857142855,
        0.5714285714285714,
        0.7142857142857143,
        0.8571428571428571,
        1.0,
        0.9887175604818206,
        0.9554359905560885,
        0.9018241671106134,
        0.8305704108364301,
        0.7452476826029011,
        0.6501344202803414,
        0.55,
        0.44986557971965857,
        0.3547523173970989,
        0.26942958916356996,
        0.19817583288938662,
        0.14456400944391146,
        0.11128243951817937,
        0.1,
        0.1,
        0.1,
        0.1,
    ]
    actual_lrs = [
        run_get_lr_cosine_schedule(
            it=it,
            max_learning_rate=max_learning_rate,
            min_learning_rate=min_learning_rate,
            warmup_iters=warmup_iters,
            cosine_cycle_iters=cosine_cycle_iters,
        )
        for it in range(25)
    ]
    numpy.testing.assert_allclose(
        numpy.array(actual_lrs),
        numpy.array(expected_lrs),
    )
