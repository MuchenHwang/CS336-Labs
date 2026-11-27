from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


_NEWTON_SCHULZ_COEFFICIENTS = (3.4445, -4.7750, 2.0315)


def zeropower_via_newtonschulz5(
    gradient: Tensor,
    steps: int = 5,
) -> Tensor:
    """Approximate the orthogonal polar factor with Newton-Schulz iteration."""
    if gradient.ndim != 2:
        raise ValueError("Muon only supports two-dimensional parameters")
    if steps < 1:
        raise ValueError("steps must be at least 1")

    original_dtype = gradient.dtype
    work_dtype = (
        torch.float32
        if original_dtype in (torch.float16, torch.bfloat16)
        else original_dtype
    )
    update = gradient.to(dtype=work_dtype)

    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.mT

    update = update / (update.norm() + 1e-7)
    a, b, c = _NEWTON_SCHULZ_COEFFICIENTS
    for _ in range(steps):
        gram = update @ update.mT
        polynomial = b * gram + c * (gram @ gram)
        update = a * update + polynomial @ update

    if transposed:
        update = update.mT
    return update.to(dtype=original_dtype)


class Muon(Optimizer):
    """Muon for matrices with an adaptive fallback for other parameters.

    Two-dimensional parameters use momentum followed by Newton-Schulz
    orthogonalization. Embeddings and gains are not distinguished from hidden
    weights because the constructor intentionally keeps its original API.
    One-dimensional and scalar parameters use an adaptive second-moment update.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if len(betas) != 2:
            raise ValueError("betas must contain exactly two values")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Muon does not support sparse gradients")

                if p.ndim == 2:
                    momentum = self.state[p].get("momentum_buffer")
                    if momentum is None:
                        momentum = torch.zeros_like(
                            p,
                            memory_format=torch.preserve_format,
                        )
                        self.state[p]["momentum_buffer"] = momentum

                    momentum.lerp_(grad, 1.0 - beta1)
                    update = grad.lerp(momentum, beta1)
                    update = zeropower_via_newtonschulz5(update)
                    update *= max(1.0, p.size(0) / p.size(1)) ** 0.5

                    p.mul_(1.0 - lr * weight_decay)
                    p.add_(update, alpha=-lr)
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p,
                        memory_format=torch.preserve_format,
                    )

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Decoupled weight decay is applied directly to parameters.
                p.mul_(1.0 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1.0 - beta2,
                )

                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                step_size = lr / bias_correction1
                bc2_sqrt = bias_correction2**0.5

                denom = (exp_avg_sq.sqrt() / bc2_sqrt).add_(eps)
                p.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Get the learning rate for linear warmup followed by cosine decay."""
    if it < 0:
        raise ValueError("iteration must be non-negative")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be >= warmup_iters")
    if not 0.0 <= min_learning_rate <= max_learning_rate:
        raise ValueError(
            "learning rates must satisfy 0 <= min_learning_rate "
            "<= max_learning_rate"
        )

    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if warmup_iters == cosine_cycle_iters:
        return min_learning_rate

    if it <= cosine_cycle_iters:
        decay_iters = cosine_cycle_iters - warmup_iters
        progress = (it - warmup_iters) / decay_iters
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (
            max_learning_rate - min_learning_rate
        )

    return min_learning_rate
