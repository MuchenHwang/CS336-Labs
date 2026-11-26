import pytest
import torch

from cs336.model import (
    RMSNorm,
    RotaryPositionalEmbedding,
    TransformerConfig,
    TransformerLM,
    cross_entropy_loss,
)


def small_config(**overrides: int | float | bool | None) -> TransformerConfig:
    values: dict[str, int | float | bool | None] = {
        "vocab_size": 64,
        "context_length": 32,
        "d_model": 32,
        "num_layers": 2,
        "num_heads": 4,
        "d_ff": 64,
        "dropout": 0.0,
    }
    values.update(overrides)
    return TransformerConfig(**values)


def test_config_rejects_incompatible_head_dimension() -> None:
    with pytest.raises(ValueError, match="divisible"):
        small_config(d_model=30)

    with pytest.raises(ValueError, match="even"):
        small_config(d_model=12, num_heads=4)


def test_rms_norm_has_expected_shape_and_scale() -> None:
    norm = RMSNorm(8)
    x = torch.randn(2, 5, 8)
    output = norm(x)

    assert output.shape == x.shape
    rms = output.float().pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones_like(rms), atol=5e-5)


def test_rope_preserves_pairwise_norm() -> None:
    rope = RotaryPositionalEmbedding(head_dim=8, context_length=16)
    x = torch.randn(2, 3, 16, 8)
    output = rope(x)

    assert output.shape == x.shape
    assert torch.allclose(
        output.float().pow(2).sum(dim=-1),
        x.float().pow(2).sum(dim=-1),
        atol=1e-5,
    )


def test_transformer_forward_loss_and_backward() -> None:
    model = TransformerLM(small_config())
    token_ids = torch.randint(0, 64, (2, 16))
    targets = torch.randint(0, 64, (2, 16))

    logits, loss = model(token_ids, targets)

    assert logits.shape == (2, 16, 64)
    assert loss.ndim == 0
    assert torch.isfinite(loss)

    loss.backward()
    assert model.token_embeddings.weight.grad is not None
    assert torch.isfinite(model.token_embeddings.weight.grad).all()


def test_transformer_is_causal() -> None:
    torch.manual_seed(0)
    model = TransformerLM(small_config())
    model.eval()

    token_ids = torch.randint(0, 64, (1, 16))
    changed_token_ids = token_ids.clone()
    changed_token_ids[:, -1] = (changed_token_ids[:, -1] + 1) % 64

    logits = model(token_ids)
    changed_logits = model(changed_token_ids)

    assert torch.allclose(logits[:, :-1], changed_logits[:, :-1], atol=1e-6)
    assert not torch.allclose(logits[:, -1], changed_logits[:, -1])


def test_tied_embeddings_and_untied_option() -> None:
    tied_model = TransformerLM(small_config(tie_embeddings=True))
    untied_model = TransformerLM(small_config(tie_embeddings=False))

    assert tied_model.lm_head.weight is tied_model.token_embeddings.weight
    assert untied_model.lm_head.weight is not untied_model.token_embeddings.weight


def test_context_length_is_enforced() -> None:
    model = TransformerLM(small_config(context_length=8))
    with pytest.raises(ValueError, match="exceeds context length"):
        model(torch.zeros((1, 9), dtype=torch.long))


def test_cross_entropy_matches_pytorch() -> None:
    logits = torch.randn(3, 7, 11)
    targets = torch.randint(0, 11, (3, 7))
    targets[0, 0] = -100

    expected = torch.nn.functional.cross_entropy(
        logits.reshape(-1, 11),
        targets.reshape(-1),
        ignore_index=-100,
    )
    assert torch.allclose(cross_entropy_loss(logits, targets), expected)
