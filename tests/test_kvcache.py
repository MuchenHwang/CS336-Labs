import pytest
import torch

from cs336.kvcache import (
    append_key_value,
    cache_sequence_length,
    kv_cache_nbytes,
    validate_kv_cache,
)
from cs336.model import TransformerConfig, TransformerLM


def small_model() -> TransformerLM:
    return TransformerLM(
        TransformerConfig(
            vocab_size=32,
            context_length=16,
            d_model=16,
            num_layers=2,
            num_heads=2,
            d_ff=32,
            dropout=0.0,
        )
    )


def test_append_key_value_and_validate_cache() -> None:
    key = torch.randn(2, 3, 4, 5)
    value = torch.randn(2, 3, 4, 5)
    past_key_value = append_key_value(None, key, value)

    added_key, added_value = append_key_value(
        past_key_value,
        torch.randn(2, 3, 2, 5),
        torch.randn(2, 3, 2, 5),
    )
    cache = ((added_key, added_value),)

    assert added_key.shape == (2, 3, 6, 5)
    assert added_value.shape == (2, 3, 6, 5)
    assert cache_sequence_length(cache) == 6
    assert (
        validate_kv_cache(
            cache,
            num_layers=1,
            batch_size=2,
            num_heads=3,
            head_dim=5,
            device=added_key.device,
            dtype=added_key.dtype,
        )
        == 6
    )
    assert kv_cache_nbytes(cache) == added_key.nbytes + added_value.nbytes


def test_cached_logits_match_full_forward() -> None:
    torch.manual_seed(0)
    model = small_model()
    model.eval()
    token_ids = torch.randint(0, 32, (2, 10))

    full_logits = model(token_ids)
    prefix = token_ids[:, :4]
    cached_logits, cache = model.forward_with_cache(prefix)

    assert cache_sequence_length(cache) == 4
    torch.testing.assert_close(
        cached_logits[:, -1],
        full_logits[:, 3],
        atol=1e-5,
        rtol=1e-4,
    )

    for position in range(4, token_ids.shape[1]):
        cached_logits, cache = model.forward_with_cache(
            token_ids[:, position : position + 1],
            cache,
        )
        assert cache_sequence_length(cache) == position + 1
        torch.testing.assert_close(
            cached_logits[:, -1],
            full_logits[:, position],
            atol=1e-5,
            rtol=1e-4,
        )


def test_cache_rejects_context_overflow() -> None:
    model = small_model()
    prompt = torch.randint(0, 32, (1, model.config.context_length))
    _, cache = model.forward_with_cache(prompt)

    with pytest.raises(ValueError, match="exceeds context length"):
        model.forward_with_cache(torch.zeros((1, 1), dtype=torch.long), cache)
