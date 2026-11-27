from __future__ import annotations

import torch
from torch import Tensor


KVCacheLayer = tuple[Tensor, Tensor]
KVCache = tuple[KVCacheLayer, ...]


def append_key_value(
    past_key_value: KVCacheLayer | None,
    key: Tensor,
    value: Tensor,
) -> KVCacheLayer:
    """Append newly projected keys and values to one layer's cache."""
    if key.ndim != 4 or value.ndim != 4:
        raise ValueError("key and value must have shape (batch, heads, seq, dim)")
    if key.shape[:-2] != value.shape[:-2]:
        raise ValueError("key and value must have matching batch/head dimensions")
    if key.shape[-1] != value.shape[-1]:
        raise ValueError("key and value must have matching head dimensions")
    if key.dtype != value.dtype:
        raise ValueError("key and value must have matching dtypes")
    if key.device != value.device:
        raise ValueError("key and value must be on the same device")

    if past_key_value is None:
        return key, value

    past_key, past_value = past_key_value
    if past_key.shape[-2] != past_value.shape[-2]:
        raise ValueError("cached key and value sequence lengths differ")
    if past_key.shape[:-2] != key.shape[:-2]:
        raise ValueError("cached and new keys must have matching dimensions")
    if past_value.shape[:-2] != value.shape[:-2]:
        raise ValueError("cached and new values must have matching dimensions")
    if past_key.shape[-1] != key.shape[-1]:
        raise ValueError("cached and new keys must have matching head dimensions")
    if past_value.shape[-1] != value.shape[-1]:
        raise ValueError("cached and new values must have matching head dimensions")
    if past_key.dtype != key.dtype or past_value.dtype != value.dtype:
        raise ValueError("cached and new tensors must have matching dtypes")
    if past_key.device != key.device or past_value.device != value.device:
        raise ValueError("cached and new tensors must be on the same device")

    return (
        torch.cat((past_key, key), dim=-2),
        torch.cat((past_value, value), dim=-2),
    )


def cache_sequence_length(cache: KVCache) -> int:
    """Return the common cached sequence length after validating the cache."""
    if not cache:
        return 0

    lengths = {layer[0].shape[-2] for layer in cache}
    if len(lengths) != 1:
        raise ValueError("all KV cache layers must have the same sequence length")
    length = lengths.pop()
    if length < 0:
        raise ValueError("KV cache sequence length must be non-negative")
    return length


def validate_kv_cache(
    cache: KVCache,
    *,
    num_layers: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> int:
    """Validate a cache against a model and return its sequence length."""
    if len(cache) != num_layers:
        raise ValueError(
            f"KV cache has {len(cache)} layers, expected {num_layers}"
        )

    for layer_index, (key, value) in enumerate(cache):
        if key.ndim != 4 or value.ndim != 4:
            raise ValueError(
                f"KV cache layer {layer_index} must be a pair of 4D tensors"
            )
        expected_prefix = (batch_size, num_heads)
        if key.shape[:2] != expected_prefix or value.shape[:2] != expected_prefix:
            raise ValueError(
                f"KV cache layer {layer_index} has incompatible batch/head shape"
            )
        if key.shape[-1] != head_dim or value.shape[-1] != head_dim:
            raise ValueError(
                f"KV cache layer {layer_index} has incompatible head dimension"
            )
        if key.shape[-2] != value.shape[-2]:
            raise ValueError(
                f"KV cache layer {layer_index} key/value lengths differ"
            )
        if device is not None and (key.device != device or value.device != device):
            raise ValueError(
                f"KV cache layer {layer_index} is on the wrong device"
            )
        if dtype is not None and (key.dtype != dtype or value.dtype != dtype):
            raise ValueError(
                f"KV cache layer {layer_index} has the wrong dtype"
            )

    return cache_sequence_length(cache)


def kv_cache_nbytes(cache: KVCache) -> int:
    """Return the number of bytes occupied by cached key/value tensors."""
    return sum(
        key.numel() * key.element_size() + value.numel() * value.element_size()
        for key, value in cache
    )
