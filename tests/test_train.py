from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from cs336.encoder import BPEEncoder
from cs336.model import TransformerConfig, TransformerLM
from cs336.optimizer import Muon
from cs336.tokenizer import save_bpe
from cs336.train import (
    TrainingConfig,
    evaluate,
    get_batch,
    get_or_build_token_cache,
    load_checkpoint,
    train,
)


def _small_model_and_optimizer() -> tuple[TransformerLM, Muon]:
    model = TransformerLM(
        TransformerConfig(
            vocab_size=32,
            context_length=8,
            d_model=16,
            num_layers=1,
            num_heads=2,
            d_ff=32,
            dropout=0.0,
        )
    )
    optimizer = Muon(model.parameters(), lr=3e-3)
    return model, optimizer


def _small_training_config() -> TrainingConfig:
    return TrainingConfig(
        context_length=8,
        batch_size=2,
        max_steps=2,
        warmup_steps=1,
        max_learning_rate=3e-3,
        min_learning_rate=3e-4,
        eval_interval=1,
        eval_batches=1,
        checkpoint_interval=1,
        log_interval=1,
        seed=0,
    )


def test_get_batch_shifts_targets_by_one_token() -> None:
    dataset = torch.arange(40, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    inputs, targets = get_batch(
        dataset,
        batch_size=3,
        context_length=8,
        device="cpu",
        generator=generator,
    )

    assert inputs.shape == (3, 8)
    assert targets.shape == (3, 8)
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_token_cache_is_reused_and_rebuilt_when_source_changes(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "corpus.txt"
    vocab_path = tmp_path / "vocab.json"
    merges_path = tmp_path / "merges.txt"
    cache_path = tmp_path / "tokens.bin"
    source_path.write_text("abc\n", encoding="utf-8")

    vocab = {token_id: bytes([token_id]) for token_id in range(256)}
    save_bpe(vocab, [], vocab_path, merges_path)
    tokenizer = BPEEncoder(vocab, [], special_tokens=[])

    tokens = get_or_build_token_cache(
        source_path,
        cache_path,
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype="uint16",
    )
    assert tokens.dtype == np.dtype("<u2")
    assert tokens.tolist() == [97, 98, 99, 10]
    assert cache_path.with_name("tokens.bin.json").is_file()
    assert cache_path.stat().st_size == tokens.size * np.dtype("<u2").itemsize

    cache_mtime_ns = cache_path.stat().st_mtime_ns
    reused_tokens = get_or_build_token_cache(
        source_path,
        cache_path,
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype="uint16",
    )
    assert reused_tokens.tolist() == tokens.tolist()
    assert cache_path.stat().st_mtime_ns == cache_mtime_ns

    source_path.write_text("defg\n", encoding="utf-8")
    rebuilt_tokens = get_or_build_token_cache(
        source_path,
        cache_path,
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype="uint16",
    )
    assert rebuilt_tokens.tolist() == [100, 101, 102, 103, 10]

    limited_tokens = get_or_build_token_cache(
        source_path,
        cache_path,
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype="uint16",
        max_tokens=3,
    )
    assert limited_tokens.tolist() == [100, 101, 102]

    int32_tokens = get_or_build_token_cache(
        source_path,
        tmp_path / "tokens-int32.bin",
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype="int32",
    )
    assert int32_tokens.dtype == np.dtype("<i4")
    assert int32_tokens.tolist() == [100, 101, 102, 103, 10]


def test_get_batch_supports_numpy_memmap(tmp_path: Path) -> None:
    cache_path = tmp_path / "tokens.bin"
    dataset = np.arange(64, dtype=np.uint16)
    dataset.tofile(cache_path)
    memmap_dataset = np.memmap(cache_path, dtype="<u2", mode="r")
    generator = torch.Generator().manual_seed(0)

    inputs, targets = get_batch(
        memmap_dataset,
        batch_size=3,
        context_length=8,
        device="cpu",
        generator=generator,
    )

    assert inputs.dtype == torch.long
    assert targets.dtype == torch.long
    assert torch.equal(inputs[:, 1:], targets[:, :-1])


def test_training_saves_and_restores_full_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    dataset = torch.randint(0, 32, (128,))
    config = _small_training_config()
    checkpoint_path = tmp_path / "checkpoint.pt"
    model, optimizer = _small_model_and_optimizer()

    completed_steps = train(
        model,
        optimizer,
        dataset,
        config,
        validation_data=dataset,
        checkpoint_path=checkpoint_path,
        device="cpu",
        log=lambda _: None,
    )

    assert completed_steps == config.max_steps
    assert checkpoint_path.is_file()
    assert torch.isfinite(
        torch.tensor(evaluate(model, dataset, config, "cpu"))
    )

    restored_model, restored_optimizer = _small_model_and_optimizer()
    restored_step = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
    )

    assert restored_step == completed_steps
    for actual, expected in zip(
        restored_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)

    restored_state = restored_optimizer.state_dict()
    expected_state = optimizer.state_dict()
    assert restored_state["param_groups"] == expected_state["param_groups"]
    for parameter_id in expected_state["state"]:
        actual_values = restored_state["state"][parameter_id]
        expected_values = expected_state["state"][parameter_id]
        assert actual_values.keys() == expected_values.keys()
        for key, expected_value in expected_values.items():
            actual_value = actual_values[key]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(actual_value, expected_value)
            else:
                assert actual_value == expected_value
