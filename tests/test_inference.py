from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from cs336.inference import (
    GenerationConfig,
    generate_token_ids,
    load_model_from_checkpoint,
    sample_next_token,
)
from cs336.model import TransformerConfig, TransformerLM
from cs336.optimizer import Muon
from cs336.train import TrainingConfig, save_checkpoint


class FixedTokenModel(nn.Module):
    def __init__(self, token_id: int, context_length: int = 8) -> None:
        super().__init__()
        self.config = TransformerConfig(
            vocab_size=16,
            context_length=context_length,
            d_model=8,
            num_layers=1,
            num_heads=2,
            d_ff=16,
        )
        self.anchor = nn.Parameter(torch.zeros(()))
        self.token_id = token_id
        self.input_lengths: list[int] = []

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        self.input_lengths.append(token_ids.shape[1])
        logits = torch.zeros(
            *token_ids.shape,
            self.config.vocab_size,
            device=token_ids.device,
        )
        logits[..., self.token_id] = 1.0
        return logits

    def forward_with_cache(
        self,
        token_ids: torch.Tensor,
        past_key_values=None,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], ...]]:
        logits = self.forward(token_ids)
        past_length = (
            0
            if past_key_values is None
            else past_key_values[0][0].shape[-2]
        )
        total_length = past_length + token_ids.shape[1]
        cache = tuple(
            (
                torch.zeros(
                    token_ids.shape[0],
                    self.config.num_heads,
                    total_length,
                    self.config.d_model // self.config.num_heads,
                    device=token_ids.device,
                ),
                torch.zeros(
                    token_ids.shape[0],
                    self.config.num_heads,
                    total_length,
                    self.config.d_model // self.config.num_heads,
                    device=token_ids.device,
                ),
            )
            for _ in range(self.config.num_layers)
        )
        return logits, cache


def test_greedy_and_top_k_sampling() -> None:
    logits = torch.tensor([1.0, 4.0, 2.0])
    greedy_config = GenerationConfig(temperature=0.0)
    assert int(sample_next_token(logits, greedy_config)) == 1

    top_k_config = GenerationConfig(
        temperature=1.0,
        top_k=1,
        top_p=1.0,
    )
    generator = torch.Generator().manual_seed(0)
    assert int(sample_next_token(logits, top_k_config, generator)) == 1


def test_generation_stops_at_end_of_text() -> None:
    model = FixedTokenModel(token_id=3)
    generated = generate_token_ids(
        model,
        [1, 2],
        GenerationConfig(max_new_tokens=10, temperature=0.0),
        eos_token_id=3,
        device="cpu",
    )
    assert generated == []


def test_generation_uses_cache_and_restores_mode() -> None:
    model = FixedTokenModel(token_id=4, context_length=8)
    model.train()

    generated = generate_token_ids(
        model,
        list(range(5)),
        GenerationConfig(max_new_tokens=3, temperature=0.0),
        device="cpu",
    )

    assert generated == [4, 4, 4]
    assert model.input_lengths == [5, 1, 1]
    assert model.training


def test_load_model_from_training_checkpoint(tmp_path: Path) -> None:
    model = TransformerLM(
        TransformerConfig(
            vocab_size=16,
            context_length=8,
            d_model=8,
            num_layers=1,
            num_heads=2,
            d_ff=16,
        )
    )
    optimizer = Muon(model.parameters(), lr=1e-3)
    training_config = TrainingConfig(
        context_length=8,
        batch_size=1,
        max_steps=1,
        warmup_steps=0,
        eval_batches=1,
        log_interval=1,
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        step=1,
        training_config=training_config,
    )

    restored, completed_steps = load_model_from_checkpoint(
        checkpoint_path,
        device="cpu",
    )

    assert completed_steps == 1
    assert restored.config == model.config
    for actual, expected in zip(
        restored.parameters(),
        model.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
