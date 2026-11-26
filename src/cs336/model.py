from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass(frozen=True)
class TransformerConfig:
    """Configuration for a decoder-only Transformer language model."""

    vocab_size: int
    context_length: int
    d_model: int = 512
    num_layers: int = 4
    num_heads: int = 8
    d_ff: int | None = None
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.d_model <= 0 or self.num_layers <= 0 or self.num_heads <= 0:
            raise ValueError("d_model, num_layers, and num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        head_dim = self.d_model // self.num_heads
        if head_dim % 2 != 0:
            raise ValueError("The attention head dimension must be even for RoPE")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.rope_theta <= 0:
            raise ValueError("rope_theta must be positive")
        if self.norm_eps <= 0:
            raise ValueError("norm_eps must be positive")

        if self.d_ff is None:
            # Match the usual 8/3 SwiGLU expansion, rounded to a friendly size.
            d_ff = math.ceil((8 * self.d_model / 3) / 64) * 64
            object.__setattr__(self, "d_ff", d_ff)
        elif self.d_ff <= 0:
            raise ValueError("d_ff must be positive")


class RMSNorm(nn.Module):
    """Root mean square layer normalization without a mean subtraction."""

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        normalized = x * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        return (normalized * self.weight.float()).to(input_dtype)


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings applied to adjacent feature pairs."""

    def __init__(
        self,
        head_dim: int,
        context_length: int,
        theta: float = 10_000.0,
    ) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")

        inverse_frequencies = 1.0 / (
            theta
            ** (
                torch.arange(0, head_dim, 2, dtype=torch.float32)
                / head_dim
            )
        )
        positions = torch.arange(context_length, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        if sequence_length > self.cos.shape[0]:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds the configured "
                f"RoPE context length {self.cos.shape[0]}"
            )

        cos = self.cos[:sequence_length].to(dtype=x.dtype)[None, None, :, :]
        sin = self.sin[:sequence_length].to(dtype=x.dtype)[None, None, :, :]

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated = torch.stack(
            (even * cos - odd * sin, even * sin + odd * cos),
            dim=-1,
        )
        return rotated.flatten(start_dim=-2)


class CausalMultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention with causal masking and RoPE."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_model % config.num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = config.num_heads
        self.head_dim = config.d_model // config.num_heads
        self.qkv_projection = nn.Linear(
            config.d_model,
            3 * config.d_model,
            bias=False,
        )
        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model,
            bias=False,
        )
        self.rope = RotaryPositionalEmbedding(
            self.head_dim,
            config.context_length,
            config.rope_theta,
        )
        self.attention_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, sequence_length, d_model = x.shape

        qkv = self.qkv_projection(x)
        query, key, value = qkv.chunk(3, dim=-1)
        query = query.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        key = key.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)
        value = value.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

        query = self.rope(query)
        key = self.rope(key)

        attention_scores = query @ key.transpose(-2, -1)
        attention_scores = attention_scores / math.sqrt(self.head_dim)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ),
            diagonal=1,
        )
        attention_scores = attention_scores.masked_fill(
            causal_mask,
            torch.finfo(attention_scores.dtype).min,
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)

        attended = attention_weights @ value
        attended = attended.transpose(1, 2).reshape(
            batch_size,
            sequence_length,
            d_model,
        )
        return self.output_projection(attended)


class SwiGLU(nn.Module):
    """The SwiGLU feed-forward network used by modern decoder models."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        if config.d_ff is None:
            raise ValueError("d_ff must be configured")
        self.gate_projection = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )
        self.up_projection = nn.Linear(
            config.d_model,
            config.d_ff,
            bias=False,
        )
        self.down_projection = nn.Linear(
            config.d_ff,
            config.d_model,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_projection(
            F.silu(self.gate_projection(x)) * self.up_projection(x)
        )


class TransformerBlock(nn.Module):
    """A pre-norm decoder block with attention and a SwiGLU MLP."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_eps)
        self.attention = CausalMultiHeadSelfAttention(config)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_eps)
        self.mlp = SwiGLU(config)
        self.residual_dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.residual_dropout(self.attention(self.attention_norm(x)))
        x = x + self.residual_dropout(self.mlp(self.mlp_norm(x)))
        return x


class TransformerLM(nn.Module):
    """A decoder-only Transformer that predicts the next token."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.d_model,
        )
        self.embedding_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_layers)
        )
        self.final_norm = RMSNorm(config.d_model, config.norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        if config.tie_embeddings:
            self.lm_head.weight = self.token_embeddings.weight

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: Tensor,
        targets: Tensor | None = None,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape (batch, sequence)")

        sequence_length = token_ids.shape[1]
        if sequence_length > self.config.context_length:
            raise ValueError(
                f"Sequence length {sequence_length} exceeds context length "
                f"{self.config.context_length}"
            )

        x = self.embedding_dropout(self.token_embeddings(token_ids))
        for layer in self.layers:
            x = layer(x)
        logits = self.lm_head(self.final_norm(x))

        if targets is None:
            return logits
        if targets.shape != token_ids.shape:
            raise ValueError("targets must have the same shape as token_ids")

        loss = F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size),
            targets.reshape(-1),
            ignore_index=-100,
        )
        return logits, loss


def cross_entropy_loss(
    logits: Tensor,
    targets: Tensor,
    ignore_index: int = -100,
) -> Tensor:
    """Compute next-token cross entropy for logits ending in vocabulary size."""
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        ignore_index=ignore_index,
    )
