from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from .decoder import BPEDecoder
from .encoder import BPEEncoder
from .model import TransformerConfig, TransformerLM
from .tokenizer import SPECIAL_TOKENS, load_bpe


END_OF_TEXT = "<|endoftext|>"


@dataclass(frozen=True)
class GenerationConfig:
    """Decoding settings for autoregressive text generation."""

    max_new_tokens: int = 128
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95
    seed: int = 0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: torch.device | str = "cpu",
) -> tuple[TransformerLM, int]:
    """Load a TransformerLM and completed step from a training checkpoint."""
    checkpoint = torch.load(
        Path(checkpoint_path),
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint must contain a dictionary")

    model_state = checkpoint.get("model")
    model_config = checkpoint.get("model_config", checkpoint.get("config"))
    if not isinstance(model_state, dict):
        raise ValueError("checkpoint is missing a model state dictionary")
    if not isinstance(model_config, dict):
        raise ValueError("checkpoint is missing the model configuration")

    model = TransformerLM(TransformerConfig(**model_config))
    model.load_state_dict(model_state)
    model.to(torch.device(device))
    model.eval()
    return model, int(checkpoint.get("step", 0))


def end_of_text_token_id(vocab: dict[int, bytes]) -> int | None:
    """Return the vocabulary ID for <|endoftext|>, if it exists."""
    token_bytes = END_OF_TEXT.encode("utf-8")
    candidates = [
        token_id
        for token_id, value in vocab.items()
        if value == token_bytes
    ]
    return min(candidates) if candidates else None


def _generator_for(
    device: torch.device,
    seed: int,
) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


def sample_next_token(
    logits: Tensor,
    config: GenerationConfig,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Sample one vocabulary index using greedy, top-k, and top-p decoding."""
    if logits.ndim != 1:
        raise ValueError("logits must have shape (vocab_size,)")
    if not torch.isfinite(logits).any():
        raise FloatingPointError("all token logits are non-finite")

    finite_logits = torch.where(
        torch.isfinite(logits),
        logits,
        torch.full_like(logits, -torch.inf),
    )
    if config.temperature == 0.0:
        return torch.argmax(finite_logits)

    scaled_logits = finite_logits.float() / config.temperature
    if config.top_k > 0:
        k = min(config.top_k, scaled_logits.numel())
        if k < scaled_logits.numel():
            top_values, top_indices = torch.topk(scaled_logits, k)
            filtered_logits = torch.full_like(scaled_logits, -torch.inf)
            filtered_logits.scatter_(0, top_indices, top_values)
            scaled_logits = filtered_logits

    if config.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(
            scaled_logits,
            descending=True,
        )
        sorted_probabilities = torch.softmax(sorted_logits, dim=-1)
        cumulative_probabilities = torch.cumsum(
            sorted_probabilities,
            dim=-1,
        )
        remove = cumulative_probabilities > config.top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -torch.inf)
        probabilities = torch.softmax(sorted_logits, dim=-1)
        sampled_position = torch.multinomial(
            probabilities,
            num_samples=1,
            generator=generator,
        )
        return sorted_indices[sampled_position[0]]

    probabilities = torch.softmax(scaled_logits, dim=-1)
    return torch.multinomial(
        probabilities,
        num_samples=1,
        generator=generator,
    )[0]


@torch.no_grad()
def generate_token_ids(
    model: nn.Module,
    prompt_ids: Sequence[int] | Tensor,
    config: GenerationConfig,
    *,
    eos_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> list[int]:
    """Generate token IDs after a non-empty prompt."""
    if device is None:
        try:
            parameter = next(model.parameters())
        except StopIteration as error:
            raise ValueError("model has no parameters") from error
        device = parameter.device
    else:
        device = torch.device(device)

    prompt = torch.as_tensor(prompt_ids, dtype=torch.long).flatten().to(device)
    if prompt.numel() == 0:
        raise ValueError("prompt must contain at least one token")
    if not hasattr(model, "forward_with_cache"):
        raise TypeError("model must implement forward_with_cache()")

    context_length = int(model.config.context_length)
    if context_length < 2:
        raise ValueError("context_length must be at least 2 for generation")
    prompt = prompt[-context_length:]
    if prompt.numel() == context_length:
        prompt = prompt[-(context_length - 1) :]
    max_new_tokens = min(
        config.max_new_tokens,
        context_length - prompt.numel(),
    )
    if max_new_tokens <= 0:
        raise ValueError("prompt leaves no room in the model context")

    generated_ids: list[int] = []
    generator = _generator_for(torch.device(device), config.seed)
    was_training = model.training
    model.eval()

    try:
        logits, past_key_values = model.forward_with_cache(
            prompt.unsqueeze(0)
        )
        for _ in range(max_new_tokens):
            next_token = sample_next_token(
                logits[0, -1],
                config,
                generator,
            )
            token_id = int(next_token.item())
            if eos_token_id is not None and token_id == eos_token_id:
                break
            generated_ids.append(token_id)
            if len(generated_ids) < max_new_tokens:
                logits, past_key_values = model.forward_with_cache(
                    next_token.reshape(1, 1),
                    past_key_values,
                )
    finally:
        model.train(was_training)

    return generated_ids


def generate_text(
    model: nn.Module,
    encoder: BPEEncoder,
    decoder: BPEDecoder,
    prompt: str,
    config: GenerationConfig,
    *,
    eos_token_id: int | None = None,
    device: torch.device | str | None = None,
) -> str:
    """Generate and decode a continuation for a text prompt."""
    prompt_ids = encoder.encode(prompt)
    if not prompt_ids:
        raise ValueError("prompt encodes to an empty token sequence")
    generated_ids = generate_token_ids(
        model,
        prompt_ids,
        config,
        eos_token_id=eos_token_id,
        device=device,
    )
    return decoder.decode(generated_ids)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate text with a trained TransformerLM checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
    )
    parser.add_argument("--vocab", type=Path, default=Path("vocab.json"))
    parser.add_argument("--merges", type=Path, default=Path("merges.txt"))
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--completion-only", action="store_true")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model, completed_steps = load_model_from_checkpoint(
        args.checkpoint,
        device=device,
    )

    vocab, merges = load_bpe(args.vocab, args.merges)
    if len(vocab) != model.config.vocab_size:
        raise ValueError(
            f"tokenizer vocabulary size {len(vocab)} does not match "
            f"checkpoint vocabulary size {model.config.vocab_size}"
        )

    encoder = BPEEncoder(vocab, merges, SPECIAL_TOKENS)
    decoder = BPEDecoder(vocab)
    completion = generate_text(
        model,
        encoder,
        decoder,
        args.prompt,
        GenerationConfig(
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            seed=args.seed,
        ),
        eos_token_id=end_of_text_token_id(vocab),
        device=device,
    )

    print(
        f"checkpoint_step={completed_steps} device={device}",
        file=sys.stderr,
    )
    if args.completion_only:
        print(completion)
    else:
        print(args.prompt + completion)


if __name__ == "__main__":
    main()
