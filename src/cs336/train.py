from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from .encoder import BPEEncoder
from .model import TransformerConfig, TransformerLM
from .optimizer import Muon, get_lr_cosine_schedule
from .tokenizer import load_bpe


_TOKEN_CACHE_FORMAT_VERSION = 1
_TOKEN_CACHE_CHUNK_SIZE = 1_000_000


@dataclass(frozen=True)
class TrainingConfig:
    """Training-loop settings shared by the CLI and tests."""

    context_length: int
    batch_size: int = 4
    max_steps: int = 100
    warmup_steps: int = 10
    max_learning_rate: float = 3e-4
    min_learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    eval_interval: int = 20
    eval_batches: int = 10
    checkpoint_interval: int = 50
    log_interval: int = 10
    seed: int = 0

    def __post_init__(self) -> None:
        if self.context_length <= 0:
            raise ValueError("context_length must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if not 0 <= self.warmup_steps < self.max_steps:
            raise ValueError("warmup_steps must be in [0, max_steps)")
        if not 0.0 <= self.min_learning_rate <= self.max_learning_rate:
            raise ValueError(
                "learning rates must satisfy 0 <= min <= max_learning_rate"
            )
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must be non-negative")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.eval_interval < 0:
            raise ValueError("eval_interval must be non-negative")
        if self.eval_batches <= 0:
            raise ValueError("eval_batches must be positive")
        if self.checkpoint_interval < 0:
            raise ValueError("checkpoint_interval must be non-negative")
        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")


def get_batch(
    dataset: Tensor | np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device | str,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample next-token prediction inputs and targets from a 1D token stream."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token stream")
    if torch.is_tensor(dataset):
        if dataset.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        ):
            raise ValueError("dataset must contain integer token IDs")
        dataset_length = dataset.numel()
    else:
        if not np.issubdtype(dataset.dtype, np.integer):
            raise ValueError("dataset must contain integer token IDs")
        dataset_length = int(dataset.size)

    if dataset_length <= context_length:
        raise ValueError("dataset is too short for the requested context length")

    num_starts = dataset_length - context_length
    starts = torch.randint(
        0,
        num_starts,
        (batch_size,),
        generator=generator,
    )
    offsets = torch.arange(context_length + 1)
    indices = starts[:, None] + offsets[None, :]

    if torch.is_tensor(dataset):
        chunks = dataset[indices]
    else:
        chunks = torch.from_numpy(np.array(dataset[indices.numpy()], copy=True))

    chunks = chunks.to(device=device, dtype=torch.long)
    return chunks[:, :-1].contiguous(), chunks[:, 1:].contiguous()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: Tensor | np.ndarray,
    config: TrainingConfig,
    device: torch.device | str,
) -> float:
    """Estimate validation loss on a fixed set of batches."""
    was_training = model.training
    model.eval()
    generator = torch.Generator().manual_seed(config.seed + 1)
    losses: list[float] = []

    try:
        for _ in range(config.eval_batches):
            inputs, targets = get_batch(
                dataset,
                config.batch_size,
                config.context_length,
                device,
                generator,
            )
            _, loss = model(inputs, targets)
            losses.append(loss.item())
    finally:
        model.train(was_training)

    return sum(losses) / len(losses)


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    training_config: TrainingConfig,
) -> None:
    """Save model, optimizer state, progress, and enough metadata to resume."""
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    model_config = getattr(model, "config", None)
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "training_config": asdict(training_config),
            "model_config": (
                asdict(model_config) if model_config is not None else None
            ),
        },
        checkpoint_path,
    )


def load_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model, optimizer, and completed-step count."""
    device = next(model.parameters()).device
    checkpoint = torch.load(
        Path(path),
        map_location=device,
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or "step" not in checkpoint:
        raise ValueError("invalid training checkpoint")

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])


def _learning_rate(
    step: int,
    config: TrainingConfig,
) -> float:
    if config.max_steps == 1:
        return config.max_learning_rate
    return get_lr_cosine_schedule(
        it=step,
        max_learning_rate=config.max_learning_rate,
        min_learning_rate=config.min_learning_rate,
        warmup_iters=config.warmup_steps,
        cosine_cycle_iters=config.max_steps - 1,
    )


def train(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_data: Tensor | np.ndarray,
    config: TrainingConfig,
    *,
    validation_data: Tensor | np.ndarray | None = None,
    checkpoint_path: str | Path | None = None,
    resume_from: str | Path | None = None,
    device: torch.device | str | None = None,
    log: Callable[[str], Any] = print,
) -> int:
    """Run next-token training and return the number of completed steps."""
    if device is None:
        device = next(model.parameters()).device
    else:
        device = torch.device(device)

    start_step = 0
    if resume_from is not None:
        start_step = load_checkpoint(resume_from, model, optimizer)
        log(f"resumed from step {start_step}")

    generator = torch.Generator().manual_seed(config.seed)
    model.train()

    for step in range(start_step, config.max_steps):
        learning_rate = _learning_rate(step, config)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(
            train_data,
            config.batch_size,
            config.context_length,
            device,
            generator,
        )
        _, loss = model(inputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at step {step}")

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=config.max_grad_norm,
        )
        optimizer.step()

        completed_steps = step + 1
        if completed_steps == 1 or completed_steps % config.log_interval == 0:
            log(
                f"step={completed_steps}/{config.max_steps} "
                f"lr={learning_rate:.2e} loss={loss.item():.4f}"
            )

        should_evaluate = (
            validation_data is not None
            and config.eval_interval > 0
            and (
                completed_steps % config.eval_interval == 0
                or completed_steps == config.max_steps
            )
        )
        if should_evaluate:
            validation_loss = evaluate(
                model,
                validation_data,
                config,
                device,
            )
            log(f"validation step={completed_steps} loss={validation_loss:.4f}")

        should_checkpoint = (
            checkpoint_path is not None
            and config.checkpoint_interval > 0
            and completed_steps % config.checkpoint_interval == 0
        )
        if should_checkpoint:
            save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                completed_steps,
                config,
            )

    if checkpoint_path is not None:
        save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            config.max_steps,
            config,
        )

    return config.max_steps


def _token_cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_name(f"{cache_path.name}.json")


def _normalize_token_dtype(dtype: str | np.dtype) -> np.dtype:
    normalized = np.dtype(dtype).newbyteorder("<")
    if normalized not in (np.dtype("<u2"), np.dtype("<i4")):
        raise ValueError("token cache dtype must be uint16 or int32")
    return normalized


def _file_signature(path: Path) -> dict[str, str | int]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _token_cache_signature(
    source_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: tuple[str, ...],
    dtype: np.dtype,
    max_tokens: int | None,
) -> dict[str, Any]:
    return {
        "format_version": _TOKEN_CACHE_FORMAT_VERSION,
        "source": _file_signature(source_path),
        "vocab": _file_signature(vocab_path),
        "merges": _file_signature(merges_path),
        "special_tokens": list(special_tokens),
        "dtype": dtype.str,
        "max_tokens": max_tokens,
    }


def _valid_cache_token_count(
    cache_path: Path,
    signature: dict[str, Any],
    dtype: np.dtype,
) -> int | None:
    metadata_path = _token_cache_metadata_path(cache_path)
    if not cache_path.is_file() or not metadata_path.is_file():
        return None

    try:
        with metadata_path.open("r", encoding="utf-8") as metadata_file:
            metadata = json.load(metadata_file)
        num_tokens = int(metadata["num_tokens"])
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None

    if metadata.get("signature") != signature:
        return None
    expected_size = num_tokens * dtype.itemsize
    if num_tokens <= 0 or cache_path.stat().st_size != expected_size:
        return None
    return num_tokens


def _write_cache_metadata(
    cache_path: Path,
    signature: dict[str, Any],
    num_tokens: int,
) -> None:
    metadata_path = _token_cache_metadata_path(cache_path)
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=metadata_path.parent,
        prefix=f".{metadata_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as metadata_file:
            json.dump(
                {
                    "signature": signature,
                    "num_tokens": num_tokens,
                },
                metadata_file,
                ensure_ascii=False,
                indent=2,
            )
            metadata_file.write("\n")
        os.replace(temporary_path, metadata_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def build_token_cache(
    source_path: str | Path,
    cache_path: str | Path,
    tokenizer: BPEEncoder,
    *,
    vocab_path: str | Path,
    merges_path: str | Path,
    dtype: str | np.dtype = "uint16",
    max_tokens: int | None = None,
) -> None:
    """Encode a text file into an atomic binary token stream and metadata."""
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    token_dtype = _normalize_token_dtype(dtype)
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    signature = _token_cache_signature(
        source_path,
        vocab_path,
        merges_path,
        tuple(tokenizer.special_tokens),
        token_dtype,
        max_tokens,
    )
    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=cache_path.parent,
        prefix=f".{cache_path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    pending: list[int] = []
    num_tokens = 0

    try:
        with (
            os.fdopen(file_descriptor, "wb") as cache_file,
            source_path.open("r", encoding="utf-8") as source_file,
        ):
            def flush_pending(limit: int | None = None) -> None:
                nonlocal num_tokens
                count = len(pending) if limit is None else min(limit, len(pending))
                if count == 0:
                    return
                token_array = np.asarray(pending[:count], dtype=token_dtype)
                cache_file.write(token_array.tobytes(order="C"))
                num_tokens += token_array.size
                del pending[:count]

            for line in source_file:
                encoded = tokenizer.encode(line)
                if max_tokens is not None:
                    remaining = max_tokens - num_tokens - len(pending)
                    if remaining <= 0:
                        break
                    encoded = encoded[:remaining]
                pending.extend(encoded)

                while len(pending) >= _TOKEN_CACHE_CHUNK_SIZE:
                    flush_pending(_TOKEN_CACHE_CHUNK_SIZE)
                if max_tokens is not None and (
                    num_tokens + len(pending) >= max_tokens
                ):
                    break
            flush_pending()

        if num_tokens == 0:
            raise ValueError(f"no tokens found in {source_path}")

        os.replace(temporary_path, cache_path)
        _write_cache_metadata(cache_path, signature, num_tokens)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def load_token_cache(
    cache_path: str | Path,
    *,
    dtype: str | np.dtype = "uint16",
) -> np.memmap:
    """Open a binary token stream without loading the full dataset in memory."""
    token_dtype = _normalize_token_dtype(dtype)
    tokens = np.memmap(
        Path(cache_path),
        dtype=token_dtype,
        mode="r",
    )
    if tokens.ndim != 1:
        raise ValueError("token cache must be a one-dimensional array")
    return tokens


def get_or_build_token_cache(
    source_path: str | Path,
    cache_path: str | Path,
    tokenizer: BPEEncoder,
    *,
    vocab_path: str | Path,
    merges_path: str | Path,
    dtype: str | np.dtype = "uint16",
    max_tokens: int | None = None,
    force: bool = False,
) -> np.memmap:
    """Reuse a valid binary token cache or rebuild it from the source text."""
    source_path = Path(source_path)
    cache_path = Path(cache_path)
    vocab_path = Path(vocab_path)
    merges_path = Path(merges_path)
    token_dtype = _normalize_token_dtype(dtype)
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError("max_tokens must be positive")

    signature = _token_cache_signature(
        source_path,
        vocab_path,
        merges_path,
        tuple(tokenizer.special_tokens),
        token_dtype,
        max_tokens,
    )
    if not force and _valid_cache_token_count(
        cache_path,
        signature,
        token_dtype,
    ) is not None:
        return load_token_cache(cache_path, dtype=token_dtype)

    build_token_cache(
        source_path,
        cache_path,
        tokenizer,
        vocab_path=vocab_path,
        merges_path=merges_path,
        dtype=token_dtype,
        max_tokens=max_tokens,
    )
    return load_token_cache(cache_path, dtype=token_dtype)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a small decoder-only Transformer language model.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("tests/fixtures/tinystories_sample.txt"),
    )
    parser.add_argument("--vocab", type=Path, default=Path("vocab.json"))
    parser.add_argument("--merges", type=Path, default=Path("merges.txt"))
    parser.add_argument(
        "--cache",
        type=Path,
        default=Path("data/tinystories_tokens.bin"),
    )
    parser.add_argument(
        "--token-dtype",
        choices=("uint16", "int32"),
        default="uint16",
    )
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/latest.pt"),
    )
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=20)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 <= args.valid_fraction < 1.0:
        raise ValueError("valid_fraction must be in [0, 1)")

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    vocab, merges = load_bpe(args.vocab, args.merges)
    tokenizer = BPEEncoder(
        vocab,
        merges,
        ["<|endoftext|>"],
    )
    if max(vocab) > np.iinfo(np.dtype(args.token_dtype)).max:
        raise ValueError(
            f"vocabulary IDs do not fit in {args.token_dtype}; "
            "use --token-dtype int32"
        )
    tokens = get_or_build_token_cache(
        args.data,
        args.cache,
        tokenizer,
        vocab_path=args.vocab,
        merges_path=args.merges,
        dtype=args.token_dtype,
        max_tokens=args.max_tokens,
        force=args.rebuild_cache,
    )

    if args.valid_fraction > 0:
        split_index = int(tokens.size * (1.0 - args.valid_fraction))
        train_data = tokens[:split_index]
        validation_data = tokens[split_index:]
    else:
        train_data = tokens
        validation_data = None

    minimum_tokens = args.context_length + 1
    if train_data.size < minimum_tokens:
        raise ValueError(
            f"training split has {train_data.size} tokens; "
            f"at least {minimum_tokens} are required"
        )
    if validation_data is not None and validation_data.size < minimum_tokens:
        raise ValueError(
            f"validation split has {validation_data.size} tokens; "
            f"at least {minimum_tokens} are required"
        )

    model_config = TransformerConfig(
        vocab_size=len(vocab),
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
    )
    model = TransformerLM(model_config).to(device)
    optimizer = Muon(
        model.parameters(),
        lr=args.max_learning_rate,
        weight_decay=args.weight_decay,
    )
    training_config = TrainingConfig(
        context_length=args.context_length,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        max_learning_rate=args.max_learning_rate,
        min_learning_rate=args.min_learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        eval_interval=args.eval_interval,
        eval_batches=args.eval_batches,
        checkpoint_interval=args.checkpoint_interval,
        log_interval=args.log_interval,
        seed=args.seed,
    )

    print(
        f"device={device} tokens={tokens.size} cache={args.cache} "
        f"train_tokens={train_data.size} "
        f"validation_tokens="
        f"{validation_data.size if validation_data is not None else 0}"
    )
    train(
        model,
        optimizer,
        train_data,
        training_config,
        validation_data=validation_data,
        checkpoint_path=args.checkpoint,
        resume_from=args.resume_from,
        device=device,
    )
    print(f"checkpoint saved to {args.checkpoint}")


if __name__ == "__main__":
    main()
