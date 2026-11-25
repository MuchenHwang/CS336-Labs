from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping


class BPEDecoder:
    """Decode token IDs with a vocabulary produced by train_bpe()."""

    def __init__(self, vocab: Mapping[int, bytes]) -> None:
        self.vocab = dict(vocab)

    def token_bytes(self, token_id: int) -> bytes:
        try:
            return self.vocab[token_id]
        except KeyError as error:
            raise ValueError(f"Unknown token ID: {token_id}") from error

    def decode(self, token_ids: Iterable[int]) -> str:
        token_bytes = b"".join(
            self.token_bytes(token_id) for token_id in token_ids
        )
        return token_bytes.decode("utf-8", errors="replace")

    def decode_iterable(
        self,
        token_id_chunks: Iterable[Iterable[int]],
    ) -> Iterator[str]:
        for token_ids in token_id_chunks:
            yield self.decode(token_ids)
