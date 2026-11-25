from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping, Sequence
import regex as re
from .tokenizer import PAT, SPECIAL_TOKENS


class BPEEncoder:
    """Encode text with the vocabulary and merges produced by train_bpe()."""

    def __init__(
        self,
        vocab: Mapping[int, bytes],
        merges: Sequence[tuple[bytes, bytes]],
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = tuple(special_tokens)

        self.token_to_id: dict[bytes, int] = {}
        for token_id, token_bytes in sorted(self.vocab.items()):
            self.token_to_id.setdefault(token_bytes, token_id)

        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        self.special_set = set(self.special_tokens)
        self._special_pattern = self._build_special_pattern(self.special_tokens)
        self._cache: dict[str, tuple[int, ...]] = {}

    @staticmethod
    def _build_special_pattern(
        special_tokens: Sequence[str],
    ) -> re.Pattern[str] | None:
        if not special_tokens:
            return None
        alternatives = "|".join(
            re.escape(token)
            for token in sorted(special_tokens, key=len, reverse=True)
        )
        return re.compile(f"({alternatives})")

    def _pre_tokenize(self, text: str) -> list[str]:
        parts = (
            self._special_pattern.split(text)
            if self._special_pattern is not None
            else [text]
        )
        pieces: list[str] = []

        for part in parts:
            if part in self.special_set:
                pieces.append(part)
            elif part:
                pieces.extend(PAT.findall(part))
        return pieces

    def _apply_merges(self, symbols: list[bytes]) -> list[bytes]:
        while len(symbols) > 1:
            best_pair: tuple[bytes, bytes] | None = None
            best_rank = len(self.merges)

            for index in range(len(symbols) - 1):
                pair = (symbols[index], symbols[index + 1])
                rank = self.merge_ranks.get(pair)
                if rank is not None and rank < best_rank:
                    best_pair = pair
                    best_rank = rank

            if best_pair is None:
                break

            merged_symbols: list[bytes] = []
            index = 0
            while index < len(symbols):
                if (
                    index < len(symbols) - 1
                    and symbols[index] == best_pair[0]
                    and symbols[index + 1] == best_pair[1]
                ):
                    merged_symbols.append(best_pair[0] + best_pair[1])
                    index += 2
                else:
                    merged_symbols.append(symbols[index])
                    index += 1
            symbols = merged_symbols

        return symbols

    def _encode_piece(self, piece: str) -> tuple[int, ...]:
        cached = self._cache.get(piece)
        if cached is not None:
            return cached

        if piece in self.special_set:
            token_bytes = piece.encode("utf-8")
            token_id = self.token_to_id.get(token_bytes)
            if token_id is None:
                raise ValueError(
                    f"Special token {piece!r} is missing from the vocabulary"
                )
            result = (token_id,)
        else:
            symbols = self._apply_merges(
                [bytes([byte]) for byte in piece.encode("utf-8")]
            )
            try:
                result = tuple(
                    self.token_to_id[token_bytes] for token_bytes in symbols
                )
            except KeyError as error:
                token_bytes = error.args[0]
                raise ValueError(
                    f"Token {token_bytes!r} is missing from the vocabulary"
                ) from error

        self._cache[piece] = result
        return result

    def encode(self, text: str) -> list[int]:
        token_ids: list[int] = []
        for piece in self._pre_tokenize(text):
            token_ids.extend(self._encode_piece(piece))
        return token_ids

    def encode_iterable(self, texts: Iterable[str]) -> Iterator[int]:
        for text in texts:
            yield from self.encode(text)