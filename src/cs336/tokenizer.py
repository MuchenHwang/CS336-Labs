import os
import regex as re
import argparse
from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool
from pathlib import Path
import json

SPECIAL_TOKENS = ["<|endoftext|>"]
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
VOCAB_SIZE = 10000
NUM_PROCESSES = 4
PATH = "data/TinyStoriesV2-GPT4-train.txt"

def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def get_chunk_ranges(boundaries: list[int]) -> list[tuple[int, int]]:
    return list(zip(boundaries[:-1], boundaries[1:]))

def pre_tokenize(text: str) -> list[str]:
    # 先把特殊 token 切出来，避免被正则吞进普通片段
    special_pattern = "(" + "|".join(re.escape(t) for t in SPECIAL_TOKENS) + ")"
    parts = re.split(special_pattern, text)
    pieces = []
    for part in parts:
        if part in SPECIAL_TOKENS:
            pieces.append(part)          # 特殊 token 保留原样
        elif part:
            pieces.extend(PAT.findall(part))
    return pieces

def process_chunk(path: str, start: int, end: int) -> Counter:
    with open(path, "rb") as f:
        f.seek(start)
        text = f.read(end - start).decode("utf-8", errors="ignore")
    return Counter(pre_tokenize(text))

def build_pretoken_counts(path: str, boundaries: list[int]) -> Counter:
    ranges = list(zip(boundaries[:-1], boundaries[1:]))
    if not ranges:
        return Counter()

    tasks = [(path, s, e) for s, e in ranges]

    with Pool(processes=len(tasks)) as pool:
        local_counters = pool.starmap(process_chunk, tasks)
    total = Counter()

    for c in local_counters:
        total.update(c)
    return total

def to_bpe_input(
    pretoken_counts: Counter,
) -> tuple[dict[tuple[bytes, ...], int], list[str]]:
    splits: dict[tuple[bytes, ...], int] = {}
    specials: list[str] = []

    for piece, freq in pretoken_counts.items():
        if piece in SPECIAL_TOKENS:
            # 特殊 token 不进入 BPE，单独收集
            specials.append(piece)
            continue
        # 字符串 -> UTF-8 字节 -> 单字节元组
        symbols = tuple(bytes([b]) for b in piece.encode("utf-8"))
        splits[symbols] = splits.get(symbols, 0) + freq

    return splits, specials

def count_pairs(splits: dict[tuple[bytes, ...], int]) -> Counter:
    pair_counts = Counter()
    for symbols, freq in splits.items():
        for i in range(len(symbols) - 1):
            pair_counts[(symbols[i], symbols[i + 1])] += freq
    return pair_counts

def merge_pair(
    splits: dict[tuple[bytes, ...], int],
    pair: tuple[bytes, bytes],
) -> dict[tuple[bytes, ...], int]:
    a, b = pair
    merged = a + b          # bytes 拼接得到新符号
    new_splits: dict[tuple[bytes, ...], int] = {}

    for symbols, freq in splits.items():
        out = []
        i = 0
        while i < len(symbols):
            if i < len(symbols) - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_splits[tuple(out)] = new_splits.get(tuple(out), 0) + freq

    return new_splits

def train_bpe(
    splits: dict[tuple[bytes, ...], int],
    vocab_size: int,
    specials: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    # 初始词表：256 个字节 + 特殊 token
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    for s in specials:
        vocab[len(vocab)] = s.encode("utf-8")

    merges: list[tuple[bytes, bytes]] = []
    num_merges = vocab_size - len(vocab)

    for _ in range(num_merges):
        pair_counts = count_pairs(splits)
        if not pair_counts:
            break
        # 最高频 pair；平局按字节序取较小者，保证可复现
        best = max(pair_counts, key=lambda p: (pair_counts[p], p))
        merges.append(best)
        vocab[len(vocab)] = best[0] + best[1]
        splits = merge_pair(splits, best)

    return vocab, merges

def save_bpe(vocab, merges, vocab_path, merges_path):
    # vocab：token_id -> bytes，存成可读格式
    with open(vocab_path, "w", encoding="utf-8") as f:
        for tid, token_bytes in vocab.items():
            f.write(json.dumps([tid, token_bytes.decode("utf-8", errors="backslashreplace")]) + "\n")

    # merges：有序的 (a, b) 规则列表
    with open(merges_path, "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', errors='backslashreplace')} "
                    f"{b.decode('utf-8', errors='backslashreplace')}\n")

## Usage
def main() -> None:
    with open(PATH, "rb") as f:
        boundaries = find_chunk_boundaries(f, NUM_PROCESSES, b"<|endoftext|>")
        pretoken_counts = build_pretoken_counts(PATH, boundaries)
        splits, specials = to_bpe_input(pretoken_counts)
        vocab, merges = train_bpe(splits, VOCAB_SIZE, SPECIAL_TOKENS)
        save_bpe(vocab, merges, "vocab.jsonl", "merges.txt")
        
if __name__ == "__main__":
    main()

   



