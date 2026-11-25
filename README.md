# TransformerLab

从零实现 BPE 分词器、编码器与解码器。当前模块包括：

- `src/cs336/tokenizer.py`：BPE 训练、预分词、merge 学习与结果保存。
- `src/cs336/encoder.py`：将文本转换为 token ID。
- `src/cs336/decoder.py`：将 token ID 还原为文本。

以下命令均在项目根目录运行。

## 环境

项目使用 Python 3.14 和 `uv`：

```bash
uv sync
```

如果当前环境已经创建 `.venv`，也可以直接使用：

```bash
source .venv/bin/activate
python -c "import cs336"
```

## 训练 BPE

当前训练入口使用：

```python
PATH = "data/TinyStoriesV2-GPT4-train.txt"
VOCAB_SIZE = 10000
NUM_PROCESSES = 4
SPECIAL_TOKENS = ["<|endoftext|>"]
```

运行完整训练：

```bash
uv run cs336
```

也可以直接运行模块：

```bash
uv run python -m cs336.tokenizer
```

训练会覆盖项目根目录下的 `vocab.json` 和 `merges.txt`。当前 TinyStories
数据较大，完整训练需要较长时间。

训练流水线如下：

```text
文件切块
  -> pre_tokenize()
  -> to_bpe_input()
  -> train_bpe()
  -> save_bpe()
```

检查指定的训练常量和数据路径：

```bash
rg -n "SPECIAL_TOKENS|VOCAB_SIZE|NUM_PROCESSES|PATH" src/cs336/tokenizer.py
test -f data/TinyStoriesV2-GPT4-train.txt && echo "training data found"
```

注意：`vocab_size` 包含 256 个基础 byte token 和 special tokens。启用
`<|endoftext|>` 时，最多训练的 merge 数量是：

```text
vocab_size - 256 - special_token_count
```

## 使用 tokenizer、encoder 与 decoder

`train_bpe()` 返回内存中的 `vocab` 和有序 `merges`。`BPEEncoder` 和
`BPEDecoder` 可以直接使用这两个对象，也可以通过 `load_bpe()` 从磁盘恢复。

`save_bpe()` 使用 GPT-2 的 byte-to-unicode 映射，将 vocab 保存为 JSON
对象，并将 merges 保存为每行两个 token。该格式可以无歧义地保存任意字节、
前导空格、换行和无效 UTF-8 byte：

```python
from cs336.tokenizer import load_bpe, save_bpe

save_bpe(vocab, merges, "vocab.json", "merges.txt")
loaded_vocab, loaded_merges = load_bpe("vocab.json", "merges.txt")

assert loaded_vocab == vocab
assert loaded_merges == merges
```

下面的小语料会实际执行训练、编码和解码：

```bash
uv run python - <<'PY'
from collections import Counter

from cs336.decoder import BPEDecoder
from cs336.encoder import BPEEncoder
from cs336.tokenizer import (
    SPECIAL_TOKENS,
    pre_tokenize,
    to_bpe_input,
    train_bpe,
)

corpus = (
    "Hello, how are you?<|endoftext|>"
    "Hello again!<|endoftext|>"
    "你好，世界。🙃"
)

pieces = pre_tokenize(corpus)
splits, _ = to_bpe_input(Counter(pieces))
vocab, merges = train_bpe(
    splits,
    vocab_size=400,
    specials=SPECIAL_TOKENS,
)

encoder = BPEEncoder(vocab, merges, SPECIAL_TOKENS)
decoder = BPEDecoder(vocab)

for text in [
    "Hello, how are you?",
    "你好，世界。🙃",
    "a<|endoftext|>b<|endoftext|><|endoftext|>",
]:
    token_ids = encoder.encode(text)
    decoded = decoder.decode(token_ids)
    assert decoded == text, (text, token_ids, decoded)
    print(repr(text), token_ids)

print(f"vocab={len(vocab)}, merges={len(merges)}")
PY
```

主要接口：

```python
pieces = pre_tokenize(text)
splits, specials = to_bpe_input(Counter(pieces))
vocab, merges = train_bpe(splits, vocab_size, specials)

encoder = BPEEncoder(vocab, merges, special_tokens=SPECIAL_TOKENS)
token_ids = encoder.encode(text)
streamed_ids = list(encoder.encode_iterable(["line 1\n", "line 2\n"]))

decoder = BPEDecoder(vocab)
text = decoder.decode(token_ids)
streamed_text = list(decoder.decode_iterable([[token_ids]]))
```

## 调试方法

先检查预分词结果，确认 special token 没有进入普通正则切片：

```bash
uv run python - <<'PY'
from cs336.tokenizer import pre_tokenize

sample = "Hello  world<|endoftext|>你好🙃"
for index, piece in enumerate(pre_tokenize(sample)):
    print(index, repr(piece))
PY
```

每次修改 tokenizer 后，可以运行以下自包含检查。它会训练一个小语料、检查基础
不变量、打印 token 对应的 bytes，并测试 round-trip：

```bash
uv run python - <<'PY'
from collections import Counter

from cs336.decoder import BPEDecoder
from cs336.encoder import BPEEncoder
from cs336.tokenizer import (
    SPECIAL_TOKENS,
    pre_tokenize,
    to_bpe_input,
    train_bpe,
)

corpus = "Hello  world<|endoftext|>你好🙃<|endoftext|>"
vocab_size = 400
pieces = pre_tokenize(corpus)
splits, _ = to_bpe_input(Counter(pieces))
vocab, merges = train_bpe(splits, vocab_size, SPECIAL_TOKENS)
encoder = BPEEncoder(vocab, merges, SPECIAL_TOKENS)
decoder = BPEDecoder(vocab)

print("pieces:", [repr(piece) for piece in pieces])

assert all(vocab[index] == bytes([index]) for index in range(256))
assert all(left in vocab.values() and right in vocab.values()
           for left, right in merges)
assert len(vocab) <= vocab_size

for text in ["", "hello", "hello\nworld", "你好🙃", "<|endoftext|>"]:
    ids = encoder.encode(text)
    assert decoder.decode(ids) == text
    print(repr(text), [(token_id, decoder.token_bytes(token_id)) for token_id in ids])

print("tokenizer invariants passed")
PY
```

检查代码是否可导入：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run python -c \
  "from cs336.tokenizer import pre_tokenize; from cs336.encoder import BPEEncoder; from cs336.decoder import BPEDecoder"
```

检查补丁格式：

```bash
git diff --check
```

项目目前没有正式测试目录，因此 `pytest` 不会提供额外测试覆盖。需要回归测试时，
应把上面的 round-trip、special-token、空白和 Unicode 用例加入 `tests/`。

## 已知问题

1. 旧版 `vocab.jsonl` 和 `merges.txt` 包含不可逆的 `backslashreplace` 编码。
   需要重新训练并生成新的 `vocab.json` 和 `merges.txt`，不要用 `load_bpe()`
   读取旧文件。

2. `encode_iterable()` 会把每个元素当作独立文本。如果某个 special token 被拆到
   两个元素中，例如 `"<|endof"` 和 `"text|>"`，它不会被识别为 special token。
   读取文件时建议逐行传入，或确保每个元素内部是完整文本。

3. `decode_iterable()` 会分别解码每个 token ID chunk。如果一个 UTF-8 字符跨越两个
   chunk，例如 emoji 的 bytes 被拆开，会得到替换字符 `�`。需要流式解码时应保留
   byte buffer，或只在完整 token 序列上调用 `decode()`。

4. `train_bpe()` 的注释写的是平局“按字节序取较小者”，当前实现实际使用
   `max(..., key=(frequency, pair))`，即平局选择字节序较大的 pair。修改代码前应先
   以课程规定的 tie-break 规则确认预期行为。
