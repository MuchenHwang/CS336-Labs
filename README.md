# TransformerLab

从零实现 BPE 分词器、编码器与解码器。当前模块包括：

- `src/cs336/tokenizer.py`：BPE 训练、预分词、merge 学习与结果保存。
- `src/cs336/encoder.py`：将文本转换为 token ID。
- `src/cs336/decoder.py`：将 token ID 还原为文本。
- `src/cs336/model.py`：带 RoPE、RMSNorm、因果注意力和 SwiGLU 的
  decoder-only Transformer 语言模型。

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

## Transformer 语言模型

`TransformerLM` 接收形状为 `(batch, sequence)` 的 token ID，并输出每个位置
对下一个 token 的 logits。传入 targets 时同时返回交叉熵损失：

```python
import torch

from cs336.model import TransformerConfig, TransformerLM

config = TransformerConfig(
    vocab_size=10_000,
    context_length=256,
    d_model=512,
    num_layers=4,
    num_heads=8,
    dropout=0.1,
)
model = TransformerLM(config)

token_ids = torch.randint(0, config.vocab_size, (2, 129))
inputs = token_ids[:, :-1]
targets = token_ids[:, 1:]
logits, loss = model(inputs, targets)

assert logits.shape == (2, 128, config.vocab_size)
assert loss.ndim == 0
```

模型默认使用预归一化残差块、RoPE、无 bias 的线性层、SwiGLU 和 tied token
embeddings。`context_length` 是位置编码允许的最大序列长度；输入更长时模型会
直接报错。

## 训练 Transformer 模型

`model.py` 只提供模型定义和前向损失，没有内置数据集、训练循环或命令行入口。
因此不能通过 `python -m cs336.model` 启动训练，需要在脚本中创建优化器，并让
`TransformerLM` 同时接收输入 token 和对应的下一 token。

训练前先确认 BPE 词表和训练样本存在：

```bash
test -f vocab.json
test -f merges.txt
test -f tests/fixtures/tinystories_sample.txt
```

下面的命令使用仓库自带的小样本完成一次可复现的训练冒烟测试，并保存
`model.pt`。它会实际执行前向传播、反向传播、梯度裁剪、AdamW 更新和
checkpoint 保存：

```bash
uv run python - <<'PY'
from pathlib import Path

import torch

from cs336.encoder import BPEEncoder
from cs336.model import TransformerConfig, TransformerLM
from cs336.tokenizer import SPECIAL_TOKENS, load_bpe

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

vocab, merges = load_bpe("vocab.json", "merges.txt")
tokenizer = BPEEncoder(vocab, merges, SPECIAL_TOKENS)
text = Path("tests/fixtures/tinystories_sample.txt").read_text(
    encoding="utf-8"
)
token_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)

context_length = 64
batch_size = 4
config = TransformerConfig(
    vocab_size=len(vocab),
    context_length=context_length,
    d_model=128,
    num_layers=2,
    num_heads=4,
    dropout=0.1,
)
model = TransformerLM(config).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

max_start = token_ids.numel() - context_length - 1
if max_start < 0:
    raise ValueError("训练文本太短，无法构造一个完整上下文")

model.train()
num_steps = 20
for step in range(1, num_steps + 1):
    starts = torch.randint(0, max_start + 1, (batch_size,))
    chunks = torch.stack(
        [
            token_ids[start : start + context_length + 1]
            for start in starts
        ]
    ).to(device)

    inputs = chunks[:, :-1]
    targets = chunks[:, 1:]
    _, loss = model(inputs, targets)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    if step == 1 or step % 5 == 0:
        print(f"step={step}, loss={loss.detach().item():.4f}")

torch.save(
    {
        "model": model.state_dict(),
        "config": {
            "vocab_size": config.vocab_size,
            "context_length": config.context_length,
            "d_model": config.d_model,
            "num_layers": config.num_layers,
            "num_heads": config.num_heads,
            "d_ff": config.d_ff,
            "dropout": config.dropout,
        },
        "step": num_steps,
    },
    "model.pt",
)
print(f"checkpoint saved to model.pt on {device}")
PY
```

训练是否能够正常推进，主要看每一步的 loss 是有限值，并确认 checkpoint 已生成：

```bash
test -f model.pt && echo "model checkpoint found"
```

完整训练 TinyStories 时，不要用 `Path.read_text()` 一次加载 2.1 GB 文本，也不要
一次保存全部 token ID。应流式读取并按块编码，将 token ID 写入二进制文件，再
通过内存映射（`numpy.memmap`）构造训练批次。每个批次满足：

- `inputs` 和 `targets` 的形状都是 `(batch_size, context_length)`。
- `targets` 是 `inputs` 在原始 token 流中的下一位，即 `chunk[:-1]` 和
  `chunk[1:]`。
- `context_length` 不能超过 `TransformerConfig.context_length`。
- 所有 token ID 必须处于 `[0, vocab_size)`。

当前训练示例没有学习率调度、验证集评估和断点恢复。正式训练时建议先实现这些
功能，再按实际显存调整 `batch_size`、`context_length`、`d_model` 和层数。

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

运行正式测试：

```bash
PYTHONDONTWRITEBYTECODE=1 uv run pytest -q
```

当前测试覆盖 RMSNorm、RoPE、模型前向与反向传播、因果掩码、上下文长度限制、
embedding tying 和交叉熵，并包含 Stanford CS336 官方的 tokenizer 与 BPE 测试。
官方 BPE 速度测试要求小语料训练在 1.5 秒内完成；当前实现约需 2.1 秒，因此完整
测试套件会在 `test_train_bpe_speed` 处失败。

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
