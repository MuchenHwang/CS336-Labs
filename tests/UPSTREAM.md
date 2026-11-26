# Stanford CS336 upstream tests

The test files, fixtures, and snapshots in this directory are adapted from:

- Repository: <https://github.com/stanford-cs336/assignment1-basics>
- Commit: `a158843b20107949f1a8d7df1b05cd33b9166712`

The local `test_model.py` belongs to this project. The upstream tokenizer, BPE,
and optimizer tests call this project's implementation through `adapters.py`.

`test_tokenizer.py` constructs tiktoken's GPT-2 reference encoding from the
checked-in vocabulary fixture so that the tests do not download data at runtime.

See `UPSTREAM_LICENSE` for the upstream license.
