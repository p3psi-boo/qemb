# Autoresearch: optimize offline embedding throughput

## Objective
Improve end-to-end offline embedding throughput for the current `qemb bench` workload without changing embedding semantics or gaming the benchmark. The active experiment code lives in the isolated worktree:
`/home/bubu/worktrees/qwen3-emb/autoresearch-throughput-2026-03-23`

This indirection exists because the main workspace already has uncommitted development changes that should not be mutated by autoresearch commit/revert cycles.

## Metrics
- **Primary**: `tok_per_s` (higher is better) - median aggregate tokens/sec across representative short, medium, and long prompts.
- **Secondary**: `short_tok_per_s`, `medium_tok_per_s`, `long_tok_per_s`.

## How to Run
`./autoresearch.sh`

## Files in Scope
- `crates/qemb-cli/src/benchmark.rs`
- `crates/qemb-cli/src/runner.rs`
- `crates/qemb-tokenizer/src/*.rs`
- `crates/qemb-cli/src/main.rs`

## Off Limits
- Model assets under `/home/bubu/models/qwen3-embedding-0.6b/`
- Benchmark-only shortcuts keyed to the fixed benchmark prompts
- Semantics changes that trade correctness for throughput

## Constraints
- Do not overfit to the benchmark and do not cheat on the benchmark.
- Keep `cargo test --workspace --quiet` passing in the worktree.
- Preserve realistic request-path behavior.

## What's Been Tried
- Created an isolated worktree and mirrored the current local, uncommitted benchmark code into it.
- Confirmed `qemb bench` is only present in local changes, not in `HEAD`.
- Built a more stable benchmark harness using short/medium/long prompts and median aggregate throughput.
- Cleared stale release artifacts in the worktree after they hid the new `bench` subcommand.
- Kept a real optimization in `OfflineModel::embed_token_ids`: removing the explicit mean-pooling scale pass before L2 normalization, because scaling by `1 / token_count` is mathematically redundant once the vector is normalized.
- Kept another embedding-path optimization by pre-decoding `embed_tokens.weight` from BF16 bytes into resident `u16` words at model load time, so the hot path skips per-element byte assembly.
- Retried `tokenizers::Tokenizer::encode_fast` after the embedding path got faster; this time it helped and became a kept change.
- A later tokenizer-side win came from cloning the fully deserialized Hugging Face tokenizer once in `Tokenizer::from_file`. This preserves semantics but gives the BPE model a fresh internal cache state, and it repeatedly benchmarks around the mid-560k tok/s range.
- Removing the tokenizer post-processor entirely for `add_special_tokens = false` produced another real gain. Sampled `qemb run` output hashes for representative ASCII inputs stayed unchanged.
- Routing ASCII inputs through a tokenizer clone with the NFC normalizer removed, while non-ASCII inputs still use the full path, produced a large gain. Since NFC is identity on ASCII, sampled ASCII outputs stayed unchanged.
- Specializing that ASCII fast path further with an ASCII-equivalent pre-tokenizer regex, gated so it only activates when the loaded tokenizer matches the Qwen split+ByteLevel structure, pushed aggregate throughput to roughly 950k-980k tok/s while keeping sampled ASCII outputs unchanged.
- The next major step was a gated ASCII ids-only fast path: reuse the configured ASCII pre-tokenizer, then tokenize each normalized segment directly with the BPE model and collect ids without building a full `Encoding`. With checks passing and sampled output hashes unchanged, this pushed aggregate throughput to roughly 1.10M tok/s.
- In that ids-only fast path, requesting split metadata with `OffsetType::None` instead of byte offsets is a free simplification and remains keep-worthy because the offsets are ignored there.
- Larger explicit BPE cache sizes looked promising at first, but follow-up tests showed the win persists even when returning to the default 10k capacity as long as the tokenizer is cloned. The simpler clone-based version is the one to keep.
- Tried skipping added-vocabulary scanning for ASCII inputs without `<`; it improved some sub-metrics but not the primary aggregate metric, even after the ASCII regex fast path, so it was discarded.
- Tried removing the reusable token-id buffer optimization after the tokenizer got much faster; it still regressed, so the explicit `encode_into` reuse remains worth keeping.
- Tried removing per-token checked arithmetic in the embedding loop; it regressed and was discarded.
- Tried reciprocal-based normalization, disabling tokenizers parallelism, and direct all-the-way-to-`f32` embedding predecode; all regressed and were discarded.
- Added a default-off phase timing mode to `qemb bench`. Manual profiling on the medium prompt shows tokenization at roughly `0.026 ms` and embedding at roughly `0.003 ms` per request, so tokenization is now about 90% of the timed path.
- Moved benchmark model assets to `/home/bubu/models/qwen3-embedding-0.6b/` because autoresearch cleanup removed untracked worktree assets during crash/discard cycles.
