# Qwen3 Embedding Bare-Metal RDNA3 Task Breakdown

This file turns `SPEC.md` into an execution plan.

Status legend:

- `[ ]` not started
- `[~]` in progress
- `[x]` done
- `BLOCKED` external dependency or unresolved design decision

## 0. Project Rules

- Target only `Qwen/Qwen3-Embedding-0.6B` in v1.
- Do not implement GGUF in v1.
- Do not implement generic quantization in v1.
- Keep the serving path free of HIP/ROCm user-space runtime dependencies.
- Prefer correctness over speed until full-model parity is reached.
- Keep external API surface limited to `/v1/embeddings` plus health/model metadata endpoints.

## 1. Milestone Map

### M0 Platform Bring-Up

- `gfx1103` target support lands.
- KFD runtime works on current machine.
- bare-metal smoke kernels dispatch reliably.

### M1 Primitive Validation

- GEMM, gather, RMSNorm, SiLU, copy kernels pass CPU-reference tests.

### M2 Single-Layer Validation

- one full Qwen3 transformer layer matches reference numerically.

### M3 Full Offline Runner

- CLI loads packed weights and emits embeddings for test strings.

### M4 Service API

- `/v1/embeddings` works end-to-end with OpenAI-compatible JSON.

### M5 Stabilization

- layer streaming is stable on 1 GiB VRAM.
- telemetry and queueing are in place.

## 2. Immediate Decisions To Lock

### 2.1 Architecture decisions

- [ ] Lock crate layout for `service`, `runtime`, `converter`, and `kernels`.
- [ ] Decide whether to keep `refs/t0-gpu` as reference-only or vendor selected pieces into main code.
- [ ] Decide whether HTTP server uses a minimal std/TCP stack or a common Rust async stack.
- [ ] Decide whether tokenizer runtime is pure Rust or generated offline into a compact format.
- [ ] Decide whether v1 supports dynamic `dimensions` or always returns 1024.

### 2.2 Runtime decisions

- [ ] Choose initial max sequence length target: `512`, `1024`, or `2048`.
- [ ] Choose whether layer weights are read directly from host-mapped memory or copied into staging VRAM buffers.
- [ ] Choose whether final pooling and L2 normalization start on CPU or GPU.
- [ ] Choose whether attention v1 is fully GPU or temporarily mixed CPU/GPU.

## 3. Repository Setup

### 3.1 Workspace skeleton

- [ ] Create crate layout for:
  - `crates/qemb-common`
  - `crates/qemb-runtime`
  - `crates/qemb-kernels`
  - `crates/qemb-convert`
  - `crates/qemb-tokenizer`
  - `crates/qemb-service`
  - `crates/qemb-cli`
- [ ] Add top-level workspace `Cargo.toml`.
- [ ] Add a `README` section describing milestone status.
- [ ] Add a `docs/` folder for implementation notes and validation logs.

### 3.2 Config and model paths

- [ ] Define runtime config file format.
- [ ] Define model bundle directory layout.
- [ ] Add startup validation for missing files, wrong target, and malformed configs.

## 4. `gfx1103` Bring-Up

### 4.1 Target metadata

- [ ] Add `gfx1103` target enum entry.
- [ ] Add `mcpu_str()` support for `gfx1103`.
- [ ] Add code object metadata emission for `amdgcn-amd-amdhsa--gfx1103`.
- [ ] Audit ELF `e_flags` and AMDGPU note metadata assumptions.

### 4.2 Scheduling and runtime detection

- [ ] Add GPU target detection that distinguishes `gfx1100` and `gfx1103`.
- [ ] Define an initial `gfx1103` schedule profile.
- [ ] Separate model logic from target-specific schedule constants.
- [ ] Add tests for target detection and target metadata generation.

### 4.3 Validation

- [ ] Dispatch a no-op or copy kernel on current hardware.
- [ ] Validate queue creation, doorbell, completion signal, and teardown.
- [ ] Record bring-up notes in `docs/gfx1103-bringup.md`.

## 5. Bare-Metal Smoke Tests

### 5.1 Runtime smoke tests

- [ ] Implement GPU memset smoke test.
- [ ] Implement GPU memcpy smoke test.
- [ ] Implement vector add smoke test.
- [ ] Add correctness checks against CPU buffers.

### 5.2 Stability checks

- [ ] Repeat smoke tests in a loop to catch queue/resource leaks.
- [ ] Add a timeout and failure classification for hung dispatches.
- [ ] Add logging for GPU target, queue size, buffer sizes, and dispatch latency.

## 6. Offline Model Converter

### 6.1 Input loading

- [ ] Define converter CLI interface.
- [ ] Load `config.json` and validate expected Qwen3-Embedding parameters.
- [ ] Load tokenizer assets.
- [ ] Load source tensors from Hugging Face export.

### 6.2 Packed format

- [ ] Implement `qembpack-v1` metadata schema.
- [ ] Implement tensor table serialization.
- [ ] Implement aligned `weights.bin` writer.
- [ ] Implement packed tensor layout descriptors.

### 6.3 Weight packing

- [ ] Pack embedding table.
- [ ] Pack per-layer RMSNorm weights.
- [ ] Pack `q_proj`, `k_proj`, `v_proj`, `o_proj`.
- [ ] Pack MLP `gate_proj`, `up_proj`, `down_proj`.
- [ ] Pack final norm weights if present.
- [ ] Pretranspose or tile-pack matrices for target GEMM kernels.

### 6.4 Converter verification

- [ ] Add `inspect` command to list packed tensors and offsets.
- [ ] Add tensor byte-size consistency checks.
- [ ] Add checksum or hash generation for packed bundle validation.
- [ ] Add a small golden conversion test for a toy model fragment.

## 7. Tokenization Layer

### 7.1 Loading and encoding

- [ ] Load tokenizer assets from the model bundle.
- [ ] Implement UTF-8 string to token ids.
- [ ] Implement request-time token count reporting.
- [ ] Enforce max sequence length.

### 7.2 Validation

- [ ] Compare tokenization results against Hugging Face reference outputs.
- [ ] Add tests for multilingual input.
- [ ] Add tests for empty strings and long strings.

## 8. Core Tensor and Buffer Abstractions

### 8.1 Host-side abstractions

- [ ] Define tensor descriptors: shape, dtype, layout, offset.
- [ ] Define typed views over packed weights.
- [ ] Define activation tensor structs for runtime buffers.

### 8.2 Device-side abstractions

- [ ] Implement persistent GPU buffer pool.
- [ ] Implement host-visible staging buffers.
- [ ] Implement reusable activation buffers for alternating layers.
- [ ] Implement buffer lifetime tracking and debug assertions.

## 9. Kernel Library: Primitive Ops

### 9.1 Already-close kernels to adapt

- [ ] embedding gather
- [ ] GEMM / linear projection
- [ ] RMSNorm forward
- [ ] SiLU
- [ ] residual add
- [ ] copy / cast helpers

### 9.2 Primitive validation harness

- [ ] Build CPU reference implementations for each primitive.
- [ ] Build synthetic input generators.
- [ ] Add per-op error metrics: max abs error, mean abs error, cosine if relevant.
- [ ] Save validation logs under `docs/validation/`.

## 10. Qwen3-Specific Missing Ops

### 10.1 Attention preparation

- [ ] Implement `q_norm` kernel or equivalent path.
- [ ] Implement `k_norm` kernel or equivalent path.
- [ ] Implement RoPE application for Q and K.
- [ ] Validate RoPE numerics against reference.

### 10.2 Attention core

- [ ] Define tensor layout for GQA with `16` Q heads and `8` KV heads.
- [ ] Implement causal attention score path.
- [ ] Implement softmax path suitable for Qwen3 attention.
- [ ] Implement attention-value accumulation path.
- [ ] Validate one-layer attention output against reference.

### 10.3 Pooling and output

- [ ] Implement last-valid-token pooling.
- [ ] Implement output truncation policy for requested `dimensions`.
- [ ] Implement final L2 normalization.
- [ ] Decide whether these begin on CPU or GPU.

## 11. Layer Execution Graph

### 11.1 Single-layer runner

- [ ] Build a single-layer forward driver with explicit op sequence.
- [ ] Support residual connections.
- [ ] Support alternating activation buffers.
- [ ] Expose per-op timing for one layer.

### 11.2 Validation

- [ ] Dump intermediate tensors for one layer.
- [ ] Compare against Python or Rust CPU reference.
- [ ] Add a layer-level cosine similarity check.
- [ ] Document any tolerated BF16 drift.

## 12. Full-Model Offline Runner

### 12.1 Execution

- [ ] Build a CLI command to run embedding inference on a single input string.
- [ ] Add support for multiple input strings executed sequentially.
- [ ] Implement layer streaming so full weights need not fit in VRAM at once.
- [ ] Add last-token pooling and normalization to produce final embedding.

### 12.2 Validation

- [ ] Compare final embeddings against a trusted reference implementation.
- [ ] Compare cosine similarities on a small sentence-pair set.
- [ ] Add a retrieval sanity test with a handful of query/document examples.

## 13. Memory Residency and Streaming

### 13.1 Streaming design

- [ ] Define which tensors stay resident across layers.
- [ ] Define per-layer upload order.
- [ ] Define staging buffer sizes.
- [ ] Define synchronization points between uploads and compute.

### 13.2 1 GiB VRAM safety

- [ ] Measure peak activation footprint by sequence length.
- [ ] Measure per-layer weight footprint after packing.
- [ ] Add admission control if requested sequence length would exceed memory budget.
- [ ] Add clear error messages for unsupported sequence lengths.

## 14. Service API

### 14.1 HTTP server

- [ ] Implement `POST /v1/embeddings`.
- [ ] Implement `GET /healthz`.
- [ ] Implement `GET /readyz`.
- [ ] Implement `GET /v1/models`.

### 14.2 Request parsing

- [ ] Accept `input` as string.
- [ ] Accept `input` as array of strings.
- [ ] Accept `model`, `encoding_format`, `dimensions`, and `user` fields.
- [ ] Reject unsupported encodings with `400`.
- [ ] Reject wrong model names with OpenAI-style error JSON.

### 14.3 Response formatting

- [ ] Emit OpenAI-compatible embedding response schema.
- [ ] Emit `usage.prompt_tokens` and `usage.total_tokens`.
- [ ] Preserve input order in the response `data` array.
- [ ] Emit structured JSON errors for all expected failure classes.

## 15. Service Runtime Integration

### 15.1 Execution path

- [ ] Load model bundle at startup.
- [ ] Initialize tokenizer at startup.
- [ ] Initialize KFD runtime at startup.
- [ ] Reject readiness until model and GPU are ready.
- [ ] Route each embedding request into the offline runner execution path.

### 15.2 Queueing

- [ ] Start with single-GPU-request serialization.
- [ ] Add bounded queue.
- [ ] Return `429` or `503` on overload.
- [ ] Add request timeout handling.

## 16. Observability

### 16.1 Metrics

- [ ] request count
- [ ] request latency
- [ ] tokenization latency
- [ ] per-layer latency
- [ ] bytes uploaded per request
- [ ] dispatch count per request
- [ ] queue depth

### 16.2 Logging

- [ ] Add request id.
- [ ] Add model name and token count.
- [ ] Add runtime path label: `gpu_full`, `gpu_mixed`, `cpu_fallback`.
- [ ] Add failure classification logs.

## 17. Testing Matrix

### 17.1 Unit tests

- [ ] packed format parser/writer
- [ ] tokenizer edge cases
- [ ] config parsing
- [ ] API schema validation
- [ ] error response formatting

### 17.2 Integration tests

- [ ] converter output loads successfully
- [ ] single primitive op runs on GPU
- [ ] one transformer layer matches reference
- [ ] full-model CLI emits stable embeddings
- [ ] `/v1/embeddings` returns expected JSON schema

### 17.3 Regression tests

- [ ] fixed prompt embedding snapshot test
- [ ] cosine similarity threshold test
- [ ] request overload handling test
- [ ] malformed input test

## 18. Documentation Tasks

- [ ] Add `docs/model-bundle-format.md`.
- [ ] Add `docs/gfx1103-bringup.md`.
- [ ] Add `docs/kernel-validation.md`.
- [ ] Add `docs/service-api.md`.
- [ ] Add `docs/runbook.md` for local serving and troubleshooting.

## 19. Recommended Implementation Order

### Phase 1: unblock hardware

- [ ] `gfx1103` target support
- [ ] KFD smoke tests
- [ ] stable buffer allocation and queue lifecycle

### Phase 2: unblock model loading

- [ ] converter CLI
- [ ] packed format
- [ ] tokenizer loading
- [ ] offline bundle inspection

### Phase 3: unblock math primitives

- [ ] GEMM
- [ ] gather
- [ ] RMSNorm
- [ ] SiLU and residual ops

### Phase 4: unblock transformer correctness

- [ ] q/k norm
- [ ] RoPE
- [ ] causal attention
- [ ] MLP path
- [ ] single-layer validation
- [ ] full-model validation

### Phase 5: expose service

- [ ] startup config
- [ ] `/v1/embeddings`
- [ ] `/healthz`, `/readyz`, `/v1/models`
- [ ] queueing and observability

## 20. Definition of Done

A task group is done only when:

- code is implemented
- relevant tests or validation scripts exist
- numerical behavior is compared against a reference where applicable
- failure cases are handled cleanly
- notes are written for any known deviations or temporary fallbacks

## 21. First Sprint Suggestion

If work starts immediately, the first sprint should aim to finish:

- [ ] workspace skeleton
- [ ] `gfx1103` target addition
- [ ] KFD smoke kernel on current machine
- [ ] converter format definition
- [ ] minimal converter stub that reads config and emits `model.json`
- [ ] service stub exposing `/healthz`, `/readyz`, and a placeholder `/v1/embeddings`

## 22. Open Blockers

- `BLOCKED` exact tokenizer runtime strategy not chosen yet
- `BLOCKED` exact source weight ingestion format and library choice not chosen yet
- `BLOCKED` attention implementation path may need a temporary CPU fallback decision
- `BLOCKED` dynamic `dimensions` behavior not finalized

## 23. Nice-To-Haves After v1

- [ ] multi-request micro-batching
- [ ] faster fused elementwise kernels
- [ ] more aggressive attention fusion
- [ ] additional RDNA3 targets beyond current machine
- [ ] optional quantized packed format v2
- [ ] support for more embedding models in the same runtime
