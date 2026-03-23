# Qwen3 Embedding Bare-Metal RDNA3 Service Design

Date: 2026-03-23

## 1. Goal

Build a Rust service that runs `Qwen/Qwen3-Embedding-0.6B` on AMD RDNA3 (780M iGPU, gfx1103) through a bare-metal stack:

- Direct `/dev/kfd` + `/dev/dri/renderD*` communication
- Hand-written GPU kernels for critical ops
- No HIP or ROCm user-space runtime in the serving path
- External OpenAI-compatible `POST /v1/embeddings` API

## 2. Architecture Decisions

### 2.1 Crate Layout

```
crates/
├── qemb-common      # Shared types, error handling, config
├── qemb-runtime     # KFD runtime, memory management, scheduling
├── qemb-kernels     # GPU kernel implementations (GEMM, attention, etc.)
├── qemb-convert     # Offline weight converter (HF → runtime format)
├── qemb-tokenizer   # Tokenization wrapper (tokenizers crate)
├── qemb-service     # HTTP server (axum on tokio)
└── qemb-cli         # CLI tool for testing and conversion
```

### 2.2 refs/t0-gpu Handling

**Decision:** Vendor selected pieces into main codebase.

- Copy KFD runtime, code object generation, and scheduling logic into `qemb-runtime` and `qemb-kernels`
- Reference implementation remains in `refs/t0-gpu` for learning, but we maintain our own copy
- Allows modifications for gfx1103 and Qwen3-specific needs

### 2.3 HTTP Server Stack

**Decision:** `axum` on `tokio`.

- Battle-tested async runtime
- Good JSON performance
- Well-documented OpenAI compatibility patterns
- Mature middleware ecosystem

### 2.4 Tokenizer Runtime

**Decision:** Pure Rust via `tokenizers` crate.

- Direct compatibility with Hugging Face tokenizer files
- Battle-tested implementation
- No external process or IPC overhead

### 2.5 Output Dimensions

**Decision:** Always return 1024 dimensions in v1.

- Ignore `dimensions` field in requests
- Simpler implementation
- Correctness first; MatRoyoshka support can be added later

## 3. Runtime Decisions

### 3.1 Max Sequence Length

**Decision:** 512 tokens.

- Lower memory pressure for initial bring-up
- Sufficient for most embedding use cases
- Can increase after v1 is stable

### 3.2 Weight Loading Strategy

**Decision:** Single unified GTT memory pool.

The AMD 780M is an integrated GPU with no dedicated VRAM. It uses system RAM via GTT (Graphics Translation Table). This means:

- Load all weights into GPU-visible memory once at startup
- No layer streaming needed for v1 (unlike discrete GPU scenario)
- Simpler code path appropriate for iGPU architecture

### 3.3 Pooling and Normalization

**Decision:** GPU kernels for both.

- Last-valid-token pooling implemented as GPU kernel
- L2 normalization implemented as GPU kernel
- No CPU-GPU transfer for post-processing
- Consistent with bare-metal approach

### 3.4 Attention Implementation

**Decision:** Fully GPU from start.

- QKV projection, attention computation, and output projection all on GPU
- No CPU-GPU data movement mid-layer
- Most complex kernel, but essential for true bare-metal inference

## 4. Target Hardware

| Property | Value |
|----------|-------|
| GPU | AMD 780M (integrated) |
| GFX version | gfx1103 |
| Memory | System RAM via GTT |
| KFD access | `/dev/kfd`, `/dev/dri/renderD*` |

Note: gfx1103 bring-up required; reference `refs/t0-gpu` targets gfx1100.

## 5. Model Specifications

Target model: `Qwen/Qwen3-Embedding-0.6B`

| Parameter | Value |
|-----------|-------|
| Hidden size | 1024 |
| Intermediate size | 3072 |
| Layers | 28 |
| Attention heads | 16 |
| Key/value heads | 8 |
| Head dimension | 128 |
| Max trained context | 32768 |
| Embedding dimension | 1024 |
| Source dtype | BF16 |

## 6. External API

### Endpoint

`POST /v1/embeddings`

### Request Format

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": "hello world",
  "encoding_format": "float"
}
```

Array input also supported:

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["first text", "second text"],
  "encoding_format": "float"
}
```

### Response Format

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.1, 0.2, ...],
      "index": 0
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 2,
    "total_tokens": 2
  }
}
```

### Health Endpoint

`GET /health` → `{"status": "ok"}`

### Model Info Endpoint

`GET /v1/models` → model metadata

## 7. Milestone Summary

| Milestone | Goal |
|-----------|------|
| M0 | Platform bring-up: gfx1103 support, KFD runtime working |
| M1 | Primitive validation: GEMM, gather, RMSNorm, SiLU kernels pass tests |
| M2 | Single-layer validation: One transformer layer matches reference |
| M3 | Full offline runner: CLI produces embeddings for test strings |
| M4 | Service API: `/v1/embeddings` works end-to-end |
| M5 | Stabilization: Memory stable, telemetry in place |

## 8. Non-Goals (v1)

- GGUF loading
- Generic quantization support
- Text generation APIs
- Batch optimization for throughput
- Full 32k context
- Dynamic output dimensions