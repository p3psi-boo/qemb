# Qwen3 Embedding Bare-Metal RDNA3 Service Spec

## 1. Goal

Build a Rust service that runs `Qwen/Qwen3-Embedding-0.6B` on AMD RDNA3 through a `refs/t0-gpu` style bare-metal stack:

- direct `/dev/kfd` + `/dev/dri/renderD*` communication
- hand-written or hand-scheduled GPU kernels for critical ops
- no HIP or ROCm user-space runtime in the serving path
- external OpenAI-compatible `POST /v1/embeddings` API

The first release explicitly does **not** target GGUF. It targets a simpler weight pipeline based on original Hugging Face weights converted offline into a custom runtime-friendly format.

## 2. Non-Goals

The first release does not aim to:

- support generic GGUF loading
- support generic quantized inference formats
- support text generation APIs such as `/v1/chat/completions`
- support batching optimized for maximum throughput
- support full 32k context on the current 1 GiB VRAM machine
- outperform mature inference stacks in the initial milestone

## 3. Why No GGUF

Dropping GGUF in v1 reduces complexity in four places:

1. No GGUF parser, metadata handling, or tensor-name compatibility layer.
2. No support for GGML block quantization layouts.
3. No dequantize-plus-matmul fused kernels in the first bring-up.
4. No need to match community-exported model variants.

This changes the problem from "build a generic GGUF runtime" to "build a dedicated Qwen3 embedding runtime".

The trade-off is higher memory pressure, so v1 must support host-resident weights and layer streaming.

## 4. Target Environment

### 4.1 Hardware assumptions

Current observed machine characteristics:

- AMD RDNA3 GPU exposed through KFD
- `gfx_target_version = 110003` (`gfx1103`)
- visible VRAM approximately 1 GiB
- Linux with accessible `/dev/kfd` and render node

### 4.2 Consequences

- `refs/t0-gpu` is currently hard-wired for `gfx1100`, so a `gfx1103` bring-up layer is required.
- Full BF16 weights do not fit entirely in VRAM, so the runtime must support host/GTT resident weights and streaming.
- The first supported workload should be short-to-medium sequence embedding, batch size 1, correctness first.

## 5. Model Scope

Target model: `Qwen/Qwen3-Embedding-0.6B`

Expected architecture parameters:

- hidden size: 1024
- intermediate size: 3072
- layers: 28
- attention heads: 16
- key/value heads: 8
- head dim: 128
- max trained context: 32768
- embedding dimension: 1024
- dtype at source: BF16

v1 runtime scope:

- single model only
- inference only
- embedding extraction only
- last-valid-token pooling
- optional L2 normalization before response

## 6. External API

The service must expose an OpenAI-compatible embeddings endpoint.

### 6.1 Endpoint

`POST /v1/embeddings`

### 6.2 Request format

Supported request body fields:

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": "hello world",
  "encoding_format": "float",
  "dimensions": 1024,
  "user": "optional-user-id"
}
```

Also support array input:

```json
{
  "model": "Qwen3-Embedding-0.6B",
  "input": ["first text", "second text"],
  "encoding_format": "float",
  "dimensions": 1024
}
```

### 6.3 Request semantics

- `model` is required and must map to the loaded embedding model.
- `input` accepts either a string or an array of strings.
- v1 may internally execute requests one item at a time even if an array is provided.
- `encoding_format` only supports `float` in v1.
- `dimensions` is optional. If absent, default to 1024.
- If `dimensions` is provided, it must be within `[32, 1024]` and be a multiple accepted by the model policy. If unsupported in v1, return `400` with a clear error.
- `user` is accepted for compatibility and logging only.

### 6.4 Response format

OpenAI-compatible response:

```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "index": 0,
      "embedding": [0.01, -0.02, 0.03]
    }
  ],
  "model": "Qwen3-Embedding-0.6B",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

### 6.5 Errors

Return JSON errors in the style:

```json
{
  "error": {
    "message": "unsupported model",
    "type": "invalid_request_error",
    "param": "model",
    "code": "model_not_found"
  }
}
```

Expected status codes:

- `200` success
- `400` invalid request
- `404` unknown route
- `429` admission control or queue full
- `500` internal runtime error
- `503` model not ready or GPU unavailable

### 6.6 Health and metadata endpoints

Recommended auxiliary endpoints:

- `GET /healthz`
- `GET /readyz`
- `GET /v1/models`

`/v1/models` may expose a minimal OpenAI-compatible listing with the single loaded embedding model.

## 7. Functional Pipeline

The end-to-end request flow is:

1. Receive HTTP request.
2. Validate request schema and model name.
3. Tokenize input on CPU.
4. Build padded token buffer and attention mask.
5. Execute Qwen3 embedding forward pass.
6. Pool last valid token hidden state.
7. Apply optional projection/truncation policy for `dimensions`.
8. L2 normalize output vector.
9. Return OpenAI-compatible JSON response.

## 8. Internal Architecture

The system is split into five layers.

### 8.1 HTTP layer

Responsibilities:

- parse and validate JSON
- expose `/v1/embeddings`
- collect timing and usage stats
- serialize OpenAI-compatible responses

### 8.2 Tokenization layer

Responsibilities:

- load tokenizer assets from Hugging Face export
- convert UTF-8 input into token ids
- enforce request token limits
- report `prompt_tokens`

This layer remains on CPU in v1.

### 8.3 Weight loader and offline converter

v1 uses original Hugging Face weights as input to an offline conversion tool.

Input artifacts:

- tokenizer files
- `config.json`
- `safetensors` weights

Output artifacts:

- one runtime config file
- one tokenizer package
- one custom packed weight bundle

The converter is allowed to:

- transpose linear weights into kernel-friendly layout
- split tensors by layer
- align sections to 64 B or 256 B boundaries
- optionally emit both host layout and GPU upload layout

### 8.4 Runtime execution layer

Responsibilities:

- own KFD device, queues, events, and buffers
- stage per-layer weights from host-visible memory to GPU-visible memory if needed
- launch bare-metal kernels through AQL
- maintain a small buffer pool to avoid repeated allocation

### 8.5 Kernel library

Responsibilities:

- provide hand-written or hand-scheduled kernels for required model ops
- support correctness-first fallback implementations
- expose per-op validation hooks against CPU reference

## 9. Weight Format

v1 defines a project-local packed format instead of GGUF.

### 9.1 Design goals

- trivial runtime parsing
- stable tensor naming for one model family
- layout chosen for our kernels, not for ecosystem compatibility
- support host streaming without costly repacking at request time

### 9.2 Proposed files

- `model.json`: model metadata and tensor index
- `tokenizer/`: tokenizer assets
- `weights.bin`: packed tensor payloads

### 9.3 `model.json` minimum fields

```json
{
  "format": "qembpack-v1",
  "model_type": "qwen3-embedding",
  "model_name": "Qwen3-Embedding-0.6B",
  "hidden_size": 1024,
  "intermediate_size": 3072,
  "num_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "embedding_dim": 1024,
  "dtype": "bf16",
  "tensors": []
}
```

### 9.4 Tensor record minimum fields

```json
{
  "name": "model.layers.0.self_attn.q_proj.weight",
  "dtype": "bf16",
  "shape": [1024, 1024],
  "layout": "row_major_transposed",
  "offset": 123456,
  "nbytes": 2097152
}
```

### 9.5 Layout policy

v1 should prepack common matrices into the exact layout expected by the target kernels:

- embedding table: row-major by token id
- linear weights: pre-transposed or tile-packed as needed
- RMSNorm gamma: contiguous row-major
- optional fused gate/up layout for SwiGLU if it reduces launches

## 10. Execution Strategy

Because VRAM is limited, v1 uses a streamed layer execution model.

### 10.1 Memory residency policy

- activations for the current layer stay in GPU-visible memory
- a small number of reusable staging buffers are allocated once
- weights may remain in host-visible mapped memory or be copied layer-by-layer
- do not assume all 28 layers are resident in VRAM simultaneously

### 10.2 Scheduling model

For batch size 1, sequence length `S`, hidden size `H`:

- tokenize on CPU
- upload token ids
- embedding gather on GPU or CPU+upload depending on bring-up stage
- for each layer:
  - RMSNorm
  - Q projection
  - K projection
  - V projection
  - q_norm and k_norm if required by the model
  - RoPE
  - causal attention
  - O projection
  - residual add
  - post-attention RMSNorm
  - gate projection
  - up projection
  - SwiGLU
  - down projection
  - residual add
- final RMSNorm if required by architecture
- gather last valid token hidden state
- normalize and truncate if needed

### 10.3 v1 sequence policy

The service should support a configurable max sequence length.

Initial recommended target:

- correctness milestone: 512 or 1024 tokens
- stretch target on current machine: 2048 tokens
- do not promise 32k in v1

## 11. Required Kernels

The first release needs the following kernels or equivalent fallback paths.

### 11.1 Already close to existing `refs/t0-gpu` capability

- embedding gather
- RMSNorm forward
- GEMM / linear projection
- SiLU / SwiGLU pieces
- residual add
- copy / cast helpers

### 11.2 Missing or must be adapted

- `gfx1103` target support in metadata and scheduling
- Qwen3 attention path instead of OCPA-specific attention
- RoPE kernel or fused RoPE inside Q/K path
- q_norm and k_norm kernels
- last-token pooling kernel or CPU fallback
- L2 normalization kernel or CPU fallback

### 11.3 CPU fallback policy

To reduce bring-up risk, these ops may temporarily fall back to CPU if needed:

- final pooling
- final L2 normalization
- very small shape edge cases

The core transformer heavy ops should still target GPU.

## 12. Kernel Bring-Up Order

Implement in this order:

1. vector add / memcpy / memset smoke tests
2. BF16 or F32 GEMM sanity tests
3. embedding gather
4. RMSNorm forward
5. SiLU and fused elementwise helpers
6. Q, K, V linear projections
7. RoPE
8. causal attention
9. MLP path
10. full single-layer forward validation
11. full 28-layer forward validation
12. HTTP serving integration

## 13. `gfx1103` Adaptation Requirements

The project must not remain hard-coded to `gfx1100`.

Required changes:

- add a target enum entry for `gfx1103`
- emit correct target metadata in generated code objects
- audit any ISA encoding assumptions verified only on `gfx1100`
- add a schedule profile for the current GPU
- keep per-target kernel configs separate from model logic

If a kernel is valid on both `gfx1100` and `gfx1103`, share source but keep target-specific metadata explicit.

## 14. Correctness Strategy

Correctness is more important than speed in v1.

### 14.1 Golden references

Use Python or Rust CPU references to validate:

- individual kernels
- per-layer outputs
- final embedding vector
- cosine similarity against reference outputs

### 14.2 Tolerances

Suggested tolerances for BF16 inference:

- per-op absolute / relative tolerance set per op
- final embedding cosine similarity against reference should be very high
- ranking behavior on a small retrieval sanity set should match reference within a narrow margin

### 14.3 Validation levels

- Level 0: kernel output matches CPU reference on synthetic inputs
- Level 1: single-layer forward matches reference tensors
- Level 2: full model forward matches reference embedding
- Level 3: HTTP response matches expected schema and numerical envelope

## 15. Performance Strategy

v1 performance priorities:

1. avoid repeated allocations
2. keep activations on GPU when possible
3. prepack weights offline
4. fuse elementwise ops where low-risk
5. minimize host-device synchronization points

v1 is allowed to be slower than optimized frameworks if it stays correct and stable.

## 16. Observability

At minimum, record:

- request count
- request latency
- tokenization latency
- per-layer execution latency
- GPU dispatch count per request
- host-to-device bytes per request
- sequence length distribution
- queue depth / in-flight requests

Recommended log fields:

- request id
- model name
- token count
- dimensions
- total latency ms
- runtime path used (`gpu_full`, `gpu_mixed`, `cpu_fallback`)

## 17. Concurrency Model

v1 should start with a simple serialized execution model:

- one request executes on the GPU at a time
- a bounded queue absorbs short bursts
- return `429` or `503` when overloaded

This keeps KFD queue management and memory residency simple during bring-up.

Later milestones may add micro-batching.

## 18. Security and Robustness

The service must:

- reject oversized request bodies
- cap input count per request
- cap token count per item
- avoid unbounded temporary allocations
- zero or safely recycle transient buffers when required
- fail closed if model files are missing or malformed

## 19. Configuration

Minimum runtime configuration:

- model bundle path
- bind address / port
- max sequence length
- max concurrent requests
- queue depth
- request timeout
- whether final normalization is enabled
- logging level

Example:

```toml
model_path = "./models/qwen3-embedding-0.6b"
bind = "0.0.0.0:8000"
max_seq_len = 1024
max_concurrent = 1
queue_depth = 16
request_timeout_ms = 30000
normalize = true
```

## 20. Milestones

### M0: Platform bring-up

- KFD runtime works on current machine
- `gfx1103` target added
- smoke kernels pass

### M1: Single-op validation

- GEMM, gather, RMSNorm, SiLU validated against CPU

### M2: Single-layer Qwen3

- one full transformer layer matches reference

### M3: Full-model offline runner

- CLI tool loads packed weights and emits embedding vectors

### M4: Service API

- `/v1/embeddings` implemented
- health endpoints implemented
- stable JSON schema and error handling

### M5: Stability and performance pass

- layer streaming tuned
- allocations pooled
- latency and memory telemetry added

## 21. Acceptance Criteria

The project is considered v1-complete when:

- a packed `Qwen3-Embedding-0.6B` bundle can be loaded
- `POST /v1/embeddings` accepts OpenAI-compatible requests
- single-string and string-array inputs both work
- outputs are numerically close to a trusted reference implementation
- service runs without HIP or ROCm user-space runtime in the inference path
- current machine can serve short-sequence embedding requests reliably

## 22. Open Questions

These items need final decisions before implementation starts:

1. Whether tokenizer integration is embedded in Rust or delegated to a sidecar conversion step.
2. Whether v1 supports dynamic `dimensions` truncation or only fixed 1024 output.
3. Whether attention stays fully on GPU in v1 or temporarily falls back to CPU for part of the path.
4. Whether weights are read directly from host-mapped memory or copied layer-by-layer into VRAM staging buffers.
5. Whether the HTTP server itself should be dependency-light or use a standard async Rust stack.

## 23. Recommendation

The recommended path is:

- skip GGUF entirely in v1
- convert Hugging Face `safetensors` offline into a custom packed format
- add `gfx1103` target support first
- bring up heavy transformer ops one by one
- expose only `/v1/embeddings` externally until correctness is solid

This keeps the first deliverable focused: a dedicated bare-metal Qwen3 embedding service, not a general-purpose LLM runtime.
