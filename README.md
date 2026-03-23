# Qwen3 Embedding Bare-Metal RDNA3 Service

A Rust service that runs `Qwen/Qwen3-Embedding-0.6B` on AMD RDNA3 GPUs through a bare-metal stack.

## Milestone Status

| Milestone | Status | Description |
|-----------|--------|-------------|
| M0 Platform Bring-Up | ✅ Complete | gfx1103 target support, KFD runtime |
| M1 Primitive Validation | ⬜ Not Started | GEMM, gather, RMSNorm, SiLU kernels |
| M2 Single-Layer Validation | ⬜ Not Started | One transformer layer matches reference |
| M3 Full Offline Runner | 🔄 In Progress | CLI produces embeddings |
| M4 Service API | ⬜ Not Started | `/v1/embeddings` works end-to-end |
| M5 Stabilization | ⬜ Not Started | Memory stable, telemetry in place |

## Architecture

```
crates/
├── qemb-common      # Shared types, error handling, config
├── qemb-runtime     # KFD runtime, memory management, scheduling
├── qemb-kernels     # GPU kernel implementations
├── qemb-convert     # Offline weight converter (HF → runtime format)
├── qemb-tokenizer   # Tokenization wrapper (tokenizers crate)
├── qemb-service     # HTTP server (axum on tokio)
└── qemb-cli         # CLI tool for testing and conversion
```

## Target Hardware

- AMD 780M iGPU (gfx1103)
- Linux with `/dev/kfd` and `/dev/dri/renderD*` access

## Quick Start

```bash
# Build
cargo build --release

# Start server (placeholder - no real inference yet)
cargo run --release -- serve --port 3000

# Health check
curl http://localhost:3000/healthz

# List models
curl http://localhost:3000/v1/models
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/healthz` | GET | Health check |
| `/readyz` | GET | Readiness check |
| `/v1/models` | GET | List available models |
| `/v1/embeddings` | POST | Create embeddings (placeholder) |

## CLI Commands

```bash
# Start the embedding service
qemb serve --port 3000 --model ./model

# Convert Hugging Face model to packed format
qemb convert --input ./hf-model --output ./model --name "Qwen3-Embedding-0.6B"

# Inspect a packed model bundle
qemb inspect --model ./model

# Run offline embedding inference (not implemented)
qemb run --text "hello world" --model ./model
```

## Packed Model Format (qembpack-v1)

The converter produces a model bundle with:

```
model/
├── model.json      # Metadata: config, tensor table, checksums
├── weights.bin     # Packed tensor weights
└── tokenizer.json  # Tokenizer (copied from source)
```

## Design Document

See [docs/plans/2026-03-23-qwen3-embedding-design.md](docs/plans/2026-03-23-qwen3-embedding-design.md) for full architecture decisions.

## Implementation Plans

- [Phase 1](docs/plans/2026-03-23-phase1-implementation.md): Workspace setup, gfx1103 support, KFD runtime
- [Phase 2](docs/plans/2026-03-23-phase2-implementation.md): Model converter, packed format, tokenizer loading

## License

MIT OR Apache-2.0