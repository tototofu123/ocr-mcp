# OCR-MCP

A local OCR model benchmarking suite that evaluates multiple vision-language models on text extraction tasks using llama.cpp. Designed for offline, privacy-preserving OCR across different model architectures.

## Models Benchmarked

| Model | Size | HuggingFace |
|-------|------|-------------|
| PaddleOCR-VL | 1.6B | `PaddlePaddle/PaddleOCR-VL-1.6` |
| LightOnOCR-2 | 1.0B | `ggml-org/LightOnOCR-2-1B-GGUF` |
| olmOCR | 7.0B | `lmstudio-community/olmOCR-2-7B-1025-GGUF` |
| Chandra OCR v1 | 8.0B | `prithivMLmods/chandra-OCR-GGUF` |
| Chandra OCR v2 | 2.0B | `prithivMLmods/chandra-ocr-2-GGUF` |
| Dots MOCR | 1.8B | `lodrick-the-lafted/dots.mocr-gguf` |
| Infinity Parser | 7.0B | `mradermacher/Infinity-Parser-7B-GGUF` |

## Usage

### Prerequisites
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with `llama-server.exe` and `llama-mtmd-cli.exe`
- GGUF quantized models downloaded to `~/.llama/models/`

### Run batch inference (all models)
```powershell
.\batch_infer.ps1
```

### Run PaddleOCR server + inference
```powershell
python run_paddle.py
```

### Rank model results
```powershell
python rank_models.py
```

### Configuration
Set env vars to customize paths:
- `$env:LLAMA_BIN` — directory containing `llama-server.exe` / `llama-mtmd-cli.exe` (default: `~/.llama/bin`)
- `$env:LLAMA_MODELS` — directory containing GGUF model files (default: `~/.llama/models`)

## Roadmap: Parallel OCR with Subagents

Current scripts process images **sequentially** — one image at a time, one model at a time. This bottlenecks on I/O and GPU inference.

### Proposed architecture: Multi-level parallelism

```
┌─────────────────────────────────────────────────────────┐
│                  Coordinator / Dispatcher               │
│  (async Python with asyncio + httpx.AsyncClient)        │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐          │
│  │ Subagent 1│  │ Subagent 2│  │ Subagent 3│  ... N   │
│  │ PaddleOCR │  │ LightOnOCR│  │ olmOCR    │          │
│  │ :18333    │  │ :18334    │  │ :18335    │          │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘          │
│        │              │              │                  │
│  ┌─────┴─────┐  ┌─────┴─────┐  ┌─────┴─────┐          │
│  │ img1..10  │  │ img1..10  │  │ img1..10  │          │
│  │ parallel  │  │ parallel  │  │ parallel  │          │
│  └───────────┘  └───────────┘  └───────────┘          │
│                                                         │
│  Results → output/agent-{name}/image-{n}-result.md     │
└─────────────────────────────────────────────────────────┘
```

### Level 1: Async inference (single server, non-blocking)
Send multiple image requests concurrently to one running server. llama.cpp server handles a queue internally.

### Level 2: Multi-model subagents (multi-server, parallel)
Start N model servers each on a unique port. Each runs as an independent subprocess. Images are distributed round-robin or by model.

### Level 3: Full subagent orchestration
Each subagent = self-contained worker (spawned via `subprocess` or a task queue). Subagents:
- Start their own model server
- Process their image batch asynchronously
- Write results, then self-terminate
- Report status back via a shared queue or filesystem

## Next Steps / Iteration Ideas

| Priority | Feature | Benefit |
|----------|---------|---------|
| P0 | Async inference (`asyncio` + `httpx.AsyncClient`) | 5-10x faster on multi-image batches |
| P0 | Multi-server launcher (`run_all_parallel.py`) | Compare all models in one command |
| P1 | Subagent dispatcher script | Non-blocking background processing |
| P1 | Progress dashboard (live results side-by-side) | See results as they arrive |
| P2 | Watch mode (`--watch` flag) | Auto-process new images in a folder |
| P2 | REST API wrapper | Use OCR as a service |
| P3 | Queue backend (Redis / file-based) | Scale across machines |
| P3 | Web UI for submitting images | No CLI needed |

## Tags

`ocr` `benchmark` `paddleocr` `lightonocr` `olmocr` `chandra-ocr` `llamacpp` `vision-language-model` `text-extraction` `local-ai` `gguf` `parallel-inference` `async` `subagents`
