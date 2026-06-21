# OCR-MCP

A local OCR MCP tool that gives coding agents (opencode, Claude, etc.) vision capabilities without sending images to external APIs. Runs multiple vision-language models via llama.cpp locally, with automatic model routing based on image content type.

## Why this exists

Free LLMs have no vision endpoint. When working in a coding agent like [opencode](https://github.com/sst/opencode), there's no way to pass a screenshot, a DevTools panel, or a scanned document into the conversation. This MCP server solves that: give the agent a file path, it calls the OCR tool, gets back text, and continues reasoning.

## Models

| Model | Size | Best for | HuggingFace |
|-------|------|----------|-----------|
| PaddleOCR-VL | 1.6B | Chinese / CJK / mixed | `PaddlePaddle/PaddleOCR-VL-1.6` |
| LightOnOCR-2 | 1.0B | General text, numbers, forms (fast) | `ggml-org/LightOnOCR-2-1B-GGUF` |
| Chandra OCR v2 | 2.0B | Chinese, handwriting | `prithivMLmods/chandra-ocr-2-GGUF` |
| Chandra OCR v1 | 8.0B | Dense text, high accuracy | `prithivMLmods/chandra-OCR-GGUF` |
| olmOCR-2 | 7.0B | Code, technical content, English | `lmstudio-community/olmOCR-2-7B-1025-GGUF` |
| Infinity Parser | 7.0B | Code, tables, structured content | `mradermacher/Infinity-Parser-7B-GGUF` |
| Dots MOCR | 1.8B | General, mixed language | `lodrick-the-lafted/dots.mocr-gguf` |

## Task-hint routing

Pass `--task-hint` and the dispatcher picks the best model automatically, with fallback to the next model on timeout or empty output.

| Hint | Primary model | Fallback chain |
|------|--------------|----------------|
| `chinese` / `cjk` | paddleocr | chandra-v2 → dots-mocr |
| `traditional-chinese` | paddleocr | chandra-v2 |
| `code` / `technical` | olmocr | infinity-parser → lightonocr |
| `screenshot` | olmocr | lightonocr → paddleocr |
| `table` | infinity-parser | olmocr → lightonocr |
| `form` / `numbers` | lightonocr | paddleocr → olmocr |
| `handwriting` | chandra-v2 | chandra-v1 → paddleocr |
| `accurate` | chandra-v1 | olmocr → infinity-parser |
| `fast` / `general` | lightonocr | paddleocr → olmocr |

## Usage

### Prerequisites
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built with `llama-server.exe`
- GGUF models downloaded to `~/.llama/models/`

### Single image (MCP / agent use case)
```powershell
# Auto-route by content type
python run_parallel.py --image screenshot.png --task-hint code
python run_parallel.py --image document.png --task-hint chinese
python run_parallel.py --image form.jpg --task-hint numbers

# Force a specific model
python run_parallel.py --image photo.png --model paddleocr
```

### Batch inference
```powershell
# Run against sources/ folder with routing
python run_parallel.py --task-hint general

# All models via batch script
.\batch_infer.ps1
```

### Rank and compare results
```powershell
python rank_models.py
```

### Environment variables
- `LLAMA_BIN` — path to `llama-server.exe` directory (default: `~/.llama/bin`)
- `LLAMA_MODELS` — path to GGUF model files (default: `~/.llama/models`)

## Architecture

```
Agent (opencode / Claude)
  └── calls OCR MCP tool with: image_path + task_hint
        └── run_parallel.py::run_with_routing()
              ├── resolve_model_order(task_hint)  →  [model1, model2, model3]
              └── for each model (with fallback):
                    ├── start llama-server on local port
                    ├── POST image as base64 to /v1/chat/completions
                    └── return text if non-empty, else try next model
```

For parallel batch benchmarking across all models simultaneously, see `batch_infer.ps1`.

## Benchmark

Evaluated on 10 images across general text, Traditional Chinese, Python code, RSA key blocks, and Pascal source. Scored on text similarity, character accuracy, word error rate, and hallucination ratio.

Run your own evaluation:
```powershell
python run_50_random.py   # run all models on 50 random images
python rank_models.py     # score and rank results
```

## Tags

`ocr` `mcp` `llama-cpp` `vision` `local-ai` `opencode` `agent` `paddleocr` `olmocr` `gguf` `routing` `task-hint` `parallel-inference`
