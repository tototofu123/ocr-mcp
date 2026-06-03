# OCR-MCP

A local OCR model benchmarking suite that evaluates multiple vision-language models on text extraction tasks using llama.cpp.

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

## Tags

`ocr` `benchmark` `paddleocr` `lightonocr` `olmocr` `chandra-ocr` `llamacpp` `vision-language-model` `text-extraction` `local-ai` `gguf`
