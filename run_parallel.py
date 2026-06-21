import asyncio, httpx, base64, json, time, sys, os, subprocess, signal, random
from pathlib import Path
from typing import List, Optional

BASE = Path(__file__).parent.resolve()
LLAMA_BIN = Path(os.environ.get('LLAMA_BIN', Path.home() / '.llama' / 'bin'))
LLAMA_MODELS = Path(os.environ.get('LLAMA_MODELS', Path.home() / '.llama' / 'models'))

PROMPT = 'Extract ALL text from this image verbatim. Preserve line breaks and structure. Output only the raw text content.'

MODELS = {
    'paddleocr': {
        'model': 'PaddleOCR-VL-1.6-GGUF.gguf',
        'mmproj': 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf',
        'port': 18333, 'ngl': 99,
        'strengths': ['chinese', 'cjk', 'mixed-language'],
    },
    'lightonocr': {
        'model': 'LightOnOCR-2-1B-GGUF-Q8_0.gguf',
        'mmproj': None,
        'port': 18334, 'ngl': 99,
        'strengths': ['general', 'numbers', 'forms', 'fast'],
    },
    'chandra-v2': {
        'model': 'chandra-ocr-2-GGUF.gguf',
        'mmproj': None,
        'port': 18335, 'ngl': 99,
        'strengths': ['chinese', 'cjk', 'handwriting'],
    },
    'olmocr': {
        'model': 'olmOCR-2-7B-1025-GGUF.gguf',
        'mmproj': None,
        'port': 18336, 'ngl': 99,
        'strengths': ['code', 'technical', 'english', 'accurate'],
    },
    'infinity-parser': {
        'model': 'Infinity-Parser-7B-GGUF.gguf',
        'mmproj': None,
        'port': 18337, 'ngl': 99,
        'strengths': ['code', 'technical', 'tables', 'structured'],
    },
    'dots-mocr': {
        'model': 'dots.mocr-gguf.gguf',
        'mmproj': None,
        'port': 18338, 'ngl': 99,
        'strengths': ['general', 'mixed-language'],
    },
    'chandra-v1': {
        'model': 'chandra-OCR-GGUF.gguf',
        'mmproj': None,
        'port': 18339, 'ngl': 99,
        'strengths': ['accurate', 'english', 'dense-text'],
    },
}

# ---------------------------------------------------------------------------
# Task-hint routing
# ---------------------------------------------------------------------------
# When calling this as an MCP tool, pass --task-hint with one of these keys.
# The router returns an ordered list of models to try (first = preferred).
# Falls back to the next model on timeout or empty output.

TASK_ROUTING: dict[str, list[str]] = {
    # Chinese / Traditional Chinese / CJK mixed
    'chinese':        ['paddleocr', 'chandra-v2', 'dots-mocr'],
    'cjk':            ['paddleocr', 'chandra-v2', 'dots-mocr'],
    'traditional-chinese': ['paddleocr', 'chandra-v2'],

    # Code and technical content
    'code':           ['olmocr', 'infinity-parser', 'lightonocr'],
    'technical':      ['olmocr', 'infinity-parser', 'lightonocr'],
    'screenshot':     ['olmocr', 'lightonocr', 'paddleocr'],

    # Structured content
    'table':          ['infinity-parser', 'olmocr', 'lightonocr'],
    'form':           ['lightonocr', 'infinity-parser', 'olmocr'],
    'numbers':        ['lightonocr', 'paddleocr', 'olmocr'],

    # General / unknown
    'general':        ['lightonocr', 'paddleocr', 'olmocr'],
    'fast':           ['lightonocr', 'dots-mocr', 'paddleocr'],
    'accurate':       ['chandra-v1', 'olmocr', 'infinity-parser'],

    # Handwriting
    'handwriting':    ['chandra-v2', 'chandra-v1', 'paddleocr'],
}

DEFAULT_ROUTE = ['lightonocr', 'paddleocr', 'olmocr']


def resolve_model_order(task_hint: Optional[str]) -> list[str]:
    """Return ordered list of model keys for a given task hint."""
    if not task_hint:
        return DEFAULT_ROUTE
    hint = task_hint.lower().strip().replace(' ', '-')
    # Direct match
    if hint in TASK_ROUTING:
        return TASK_ROUTING[hint]
    # Partial match — find first routing key that contains the hint
    for key, models in TASK_ROUTING.items():
        if hint in key or key in hint:
            return models
    return DEFAULT_ROUTE


# ---------------------------------------------------------------------------
# Server / inference helpers (unchanged from original)
# ---------------------------------------------------------------------------

active_servers = []


async def process_one(
    client: httpx.AsyncClient,
    port: int,
    img_path: Path,
    out_dir: Path,
    sem: asyncio.Semaphore,
    idx: int,
):
    async with sem:
        ext = img_path.suffix.lower()
        media_type = 'image/png' if ext == '.png' else 'image/jpeg'
        b64 = base64.b64encode(img_path.read_bytes()).decode()
        body = {
            'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{b64}'}},
                {'type': 'text', 'text': PROMPT}
            ]}],
            'temperature': 0, 'max_tokens': 3800,
        }
        t0 = time.time()
        try:
            r = await client.post(
                f'http://127.0.0.1:{port}/v1/chat/completions',
                json=body, timeout=180
            )
            result = r.json()['choices'][0]['message']['content']
            if not result.strip():
                raise ValueError('Empty response from model')
            out_path = out_dir / f'{img_path.stem}-output.md'
            out_path.write_text(result, encoding='utf-8')
            elapsed = time.time() - t0
            print(f'  [{idx:02d}] \u2713 {img_path.name}: {len(result)} chars in {elapsed:.1f}s', flush=True)
            return True
        except Exception as e:
            elapsed = time.time() - t0
            print(f'  [{idx:02d}] \u2717 {img_path.name}: FAILED after {elapsed:.1f}s - {e}', flush=True)
            return False


async def run_model(port: int, images: List[Path], out_dir: Path, concurrency: int = 4):
    out_dir.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=180) as client:
        tasks = [
            process_one(client, port, img, out_dir, sem, i + 1)
            for i, img in enumerate(images)
        ]
        results = await asyncio.gather(*tasks)
    return sum(results), len(results)


def start_server(model_key: str, port: int) -> subprocess.Popen:
    cleanup_stale_servers()
    cfg = MODELS[model_key]
    log_path = BASE / f'srv_{model_key}.log'
    try:
        log_path.unlink(missing_ok=True)
    except Exception:
        pass
    logfile = open(log_path, 'w', buffering=1)
    cmd = [str(LLAMA_BIN / 'llama-server.exe')]
    cmd += ['-m', str(LLAMA_MODELS / cfg['model'])]
    if cfg['mmproj']:
        cmd += ['--mmproj', str(LLAMA_MODELS / cfg['mmproj'])]
    cmd += ['--port', str(port), '--host', '127.0.0.1']
    cmd += ['--temp', '0', '--top-k', '1', '-ngl', str(cfg['ngl'])]
    cmd += ['--ctx-size', '8192', '--no-warmup']
    print(f'  Starting {model_key} on :{port}...', flush=True)
    proc = subprocess.Popen(
        cmd, stdout=logfile, stderr=logfile, text=True, cwd=str(LLAMA_BIN)
    )
    active_servers.append(proc)
    return proc


def cleanup_stale_servers():
    try:
        subprocess.run(
            ['taskkill', '/f', '/im', 'llama-server.exe'],
            capture_output=True, timeout=5
        )
    except Exception:
        pass


async def wait_for_server(port: int, timeout: int = 60):
    async with httpx.AsyncClient(timeout=10) as c:
        for i in range(timeout):
            try:
                r = await c.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]},
                    headers={'Content-Type': 'application/json'},
                )
                if r.status_code == 200:
                    print(f'  Server ready after {i + 1}s', flush=True)
                    return True
                print(f'  Waiting... status={r.status_code}', flush=True)
            except Exception as e:
                if i == 0 or (i + 1) % 10 == 0:
                    print(f'  Waiting... ({i + 1}s) {e.__class__.__name__}', flush=True)
            await asyncio.sleep(1)
    return False


def cleanup():
    for p in active_servers:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            pass
    active_servers.clear()


# ---------------------------------------------------------------------------
# Routed single-image inference (for MCP tool use)
# ---------------------------------------------------------------------------

async def run_with_routing(
    img_path: Path,
    task_hint: Optional[str] = None,
    fallback: bool = True,
) -> Optional[str]:
    """
    Run OCR on a single image using the best model for the given task hint.
    Falls back to the next model in the routing list on failure or empty output.

    Returns extracted text string, or None if all models failed.

    Usage from MCP tool:
        result = asyncio.run(run_with_routing(Path('screenshot.png'), task_hint='code'))
    """
    model_order = resolve_model_order(task_hint)
    for model_key in model_order:
        if model_key not in MODELS:
            continue
        cfg = MODELS[model_key]
        port = cfg['port']
        print(f'  Trying model: {model_key} (hint={task_hint})', flush=True)
        proc = start_server(model_key, port)
        try:
            if not await wait_for_server(port, timeout=90):
                print(f'  {model_key}: server startup timed out, trying next model', flush=True)
                proc.terminate()
                continue
            ext = img_path.suffix.lower()
            media_type = 'image/png' if ext == '.png' else 'image/jpeg'
            b64 = base64.b64encode(img_path.read_bytes()).decode()
            body = {
                'messages': [{'role': 'user', 'content': [
                    {'type': 'image_url', 'image_url': {'url': f'data:{media_type};base64,{b64}'}},
                    {'type': 'text', 'text': PROMPT},
                ]}],
                'temperature': 0, 'max_tokens': 3800,
            }
            async with httpx.AsyncClient(timeout=120) as client:
                r = await client.post(
                    f'http://127.0.0.1:{port}/v1/chat/completions',
                    json=body,
                )
                result = r.json()['choices'][0]['message']['content'].strip()
            if result:
                print(f'  {model_key}: success ({len(result)} chars)', flush=True)
                cleanup()
                return result
            print(f'  {model_key}: empty output, trying next model', flush=True)
        except Exception as e:
            print(f'  {model_key}: error ({e}), trying next model', flush=True)
        finally:
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                pass
        if not fallback:
            break
    cleanup()
    return None


# ---------------------------------------------------------------------------
# CLI entrypoint (batch mode, unchanged behaviour)
# ---------------------------------------------------------------------------

async def main():
    model_key = None
    task_hint = None
    single_image = None
    custom_images = []
    concurrency = 4

    argv = sys.argv[1:]
    while argv:
        a = argv.pop(0)
        if a == '--model' and argv:
            model_key = argv.pop(0)
        elif a == '--task-hint' and argv:
            task_hint = argv.pop(0)
        elif a == '--image' and argv:
            single_image = Path(argv.pop(0))
        elif a == '--images' and argv:
            custom_images = argv.pop(0).split(',')
        elif a == '--concurrency' and argv:
            concurrency = int(argv.pop(0))
        elif a == '--help':
            print('Usage: python run_parallel.py [OPTIONS]')
            print()
            print('Options:')
            print('  --model MODEL         Force a specific model key')
            print('  --task-hint HINT      Auto-select model by task type')
            print('  --image FILE          Single image (uses routing + fallback)')
            print('  --images DIR|F1,F2    Batch of images')
            print('  --concurrency N       Parallel requests per server (default: 4)')
            print()
            print(f'Models: {", ".join(MODELS.keys())}')
            print()
            print('Task hints:', ', '.join(TASK_ROUTING.keys()))
            return

    # Single-image routed mode (MCP use case)
    if single_image:
        text = await run_with_routing(single_image, task_hint=task_hint)
        if text:
            print(text)
        else:
            print('OCR failed: all models returned empty or timed out', file=sys.stderr)
            sys.exit(1)
        return

    # Batch mode — explicit model required (or derive from task hint)
    if not model_key:
        if task_hint:
            model_key = resolve_model_order(task_hint)[0]
            print(f'  Routing hint "{task_hint}" -> model: {model_key}', flush=True)
        else:
            model_key = DEFAULT_ROUTE[0]
            print(f'  No model or hint specified, defaulting to: {model_key}', flush=True)

    if model_key not in MODELS:
        print(f'Unknown model: {model_key}. Available: {", ".join(MODELS.keys())}')
        return

    if custom_images:
        images = []
        for ref in custom_images:
            p = Path(ref)
            if p.is_dir():
                images.extend(sorted(p.glob('*.*')))
            elif p.is_file():
                images.append(p)
    else:
        img_dir = BASE / 'sources'
        images = sorted(img_dir.glob('*.png')) + sorted(img_dir.glob('*.jpg'))

    if not images:
        print('No images found')
        return

    images = [i for i in images if i.suffix.lower() in ('.png', '.jpg', '.jpeg')]
    random.seed(42)
    images = random.sample(images, min(10, len(images)))

    print(f'\n=== Parallel OCR Dispatcher ===')
    print(f'  Model:       {model_key}')
    print(f'  Images:      {len(images)}')
    print(f'  Concurrency: {concurrency}')
    print()

    port = MODELS[model_key]['port']
    out_dir = BASE / 'output' / model_key

    proc = start_server(model_key, port)
    if not await wait_for_server(port):
        print('Server failed to start')
        proc.terminate()
        return

    print('  Server ready! Processing...\n')
    t0 = time.time()
    success, total = await run_model(port, images, out_dir, concurrency)
    elapsed = time.time() - t0

    print(f'\n  Done! {success}/{total} images in {elapsed:.1f}s ({elapsed / max(success, 1):.1f}s avg)')
    print(f'  Results: {out_dir}')
    cleanup()


if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        cleanup()
