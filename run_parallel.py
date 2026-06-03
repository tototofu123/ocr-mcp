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
    },
    'lightonocr': {
        'model': 'LightOnOCR-2-1B-GGUF-Q8_0.gguf',
        'mmproj': None,
        'port': 18334, 'ngl': 99,
    },
}

active_servers = []

def pick_10_random_images(all_imgs: List[Path]) -> List[Path]:
    return random.sample(all_imgs, min(10, len(all_imgs)))

async def process_one(client: httpx.AsyncClient, port: int, img_path: Path, out_dir: Path, sem: asyncio.Semaphore, idx: int):
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
            r = await client.post(f'http://127.0.0.1:{port}/v1/chat/completions', json=body, timeout=180)
            result = r.json()['choices'][0]['message']['content']
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
        tasks = [process_one(client, port, img, out_dir, sem, i+1) for i, img in enumerate(images)]
        results = await asyncio.gather(*tasks)
    return sum(results), len(results)

def start_server(model_key: str, port: int) -> subprocess.Popen:
    cleanup_stale_servers()
    cfg = MODELS[model_key]
    log_path = BASE / f'srv_{model_key}.log'
    try: log_path.unlink(missing_ok=True)
    except: pass
    logfile = open(log_path, 'w', buffering=1)
    cmd = [str(LLAMA_BIN / 'llama-server.exe')]
    cmd += ['-m', str(LLAMA_MODELS / cfg['model'])]
    if cfg['mmproj']:
        cmd += ['--mmproj', str(LLAMA_MODELS / cfg['mmproj'])]
    cmd += ['--port', str(port), '--host', '127.0.0.1']
    cmd += ['--temp', '0', '--top-k', '1', '-ngl', str(cfg['ngl'])]
    cmd += ['--ctx-size', '8192', '--no-warmup']
    print(f'  Starting {model_key} on :{port}...', flush=True)
    proc = subprocess.Popen(cmd, stdout=logfile, stderr=logfile, text=True, cwd=str(LLAMA_BIN))
    active_servers.append(proc)
    return proc

def cleanup_stale_servers():
    try:
        subprocess.run(['taskkill', '/f', '/im', 'llama-server.exe'], capture_output=True, timeout=5)
    except: pass

async def wait_for_server(port: int, timeout: int = 60):
    async with httpx.AsyncClient(timeout=10) as c:
        for i in range(timeout):
            try:
                r = await c.post(f'http://127.0.0.1:{port}/v1/chat/completions',
                    json={'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]},
                    headers={'Content-Type': 'application/json'})
                if r.status_code == 200:
                    print(f'  Server ready after {i+1}s', flush=True)
                    return True
                print(f'  Waiting... status={r.status_code}', flush=True)
            except Exception as e:
                if i == 0 or (i+1) % 10 == 0:
                    print(f'  Waiting... ({i+1}s) {e.__class__.__name__}', flush=True)
            await asyncio.sleep(1)
    return False

def cleanup():
    for p in active_servers:
        try: p.terminate(); p.wait(timeout=3)
        except: pass
    active_servers.clear()

async def main():
    model_key = 'paddleocr'
    custom_images = []
    concurrency = 4

    argv = sys.argv[1:]
    while argv:
        a = argv.pop(0)
        if a == '--model' and argv: model_key = argv.pop(0)
        elif a == '--images' and argv: custom_images = argv.pop(0).split(',')
        elif a == '--concurrency' and argv: concurrency = int(argv.pop(0))
        elif a == '--help':
            print(f'Usage: python run_parallel.py [--model MODEL] [--images DIR|FILE1,FILE2,...] [--concurrency N]')
            print(f'Models: {", ".join(MODELS.keys())}')
            return

    if model_key not in MODELS:
        print(f'Unknown model: {model_key}. Available: {", ".join(MODELS.keys())}')
        return

    if custom_images:
        images = []
        for ref in custom_images:
            p = Path(ref)
            if p.is_dir(): images.extend(sorted(p.glob('*.*')))
            elif p.is_file(): images.append(p)
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
    print(f'  Model:      {model_key}')
    print(f'  Images:     {len(images)}')
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

    print(f'\n  Done! {success}/{total} images in {elapsed:.1f}s ({elapsed/max(success,1):.1f}s avg)')
    print(f'  Results: {out_dir}')
    cleanup()

if __name__ == '__main__':
    try:
        asyncio.run(main())
    finally:
        cleanup()
