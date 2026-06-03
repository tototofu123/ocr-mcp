import subprocess, httpx, base64, json, time, sys, os
from pathlib import Path

BASE = Path(__file__).parent.resolve()
SOURCES = BASE / 'sources'

LLAMA_BIN = Path(os.environ.get('LLAMA_BIN', Path.home() / '.llama' / 'bin'))
LLAMA_MODELS = Path(os.environ.get('LLAMA_MODELS', Path.home() / '.llama' / 'models'))

logfile = open(BASE / 'paddle_srv2.log', 'w')
proc = subprocess.Popen([
    str(LLAMA_BIN / 'llama-server.exe'),
    '-m', str(LLAMA_MODELS / 'PaddleOCR-VL-1.6-GGUF.gguf'),
    '--mmproj', str(LLAMA_MODELS / 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf'),
    '--port', '18333', '--host', '127.0.0.1',
    '--temp', '0', '--top-k', '1', '-ngl', '99',
    '--ctx-size', '8192', '--no-warmup',
], stdout=logfile, stderr=logfile, text=True)

time.sleep(10)

img_path = SOURCES / 'image1.png'
b64 = base64.b64encode(img_path.read_bytes()).decode()
body = {
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
        {'type': 'text', 'text': 'Extract ALL text from this image verbatim. Preserve line breaks and structure.'}
    ]}],
    'temperature': 0,
    'max_tokens': 3800,
}

print('Sending image request (120s timeout)...', flush=True)
t0 = time.time()
try:
    with httpx.Client(timeout=120) as client:
        r = client.post('http://127.0.0.1:18333/v1/chat/completions', json=body)
        result = r.json()['choices'][0]['message']['content']
    print(f'image1: {len(result)} chars in {time.time()-t0:.1f}s', flush=True)
    print(f'Result: {result[:500]}', flush=True)
except Exception as e:
    print(f'FAILED: {e} ({time.time()-t0:.1f}s)', flush=True)

proc.terminate()
