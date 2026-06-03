import subprocess, httpx, base64, json, time, sys, os
from pathlib import Path

BASE = Path(__file__).parent.resolve()
SOURCES = BASE / 'sources'
OUT = BASE / 'output' / 'paddleocr-vl'
OUT.mkdir(parents=True, exist_ok=True)

LLAMA_BIN = Path(os.environ.get('LLAMA_BIN', Path.home() / '.llama' / 'bin'))
LLAMA_MODELS = Path(os.environ.get('LLAMA_MODELS', Path.home() / '.llama' / 'models'))

logfile = open(BASE / 'paddle_srv.log', 'w')
proc = subprocess.Popen([
    str(LLAMA_BIN / 'llama-server.exe'),
    '-m', str(LLAMA_MODELS / 'PaddleOCR-VL-1.6-GGUF.gguf'),
    '--mmproj', str(LLAMA_MODELS / 'PaddleOCR-VL-1.6-GGUF-mmproj.gguf'),
    '--port', '18333', '--host', '127.0.0.1',
    '--temp', '0', '--top-k', '1', '-ngl', '99',
    '--ctx-size', '8192', '--no-warmup',
], stdout=logfile, stderr=logfile, text=True)

print('Waiting for server...', flush=True)
for attempt in range(30):
    time.sleep(1)
    try:
        with httpx.Client(timeout=5) as c:
            r = c.post('http://127.0.0.1:18333/v1/chat/completions',
                       json={'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'hi'}]}]})
            if r.status_code == 200:
                print(f'Server up after {attempt+1}s', flush=True)
                break
    except:
        pass
else:
    print('Server failed to start', flush=True)
    logfile.flush()
    with open(BASE / 'paddle_srv.log') as f:
        print(f.read()[-2000:])
    sys.exit(1)

prompt = 'Extract ALL text from this image verbatim. Preserve line breaks and structure.'

for i in range(1, 11):
    img_path = SOURCES / f'image{i}.png'
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    body = {
        'messages': [{'role': 'user', 'content': [
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{b64}'}},
            {'type': 'text', 'text': prompt}
        ]}],
        'temperature': 0,
        'max_tokens': 3800,
    }
    t0 = time.time()
    try:
        with httpx.Client(timeout=180) as client:
            r = client.post('http://127.0.0.1:18333/v1/chat/completions', json=body)
            result = r.json()['choices'][0]['message']['content']
        (OUT / f'image{i}-output.md').write_text(result, encoding='utf-8')
        print(f'image{i}: {len(result)} chars in {time.time()-t0:.1f}s', flush=True)
    except Exception as e:
        print(f'image{i}: FAILED - {e}', flush=True)

proc.terminate()
print('PaddleOCR batch done!', flush=True)
