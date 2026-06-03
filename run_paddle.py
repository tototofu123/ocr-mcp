import httpx, json, base64, time, sys
from pathlib import Path

BASE = Path(__file__).parent.resolve()
SOURCES = BASE / 'sources'
OUT = BASE / 'output' / 'paddleocr-vl'
OUT.mkdir(parents=True, exist_ok=True)

LLAMA_BIN = Path(os.environ.get('LLAMA_BIN', Path.home() / '.llama' / 'bin'))
LLAMA_MODELS = Path(os.environ.get('LLAMA_MODELS', Path.home() / '.llama' / 'models'))

url = 'http://127.0.0.1:18333/v1/chat/completions'
prompt = 'Extract ALL text from this image verbatim. Preserve line breaks and structure. Output only the raw text content.'

print('Testing connectivity...')
t0 = time.time()
try:
    with httpx.Client(timeout=30) as client:
        r = client.post(url, json={'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'say hello'}]}]})
        data = r.json()
        print(f'Connected: {data["choices"][0]["message"]["content"][:50]} ({time.time()-t0:.1f}s)')
except Exception as e:
    print(f'FAILED to connect: {e}')
    sys.exit(1)

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
            r = client.post(url, json=body)
            result = r.json()['choices'][0]['message']['content']
        (OUT / f'image{i}-output.md').write_text(result, encoding='utf-8')
        print(f'image{i}: {len(result)} chars in {time.time()-t0:.1f}s')
    except Exception as e:
        print(f'image{i}: FAILED - {e}')

print('PaddleOCR batch done!')
