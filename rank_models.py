import json, os, re
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict

BASE = Path(__file__).parent.resolve()
OUTPUT = BASE / 'output'
OFFICIAL = OUTPUT / 'official-answer'

models = ['lightonocr-2-1b', 'chandra-ocr-2', 'dots-mocr', 'chandra-v1', 'olmocr-2-7b', 'infinity-parser-7b', 'paddleocr-vl']

def read_file(p):
    try:
        return p.read_text(encoding='utf-8').strip()
    except:
        return ''

def normalize(text):
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9\u4e00-\u9fff\s]', '', text)
    return text

def char_accuracy(gt, pred):
    if not gt:
        return 0.0
    gt_norm = re.sub(r'\s+', '', gt)
    pred_norm = re.sub(r'\s+', '', pred)
    if not gt_norm:
        return 0.0
    matches = sum(1 for i, c in enumerate(pred_norm) if i < len(gt_norm) and c == gt_norm[i])
    return matches / len(gt_norm)

def text_similarity(a, b):
    return SequenceMatcher(None, normalize(a), normalize(b)).ratio()

def word_error_rate(gt, pred):
    if not gt:
        return 1.0
    gtw = gt.split()
    pw = pred.split()
    if not gtw:
        return 0.0
    d = [[0]*(len(pw)+1) for _ in range(len(gtw)+1)]
    for i in range(len(gtw)+1): d[i][0] = i
    for j in range(len(pw)+1): d[0][j] = j
    for i in range(1, len(gtw)+1):
        for j in range(1, len(pw)+1):
            cost = 0 if gtw[i-1] == pw[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(gtw)][len(pw)] / max(len(gtw), 1)

results = defaultdict(lambda: defaultdict(dict))
model_scores = defaultdict(lambda: {'sim': [], 'char_acc': [], 'wer': [], 'len': [], 'halluc': []})

for i in range(1, 11):
    gt = read_file(OFFICIAL / f'image{i}-official.txt')
    gt_norm = normalize(gt)
    
    for m in models:
        pred = read_file(OUTPUT / m / f'image{i}-output.md')
        if not pred:
            results[m][i] = {'sim': 0, 'char_acc': 0, 'wer': 1, 'pred_len': 0, 'halluc_ratio': 0}
            continue
        
        sim = text_similarity(gt, pred)
        ca = char_accuracy(gt, pred)
        wer = word_error_rate(gt, pred)
        
        pred_words = set(normalize(pred).split())
        gt_words = set(normalize(gt).split())
        extra = len(pred_words - gt_words)
        hall_ratio = extra / max(len(pred_words), 1) if pred_words else 0
        
        results[m][i] = {
            'sim': round(sim, 3),
            'char_acc': round(ca, 3),
            'wer': round(wer, 3),
            'pred_len': len(pred),
            'halluc_ratio': round(hall_ratio, 3),
        }
        model_scores[m]['sim'].append(sim)
        model_scores[m]['char_acc'].append(ca)
        model_scores[m]['wer'].append(wer)
        model_scores[m]['len'].append(len(pred))
        model_scores[m]['halluc'].append(hall_ratio)

print('=' * 120)
print(f"{'Model':22s} {'Size':6s} {'Sim':6s} {'CharAcc':7s} {'WER':6s} {'Halluc':7s} {'AvgLen':7s}")
print('-' * 120)

model_sizes = {
    'lightonocr-2-1b': '1.0B',
    'chandra-ocr-2': '2.0B',
    'dots-mocr': '1.8B',
    'chandra-v1': '8.0B',
    'olmocr-2-7b': '7.0B',
    'infinity-parser-7b': '7.0B',
    'paddleocr-vl': '1.6B',
}

ranked = sorted(models, key=lambda m: sum(model_scores[m]['sim'])/max(len(model_scores[m]['sim']),1), reverse=True)

for rank, m in enumerate(ranked, 1):
    s = model_scores[m]
    avg_sim = sum(s['sim']) / len(s['sim'])
    avg_ca = sum(s['char_acc']) / len(s['char_acc'])
    avg_wer = sum(s['wer']) / len(s['wer'])
    avg_hall = sum(s['halluc']) / len(s['halluc'])
    avg_len = sum(s['len']) / len(s['len'])
    print(f"#{rank:<2} {m:22s} {model_sizes[m]:6s} {avg_sim:.3f}  {avg_ca:.3f}  {avg_wer:.3f}  {avg_hall:.3f}  {avg_len:6.0f}")

print('=' * 120)
print()

print('Per-Image Best Model:')
for i in range(1, 11):
    best = max(models, key=lambda m: results[m][i]['sim'])
    print(f"  image{i}: {best} (sim={results[best][i]['sim']:.3f})")

print()

print('Traditional Chinese (image8) ranking:')
cjks = sorted(models, key=lambda m: results[m][8]['sim'], reverse=True)
for m in cjks:
    print(f"  {m:22s} sim={results[m][8]['sim']:.3f} char_acc={results[m][8]['char_acc']:.3f} len={results[m][8]['pred_len']}")

print()

print('Code/Technical Content (images 2,6,7) ranking:')
for img_label, img_nums in [('image2 (Python)', [2]), ('image6 (RSA)', [6]), ('image7 (Pascal)', [7]), ('All Code', [2,6,7])]:
    scored = []
    for m in models:
        sims = [results[m][i]['sim'] for i in img_nums]
        scored.append((sum(sims)/len(sims), m))
    scored.sort(reverse=True)
    best_m = scored[0][1]
    print(f"  {img_label}:")
    for s, m in scored[:4]:
        print(f"    {m:22s} avg_sim={s:.3f}")
