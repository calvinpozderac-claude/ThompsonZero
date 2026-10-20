"""Rank lab results: summary.py [split] [sims] [ref-json]"""
import json, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import elo
split = sys.argv[1] if len(sys.argv) > 1 else 'screen'
sims = int(sys.argv[2]) if len(sys.argv) > 2 else 400
rows = {}
for ln in open('runs/lab/results.jsonl'):
    r = json.loads(ln)
    if r['split'] != split or r['sims'] != sims or r['b'] != {} or r.get('sims_a') or r.get('sims_b'):
        continue
    rows.setdefault(r['name'], {})[r['net']] = r
nets = sorted({n for v in rows.values() for n in v})
print(f'{"variant":<40}' + ''.join(f'{n:>16}' for n in nets) + '    mean')
out = []
for name, v in rows.items():
    es = {n: elo(v[n]['w'], v[n]['d'], v[n]['l']) for n in v}
    m = sum(e[0] for e in es.values()) / len(es)
    out.append((m, name, es))
for m, name, es in sorted(out, reverse=True):
    print(f'{name:<40}' + ''.join(f'{es[n][0]:+9.0f} ±{es[n][1]:3.0f}  ' if n in es else ' ' * 16 for n in nets) + f'  {m:+5.0f}')
