"""Final round robin on held-out openings; Bradley-Terry ratings (PUCT = 0)."""
import sys, json; sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *
import numpy as np

BEST = dict(pick='q', pick_frac=0.2, eager=0, fpu=0.0, backup='mixmax', bparam=0.25)
# best stage-3 tweak by mean Elo over both nets, if it beat BEST's stage-2 score
recs = [r for r in load().values() if r['split'] == 'tune' and r['sims'] == 400 and not r["b"] and not r["sims_b"]]
score = {}
for r in recs:
    score.setdefault(json.dumps(r['a'], sort_keys=True), []).append(elo(r['w'], r['d'], r['l'])[0])
tuned = max(((np.mean(v), k) for k, v in score.items() if len(v) == 2), default=None)
C = {'puct': {}, 'pq': dict(pick='q', pick_frac=0.2),
     'pq+mixmax': dict(pick='q', pick_frac=0.2, backup='mixmax', bparam=0.25),
     'best': BEST, 'puct+ts fit': dict(sel='ts', kappa=1.0, sigma='fit', ts_c=0.1)}
if tuned and json.loads(tuned[1]) not in C.values():
    C['tuned'] = json.loads(tuned[1])
print('candidates:', json.dumps(C), flush=True)
names = list(C)
for sims in (400, 100):
    for net in ('scalar', 'wdl'):
        W = np.zeros((len(names), len(names)))
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                r = match(f'{names[i]} vs {names[j]}', C[names[i]], b=C[names[j]], net=net,
                          sims=sims, split='final200')
                W[i, j] += r['w'] + r['d'] / 2; W[j, i] += r['l'] + r['d'] / 2
        # Bradley-Terry by MM iterations
        p = np.ones(len(names))
        for _ in range(2000):
            for i in range(len(names)):
                den = sum((W[i, j] + W[j, i]) / (p[i] + p[j]) for j in range(len(names)) if j != i)
                p[i] = W[i].sum() / den
            p /= p[0]
        el = 400 * np.log10(p)
        print(f'== {net} @{sims} Bradley-Terry Elo (puct = 0)', flush=True)
        for k in np.argsort(-el):
            print(f'   {names[k]:<14} {el[k]:+6.0f}', flush=True)
