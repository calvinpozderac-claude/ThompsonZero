"""Separate a run's network gain from its search gain: run A's net vs run B's
net, each with plain PUCT and with A's search options, on held-out openings.
Usage: net_vs_search.py <netA.pt> <netB.pt> [opts-json]"""
import json, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *

a_path, b_path = sys.argv[1], sys.argv[2]
opts = json.loads(sys.argv[3]) if len(sys.argv) > 3 else dict(pick='q', pick_frac=0.2, backup='mixmax', bparam=0.25)
evs = []
for p in (a_path, b_path):
    n, cfg = load_net(p, 'cuda')
    evs.append(Evaluator(n, 'cuda', cfg.amp))
ops = openings('final200')
for sims in (400, 100):
    for name, oa, ob in [('A puct vs B puct', {}, {}), ('A search vs B puct', opts, {}), ('A search vs B search', opts, opts)]:
        sa, sb = LabSide('othello', **oa), LabSide('othello', **ob)
        res = play_games([(sa, evs[0]), (sb, evs[1])], 2 * len(ops), sims, openings=ops)
        w = sum(r[2] > 0 for r in res); l = sum(r[2] < 0 for r in res); d = len(res) - w - l
        e, se = elo(w, d, l)
        print(f'{sims:>4} {name:<22} W{w:>4} D{d:>3} L{l:>4}  {e:+5.0f} ±{se:3.0f}', flush=True)
