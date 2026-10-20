"""Lab experiment driver: variant-vs-reference matches with a fixed net,
results appended to runs/lab/results.jsonl (resumable)."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, 'python')
from thompsonzero import edax
from thompsonzero.arena import play_games
from thompsonzero.inference import Evaluator
from thompsonzero.train import load_net
from thompsonzero.lab import LabSide

NETS = {'scalar': 'runs/az50k_128/bench_50000.pt', 'wdl': 'runs/azwdl50k/bench_50000.pt',
        'scalar20': 'runs/az50k_128/bench_20000.pt', 'wdl20': 'runs/azwdl50k/bench_20000.pt'}
OUT = 'runs/lab/results.jsonl'
PUCT = {}

_ops = None
def openings(split):
    global _ops
    if _ops is None:
        with open(edax.DEFAULT_DIR + '/xot/openingslarge.txt') as f:
            seqs = [ln.strip() for ln in f if ln.strip()]
        idx = np.random.default_rng(7).permutation(len(seqs))
        _ops = {'screen': idx[:100], 'tune': idx[100:200], 'final': idx[200:600], 'calib': idx[600:1100]}
        _ops = {k: [edax.parse_moves(seqs[i]) for i in v] for k, v in _ops.items()}
        _ops['final200'] = _ops['final'][:200]
    return _ops[split]

_evs = {}
def evaluator(net):
    if net not in _evs:
        n, cfg = load_net(NETS[net], 'cuda')
        _evs[net] = Evaluator(n, 'cuda', cfg.amp)
    return _evs[net]

def load():
    done = {}
    if os.path.exists(OUT):
        for ln in open(OUT):
            r = json.loads(ln)
            done[r['key']] = r
    return done

def elo(w, d, l):
    n = w + d + l
    s = (w + d / 2) / n
    var = (w * (1 - s) ** 2 + d * (0.5 - s) ** 2 + l * s ** 2) / n / n
    f = lambda x: -400 * math.log10(1 / min(max(x, 1e-3), 1 - 1e-3) - 1)
    return f(s), (f(s + math.sqrt(var)) - f(s - math.sqrt(var))) / 2

def match(name, a, b=PUCT, net='scalar', sims=400, split='screen', sims_a=None, sims_b=None, quiet=False):
    """Variant `a` (lab options) vs `b` with network `net`; every opening of
    `split` from both colours.  Returns the stored record."""
    key = json.dumps([net, a, b, sims, split, sims_a, sims_b], sort_keys=True)
    done = load()
    if key in done:
        r = done[key]
    else:
        ops = openings(split)
        sa, sb = LabSide('othello', sims=sims_a, **a), LabSide('othello', sims=sims_b, **b)
        ev = evaluator(net)
        t0 = time.time()
        res = play_games([(sa, ev), (sb, ev)], 2 * len(ops), sims, openings=ops)
        w = sum(r[2] > 0 for r in res); l = sum(r[2] < 0 for r in res); d = len(res) - w - l
        (ea, na), (eb, nb) = sa.evals(), sb.evals()
        r = dict(key=key, name=name, net=net, sims=sims, split=split, a=a, b=b, sims_a=sims_a,
                 sims_b=sims_b, w=w, d=d, l=l, evals_a=ea / max(na, 1), evals_b=eb / max(nb, 1),
                 secs=time.time() - t0)
        with open(OUT, 'a') as f:
            f.write(json.dumps(r) + '\n')
    e, se = elo(r['w'], r['d'], r['l'])
    if not quiet:
        print(f"{r['net']:>8} {r['sims']:>5} {r['name']:<40} W{r['w']:>3} D{r['d']:>3} L{r['l']:>3} "
              f"{e:+6.0f} ±{se:3.0f}  evals/move {r['evals_a']:6.0f} vs {r['evals_b']:6.0f}  {r['secs']:4.0f}s", flush=True)
    return r
