"""Strength against Edax, measured the way OLIVAW was (arXiv 2103.17228, VII-B).

PROTOCOL
  * Edax E_k is `lEdax -level k` (alpha-beta depth k), no opening book, one
    thread per process; parallelism is across games.
  * Random XOT openings: the first 8 moves come from the 10,784-sequence
    "large" XOT list (positions Edax rates within 2 discs at depth 16), which
    is what the paper used to stop deterministic Edax replaying one game.
  * Our side plays its evaluation-time pick from a fresh `sims`-simulation
    search (O-1000 = 1000): AlphaZero's most visited move, ThompsonZero's
    greedy posterior move; no noise, no tree reuse, so `sims` bounds its tree.
  * Each opening is played from both colours (the paper does not say; pairing
    cancels the opening's bias).  Openings are drawn with a fixed seed, so
    every checkpoint and level sees the same set.

THE TARGET: OLIVAW's O-1000 in 100-game series (Fig. 5a; Fig. 6 agrees):
    E4 ~97% wins   E6 ~75%   E8 ~47% W / 10% D / 43% L   E10 ~25 / 8 / 67

    thompsonzero edax --run-dir runs/othello_az --levels 4 6 8 10
    thompsonzero edax --run-dir runs/othello_az --bench 25000 50000 --openings 25
    thompsonzero edax --run-dir runs/othello_az --watch --every 5000   # during training

Edax lives in --edax-dir (default $THOMPSONZERO_EDAX, else ~/code/edax-dist):
lEdax-x86-64-v3 and data/eval.dat from the v4.6 release, and
xot/openingslarge.txt from https://berg.earthlingz.de/xot/downloads/.
"""
import dataclasses
import json
import math
import os
import queue
import re
import subprocess
import time

import numpy as np

from .arena import play_games
from .inference import Evaluator
from .train import load_net, pick_device

DEFAULT_DIR = os.environ.get('THOMPSONZERO_EDAX', os.path.expanduser('~/code/edax-dist'))
PASS = 64
_PLAYS = re.compile(r'Edax plays (\w+)', re.I)


def parse_moves(seq):
    """'f5d6c4d3' -> [37, 43, 26, 19] (action = row * 8 + col)."""
    seq = seq.strip().lower()
    return [(int(seq[i + 1]) - 1) * 8 + ord(seq[i]) - ord('a') for i in range(0, len(seq), 2)]


def load_openings(path, n, seed=0):
    """`n` distinct XOT openings drawn with `seed`, as action lists."""
    with open(path) as f:
        seqs = [ln.strip() for ln in f if ln.strip()]
    pick = np.random.default_rng(seed).choice(len(seqs), size=n, replace=False)
    return [parse_moves(seqs[i]) for i in sorted(pick)]


class Edax:
    """One persistent Edax process at a fixed level, driven by `setboard` +
    `go` (stateless: the whole position is sent every move)."""

    def __init__(self, level, edax_dir=DEFAULT_DIR, exe='lEdax-x86-64-v3'):
        self.proc = subprocess.Popen(
            [os.path.join(edax_dir, exe), '-q', '-level', str(level), '-n', '1',
             '-book-usage', 'off', '-eval-file', os.path.join(edax_dir, 'data', 'eval.dat')],
            cwd=edax_dir, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, bufsize=1)

    def move(self, state):
        """Edax's action for `state` (a GameState); a forced pass is not
        sent to Edax."""
        legal = state.legal_actions()
        if legal == [PASS]:
            return PASS
        self.proc.stdin.write(f'setboard {state.edax_board()}\ngo\n')
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(f'Edax exited (code {self.proc.poll()})')
            m = _PLAYS.search(line)
            if m:
                tok = m.group(1).lower()
                a = PASS if tok in ('ps', 'pa', 'pass') else parse_moves(tok)[0]
                if a not in legal:
                    raise RuntimeError(f'Edax played {tok} (action {a}), legal: {legal}')
                return a

    def close(self):
        try:
            self.proc.stdin.write('quit\n')
            self.proc.stdin.flush()
            self.proc.wait(timeout=5)
        except Exception:  # noqa: BLE001 -- best effort
            self.proc.kill()


def wilson(score, n, z=1.96):
    """95% Wilson interval for a score fraction over n games."""
    if n == 0:
        return 0.0, 1.0
    c = (score + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(score * (1 - score) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, c - h), min(1.0, c + h)


def play(net, cfg, device, level, sims, openings, workers=16, edax_dir=DEFAULT_DIR, seed=0):
    """Play every opening from both colours against Edax at `level` (any
    algorithm: our side searches with its own engine, see
    `arena.play_games`).  Returns [(game, our player, result for us)]."""
    ev = Evaluator(net, device, cfg.amp)
    games = 2 * len(openings)
    pool = queue.Queue()
    for _ in range(min(workers, games)):
        pool.put(Edax(level, edax_dir))

    def edax_move(state):
        e = pool.get()
        try:
            return e.move(state)
        finally:
            pool.put(e)

    try:
        return play_games([(cfg, ev), edax_move], games, sims, openings=openings,
                          seed=seed, workers=pool.qsize())
    finally:
        while not pool.empty():
            pool.get().close()


def summarize(results):
    w = sum(r > 0 for _, _, r in results)
    l = sum(r < 0 for _, _, r in results)
    d = len(results) - w - l
    return w, d, l


def _benchmarks(run_dir):
    return sorted(int(f[6:-3]) for f in os.listdir(run_dir)
                  if f.startswith('bench_') and f.endswith('.pt'))


def run(a):
    device = pick_device(a.device)
    openings = load_openings(a.xot or os.path.join(a.edax_dir, 'xot', 'openingslarge.txt'),
                             a.openings, a.seed)
    out_path = os.path.join(a.run_dir, 'edax.json')
    rows = []
    if os.path.exists(out_path):
        with open(out_path) as f:
            rows = json.load(f)
    over = {}
    for kv in a.set or ():
        k, _, v = kv.partition('=')
        over[k] = json.loads(v)
    if not a.watch:
        _evaluate(a, a.bench or _benchmarks(a.run_dir)[-1:], openings, over, rows, out_path, device)
        return
    # watch: the first benchmark past each multiple of `every`, as it appears
    # (bench_0 is the untrained net: skipped)
    done = {r['bench'] for r in rows if r['sims'] == a.sims and r['openings'] == a.openings
            and r['seed'] == a.seed and r.get('overrides', {}) == over}
    while True:
        last = max(done, default=0)
        new = [b for b in _benchmarks(a.run_dir) if b not in done and b // a.every > last // a.every]
        if new:
            _evaluate(a, new[:1], openings, over, rows, out_path, device)
            done.add(new[0])
            continue
        time.sleep(60)


def _evaluate(a, labels, openings, over, rows, out_path, device):
    for lab in labels:
        net, cfg = load_net(os.path.join(a.run_dir, f'bench_{lab}.pt'), device)
        cfg = dataclasses.replace(cfg, **over)
        for level in a.levels:
            t0 = time.time()
            res = play(net, cfg, device, level, a.sims, openings, a.workers, a.edax_dir, a.seed)
            w, d, l = summarize(res)
            n = w + d + l
            score = (w + 0.5 * d) / max(n, 1)
            lo, hi = wilson(score, n)
            black = [r for _, p, r in res if p == 0]
            tag = ''.join(f' {k}={v}' for k, v in over.items())
            print(f'bench_{lab}{tag} @{a.sims} sims vs E{level}: W{w} D{d} L{l} = {score:.1%} '
                  f'[{lo:.0%}, {hi:.0%}] n={n} | as black {sum(r > 0 for r in black)}/{len(black)} '
                  f'wins ({time.time() - t0:.0f}s)', flush=True)
            rows.append(dict(bench=lab, sims=a.sims, level=level, openings=a.openings,
                             seed=a.seed, overrides=over, wins=w, draws=d, losses=l, score=score,
                             games=[list(r) for r in res], time=time.time()))
            tmp = out_path + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(rows, f, indent=1)
            os.replace(tmp, out_path)


def add_parser(sub):
    e = sub.add_parser('edax', help='play benchmark checkpoints against Edax (OLIVAW protocol)')
    e.add_argument('--run-dir', required=True)
    e.add_argument('--bench', type=int, nargs='*', help='benchmark episodes (default: the latest)')
    e.add_argument('--levels', type=int, nargs='+', default=[4, 6, 8, 10])
    e.add_argument('--sims', type=int, default=1000, help='our simulations per move (O-1000)')
    e.add_argument('--openings', type=int, default=50, help='XOT openings, each played twice')
    e.add_argument('--seed', type=int, default=0, help='which openings')
    e.add_argument('--workers', type=int, default=16, help='Edax processes')
    e.add_argument('--edax-dir', default=DEFAULT_DIR)
    e.add_argument('--xot', help='XOT list (default: <edax-dir>/xot/openingslarge.txt)')
    e.add_argument('--device', default='auto')
    e.add_argument('--watch', action='store_true', help='keep rating new benchmarks as they appear')
    e.add_argument('--every', type=int, default=5000,
                   help='--watch: rate the first benchmark past each multiple of this')
    e.add_argument('--set', action='append', metavar='KEY=VALUE',
                   help='override a search setting of the saved config (e.g. c_puct=2.5)')
    e.set_defaults(fn=run)
