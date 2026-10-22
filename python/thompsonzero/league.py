"""A Bradley-Terry league across runs and algorithms.

    thompsonzero league --dir runs/league --player az=runs/az50k_128 \\
        --player tz=runs/tz50k --every 10000 --watch

Players are the first benchmark of each run past every multiple of `every`
(bench_0, the untrained net, is skipped), named `<prefix>@<k>k`.  Every new
player plays every player already in the league -- an incremental round
robin -- with `arena.play_pair`: `openings` XOT openings, each from both
colours, `sims` simulations a move, each side searching with its own
algorithm.  After each player's matches the ratings are refitted from all
stored W/D/L (never updated incrementally), printed, and appended to
`ratings.jsonl`.

`<dir>/league.json` holds the settings, the players and the per-pair
results; a league is resumed with the same settings or refused.
"""
import json
import os
import time

import numpy as np

from . import edax
from .arena import bradley_terry, play_pair
from .inference import Evaluator
from .train import load_net, pick_device


def milestones(run_dir, every):
    """{multiple of `every`: first benchmark label at or past it (and
    before the next multiple)}."""
    if not os.path.isdir(run_dir):
        return {}                       # not started yet
    labels = sorted(int(f[6:-3]) for f in os.listdir(run_dir)
                    if f.startswith('bench_') and f.endswith('.pt'))
    out = {}
    for lab in labels:
        m = lab // every * every
        if m and m not in out:
            out[m] = lab
    return out


def finished(run_dir):
    if not os.path.exists(os.path.join(run_dir, 'config.json')):
        return False
    with open(os.path.join(run_dir, 'config.json')) as f:
        c = json.load(f)
        episodes = c.get('stop_at') or c['episodes']
    labels = [int(f[6:-3]) for f in os.listdir(run_dir) if f.startswith('bench_') and f.endswith('.pt')]
    return max(labels, default=0) >= episodes


def standard_errors(pairs, ratings, anchor):
    """Elo standard errors relative to `anchor`, from the Fisher information
    of the Bradley-Terry fit (draws counted as half a win each way)."""
    players = sorted(ratings)
    ix = {p: i for i, p in enumerate(players)}
    n = len(players)
    info = np.zeros((n, n))
    scale = np.log(10) / 400
    for (a, b), (w, d, l) in pairs.items():
        g = w + d + l
        p = 1 / (1 + 10 ** ((ratings[b] - ratings[a]) / 400))
        f = g * p * (1 - p)
        i, j = ix[a], ix[b]
        info[i, i] += f
        info[j, j] += f
        info[i, j] -= f
        info[j, i] -= f
    keep = [i for i in range(n) if players[i] != anchor]
    cov = np.linalg.pinv(info[np.ix_(keep, keep)])
    se = {anchor: 0.0}
    for k, i in enumerate(keep):
        se[players[i]] = float(np.sqrt(max(cov[k, k], 0.0))) / scale
    return se


class League:
    def __init__(self, path, settings):
        self.path = path
        self.players, self.pairs = {}, {}
        if os.path.exists(path):
            with open(path) as f:
                d = json.load(f)
            if d['settings'] != settings:
                raise SystemExit(f'{path} was played with {d["settings"]}, not {settings}')
            self.players = d['players']
            self.pairs = {tuple(k.split('|')): v for k, v in d['pairs'].items()}
        self.settings = settings

    def save(self):
        d = dict(settings=self.settings, players=self.players,
                 pairs={'|'.join(k): v for k, v in self.pairs.items()})
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(d, f, indent=1)
        os.replace(tmp, self.path)

    def played(self, a, b):
        return (a, b) in self.pairs or (b, a) in self.pairs

    def ratings(self, anchor=None):
        r = bradley_terry(self.pairs)
        if anchor in r:
            r = {k: v - r[anchor] for k, v in r.items()}
        return r

    def table(self, anchor=None):
        r = self.ratings(anchor)
        ref = anchor if anchor in r else min(r, key=lambda k: abs(r[k]))
        se = standard_errors(self.pairs, r, ref)
        rows = []
        for p in sorted(r, key=lambda k: -r[k]):
            g = s = 0.0
            for (a, b), (w, d, l) in self.pairs.items():
                if p in (a, b):
                    g += w + d + l
                    s += (w if p == a else l) + 0.5 * d
            rows.append((p, r[p], se[p], s / max(g, 1), int(g)))
        return rows


def run(a):
    device = pick_device(a.device)
    os.makedirs(a.dir, exist_ok=True)
    runs = []
    for spec in a.player:
        prefix, _, path = spec.partition('=')
        runs.append((prefix, path))
    settings = dict(sims=a.sims, openings=a.openings, seed=a.seed, every=a.every)
    league = League(os.path.join(a.dir, 'league.json'), settings)
    ops = edax.load_openings(a.xot or os.path.join(a.edax_dir, 'xot', 'openingslarge.txt'),
                             a.openings, a.seed)
    nets = {}

    def side(name):
        if name not in nets:
            net, cfg = load_net(league.players[name], device)
            nets[name] = (cfg, Evaluator(net, device, cfg.amp))
        return nets[name]

    def report(name):
        rows = league.table(a.anchor)
        print(f'--- ratings after {name} ({len(league.players)} players, '
              f'{sum(sum(v) for v in league.pairs.values())} games; Elo, anchor '
              f'{a.anchor if a.anchor in league.players else "nearest 0"})', flush=True)
        for p, r, se, score, g in rows:
            print(f'  {p:>12} {r:+6.0f} ± {se:3.0f}   score {score:5.1%} over {g} games', flush=True)
        with open(os.path.join(a.dir, 'ratings.jsonl'), 'a') as f:
            f.write(json.dumps(dict(after=name, time=time.time(), ratings=[
                dict(player=p, elo=r, se=se, score=s, games=g) for p, r, se, s, g in rows])) + '\n')

    while True:
        # discovery order: by milestone, then by --player order
        found = []
        for i, (prefix, path) in enumerate(runs):
            for m, lab in milestones(path, a.every).items():
                tag = f'{m // 1000}k' if m % 1000 == 0 else str(m)
                found.append((m, i, f'{prefix}@{tag}', os.path.join(path, f'bench_{lab}.pt')))
        progressed = False
        for _, _, name, path in sorted(found):
            if name not in league.players:
                league.players[name] = path
                league.save()
            opponents = [p for p in league.players if p != name and not league.played(name, p)]
            # a player plays only those that joined before it
            order = list(league.players)
            opponents = [p for p in opponents if order.index(p) < order.index(name)]
            for opp in opponents:
                t0 = time.time()
                w, d, l = play_pair([side(name), side(opp)], 2 * len(ops), a.sims, openings=ops)
                league.pairs[(name, opp)] = [w, d, l]
                league.save()
                print(f'{name} vs {opp} @{a.sims}: W{w} D{d} L{l} '
                      f'({(w + d / 2) / (w + d + l):.1%}, {time.time() - t0:.0f}s)', flush=True)
            if opponents:
                report(name)
                progressed = True
        if progressed:
            continue
        if not a.watch or all(finished(path) for _, path in runs):
            return
        time.sleep(60)


def add_parser(sub):
    e = sub.add_parser('league', help='round-robin Bradley-Terry league across runs and algorithms')
    e.add_argument('--dir', required=True, help='league directory (league.json, ratings.jsonl)')
    e.add_argument('--player', action='append', required=True, metavar='PREFIX=RUN_DIR',
                   help='a run whose benchmarks join the league (repeatable)')
    e.add_argument('--every', type=int, default=10000, help='one player per this many episodes')
    e.add_argument('--sims', type=int, default=400)
    e.add_argument('--openings', type=int, default=100, help='XOT openings, each played twice')
    e.add_argument('--seed', type=int, default=1, help='which openings (the Edax evaluation uses 0)')
    e.add_argument('--anchor', help='player rated 0 (default: mean 0)')
    e.add_argument('--edax-dir', default=edax.DEFAULT_DIR, help='for the XOT list')
    e.add_argument('--xot', help='XOT list (default: <edax-dir>/xot/openingslarge.txt)')
    e.add_argument('--device', default='auto')
    e.add_argument('--watch', action='store_true',
                   help='keep adding benchmarks as they appear, until every run has finished')
    e.set_defaults(fn=run)
