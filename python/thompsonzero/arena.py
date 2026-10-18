"""Matches between networks, and a Bradley-Terry ladder over checkpoints.

`play_match` runs one ThompsonZero engine in arena mode: both sides search
with `sims` simulations using their own network, play the greedy posterior
move, and alternate colours; a few random opening plies give the games
variety.  With sims=0 it is a search-free match read straight off the action
heads.

`play_games` / `play_pair` play any two sides -- ThompsonZero or AlphaZero
networks in any combination, or an external player such as Edax -- with
each network's own engine running as a search service; the games themselves
live here.  Every opening is played from both colours.

Ratings are always refitted from the stored per-pair W/D/L (Bradley-Terry,
draws as half points), never accumulated incrementally.
"""
import concurrent.futures as cf
import json
import math
import os
import random

import numpy as np

from . import _core, az
from .inference import evaluate_split


def play_match(cfg, evaluators, games, sims, seed=0, opening_plies=2):
    """Play `games` between evaluators[0] and evaluators[1]; returns
    (wins, draws, losses) for evaluators[0]."""
    eng = _core.Engine(
        cfg.game, num_games=min(games, 512), groups=1, seed=seed, sims=sims, wave=4,
        search=cfg.search_rule, target=cfg.target_rule, rho_search=cfg.rho_search,
        rho_target=cfg.rho_target, virtual_loss=cfg.virtual_loss,
        descent_temp=eval_descent_temp(cfg), max_plies=cfg.max_plies or None,
        arena_games=games, opening_plies=opening_plies)
    while not eng.done:
        obs, nets = eng.collect(0)
        eng.apply(0, *evaluate_split(evaluators, obs, nets))
    s = eng.stats()
    return s['wins'], s['games'] - s['wins'] - s['losses'], s['losses']


def eval_descent_temp(cfg):
    """TZ's in-tree descent temperature in matches."""
    return cfg.eval_descent_temp or cfg.descent_temp


def search_engine(cfg, slots, sims, seed=0):
    """`cfg`'s search (TZ or AZ, as at evaluation time) as a search service
    with `slots` slots: `search(slot, state)`, collect/apply, `moves()`."""
    if cfg.algo == 'az':
        return az.make_engine(cfg, seed, sims, slots=slots, service=True)
    return _core.Engine(
        cfg.game, num_games=slots, groups=1, seed=seed, sims=sims, wave=cfg.wave,
        search=cfg.search_rule, target=cfg.target_rule, rho_search=cfg.rho_search,
        rho_target=cfg.rho_target, virtual_loss=cfg.virtual_loss,
        descent_temp=eval_descent_temp(cfg), max_plies=cfg.max_plies or None, service=True)


def opening_states(game, games, openings=None, opening_plies=2, seed=0):
    """Start positions: game k from openings[(k // 2) % len] (action id
    sequences), or `opening_plies` random plies shared by games 2j, 2j+1."""
    rng = random.Random(seed)
    out = []
    for k in range(games):
        if k % 2:
            out.append(out[-1].clone())
            continue
        while True:
            st = _core.GameState(game)
            seq = openings[(k // 2) % len(openings)] if openings else None
            for i in range(len(seq) if seq else opening_plies):
                st.apply(seq[i] if seq else rng.choice(st.legal_actions()))
                if st.is_terminal():
                    break
            if not st.is_terminal():
                break
            if seq:
                raise ValueError(f'opening {(k // 2) % len(openings)} ends the game')
        out.append(st)
    return out


def play_games(sides, games, sims, openings=None, opening_plies=2, seed=0, workers=16):
    """Play `games` games between sides[0] and sides[1]; side 0 is player
    k % 2 in game k.  A side is a (cfg, evaluator) pair, whose move is its
    own evaluation-time pick from a fresh `sims`-simulation search, or a
    blocking callable state -> action (an external player such as Edax),
    called from up to `workers` threads.  Forced moves are played without
    asking either.  Returns [(game, side 0's player, result for side 0)]."""
    game = next(s[0].game for s in sides if isinstance(s, tuple))
    states = opening_states(game, games, openings, opening_plies, seed)
    engines = [search_engine(s[0], games, sims, seed + i) if isinstance(s, tuple) else None
               for i, s in enumerate(sides)]
    pool = (cf.ThreadPoolExecutor(max_workers=min(workers, games))
            if None in engines else None)
    busy = {}          # game -> side to move, while it is being decided
    inflight = {}      # game -> external move future
    live = set(range(games))
    results = []
    try:
        while live:
            for k in sorted(live - busy.keys()):
                st = states[k]
                while not st.is_terminal() and len(st.legal_actions()) == 1:
                    st.apply(st.legal_actions()[0])
                if st.is_terminal():
                    results.append((k, k % 2, int(st.returns()[k % 2])))
                    live.discard(k)
                    continue
                side = 0 if st.current_player() == k % 2 else 1
                if engines[side] is None:
                    inflight[k] = pool.submit(sides[side], st.clone())
                else:
                    engines[side].search(k, st)
                busy[k] = side
            # one wave of each engine; the second's tree work overlaps the
            # first's batch on the GPU
            handles, rows = [], 0
            for i, eng in enumerate(engines):
                if eng is not None:
                    obs, _ = eng.collect(0)
                    rows += len(obs)
                    handles.append((eng, sides[i][1], sides[i][1].submit(obs)))
            for eng, ev, h in handles:
                eng.apply(0, *ev.result(h))
                for k, a in eng.moves():
                    states[k].apply(a)
                    del busy[k]
            if inflight and rows == 0:
                cf.wait(list(inflight.values()), timeout=0.05, return_when=cf.FIRST_COMPLETED)
            for k, fut in list(inflight.items()):
                if fut.done():
                    del inflight[k]
                    states[k].apply(fut.result())
                    del busy[k]
    finally:
        if pool is not None:
            pool.shutdown(cancel_futures=True)
    return sorted(results)


def play_pair(sides, games, sims, openings=None, opening_plies=2, seed=0):
    """`play_games` between two (cfg, evaluator) sides; returns (wins,
    draws, losses) for side 0."""
    res = [r for _, _, r in play_games(sides, games, sims, openings, opening_plies, seed)]
    w, l = sum(r > 0 for r in res), sum(r < 0 for r in res)
    return w, len(res) - w - l, l


def bradley_terry(pairs, iters=2000, prior_games=1.0):
    """Elo-scale ratings (mean 0) from {(a, b): (w, d, l)}.  Every player gets
    `prior_games` virtual draws against an average opponent so undefeated or
    winless players stay finite."""
    players = sorted({p for ab in pairs for p in ab})
    if not players:
        return {}
    ix = {p: i for i, p in enumerate(players)}
    n = len(players)
    score = np.zeros((n, n))
    for (a, b), (w, d, l) in pairs.items():
        i, j = ix[a], ix[b]
        score[i, j] += w + 0.5 * d
        score[j, i] += l + 0.5 * d
    games = score + score.T
    g = np.ones(n)
    for _ in range(iters):
        num = score.sum(1) + 0.5 * prior_games
        den = (games / (g[:, None] + g[None, :])).sum(1) + prior_games / (g + 1.0)
        new = num / den
        new /= np.exp(np.log(new).mean())
        if np.max(np.abs(new - g)) < 1e-10:
            g = new
            break
        g = new
    return {p: 400.0 * math.log10(g[ix[p]]) for p in players}


class Ladder:
    """Per-pair results stored in `<run_dir>/ladder.json`."""

    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, 'ladder.json')
        self.pairs = {}
        if os.path.exists(self.path):
            with open(self.path) as f:
                self.pairs = {tuple(k.split('|')): v for k, v in json.load(f).items()}

    def add(self, a, b, w, d, l):
        cur = self.pairs.get((a, b), [0, 0, 0])
        self.pairs[(a, b)] = [cur[0] + w, cur[1] + d, cur[2] + l]
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump({'|'.join(k): v for k, v in self.pairs.items()}, f, indent=1)
        os.replace(tmp, self.path)

    def rated(self):
        return {p for ab in self.pairs for p in ab}

    def ratings(self):
        return bradley_terry(self.pairs)
