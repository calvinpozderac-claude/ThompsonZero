"""Matches between networks, and a Bradley-Terry ladder over checkpoints.

A match runs the same Rust engine in arena mode: both sides search with
`sims` simulations using their own network, play the greedy posterior move,
and alternate colours; a few random opening plies give the games variety.
With sims=0 it is a search-free match read straight off the action heads.

Ratings are always refitted from the stored per-pair W/D/L (Bradley-Terry,
draws as half points), never accumulated incrementally.
"""
import json
import math
import os

import numpy as np

from . import _core
from .inference import evaluate_split


def play_match(cfg, evaluators, games, sims, seed=0, opening_plies=2):
    """Play `games` between evaluators[0] and evaluators[1]; returns
    (wins, draws, losses) for evaluators[0]."""
    eng = _core.Engine(
        cfg.game, num_games=min(games, 512), groups=1, seed=seed, sims=sims, wave=4,
        search=cfg.search_rule, target=cfg.target_rule, rho_search=cfg.rho_search,
        rho_target=cfg.rho_target, virtual_loss=cfg.virtual_loss, descent_temp=cfg.descent_temp,
        max_plies=cfg.max_plies or None, arena_games=games, opening_plies=opening_plies)
    while not eng.done:
        obs, nets = eng.collect(0)
        eng.apply(0, *evaluate_split(evaluators, obs, nets, eng.num_actions))
    s = eng.stats()
    return s['wins'], s['games'] - s['wins'] - s['losses'], s['losses']


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
