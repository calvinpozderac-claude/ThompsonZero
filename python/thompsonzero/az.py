"""The AlphaZero baseline: engine factory, loss, matches.

This is AlphaGo Zero as OLIVAW (arXiv 2103.17228) ran it on Othello: PUCT
search over a policy + value network, self-play with root Dirichlet noise
(alpha = min(1, 10 / legal moves), eps 0.25), moves sampled by visit count
for the first 20 plies and the most visited after, OLIVAW's doubling, and
the loss

    L = (z - v)^2 - pi . log p + c ||theta||^2         (c in the optimizer)

or, with `value_head=wdl`, a win/draw/loss cross-entropy in place of the
squared error (see `compute`).

trained with SGD + momentum 0.9 on a step schedule (OLIVAW's shape, 30x its
LR, and 3 steps per game rather than ~6: see the `az` entry of RULES).
Training positions are augmented with a random one of the eight board
symmetries, as AlphaGo Zero did (`symmetry.augment`).
"""
import torch

from . import _core
from .inference import evaluate_split


def make_engine(cfg, seed, sims, slots=None, **arena):
    """`arena`: arena_games / external / openings / opening_plies for
    matches, or service=True for a search service with `slots` slots;
    without it, a self-play engine."""
    kw = dict(num_games=cfg.num_games, groups=cfg.groups, wave=cfg.wave)
    if arena.get('arena_games'):
        kw.update(num_games=min(arena['arena_games'], cfg.num_games), groups=1)
    if slots:
        kw.update(num_games=slots, groups=1)
    return _core.AzEngine(
        cfg.game, seed=seed, sims=sims, c_puct=cfg.c_puct, fpu_reduction=cfg.fpu_reduction,
        virtual_loss=cfg.virtual_loss, noise_alpha=cfg.noise_alpha, noise_frac=cfg.noise_frac,
        temp_threshold=cfg.temp_threshold, max_plies=cfg.max_plies or None,
        reuse_tree=cfg.reuse_tree, double_aux=cfg.double_aux,
        double_min_visits=int(cfg.double_min_visits), **kw, **arena)


def wdl_target(z):
    """z in [-1, 1] -> (win, draw, loss) with mean z: exact for game results,
    and a doubled sample's q splits between draw and its sign."""
    return torch.stack([z.clamp_min(0), 1 - z.abs(), (-z).clamp_min(0)], -1)


def compute(v, logits, b, value_weight=1.0):
    """Mean (z - v)^2 over valued rows -- or with (B, 3) win/draw/loss
    logits, their cross-entropy to `wdl_target(z)` -- plus the policy
    cross-entropy over rows with a search target.  Returns (total, dict of
    detached scalars)."""
    zw = b['z_w']
    z = b['z'].float()
    if v.dim() == 2:
        logq = torch.log_softmax(v.float(), dim=-1)
        err = -(wdl_target(z) * logq).sum(-1)
        v = logq.exp()[:, 0] - logq.exp()[:, 2]
    else:
        err = (v.float() - z) ** 2
    mse = (zw * err).sum() / zw.sum().clamp_min(1.0)
    pi = b['pi']
    has_pi = (pi.sum(-1) > 0).float()
    logp = torch.log_softmax(logits.float(), dim=-1)
    ce = -(pi * logp).sum(-1)
    policy = (has_pi * ce).sum() / has_pi.sum().clamp_min(1.0)
    total = value_weight * mse + policy
    with torch.no_grad():
        # target entropy: the irreducible part of `policy`
        ent = -(pi * torch.log(pi.clamp_min(1e-12))).sum(-1)
        parts = {'loss': total.detach(), 'value': mse.detach(), 'policy': policy.detach(),
                 'pi_entropy': (has_pi * ent).sum() / has_pi.sum().clamp_min(1.0),
                 'v_mean_abs': v.float().abs().mean()}
    return total, parts


def play_match(cfg, evaluators, games, sims, seed=0, opening_plies=2, openings=None):
    """Play `games` between evaluators[0] and evaluators[1] (most visited
    move of a fresh `sims`-simulation search, no noise); returns (wins,
    draws, losses) for evaluators[0]."""
    eng = make_engine(cfg, seed, sims, arena_games=games, opening_plies=opening_plies,
                      openings=openings)
    while not eng.done:
        obs, nets = eng.collect(0)
        eng.apply(0, *evaluate_split(evaluators, obs, nets))
    s = eng.stats()
    return s['wins'], s['games'] - s['wins'] - s['losses'], s['losses']
