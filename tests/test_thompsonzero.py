"""End-to-end checks of the Python side (CPU only, a few seconds each).

The game rules, belief maths and tree are unit-tested in Rust (`cargo test`).
"""
import numpy as np
import pytest
import torch

from thompsonzero import Engine, GameState, preset
from thompsonzero.arena import bradley_terry, play_match
from thompsonzero.inference import Evaluator
from thompsonzero.loss import dirichlet_kl
from thompsonzero.network import build
from thompsonzero.train import train

GAMES = ['connect_four', 'othello', 'boop']


def uniform(n, num_actions):
    v = np.tile(np.float32([1 / 3, 1 / 3, 1 / 3, 1.5]), (n, 1))
    return v, np.broadcast_to(v[:, None, :], (n, num_actions, 4)).copy()


@pytest.mark.parametrize('game', GAMES)
def test_game_state(game):
    g = GameState(game)
    assert g.observation().shape == g.obs_shape
    legal = g.legal_actions()
    assert legal and max(legal) < g.num_actions
    with pytest.raises(ValueError):
        g.apply(g.num_actions + 5)
    rng = np.random.default_rng(0)
    while not g.is_terminal():
        g.apply(int(rng.choice(g.legal_actions())))
    assert g.returns()[0] == -g.returns()[1]


@pytest.mark.parametrize('game', GAMES)
@pytest.mark.parametrize('rules', [('additive_mle', 'avgconc'), ('corr', 'corr')])
def test_selfplay_samples(game, rules):
    eng = Engine(game, num_games=8, groups=2, sims=8, wave=4, search=rules[0], target=rules[1])
    A = eng.num_actions
    while eng.stats()['games'] < 2:
        for k in range(2):
            obs, nets = eng.collect(k)
            assert obs.shape[1:] == eng.obs_shape and (nets == 0).all()
            eng.apply(k, *uniform(len(obs), A))
    s = eng.drain()
    n = len(s['obs'])
    assert n > 0 and s['ev'].shape == (n, A, 3) and s['legal'].shape == (n, A)
    assert not (s['ev_mask'] & ~s['legal']).any()
    played = s['played'][s['played'] >= 0]
    assert s['legal'][s['played'] >= 0, played].all()
    assert (s['has_next'] != (s['next_term'] >= 0))[s['played'] >= 0].all()
    assert np.isfinite(s['v']).all() and (s['v'] > 0).all()
    with pytest.raises(ValueError):
        eng.collect(0)
        eng.apply(0, *uniform(10_000, A))   # wrong number of rows


def test_arena_counts_every_game():
    cfg = preset('connect_four', 'tz', channels=8, blocks=1)
    g = GameState(cfg.game)
    nets = [build(cfg, g.obs_shape, g.num_actions) for _ in range(2)]
    evs = [Evaluator(n, 'cpu', amp=False) for n in nets]
    w, d, l = play_match(cfg, evs, games=6, sims=4)
    assert w + d + l == 6


def test_dirichlet_kl():
    a = torch.tensor([[2.0, 1.0, 0.5], [30.0, 0.1, 0.1]])
    b = torch.tensor([[1.0, 1.0, 1.0], [5.0, 5.0, 5.0]])
    assert torch.allclose(dirichlet_kl(a, a), torch.zeros(2), atol=1e-5)
    kl = dirichlet_kl(a, b)
    assert (kl > 0).all()
    # Monte-Carlo check of the closed form: E_a[log p_a(x) - log p_b(x)]
    da, db = torch.distributions.Dirichlet(a[0]), torch.distributions.Dirichlet(b[0])
    x = da.sample((200_000,))
    mc = (da.log_prob(x) - db.log_prob(x)).mean()
    assert abs(float(mc) - float(kl[0])) < 0.02


def test_bradley_terry():
    r = bradley_terry({('a', 'b'): (5, 0, 5)})
    assert abs(r['a'] - r['b']) < 1e-6
    r = bradley_terry({('a', 'b'): (9, 0, 1), ('b', 'c'): (9, 0, 1)})
    assert r['a'] > r['b'] > r['c']


def test_training_runs_and_resumes(tmp_path):
    kw = dict(run_dir=str(tmp_path), episodes=24, channels=8, blocks=1, head_channels=4,
              num_games=16, batch_size=32, buffer_size=2000, sims_ramp=((0.0, 8),),
              log_every=8, checkpoint_every=12, benchmark_every=24, quick_eval_games=4,
              train_steps_per_episode=1.0, device='cpu')
    train(preset('connect_four', 'tz', **kw))
    ck = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert ck['episode'] >= 24 and ck['steps'] > 0
    assert (tmp_path / 'buffer.pt').exists() and (tmp_path / 'bench_0.pt').exists()
    kw['episodes'] = 40
    train(preset('connect_four', 'tz', **kw))
    assert torch.load(tmp_path / 'latest.pt', weights_only=False)['steps'] > ck['steps']
