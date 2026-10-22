"""End-to-end checks of the Python side (CPU only, a few seconds each).

The game rules, belief maths and tree are unit-tested in Rust (`cargo test`).
"""
import json
import os

import numpy as np
import pytest
import torch

from thompsonzero import AzEngine, Engine, GameState, preset
from thompsonzero.edax import DEFAULT_DIR as EDAX_DIR
from thompsonzero.arena import bradley_terry, play_match, play_pair
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


# ── AlphaZero baseline ──────────────────────────────────────────────────────

def az_uniform(n, num_actions):
    return np.zeros(n, np.float32), np.zeros((n, num_actions), np.float32)


@pytest.mark.parametrize('game', GAMES)
def test_az_selfplay_samples(game):
    eng = AzEngine(game, num_games=8, groups=2, sims=8, wave=4)
    A = eng.num_actions
    while eng.stats()['games'] < 2:
        for k in range(2):
            obs, nets = eng.collect(k)
            assert obs.shape[1:] == eng.obs_shape and (nets == 0).all()
            eng.apply(k, *az_uniform(len(obs), A))
    s = eng.drain()
    n = len(s['obs'])
    assert n > 0 and s['pi'].shape == (n, A)
    assert np.allclose(s['pi'].sum(1), 1.0, atol=1e-4)
    assert (np.abs(s['z']) <= 1).all() and (s['z_w'] == 1).all()
    with pytest.raises(ValueError):
        eng.collect(0)
        eng.apply(0, *az_uniform(10_000, A))   # wrong number of rows


def test_az_arena_and_external_play():
    from thompsonzero import az
    cfg = preset('othello', 'az', channels=8, blocks=1)
    g = GameState(cfg.game)
    evs = [Evaluator(build(cfg, g.obs_shape, g.num_actions), 'cpu', amp=False) for _ in range(2)]
    w, d, l = az.play_match(cfg, evs, games=6, sims=4)
    assert w + d + l == 6
    opening = [[37, 43, 26, 19]]
    with pytest.raises(ValueError):
        AzEngine('othello', arena_games=2, external=True, openings=[[0]])  # illegal opening
    eng = az.make_engine(cfg, 0, 4, arena_games=4, external=True, openings=opening)
    rng = np.random.default_rng(0)
    while not eng.done:
        obs, _ = eng.collect(0)
        eng.apply(0, *evs[0].evaluate(obs))
        for slot, st in eng.waiting():
            assert st.current_player() != slot % 2   # net 0 plays black in even games
            eng.play_external(slot, int(rng.choice(st.legal_actions())))
    res = eng.results()
    assert sorted(r[0] for r in res) == [0, 1, 2, 3]
    assert [r[1] for r in sorted(res)] == [0, 1, 0, 1]


def test_az_augment_moves_obs_and_policy_together():
    from thompsonzero.symmetry import augment
    n = 64
    obs = torch.zeros(n, 3, 8, 8)
    pi = torch.zeros(n, 65)
    sq = torch.arange(n) % 64
    obs[torch.arange(n), 0, sq // 8, sq % 8] = 1.0
    pi[torch.arange(n), sq] = 0.75
    pi[:, 64] = 0.25
    out = augment({'obs': obs, 'pi': pi, 'z': torch.zeros(n)}, 'othello')
    assert torch.equal(out['obs'][:, 0].reshape(n, 64).argmax(1), out['pi'][:, :64].argmax(1))
    assert torch.all(out['pi'][:, 64] == 0.25) and torch.allclose(out['pi'].sum(1), torch.ones(n))
    # the start position is invariant under the diagonal reflections only
    g = GameState('othello')
    start = torch.from_numpy(g.observation())[None].expand(256, -1, -1, -1)
    aug = augment({'obs': start, 'pi': torch.zeros(256, 65)}, 'othello')['obs']
    same = (aug == start).flatten(1).all(1).float().mean()
    assert 0.3 < same < 0.7


def test_tz_augment_keeps_samples_consistent():
    from thompsonzero.symmetry import augment
    eng = Engine('othello', num_games=8, groups=1, sims=8, wave=4)
    while eng.stats()['games'] < 2:
        obs, _ = eng.collect(0)
        eng.apply(0, *uniform(len(obs), eng.num_actions))
    b = {k: torch.from_numpy(v) for k, v in eng.drain().items()}
    n = len(b['obs'])
    for k in range(8):
        out = augment(b, 'othello', k=torch.full((n,), k))
        empty = out['obs'][:, 2].reshape(n, 64) > 0
        assert not (out['legal'][:, :64] & ~empty).any()      # legal squares are empty
        assert torch.equal(out['legal'].sum(1), b['legal'].sum(1))
        assert not (out['ev_mask'] & ~out['legal']).any()
        rows = torch.nonzero((out['played'] >= 0) & (out['played'] < 64)).squeeze(1)
        p = out['played'][rows]
        assert out['legal'][rows, p].all()
        # after the move the played square holds the (new) opponent's disc
        nxt = rows[b['has_next'][rows]]
        assert (out['next_obs'][nxt, 1].reshape(-1, 64)[torch.arange(len(nxt)), out['played'][nxt]] == 1).all()
        assert torch.equal(out['ev'][rows, p], b['ev'][rows, b['played'][rows]])
    assert torch.equal(augment(b, 'othello', k=torch.zeros(n, dtype=torch.long))['played'], b['played'])


def test_play_pair_mixes_algorithms():
    sides = []
    for rule in ('az', 'tz'):
        cfg = preset('othello', rule, channels=8, blocks=1)
        g = GameState('othello')
        sides.append((cfg, Evaluator(build(cfg, g.obs_shape, g.num_actions), 'cpu', False)))
    w, d, l = play_pair(sides, 6, sims=8, openings=[[37, 43], [37, 45], [37, 29]])
    assert w + d + l == 6
    w, d, l = play_pair(sides[::-1], 4, sims=0, opening_plies=3)
    assert w + d + l == 4


def test_league_round_robin(tmp_path):
    import argparse
    from thompsonzero import league
    g = GameState('othello')
    for rule, labels in (('az', (0, 10, 21)), ('tz', (0, 12))):
        cfg = preset('othello', rule, channels=8, blocks=1, episodes=21)
        d = tmp_path / rule
        d.mkdir()
        (d / 'config.json').write_text(cfg.to_json())
        net = build(cfg, g.obs_shape, g.num_actions)
        for lab in labels:
            torch.save({'model': net.state_dict(), 'config': json.loads(cfg.to_json())},
                       d / f'bench_{lab}.pt')
    assert league.milestones(str(tmp_path / 'az'), 10) == {10: 10, 20: 21}
    xot = tmp_path / 'xot.txt'
    xot.write_text('f5d6c4d3c2b3b4b5\nf5f4g3g6f3g4e3e2\n')
    a = argparse.Namespace(dir=str(tmp_path / 'lg'), player=[f'az={tmp_path / "az"}', f'tz={tmp_path / "tz"}'],
                           every=10, sims=4, openings=2, seed=0, anchor='az@20', edax_dir='',
                           xot=str(xot), device='cpu', watch=False)
    league.run(a)
    lg = league.League(str(tmp_path / 'lg' / 'league.json'), dict(sims=4, openings=2, seed=0, every=10))
    assert set(lg.players) == {'az@10', 'az@20', 'tz@10'}
    assert len(lg.pairs) == 3 and all(sum(v) == 4 for v in lg.pairs.values())
    rows = lg.table('az@20')
    assert dict((p, r) for p, r, *_ in rows)['az@20'] == 0.0
    league.run(a)                       # resumes: nothing left to play
    assert len(league.League(lg.path, lg.settings).pairs) == 3


def test_az_training_runs_and_resumes(tmp_path):
    kw = dict(run_dir=str(tmp_path), episodes=24, channels=8, blocks=1,
              num_games=16, batch_size=32, buffer_size=2000, sims_ramp=((0.0, 8),),
              log_every=8, checkpoint_every=12, benchmark_every=24, quick_eval_games=4,
              train_steps_per_episode=1.0, device='cpu')
    train(preset('othello', 'az', **kw))
    ck = torch.load(tmp_path / 'latest.pt', weights_only=False)
    assert ck['episode'] >= 24 and ck['steps'] > 0
    assert (tmp_path / 'buffer.pt').exists() and (tmp_path / 'bench_0.pt').exists()
    kw['episodes'] = 40
    train(preset('othello', 'az', **kw))
    assert torch.load(tmp_path / 'latest.pt', weights_only=False)['steps'] > ck['steps']


def test_edax_board_and_moves():
    from thompsonzero.edax import parse_moves
    assert parse_moves('f5d6c4d3') == [37, 43, 26, 19]
    g = GameState('othello')
    assert g.edax_board() == '-' * 27 + 'OX------XO' + '-' * 27 + ' X'
    for a in parse_moves('f5d6'):
        g.apply(a)
    board, side = g.edax_board().split()
    assert side == 'X' and board.count('X') == 3 and board.count('O') == 3


@pytest.mark.skipif(not os.path.exists(os.path.join(EDAX_DIR, 'data', 'eval.dat')),
                    reason='Edax not installed')
def test_edax_plays_a_full_game():
    from thompsonzero.edax import Edax
    e = Edax(1, EDAX_DIR)
    g = GameState('othello')
    rng = np.random.default_rng(0)
    try:
        while not g.is_terminal():
            a = e.move(g) if g.current_player() == 0 else int(rng.choice(g.legal_actions()))
            g.apply(a)
    finally:
        e.close()
    assert g.returns()[0] == 1   # Edax as black beats a random mover


@pytest.mark.skipif(not os.path.exists(os.path.join(EDAX_DIR, 'data', 'eval.dat')), reason='no Edax')
@pytest.mark.parametrize('rule', ['az', 'tz'])
def test_edax_match_any_algorithm(rule):
    from thompsonzero.edax import parse_moves, play
    cfg = preset('othello', rule, channels=8, blocks=1)
    g = GameState('othello')
    net = build(cfg, g.obs_shape, g.num_actions)
    ops = [parse_moves('f5d6c4d3c2b3b4b5'), parse_moves('f5f4g3g6f3g4e3e2')]
    res = play(net, cfg, 'cpu', 1, 4, ops, workers=2)
    assert [(k, p) for k, p, _ in res] == [(0, 0), (1, 1), (2, 0), (3, 1)]
    assert all(r in (-1, 0, 1) for *_, r in res)


def test_replay_window_samples_only_the_newest_rows():
    from thompsonzero.replay import AZ_SCHEMA, ReplayBuffer
    schema = {k: AZ_SCHEMA[k] for k in ('obs', 'pi', 'z', 'z_w')}
    buf = ReplayBuffer(10, (3, 8, 8), 65, 'cpu', schema)
    for start in (0, 6, 12):   # 18 rows through a 10-row ring
        n = 6
        buf.add({'obs': np.zeros((n, 3, 8, 8), np.float32), 'pi': np.zeros((n, 65), np.float32),
                 'z': np.arange(start, start + n, dtype=np.float32), 'z_w': np.ones(n, np.float32)})
    assert len(buf) == 10
    assert set(buf.sample(500, window=3)['z'].tolist()) == {15.0, 16.0, 17.0}
    assert set(buf.sample(2000)['z'].tolist()) == set(range(8, 18))


def test_split_kl_trains_mean_and_concentration_apart():
    from thompsonzero.loss import split_kl
    t = torch.tensor([[60.0, 10.0, 30.0]])
    for c_raw in (2.0, 50.0):
        raw = torch.tensor([[0.3, -0.2, 0.1, c_raw]], requires_grad=True)
        split_kl(t, raw).sum().backward()
        # the mean sees only the cross-entropy to the target mean: p - m,
        # whatever the predicted concentration
        p = torch.softmax(raw.detach()[0, :3], -1)
        assert torch.allclose(raw.grad[0, :3], p - t[0] / t.sum(), atol=1e-5)
        assert raw.grad[0, 3] != 0
    # an outcome the search ruled out (~1e-9) must not blow the KL up
    tiny = torch.tensor([[5.0, 1e-9, 3.0]])
    assert split_kl(tiny, torch.tensor([[0.5, -2.0, 0.3, 20.0]])).item() < 100


def test_cal_channel_learns_on_the_side():
    """The calibrated channel is a fifth number per belief; its loss never
    reaches the trunk or the means, and `use_cal` swaps it into the engine
    format."""
    from thompsonzero import loss
    from thompsonzero.network import DirichletNet
    torch.manual_seed(0)
    net = DirichletNet((3, 8, 8), 65, channels=8, blocks=1, cal=True)
    x = torch.randn(4, 3, 8, 8)
    v, a = net(x)
    assert v.shape == (4, 5) and a.shape == (4, 65, 5)
    t = torch.tensor([[60.0, 10.0, 30.0]]).expand(4, 3)
    loss.cal_kl(t, v, 1000.0).sum().backward()
    assert net.v_cal.weight.grad.abs().sum() > 0
    assert all(p.grad is None or p.grad.abs().sum() == 0
               for n, p in net.named_parameters() if not n.startswith(('v_cal', 'a_cal')))
    with torch.no_grad():
        v, a = net(x)
        own = net.postprocess(v, a)[0][:, 3]
        net.use_cal = True
        cal = net.postprocess(v, a)[0][:, 3]
    assert torch.allclose(cal, torch.exp(v[:, 4])) and not torch.allclose(own, cal)


def test_tz_preset_matches_use_the_calibrated_search(tmp_path):
    """The tz preset learns the calibrated channel on the side: self-play
    leaves it off, a saved net loads with it on, and matches search with the
    eval rule."""
    from thompsonzero.arena import eval_search_rule
    from thompsonzero.train import load_net
    cfg = preset('othello', 'tz', channels=8, blocks=1)
    g = GameState('othello')
    net = build(cfg, g.obs_shape, g.num_actions)
    assert net.cal and not net.use_cal and eval_search_rule(cfg) == 'point_400'
    path = str(tmp_path / 'bench.pt')
    torch.save(dict(config=json.loads(cfg.to_json()), model=net.state_dict()), path)
    loaded, lcfg = load_net(path)
    assert loaded.use_cal
    evs = [Evaluator(loaded, 'cpu', amp=False)] * 2
    w, d, l = play_match(lcfg, evs, games=4, sims=16)
    assert w + d + l == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs CUDA')
def test_evaluator_first_batch_sees_the_weights():
    """A fresh Evaluator's first batch must see the copied weights: the copy
    and cast run on the current stream, inference on the Evaluator's own."""
    from thompsonzero.inference import Evaluator
    from thompsonzero.network import DirichletNet
    obs = np.random.default_rng(0).standard_normal((256, 3, 8, 8)).astype(np.float32)
    for seed in range(3):
        torch.manual_seed(seed)
        net = DirichletNet((3, 8, 8), 65, channels=128, blocks=10, trunk='az').cuda().eval()
        ref = Evaluator(net, 'cuda', amp=True)
        torch.cuda.synchronize()
        want = ref.evaluate(obs)[0]
        got = Evaluator(net, 'cuda', amp=True).evaluate(obs)[0]
        assert np.array_equal(got, want)


def test_wdl_value_head():
    """value_head=wdl: three logits per position, trained by cross-entropy to
    (win, draw, loss) with mean z, and handed to the search as p_win - p_loss."""
    from thompsonzero import az
    cfg = preset('othello', 'az', channels=8, blocks=1, value_head='wdl')
    g = GameState('othello')
    net = build(cfg, g.obs_shape, g.num_actions).eval()
    x = torch.randn(4, *g.obs_shape)
    v, logits = net(x)
    assert v.shape == (4, 3)
    sv, _, d = net.postprocess(v, logits)
    p = torch.softmax(v, -1)
    assert torch.allclose(sv, p[:, 0] - p[:, 2]) and torch.allclose(d, p[:, 1])
    z = torch.tensor([1.0, 0.0, -1.0, 0.5])
    assert torch.allclose(az.wdl_target(z), torch.tensor(
        [[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [0.5, 0.5, 0]]))
    b = dict(z=z, z_w=torch.ones(4), pi=torch.full((4, g.num_actions), 1 / g.num_actions))
    total, parts = az.compute(v, logits, b)
    total.backward()
    assert torch.isfinite(total) and parts['value'] > 0
    evs = [Evaluator(net, 'cpu', amp=False)] * 2
    w, d, l = az.play_match(cfg, evs, games=4, sims=16)
    assert w + d + l == 4


def test_az_thompson_selection():
    """az_select=thompson searches with Thompson draws from the WDL head's
    draw probability, az_select=gaussian with normal draws (after ts_from of
    each search by PUCT); matches and self-play run unchanged around them."""
    from thompsonzero import az
    for head, select, ts_from in [('wdl', 'thompson', 0.0), ('scalar', 'thompson', 0.0),
                                  ('scalar', 'gaussian', 0.0), ('scalar', 'gaussian', 0.5)]:
        cfg = preset('othello', 'az', channels=8, blocks=1, value_head=head,
                     az_select=select, ts_c=0.5, ts_from=ts_from, num_games=8, groups=2)
        g = GameState('othello')
        net = build(cfg, g.obs_shape, g.num_actions).eval()
        evs = [Evaluator(net, 'cpu', amp=False)] * 2
        w, d, l = az.play_match(cfg, evs, games=4, sims=32)
        assert w + d + l == 4
        eng = az.make_engine(cfg, 0, 16)
        for _ in range(40):
            for k in range(eng.groups):
                obs, _ = eng.collect(k)
                eng.apply(k, *evs[0].evaluate(obs))
        assert eng.stats()['sims'] > 0


def test_az_mixmax_and_q_pick():
    """mixmax / pick_frac change the backup and the move played; matches and
    self-play run unchanged around them."""
    from thompsonzero import az
    cfg = preset('othello', 'az', channels=8, blocks=1, mixmax=0.25, pick_frac=0.2,
                 num_games=8, groups=2, temp_threshold=2)
    g = GameState('othello')
    net = build(cfg, g.obs_shape, g.num_actions).eval()
    evs = [Evaluator(net, 'cpu', amp=False)] * 2
    w, d, l = az.play_match(cfg, evs, games=4, sims=32)
    assert w + d + l == 4
    eng = az.make_engine(cfg, 0, 16)
    for _ in range(40):
        for k in range(eng.groups):
            obs, _ = eng.collect(k)
            eng.apply(k, *evs[0].evaluate(obs))
    assert eng.stats()['sims'] > 0


def test_az_auxiliary_targets():
    """Generalised doubling, z/q blending and every auxiliary head: samples
    carry the targets, the buffer and augmentation keep them, the loss is
    finite and trains, and inference still sees (v, logits)."""
    from thompsonzero import az, symmetry
    from thompsonzero.train import make_buffer
    cfg = preset('othello', 'az', channels=8, blocks=1, num_games=8, groups=2,
                 double_aux=4, double_min_visits=2, aux_depth=2, aux_pow=1.0, q_lambda=0.5,
                 q_blend=0.5, aux_to_main=False, q_head=1.0, stv_head=1.0, own_head=1.0,
                 reply_head=0.15, var_head=1.0, buffer_size=4096)
    g = GameState('othello')
    net = build(cfg, g.obs_shape, g.num_actions)
    evs = [Evaluator(net.eval(), 'cpu', amp=False)] * 2
    eng = az.make_engine(cfg, 0, 16)
    while eng.stats()['games'] < 2:
        for k in range(eng.groups):
            obs, _ = eng.collect(k)
            eng.apply(k, *evs[0].evaluate(obs))
    s = eng.drain()
    main = s['main'] > 0
    assert main.any() and (~main).any()
    assert (s['own_w'][main] == 1).all() and (s['reply_w'][~main] == 0).all()
    assert np.allclose(s['reply'][s['reply_w'] > 0].sum(-1), 1, atol=1e-3)
    buf = make_buffer(cfg, g.obs_shape, g.num_actions, 'cpu')
    buf.add(s)
    b = symmetry.augment(buf.sample(64), 'othello')
    net.train()
    v, logits, extra = net(b['obs'], aux=True)
    assert set(extra) == {'q', 'stv', 'own', 'score', 'reply', 'var'}
    total, parts = az.compute(v, logits, b, 1.0, extra, cfg)
    total.backward()
    assert torch.isfinite(total) and all(torch.isfinite(x) for x in parts.values())
    assert len(net.eval()(b['obs'])) == 2
