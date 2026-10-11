"""Command line: train, ladder, bench.

    thompsonzero train --game othello --rule corr
    thompsonzero train --run-dir runs/othello_corr          # resume
    thompsonzero train --game boop --set num_games=2048 --set wave=2
    thompsonzero ladder --run-dir runs/othello_corr --watch
    thompsonzero bench --game othello
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from . import _core
from .arena import Ladder, play_match
from .config import Config, preset
from .inference import Evaluator
from .network import build
from .train import load_net, make_step, pick_device, train
from .replay import ReplayBuffer


def _overrides(pairs):
    out = {}
    for p in pairs or ():
        k, _, v = p.partition('=')
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_train(a):
    over = _overrides(a.set)
    cfg_path = os.path.join(a.run_dir or '', 'config.json')
    if a.run_dir and os.path.exists(cfg_path) and not a.game:
        with open(cfg_path) as f:
            d = json.load(f)
        d.update(over)
        cfg = Config.from_dict(d)
    else:
        if not a.game:
            raise SystemExit('--game is required for a new run')
        if a.run_dir:
            over['run_dir'] = a.run_dir
        cfg = preset(a.game, a.rule, **over)
    train(cfg)


def cmd_ladder(a):
    ladder = Ladder(a.run_dir)
    device = pick_device(a.device)
    while True:
        labels = sorted(int(f[6:-3]) for f in os.listdir(a.run_dir)
                        if f.startswith('bench_') and f.endswith('.pt'))
        new = [x for x in labels if str(x) not in ladder.rated()]
        if len(labels) < 2 and not a.watch:
            print('need at least two benchmarks')
            return
        for lab in new:
            older = [x for x in labels if x < lab][-a.opponents:]
            if not older:
                continue
            net, cfg = load_net(os.path.join(a.run_dir, f'bench_{lab}.pt'), device)
            for opp in older:
                ref, _ = load_net(os.path.join(a.run_dir, f'bench_{opp}.pt'), device)
                evs = [Evaluator(net, device, cfg.amp), Evaluator(ref, device, cfg.amp)]
                w, d, l = play_match(cfg, evs, a.games, a.sims, seed=lab + opp)
                ladder.add(str(lab), str(opp), w, d, l)
                print(f'{lab} vs {opp}: W{w} D{d} L{l}', flush=True)
            r = ladder.ratings()
            print('ratings: ' + '  '.join(f'{k}={v:.0f}' for k, v in
                                          sorted(r.items(), key=lambda kv: int(kv[0]))), flush=True)
        if not a.watch:
            return
        time.sleep(60)


def cmd_bench(a):
    """Throughput of each stage in isolation, to size num_games/wave/batch."""
    cfg = preset(a.game, a.rule)
    device = pick_device(a.device)
    g = _core.GameState(cfg.game)
    A = g.num_actions

    # 1) engine alone, with a constant (uniform) network: pure CPU cost
    eng = _core.Engine(cfg.game, num_games=a.games, groups=2, sims=a.sims, wave=cfg.wave,
                       search=cfg.search_rule, target=cfg.target_rule)
    t0, rows, calls = time.time(), 0, 0
    pend = [None, None]
    while time.time() - t0 < a.seconds:
        for k in range(2):
            if pend[k] is not None:
                eng.apply(k, *pend[k])
            obs, _ = eng.collect(k)
            n = len(obs)
            rows += n
            calls += 1
            v = np.tile(np.float32([1 / 3, 1 / 3, 1 / 3, 1.5]), (n, 1))
            pend[k] = (v, np.broadcast_to(v[:, None, :], (n, A, 4)).copy())
    s, dt = eng.stats(), time.time() - t0
    print(f'engine ({a.games} games, {a.sims} sims, {os.cpu_count()} threads): '
          f'{s["sims"] / dt / 1e3:.0f}k sims/s, {rows / dt / 1e3:.0f}k evals/s, '
          f'{s["games"] / dt:.1f} games/s, mean batch {rows / max(calls, 1):.0f} rows')

    # 2) network inference
    net = build(cfg, g.obs_shape, A).to(device)
    for amp, comp in ([(False, False), (True, False), (True, True)] if device.type == 'cuda' else [(False, False)]):
        ev = Evaluator(net, device, amp=amp, compile=comp)
        for bs in (256, 1024, 2048, 4096):
            x = np.random.rand(bs, *g.obs_shape).astype(np.float32)
            for _ in range(3):
                ev.evaluate(x, A)
            t0, n = time.time(), 0
            while time.time() - t0 < 2.0:
                ev.evaluate(x, A)
                n += bs
            print(f'inference amp={amp} compile={comp} batch {bs}: {n / (time.time() - t0) / 1e3:.0f}k rows/s')

    # 3) training step
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    buf = ReplayBuffer(4 * cfg.batch_size, g.obs_shape, A, device)
    eng = _core.Engine(cfg.game, num_games=64, groups=1, sims=8, wave=4)
    while len(buf) < cfg.batch_size:
        obs, _ = eng.collect(0)
        n = len(obs)
        v = np.tile(np.float32([1 / 3, 1 / 3, 1 / 3, 1.5]), (n, 1))
        eng.apply(0, v, np.broadcast_to(v[:, None, :], (n, A, 4)).copy())
        buf.add(eng.drain())
    step = make_step(net, opt, cfg, device, buf)
    for _ in range(3):
        step(buf.sample(cfg.batch_size))
    t0, n = time.time(), 0
    while time.time() - t0 < 3.0:
        step(buf.sample(cfg.batch_size))
        n += 1
    if device.type == 'cuda':
        torch.cuda.synchronize()
    print(f'training: {n / (time.time() - t0):.1f} steps/s at batch {cfg.batch_size}')


def main(argv=None):
    p = argparse.ArgumentParser(prog='thompsonzero')
    sub = p.add_subparsers(dest='cmd', required=True)

    t = sub.add_parser('train', help='train (or resume) a run')
    t.add_argument('--game', choices=['connect_four', 'connect4', 'othello', 'boop'])
    t.add_argument('--rule', default='tz', choices=['tz', 'corr'])
    t.add_argument('--run-dir')
    t.add_argument('--set', action='append', metavar='KEY=VALUE', help='override a Config field')
    t.set_defaults(fn=cmd_train)

    l = sub.add_parser('ladder', help='rate benchmark checkpoints (Bradley-Terry)')
    l.add_argument('--run-dir', required=True)
    l.add_argument('--sims', type=int, default=400)
    l.add_argument('--games', type=int, default=24)
    l.add_argument('--opponents', type=int, default=4)
    l.add_argument('--device', default='auto')
    l.add_argument('--watch', action='store_true', help='keep rating new benchmarks')
    l.set_defaults(fn=cmd_ladder)

    b = sub.add_parser('bench', help='measure engine / inference / training throughput')
    b.add_argument('--game', default='othello')
    b.add_argument('--rule', default='tz', choices=['tz', 'corr'])
    b.add_argument('--games', type=int, default=1024)
    b.add_argument('--sims', type=int, default=400)
    b.add_argument('--seconds', type=float, default=10.0)
    b.add_argument('--device', default='auto')
    b.set_defaults(fn=cmd_bench)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == '__main__':
    main()
