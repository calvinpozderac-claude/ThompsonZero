"""The training loop: self-play thread + GPU replay buffer + optimizer.

Self-play (Rust engine + inference network copy) runs continuously in its own
thread.  The trainer takes finished games from it, keeps exactly
`train_steps_per_episode` optimizer steps per finished game (the replay
ratio), and pushes fresh weights back to the self-play network every
`sync_every_steps` steps.  Self-play blocks when it gets `max_pending`
batches ahead, so the replay ratio holds whichever side is faster.

Run directory:
    config.json      the Config of the run
    train.log        human-readable log
    metrics.jsonl    one JSON line per log interval
    latest.pt        model + optimizer + counters (resumable)
    buffer.pt        replay buffer (saved at benchmarks and on exit)
    bench_<ep>.pt    frozen networks for the ladder (`thompsonzero ladder`)
"""
import copy
import glob
import json
import os
import queue
import sys
import time
from collections import defaultdict

import torch

from . import _core, az, loss, symmetry
from .arena import play_match
from .config import Config
from .inference import Evaluator, SelfPlay
from .network import build
from .replay import TZ_SCHEMA, ReplayBuffer, az_schema


def pick_device(pref='auto'):
    if pref == 'auto':
        pref = 'cuda' if torch.cuda.is_available() else 'cpu'
    return torch.device(pref)


def make_engine(cfg, seed, sims):
    if cfg.algo == 'az':
        return az.make_engine(cfg, seed, sims)
    return _core.Engine(
        cfg.game, num_games=cfg.num_games, groups=cfg.groups, seed=seed, sims=sims,
        wave=cfg.wave, search=cfg.search_rule, target=cfg.target_rule,
        rho_search=cfg.rho_search, rho_target=cfg.rho_target, sample_leaf=cfg.sample_leaf,
        best_value=cfg.best_value,
        virtual_loss=cfg.virtual_loss, descent_temp=cfg.descent_temp,
        temp_threshold=cfg.temp_threshold, root_temp=cfg.root_temp,
        max_plies=cfg.max_plies or None, double_aux=cfg.double_aux, double_min_visits=cfg.double_min_visits,
        solved_aux=cfg.solved_aux)


def make_buffer(cfg, obs_shape, num_actions, device):
    if cfg.algo == 'az':
        schema = az_schema(reply=cfg.reply_head > 0, own=cfg.own_head > 0,
                           squares=obs_shape[1] * obs_shape[2])
    else:
        schema = TZ_SCHEMA
    return ReplayBuffer(cfg.buffer_size, obs_shape, num_actions, device, schema)


def make_optimizer(cfg, net, device):
    lr = cfg.lr_steps[0][1]
    if cfg.optimizer == 'sgd':
        return torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                               weight_decay=cfg.weight_decay)
    if cfg.optimizer == 'adamw':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=cfg.weight_decay,
                                 fused=device.type == 'cuda')
    raise ValueError(f'unknown optimizer {cfg.optimizer!r}')


def make_step(net, opt, cfg, device, buffer):
    """One optimizer step on a replay batch; returns detached loss parts.
    `step.eval(b)` is the same loss in eval mode, without a step."""
    amp = cfg.amp and device.type == 'cuda'

    if cfg.algo == 'az':
        def forward_loss(b):
            with torch.autocast('cuda', torch.bfloat16, enabled=amp):
                v, logits, extra = net(b['obs'], aux=True)
            return az.compute(v, logits, b, cfg.value_weight, extra, cfg)
    else:
        weights = {k: float(w) for k, w in cfg.loss_weights.items() if w}
        unknown = set(weights) - {'klv', 'kla', 'cev', 'cea', 'cons'}
        if unknown:
            raise ValueError(f'unknown loss components {sorted(unknown)}')

        def forward_loss(b):
            # batch positions and their successors in ONE forward pass
            x = torch.cat([b['obs'], b['next_obs']])
            with torch.autocast('cuda', torch.bfloat16, enabled=amp):
                v, a = net(x)
            n = b['obs'].shape[0]
            return loss.compute(v[:n], a[:n], v[n:], b, weights, split=cfg.loss_split,
                                mean_w=cfg.split_mean_weight, cal_c=cfg.cal_conc)

    # Mid-run diagnostics use the eager loss: a recompile on this thread would
    # FX-trace while the self-play thread runs its compiled network, and
    # torch.compile does not support that.
    eager_loss = forward_loss
    if cfg.compile and device.type == 'cuda':
        # compile + warm up now, on this thread, before self-play starts
        compiled = torch.compile(forward_loss)
        try:
            total, _ = compiled(buffer.placeholder_batch(cfg.batch_size))
            total.backward()
            opt.zero_grad(set_to_none=True)
            forward_loss = compiled
        except Exception as e:  # noqa: BLE001 -- any compiler failure
            opt.zero_grad(set_to_none=True)
            print(f'torch.compile unavailable for training ({type(e).__name__}: {e}); using eager')

    def step(b):
        if cfg.augment:
            b = symmetry.augment(b, cfg.game)
        total, parts = forward_loss(b)
        opt.zero_grad(set_to_none=True)
        total.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(net.parameters(), cfg.grad_clip)
        # a single non-finite step would write NaN into every weight: skip it
        ok = bool(torch.isfinite(gnorm) & torch.isfinite(total))
        if ok:
            opt.step()
        parts['skipped'] = torch.tensor(0.0 if ok else 1.0)
        return parts

    @torch.no_grad()
    def evaluate(b):
        was = net.training
        net.eval()
        try:
            return eager_loss(b)[1]
        finally:
            net.train(was)

    step.eval = evaluate
    return step


class Run:
    """Files of one run directory."""

    def __init__(self, cfg):
        self.cfg, self.dir = cfg, cfg.run_dir
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, 'config.json'), 'w') as f:
            f.write(cfg.to_json())
        self._log = open(os.path.join(self.dir, 'train.log'), 'a', buffering=1)
        self._metrics = open(os.path.join(self.dir, 'metrics.jsonl'), 'a', buffering=1)

    def log(self, msg):
        line = f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {msg}'
        print(line, flush=True)
        self._log.write(line + '\n')

    def metrics(self, d):
        self._metrics.write(json.dumps(d) + '\n')

    def path(self, name):
        return os.path.join(self.dir, name)

    def save(self, name, obj):
        tmp = self.path(name + '.tmp')
        torch.save(obj, tmp)
        os.replace(tmp, self.path(name))

    def benchmarks(self):
        files = glob.glob(self.path('bench_*.pt'))
        return sorted(int(os.path.basename(f)[6:-3]) for f in files)


def load_net(path, device='cpu'):
    """Rebuild a saved network (benchmark or latest checkpoint), set up for
    matches (`eval_cal_search`)."""
    ck = torch.load(path, map_location='cpu', weights_only=False)
    cfg = Config.from_dict(ck['config'])
    g = _core.GameState(cfg.game)
    net = build(cfg, g.obs_shape, g.num_actions)
    net.load_state_dict(ck['model'])
    if cfg.algo != 'az':
        net.use_cal = cfg.cal_search or cfg.eval_cal_search
    return net.to(device).eval(), cfg


def train(cfg: Config):
    run = Run(cfg)
    device = pick_device(cfg.device)
    torch.manual_seed(cfg.seed)
    # The self-play thread needs the GIL only in short bursts between Rust
    # calls; a 1 ms switch interval (default 5 ms) bounds how long it can wait
    # behind the trainer thread before the next GPU batch is launched.
    sys.setswitchinterval(1e-3)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    g = _core.GameState(cfg.game)
    obs_shape, num_actions = g.obs_shape, g.num_actions
    net = build(cfg, obs_shape, num_actions).to(device)
    opt = make_optimizer(cfg, net, device)
    buffer = make_buffer(cfg, obs_shape, num_actions, device)
    episode = steps = 0

    if os.path.exists(run.path('latest.pt')):
        ck = torch.load(run.path('latest.pt'), map_location=device, weights_only=False)
        net.load_state_dict(ck['model'])
        opt.load_state_dict(ck['optim'])
        episode, steps = ck['episode'], ck['steps']
        if os.path.exists(run.path('buffer.pt')):
            buffer.load_state_dict(torch.load(run.path('buffer.pt'), weights_only=False))
        run.log(f'resumed at episode {episode} (step {steps}, buffer {len(buffer):,})')

    def checkpoint(with_buffer=False):
        run.save('latest.pt', {'model': net.state_dict(), 'optim': opt.state_dict(),
                               'episode': episode, 'steps': steps,
                               'config': json.loads(cfg.to_json())})
        if with_buffer and cfg.save_buffer:
            run.save('buffer.pt', buffer.state_dict())

    def benchmark():
        run.save(f'bench_{episode}.pt', {'model': net.state_dict(),
                                         'config': json.loads(cfg.to_json())})

    if not run.benchmarks():
        benchmark()

    n_params = sum(p.numel() for p in net.parameters())
    if cfg.algo == 'az':
        algo = (f'alphazero {cfg.channels}ch x {cfg.blocks} c_puct={cfg.c_puct} '
                f'fpu={cfg.fpu_reduction} {cfg.optimizer} augment={cfg.augment}')
    else:
        algo = f'search={cfg.search_rule} target={cfg.target_rule}'
    run.log(f'{cfg.game}: {n_params:,} params on {device} | {algo} '
            f'| {cfg.num_games} games x wave {cfg.wave} | episodes {episode}/{cfg.episodes}')

    step = make_step(net, opt, cfg, device, buffer)
    evaluator = Evaluator(net, device, cfg.amp, cfg.compile and device.type == 'cuda')
    selfplay = SelfPlay(make_engine(cfg, cfg.seed + episode, cfg.sims_at(episode)),
                        evaluator, cfg.max_pending)
    selfplay.start()
    gen = torch.Generator(device).manual_seed(cfg.seed + steps)

    acc = defaultdict(float)
    n_acc = 0
    gap = defaultdict(list)
    owed = 0.0
    window = dict(t=time.time(), episode=episode, steps=steps, stats=None)
    added_rows = added_main = 0
    try:
        while episode < (cfg.stop_at or cfg.episodes):
            if selfplay.error is not None:
                raise RuntimeError('self-play thread failed') from selfplay.error
            try:
                samples, games = selfplay.out.get(timeout=1.0)
            except queue.Empty:
                continue
            rows = buffer.add(samples)
            added_rows += len(rows)
            added_main += float(samples['main'].sum()) if 'main' in samples else len(rows)
            prev, episode = episode, episode + games
            if cfg.fresh_eval_every and episode // cfg.fresh_eval_every > prev // cfg.fresh_eval_every:
                # generalisation gap: loss on games not yet trained on vs on
                # replay data, both in eval mode
                fresh = step.eval(buffer.gather(rows[:cfg.batch_size]))
                seen = step.eval(buffer.sample(cfg.batch_size, gen))
                for k in fresh.keys() & {'loss', 'value', 'policy', 'klv', 'cev'}:
                    gap['fresh_' + k].append(float(fresh[k]))
                    gap['seen_' + k].append(float(seen[k]))
            selfplay.sims = cfg.sims_at(episode)
            for g in opt.param_groups:
                g['lr'] = cfg.lr_at(episode)

            owed += games * cfg.train_steps_per_episode
            if len(buffer) < cfg.batch_size:
                owed = 0.0
            net.train()
            while owed >= 1.0:
                win = cfg.window_at(episode)
                if win and cfg.window_main and added_main:
                    win = int(win * added_rows / added_main)
                parts = step(buffer.sample(cfg.batch_size, gen, win))
                for k, v in parts.items():
                    acc[k] += v           # stays on device until logged
                n_acc += 1
                steps += 1
                owed -= 1.0
                if steps % cfg.sync_every_steps == 0:
                    evaluator.load_from(net)

            def crossed(every):
                return every and episode // every > prev // every

            if crossed(cfg.log_every):
                window = _log(run, cfg, episode, steps, buffer, acc, n_acc, gap, selfplay, window, opt)
                acc, n_acc, gap = defaultdict(float), 0, defaultdict(list)
            if crossed(cfg.benchmark_every):
                benchmark()
                checkpoint(with_buffer=True)
            elif crossed(cfg.checkpoint_every):
                checkpoint()
                _quick_eval(run, cfg, net, device, episode)
    finally:
        selfplay.stop()
        selfplay.join(timeout=10)
        checkpoint(with_buffer=True)
        run.log(f'stopped at episode {episode}')
    return net


def _log(run, cfg, episode, steps, buffer, acc, n_acc, gap, selfplay, window, opt):
    now = time.time()
    dt = max(now - window['t'], 1e-9)
    st = dict(selfplay.stats or {})
    prev = window['stats'] or {k: 0 for k in st}
    d = {k: st.get(k, 0) - prev.get(k, 0) for k in st}
    m = {k: float(v) / max(n_acc, 1) for k, v in acc.items()}
    m.update({k: sum(v) / len(v) for k, v in gap.items() if v})
    games = max(d.get('games', 0), 1)
    rec = dict(episode=episode, steps=steps, time=now, lr=opt.param_groups[0]['lr'],
               buffer=len(buffer), sims=cfg.sims_at(episode),
               games_per_s=(episode - window['episode']) / dt,
               steps_per_s=(steps - window['steps']) / dt,
               evals_per_s=d.get('evals', 0) / dt,
               plies=d.get('plies', 0) / games, draws=d.get('draws', 0) / games,
               aux_solved=d.get('solved_aux', 0) / games, **m)
    run.metrics(rec)
    if cfg.algo == 'az':
        terms = (f'loss {m.get("loss", float("nan")):.3f} (value {m.get("value", 0):.3f} '
                 f'policy {m.get("policy", 0):.3f} | pi entropy {m.get("pi_entropy", 0):.3f}) '
                 f'| |v| {m.get("v_mean_abs", 0):.2f} doubles {d.get("double_aux", 0) / games:.0f}/game')
        extra = ' '.join(f'{k[4:]} {x:.3f}' for k, x in m.items() if k.startswith('aux_'))
        if extra:
            terms += f' | aux {extra}'
    else:
        w = cfg.loss_weights
        terms = ' '.join(f'{k} {m.get(k, float("nan")):.3f}' for k in ('klv', 'kla', 'cev', 'cea', 'cons') if w.get(k))
        terms = (f'loss {m.get("loss", float("nan")):.3f} ({terms}) '
                 f'| conc v {m.get("v_conc", 0):.1f}/{m.get("v_conc_tgt", 0):.1f} '
                 f'a {m.get("a_conc", 0):.1f}/{m.get("a_conc_tgt", 0):.1f}')
        if 'v_cal' in m:
            terms += (f' | cal v {m["v_cal"]:.0f} a {m["a_cal"]:.0f} '
                      f'(kl {m.get("calv", 0):.2f}/{m.get("cala", 0):.2f})')
    if 'fresh_loss' in m:
        k = ('value', 'policy') if cfg.algo == 'az' else ('klv', 'cev')
        terms += (f' | fresh/replay {k[0]} {m.get("fresh_" + k[0], 0):.3f}/{m.get("seen_" + k[0], 0):.3f} '
                  f'{k[1]} {m.get("fresh_" + k[1], 0):.3f}/{m.get("seen_" + k[1], 0):.3f}')
    run.log(f'ep {episode:7d} | {terms} '
            f'| {rec["games_per_s"]:.2f} games/s {rec["steps_per_s"]:.1f} steps/s '
            f'{rec["evals_per_s"] / 1000:.1f}k evals/s | ply {rec["plies"]:.0f} '
            f'draw {100 * rec["draws"]:.0f}% | sims {rec["sims"]} buf {len(buffer) // 1000}k '
            f'lr {rec["lr"]:.1e}' + (f' | SKIPPED {m["skipped"] * n_acc:.0f} steps' if m.get('skipped') else ''))
    return dict(t=now, episode=episode, steps=steps, stats=st)


def _quick_eval(run, cfg, net, device, episode):
    """Search-free match against the latest benchmark: a cheap pulse."""
    if not cfg.quick_eval_games:
        return
    label = run.benchmarks()[-1]
    ref, _ = load_net(run.path(f'bench_{label}.pt'), device)
    cur = copy.deepcopy(net).eval()
    evs = [Evaluator(cur, device, cfg.amp), Evaluator(ref, device, cfg.amp)]
    match = az.play_match if cfg.algo == 'az' else play_match
    w, d, l = match(cfg, evs, cfg.quick_eval_games, sims=0, seed=episode)
    run.log(f'         vs bench_{label} (no search): W{w} D{d} L{l}')
