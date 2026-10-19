"""Run configuration and the per-game presets.

Every field has the value the research runs used (the "OLIVAW recipe" from
`olivaw_common.py`), so `Config(game=...)` is a faithful starting point and a
run script only states what it changes.
"""
import dataclasses
import json
from dataclasses import dataclass, field

# The training budget of the AlphaZero baseline, shared by the TZ run compared
# with it so that only the algorithm differs: AlphaGo Zero's trunk at 128
# channels x 10 blocks, SGD with momentum and L2 c = 1e-4 (weight decay 2c),
# symmetry augmentation, OLIVAW's sims ramp and training window.  Two
# deliberate departures from OLIVAW, both measured in head-to-head matches
# and against Edax (Oct 2026 sweeps): its LR (3e-3) is far too low here --
# 10x beat it by ~600 Elo at 5k episodes, 30x by another ~70 Elo at 10k --
# and 2-3 optimizer steps per game beat its ~6 (fewer replays of each
# sample: less overfitting).  OLIVAW never gives its width; at 15k episodes
# 128 channels led 64 and 256 (which learns more slowly and overfits at 3
# steps per game), and at 50k 256 only tied 128 (Edax and head-to-head) for
# 3.4x the time.
BASELINE = dict(
    channels=128, trunk='az', optimizer='sgd', momentum=0.9, weight_decay=2e-4, augment=True,
    lr_steps=((0.0, 0.1), (0.20, 0.03), (0.55, 0.003)), train_steps_per_episode=2.0,
    # OLIVAW: 200 sims from generation 5/20, 400 from 12/20; train on the
    # last 2 generations growing to the last 5 (1 generation = 2500 games ~
    # 290k samples with doubling)
    sims_ramp=((0.0, 100), (0.20, 200), (0.55, 400)),
    buffer_size=1_450_000,
    window_ramp=((0.0, 575_000), (0.20, 860_000), (0.40, 1_150_000), (0.60, 1_440_000)))

# Rule sets (`--rule`):
#   az    the AlphaZero baseline (src/puct.rs, src/az.rs, az.py) on BASELINE
#   tz    ThompsonZero on BASELINE: search additive_mle + target avgconc,
#         Dirichlet heads on the AlphaZero trunk.  Its own settings, from
#         the Oct 2026 sweeps (10k episodes, head-to-head vs an AlphaZero net
#         trained identically): in-tree descent sharpened to Dir(32 alpha) in
#         self-play (+90 Elo over 1, while near-greedy 1000 lost exploration)
#         and to Dir(1000 alpha) in matches (+150-260 Elo at play time);
#         opening moves drawn from Dir(1024 alpha) (+110-180 over 1).  A
#         calibrated-concentration channel learned on the side (cal_conc)
#         is read only in matches, with point_400 evidence at Dir(4 alpha):
#         +40 Elo at 10k, +111 on the 50k net (calibrated self-play lost
#         ~100 Elo, so self-play keeps the search above).
#   corr  the research recipe of record (64ch SE trunk, AdamW, 6 steps per
#         game): search corr + target corr, rho 0.8 / 0.999 (CORR-TZ)
RULES = {
    'az': dict(algo='az', **BASELINE),
    'tz': dict(search_rule='additive_mle', target_rule='avgconc',
               descent_temp=32.0, root_temp=1024.0, cal_conc=1000.0,
               eval_cal_search=True, eval_search_rule='point_400', eval_descent_temp=4.0,
               **BASELINE),
    'corr': dict(search_rule='corr', target_rule='corr', rho_search=0.8, rho_target=0.999),
}


@dataclass
class Config:
    game: str = 'othello'                 # connect_four | othello | boop
    run_dir: str = 'runs/othello'
    seed: int = 0
    algo: str = 'tz'                      # tz (Thompson / Dirichlet) | az (AlphaZero)

    # ── network ─────────────────────────────────────────────────────────────
    channels: int = 64
    blocks: int = 10
    head_channels: int = 8
    trunk: str = 'se'                     # tz: se (GroupNorm + SE) | az (AlphaGo Zero's, BatchNorm)

    # ── search (see src/tree.rs and src/belief.rs) ──────────────────────────
    search_rule: str = 'additive_mle'     # mixture | additive | additive_x<k> | additive_mle | mean_mle
                                          # | avgconc | sumconc | point | point_<c> | corr
    target_rule: str = 'avgconc'
    rho_search: float = 0.8               # corr only: leaf-error correlation
    rho_target: float = 0.999
    sample_leaf: bool = False             # back up a Dirichlet draw, not the mean
    best_value: bool = False              # V(s) target = the most-visited edge's (one-ply max)
    virtual_loss: float = 1.0
    descent_temp: float = 1.0             # in-tree sampling from Dir(temp * alpha)
    eval_descent_temp: float = 0.0        # the same in matches (arena, league, Edax); 0 = descent_temp
    eval_search_rule: str = ''            # search_rule in matches; '' = search_rule
    eval_cal_search: bool = False         # matches read the calibrated channel (see cal_conc)
    # ((fraction of run, sims per move), ...): 100 -> 200 -> 400 in thirds
    sims_ramp: tuple = ((0.0, 100), (1 / 3, 200), (2 / 3, 400))
    temp_threshold: int = 20              # Thompson-sampled moves, then greedy
    root_temp: float = 1.0                # tz: those moves drawn from Dir(root_temp * alpha)
    double_aux: int = 1                   # OLIVAW doubling: unplayed children per move
    double_min_visits: int = 8
    solved_aux: bool = True               # exact samples for solver-proven nodes
    max_plies: int = 0                    # 0 = the game's own cap

    # ── AlphaZero search (algo = az; see src/puct.rs) ───────────────────────
    c_puct: float = 1.5
    fpu_reduction: float = 0.2            # unvisited Q = node mean - this
    noise_alpha: float = 10.0             # root Dir(min(1, 10 / legal moves))
    noise_frac: float = 0.25
    reuse_tree: bool = True
    value_head: str = 'scalar'            # scalar (tanh, MSE to z) | wdl (win/draw/loss softmax,
                                          # CE; the search reads p_win - p_loss)
    az_select: str = 'puct'               # puct | thompson: draw from a win/draw/loss Dirichlet
                                          # with sd = PUCT's U (src/puct.rs)

    # ── self-play engine ────────────────────────────────────────────────────
    num_games: int = 1024                 # concurrent games (all CPU cores share them)
    groups: int = 2                       # pipelined halves: CPU on one, GPU on the other
    wave: int = 4                         # simulations per game per network batch

    # ── training ────────────────────────────────────────────────────────────
    episodes: int = 50_000
    batch_size: int = 1024
    train_steps_per_episode: float = 6.0
    buffer_size: int = 600_000
    # ((fraction of run, newest samples drawn from), ...); () = whole buffer
    window_ramp: tuple = ()
    lr_warmup_episodes: int = 100
    # ((fraction of run, lr), ...): 3e-3, 1e-3 from 20%, 1e-4 from 55%
    lr_steps: tuple = ((0.0, 3e-3), (0.20, 1e-3), (0.55, 1e-4))
    weight_decay: float = 1e-4
    optimizer: str = 'adamw'              # adamw | sgd (momentum below)
    momentum: float = 0.9
    augment: bool = False                 # random board symmetry per sample (othello)
    value_weight: float = 1.0             # az: (z - v)^2 weight vs policy CE
    grad_clip: float = 1.0
    cal_conc: float = 0.0                 # tz, > 0: a calibrated-concentration channel trained on
                                          # point targets of this concentration (network.py)
    cal_search: bool = False              # ...and search reads it (False: learned on the side)
    loss_split: bool = False              # tz: KL terms train mean and concentration apart
    split_mean_weight: float = 1.0        # ...with the search means' pull on the mean scaled by this
    loss_weights: dict = field(default_factory=lambda: {
        'klv': 1.0, 'kla': 2.0, 'cev': 1.0, 'cea': 1.0, 'cons': 0.5})
    sync_every_steps: int = 6             # copy weights to the self-play net
    max_pending: int = 4                  # self-play batches allowed ahead of training

    # ── hardware ────────────────────────────────────────────────────────────
    device: str = 'auto'                  # auto | cuda | cpu
    amp: bool = True                      # bf16 network on CUDA
    compile: bool = True                  # torch.compile training step + inference

    # ── logging / checkpoints / evaluation ──────────────────────────────────
    log_every: int = 500                  # episodes
    checkpoint_every: int = 1000
    benchmark_every: int = 5000           # frozen nets for the ladder
    save_buffer: bool = True              # replay buffer goes in the checkpoint
    quick_eval_games: int = 64            # search-free match vs last benchmark
    fresh_eval_every: int = 50            # episodes: loss on untrained games vs replay
    eval_sims: int = 400                  # ladder settings
    eval_games: int = 24
    eval_opponents: int = 4

    def lr_at(self, episode):
        if episode < self.lr_warmup_episodes:
            return self.lr_steps[0][1] * episode / max(self.lr_warmup_episodes, 1)
        frac = episode / max(self.episodes, 1)
        return [lr for f, lr in self.lr_steps if frac >= f][-1]

    def sims_at(self, episode):
        frac = episode / max(self.episodes, 1)
        return [s for f, s in self.sims_ramp if frac >= f][-1]

    def window_at(self, episode):
        frac = episode / max(self.episodes, 1)
        return ([w for f, w in self.window_ramp if frac >= f] or [None])[-1]

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: (tuple(map(tuple, v)) if k in ('sims_ramp', 'lr_steps', 'window_ramp') else v)
             for k, v in d.items() if k in known}
        return cls(**d)


# Per-game differences, applied over the rule set (BASELINE is sized for
# Othello).  Symmetry augmentation is implemented for Othello only.
GAMES = {
    'othello': dict(),
    'connect_four': dict(channels=64, blocks=5, head_channels=16, temp_threshold=12,
                         buffer_size=150_000, window_ramp=(), episodes=20_000, augment=False),
    'boop': dict(temp_threshold=20, augment=False),
}


def preset(game, rule='tz', **overrides):
    """Config for `game` with rule set `rule` (a key of RULES: az, tz, corr)."""
    game = 'connect_four' if game == 'connect4' else game
    if game not in GAMES:
        raise ValueError(f'unknown game {game!r}; expected one of {sorted(GAMES)}')
    if rule not in RULES:
        raise ValueError(f'unknown rule set {rule!r}; expected one of {sorted(RULES)}')
    kw = dict(game=game, run_dir=f'runs/{game}_{rule}')
    kw.update(RULES[rule])
    kw.update(GAMES[game])
    kw.update(overrides)
    return Config(**kw)
