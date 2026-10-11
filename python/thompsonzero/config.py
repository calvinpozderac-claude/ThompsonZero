"""Run configuration and the per-game presets.

Every field has the value the research runs used (the "OLIVAW recipe" from
`olivaw_common.py`), so `Config(game=...)` is a faithful starting point and a
run script only states what it changes.
"""
import dataclasses
import json
from dataclasses import dataclass, field

# Evidence rules implemented in src/belief.rs.  The two runs of record:
#   tz    search additive_mle + target avgconc   (OLIVAW-TZ)
#   corr  search corr + target corr, rho 0.8 / 0.999   (CORR-TZ, the leader)
RULES = {
    'tz': dict(search_rule='additive_mle', target_rule='avgconc'),
    'corr': dict(search_rule='corr', target_rule='corr', rho_search=0.8, rho_target=0.999),
}


@dataclass
class Config:
    game: str = 'othello'                 # connect_four | othello | boop
    run_dir: str = 'runs/othello'
    seed: int = 0

    # ── network ─────────────────────────────────────────────────────────────
    channels: int = 64
    blocks: int = 10
    head_channels: int = 8

    # ── search (see src/tree.rs and src/belief.rs) ──────────────────────────
    search_rule: str = 'additive_mle'     # mixture | additive | additive_mle | avgconc | corr
    target_rule: str = 'avgconc'
    rho_search: float = 0.8               # corr only: leaf-error correlation
    rho_target: float = 0.999
    sample_leaf: bool = False             # back up a Dirichlet draw, not the mean
    virtual_loss: float = 1.0
    descent_temp: float = 1.0             # in-tree sampling from Dir(temp * alpha)
    # ((fraction of run, sims per move), ...): 100 -> 200 -> 400 in thirds
    sims_ramp: tuple = ((0.0, 100), (1 / 3, 200), (2 / 3, 400))
    temp_threshold: int = 20              # Thompson-sampled moves, then greedy
    double_aux: int = 1                   # OLIVAW doubling: unplayed children per move
    double_min_visits: int = 8
    solved_aux: bool = True               # exact samples for solver-proven nodes
    max_plies: int = 0                    # 0 = the game's own cap

    # ── self-play engine ────────────────────────────────────────────────────
    num_games: int = 1024                 # concurrent games (all CPU cores share them)
    groups: int = 2                       # pipelined halves: CPU on one, GPU on the other
    wave: int = 4                         # simulations per game per network batch

    # ── training ────────────────────────────────────────────────────────────
    episodes: int = 50_000
    batch_size: int = 1024
    train_steps_per_episode: float = 6.0
    buffer_size: int = 600_000
    lr_warmup_episodes: int = 100
    # ((fraction of run, lr), ...): 3e-3, 1e-3 from 20%, 1e-4 from 55%
    lr_steps: tuple = ((0.0, 3e-3), (0.20, 1e-3), (0.55, 1e-4))
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
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

    def to_json(self):
        return json.dumps(dataclasses.asdict(self), indent=2)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in dataclasses.fields(cls)}
        d = {k: (tuple(map(tuple, v)) if k in ('sims_ramp', 'lr_steps') else v)
             for k, v in d.items() if k in known}
        return cls(**d)


# Per-game differences from the defaults above.
GAMES = {
    'othello': dict(),
    'connect_four': dict(channels=64, blocks=5, head_channels=16, temp_threshold=12,
                         buffer_size=150_000, episodes=20_000),
    'boop': dict(temp_threshold=20),
}


def preset(game, rule='tz', **overrides):
    """Config for `game` with evidence rule set `rule` ('tz' or 'corr')."""
    game = 'connect_four' if game == 'connect4' else game
    if game not in GAMES:
        raise ValueError(f'unknown game {game!r}; expected one of {sorted(GAMES)}')
    if rule not in RULES:
        raise ValueError(f'unknown rule set {rule!r}; expected one of {sorted(RULES)}')
    kw = dict(game=game, run_dir=f'runs/{game}_{rule}')
    kw.update(GAMES[game])
    kw.update(RULES[rule])
    kw.update(overrides)
    return Config(**kw)
