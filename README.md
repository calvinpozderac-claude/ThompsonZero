# ThompsonZero

Self-play reinforcement learning where search picks moves by **Thompson
sampling over Dirichlet value beliefs** instead of PUCT over a policy.
This repository is the cleaned-up implementation of the research code
developed in `calvinpozderac-claude/open_spiel` (branches `claude/correlated`
and earlier), with the search engine and games rewritten in Rust and training
in PyTorch.

Three games are included: **Connect Four**, **Othello** and **Boop**.

## The method in one screen

* **Network.** A residual trunk with two Dirichlet heads over (win, draw, loss)
  from the mover's perspective: a STATE head V(s) and an ACTION head Q(s,a)
  for every action.  No policy head.
* **Search.** At each node, draw one value `p_win - p_loss` per legal edge
  from that edge's current belief and descend the argmax.  Expand the leaf,
  then back up its V(leaf) (or a proven result), flipped into each node's
  perspective.  Every node and searched edge keeps an O(1) evidence
  accumulator, and an **evidence rule** collapses it back into one Dirichlet,
  separately for selection and for training targets.  An MCTS-solver overlay
  proves won/lost/drawn edges.
* **Self-play.** Thompson-sample the root for the first `temp_threshold`
  plies, then play the best posterior mean.  Tree reuse between moves.
* **Training targets.** The search state belief (`klv`), the searched edge
  beliefs (`kla`), the game outcome (`cev`, `cea`), and one-ply consistency
  `Q(s,a*) ~ flip V(s')` (`cons`).  Solver-proven nodes add exact-label
  samples; OLIVAW "doubling" adds the most-visited unplayed child per move.

The rule sets (`--rule`):

| rule set | algorithm | selection rule | target rule | training recipe |
|---|---|---|---|---|
| `az`   | AlphaZero (PUCT) | — | — | the baseline budget (below) |
| `tz`   | ThompsonZero | `additive_mle` | `avgconc` | the baseline budget, Dirichlet heads on the AlphaZero trunk |
| `corr` | ThompsonZero | `corr` (ρ = 0.8) | `corr` (ρ = 0.999) | the research recipe; led TZ from ep15k in the research runs |

The **baseline budget** (`BASELINE` in `config.py`) is what the AlphaZero
baseline was tuned to, and `tz` shares all of it so that only the algorithm
differs: AlphaGo Zero's residual trunk (BatchNorm, no SE) at 128 channels x
10 blocks, SGD with momentum 0.9, L2 1e-4 and LR 0.1 -> 0.03 (20%) -> 0.003
(55%), 2 optimizer steps of batch 1024 per game, 8-fold symmetry
augmentation, sims 100 -> 200 (20%) -> 400 (55%), and a 1.45M buffer whose
training window grows from 575k to 1.44M samples.  TZ's own settings are its
in-tree descent temperature, sharpened to Dir(32 α) in self-play and Dir(1000
α) in matches (`descent_temp`, `eval_descent_temp`); both were measured in
10k-episode sweeps against an AlphaZero net trained identically.  After 50k
episodes on this budget TZ rates -451 ± 17 Elo against the AlphaZero
baseline's 50k net (the AlphaZero 10k net's level) in the league, and scores
67% / 33% / 9% / 2% against Edax E4 / E6 / E8 / E10 (200 games each).

**Calibrated concentrations** (Oct 2026, against Edax-scored positions).
The concentrations `tz` learns are ~16-64x too low for its Thompson picks to
be calibrated (the temperatures above compensate), because `avgconc`
targets inherit the network's own leaf concentrations.  Feeding the main
loss confident targets (`target_rule=point`, `loss_split`) breaks how the
means learn, so `cal_conc=1000` instead adds a separate log-concentration
channel per belief, read off the trunk with the gradient stopped and trained
on point targets (search mean at concentration 1000); the means learn
exactly as before.  Searching with it (`cal_search`) needs evidence that
outweighs the now-confident prior (`search_rule=point_400`).  At evaluation
(Dir(4 α) descent) that is +40 Elo on a 10k net and +110 Elo on the 50k net,
whose Edax scores at 1000 sims rise to 85% / 55% / 26% / 8% (channel fitted
post hoc on its replay buffer).  Calibrated search in SELF-PLAY trained
weaker nets (-65 to -150 Elo at 10k), so self-play keeps the temperatures.

The research recipe (`corr`, and every `Config` default): 64ch x 10 SE
blocks, AdamW, sims 100 -> 200 -> 400 in thirds, 6 optimizer steps of batch
1024 per game, 600k buffer, LR steps 3e-3 -> 1e-3 (20%) -> 1e-4 (55%).

## Layout

```
src/                     Rust crate (compiled into thompsonzero._core)
  games/{connect4,othello,boop}.rs   bitboard games behind the `Game` trait
  belief.rs              Dirichlet maths, accumulators, evidence rules
  tree.rs                Thompson-sampling tree, backup, solver, targets
  engine.rs              batched multi-threaded self-play / arena / search-service driver
  puct.rs                AlphaZero baseline: PUCT tree
  az.rs                  AlphaZero baseline: self-play / arena / external-opponent /
                         search-service driver
  python.rs              PyO3 bindings (Engine, AzEngine, GameState)
python/thompsonzero/
  config.py              Config + per-game presets
  network.py             DirichletNet, AZNet (AlphaGo Zero policy/value net)
  loss.py                the training loss
  az.py                  AlphaZero engine factory, loss, matches
  symmetry.py            board-symmetry augmentation of training batches (both algorithms)
  replay.py              replay buffer resident on the GPU
  inference.py           GPU evaluator + pipelined self-play thread
  train.py               training loop, checkpoints, benchmarks
  arena.py               matches between any two algorithms (or Edax) + Bradley-Terry
  league.py              round-robin Bradley-Terry league across runs and algorithms
  edax.py                matches against Edax, OLIVAW's protocol
  cli.py                 `thompsonzero train | ladder | league | bench | edax`
tests/                   Python end-to-end tests (Rust tests live in src/)
```

## Install (Linux, CUDA)

The RTX 5070 Ti is a Blackwell GPU (sm_120): use a PyTorch build for
**CUDA 12.8 or newer**.

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install maturin pytest
maturin develop --release        # builds src/ with -C target-cpu=native
```

`.cargo/config.toml` compiles for the host CPU (AVX-512 / BMI2 on the
Ryzen 9950X); build on the machine you train on.

## Use

```bash
thompsonzero train --game othello --rule corr          # -> runs/othello_corr
thompsonzero train --game boop --rule tz --set num_games=2048 --set wave=2
thompsonzero train --run-dir runs/othello_corr         # resume (config.json)
thompsonzero ladder --run-dir runs/othello_corr --watch   # rate benchmarks
thompsonzero league --dir runs/league --player az=runs/az50k_128 \
    --player tz=runs/tz50k --anchor az@50k --watch       # across runs / algorithms
thompsonzero bench --game othello                      # throughput per stage
```

Any `Config` field can be overridden with `--set key=value` (JSON values).
A run directory holds `config.json`, `train.log`, `metrics.jsonl` (one JSON
record per log interval), `latest.pt` (resumable), `buffer.pt` (the replay
buffer, so a resume does not start from an empty buffer) and `bench_<ep>.pt`
frozen networks.  The ladder plays each new benchmark against the previous
`--opponents` at `--sims` simulations and refits Bradley-Terry over all stored
W/D/L.

The league takes one benchmark per `--every` episodes (default 10k) from each
`--player` run, AlphaZero or ThompsonZero, and plays every newcomer against
everyone already in it: `--openings` XOT openings (default 100) from both
colours at `--sims` (default 400), each side searching with its own
algorithm and playing its evaluation-time pick.  After each newcomer it
refits Bradley-Terry over every stored W/D/L, prints Elo with standard errors
relative to `--anchor`, and appends to `ratings.jsonl`.  Matches between
different engines work because both Rust engines can run as a *search
service*: Python owns the games and hands each engine the positions where its
side is to move (`arena.play_games`); Edax matches use the same driver.

## AlphaZero baseline (`--rule az`)

A plain AlphaZero, for comparison with ThompsonZero and for reproducing
OLIVAW (Norelli & Panconesi, arXiv 2103.17228), an AlphaGo Zero-style Othello
engine that reached roughly Edax depth 8 at 1000 nodes per move.

* **Network** (`AZNet`): convolutional block, tower of AlphaGo Zero residual
  blocks (BatchNorm, no SE), policy head over all 65 action ids, tanh value
  head.
* **Search** (`src/puct.rs`): PUCT, `c_puct * P * sqrt(1 + N) / (1 + n)`,
  priors a softmax over the legal moves, unvisited children at the node's
  mean value minus `fpu_reduction`, virtual loss, terminal results exact.
* **Self-play** (`src/az.rs`): root Dirichlet noise alpha = min(1, 10 / legal
  moves), eps 0.25; moves sampled by visit count for 20 plies, most visited
  after; subtree reuse; OLIVAW's doubling (the most visited unplayed root
  child, visited at least 8 times, with its own visit distribution and its
  search value q as targets).  Forced moves are played without search.
* **Training**: `(z - v)^2 - pi . log p`, SGD with momentum 0.9 and L2
  1e-4, a random one of the 8 board symmetries per sample, the OLIVAW sims
  ramp (100, 200 from 20% of the run, 400 from 55%) and LR step points, and a
  training window that grows from the newest 2 to the newest 5 OLIVAW
  generations (2500 games each).
* **Where it departs from OLIVAW**, each measured in 5k-15k-episode sweeps
  (head-to-head matches plus Edax): the LR is 30x theirs (0.1, 0.03, 0.003;
  their 0.003 peak lost ~600 Elo), and training does 2 optimizer steps of
  batch 1024 per game instead of their ~6.4, which overfit.  The paper does
  not give the network width; 128 channels x 10 blocks led 64 and 256 at
  15k episodes, and at 50k 256 channels only drew level (same Edax scores,
  an even head-to-head) for 3.4x the training time.
* **Measured against Edax** (`thompsonzero edax`), the paper's protocol:
  Edax `-level k` with no book, random 8-move XOT openings from the
  10,784-sequence list, our side the most visited move of a fresh 1000-sim
  search, every opening from both colours.  OLIVAW's O-1000 scored about
  97% / 75% / 52% / 29% against E4 / E6 / E8 / E10 (Fig. 5a).  The `az`
  preset after 50k episodes (1h41m on an RTX 5070 Ti): 99.5% / 91% / 68.5% /
  46%, 200 games per level.

```bash
thompsonzero train --game othello --rule az            # -> runs/othello_az
thompsonzero edax --run-dir runs/othello_az --levels 4 6 8 10
```

Edax is not bundled: put `lEdax-x86-64-v3` and `data/eval.dat` from the
[v4.6 release](https://github.com/abulmo/edax-reversi/releases/tag/v4.6) and
`xot/openingslarge.txt` from
[the XOT site](https://berg.earthlingz.de/xot/downloads/openingslarge.txt) in
`~/code/edax-dist` (or `$THOMPSONZERO_EDAX`, or pass `--edax-dir`).

## Performance design

* **Rust engine, all cores.** Thousands of concurrent games, each with its
  own arena-allocated tree (nodes store their game state, so descent never
  replays moves).  `collect`/`apply` run in parallel over games with rayon
  (observation encoding included) and release the GIL.  Measured CPU cost per
  simulation, one thread: ~2-3 us for `corr`, `additive`, `mixture` and
  `avgconc` (~5-6 us on Boop, whose ~50 legal moves are each sampled at every
  node), ~7-8 us for `additive_mle` (~11 us on Boop), whose MLE refresh
  dominates.  A 16-core 9950X therefore searches well over 1M simulations/s,
  more than the GPU can evaluate: the GPU is the bottleneck by design.
* **Pipelined GPU inference.** Games are split into `groups`; while one
  group's batch is on the GPU (bf16, dedicated CUDA stream, `torch.compile`d
  with a symbolic batch size, padded only to a multiple of 128), the CPU
  searches the next group.  Batch size is up to `num_games / groups * wave`.
  With a simulated GPU that is the bottleneck, the self-play thread keeps it
  98-100% busy, including while a trainer thread competes for the GIL.
* **Self-play and training run concurrently.** Self-play has its own bf16
  copy of the network; the trainer pushes weights into it every
  `sync_every_steps` steps via a staging buffer, without either side
  stalling.  The replay ratio is fixed (`train_steps_per_episode`) and
  self-play blocks if it gets `max_pending` batches ahead.
* **GPU-resident replay buffer.** Samples are copied to the GPU once; batches
  are gathered there, so a training step moves nothing over PCIe.  The
  forward/loss is compiled and runs in bf16 autocast with fp32 loss maths.

Tuning: run `thompsonzero bench` (engine, inference and training throughput
in isolation).  If the GPU is underfed (small batches), raise `num_games`
(about 1 MB of RAM per concurrent Othello or Boop game at 400 sims, 0.1 MB for
Connect Four) or `wave` (more virtual-loss distortion per batch).  If
training cannot keep up, self-play simply waits.  `RAYON_NUM_THREADS` caps the
engine's threads.

## Extending

**A new backup / evidence rule** (`src/belief.rs`): add a `Rule` variant and
its name in `Rule::parse`; implement its collapse in `Rule::observed` (or give
it its own arm in `search_belief` / `edge_target` / `state_target` if it
combines with the network prior, like `corr`); if it needs a new running
statistic, add it to `Acc` behind a `Track` flag.  It is then selectable with
`--set search_rule=... --set target_rule=...`.  Leaf sampling instead of mean
backup is already a switch (`sample_leaf`).

**A new game:** implement `games::Game` (legal actions, apply, outcome,
mover-relative encoding), then list it in the `with_game!` macro and the
enums in `src/python.rs` and in `GAMES` in `config.py`.  The engine assumes a
two-player, zero-sum, deterministic, perfect-information game; players need
not alternate (perspective flips are by player id).

## Tests

```bash
cargo test --release          # games (perft, rules), beliefs, tree, engine
pytest tests                  # bindings, self-play samples, arena, loss, training
```

The Rust games were also cross-checked against the originals on random
playouts (legal actions, results and observations at every ply): 3000
Connect Four and 1000 Othello games against open_spiel's C++ games, 400
Boop games against the Python Boop of `boop_thompson_worker.py`.  The search
was checked the same way: the original `connect4_dirichlet_utils` search and
this tree, run 120-150 times each on the same Othello positions with the same
fixed network, give statistically indistinguishable root visit distributions,
state and edge targets, and solver results for all five evidence rules
(opening, middlegame and solved endgames).

## Relation to the open_spiel research code

Kept: the Dirichlet two-head network (init, concentration activation,
SE-ResNet trunk), Thompson selection with virtual loss, the five evidence
rules behind the runs of record (`mixture`, `additive`, `additive_mle`,
`avgconc`, `corr`), exact-terminal backups under `corr`, the MCTS solver and
its exact-label samples, tree reuse, doubling, the sims ramp, the step LR
schedule, the loss of record, and Bradley-Terry rating from stored W/D/L.

Dropped as experiment scaffolding: the unused loss terms (evidential, z-KL,
concentration MSE, ρ-drift and the ρ head), the `mean`/`sum`/`pathsum`/
`pathavg` rules, Gaussian selection, adaptive simulation budgets,
curricula/restarts, opponent pools, DirectML workarounds, notebooks,
Edax/XOT benchmarks, play servers, and the per-experiment run scripts.

Behavioural differences:

* **Boop line choice.** When several lines graduate at once, the original
  picked one at random, which made transitions stochastic and silently broke
  tree reuse and solver proofs.  Here the line with the most kittens is chosen
  (first in scan order on ties).  Everything else matches the original rules.
* **Perspective flips use player ids**, not ply parity, so games where a
  player can move twice are handled correctly.
* The replay buffer is saved with the checkpoint, so resuming no longer
  refills it from scratch.
