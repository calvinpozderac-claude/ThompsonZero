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

The two configurations of record (`--rule`):

| rule set | selection rule | target rule | notes |
|---|---|---|---|
| `tz`   | `additive_mle` | `avgconc` | OLIVAW-TZ arm |
| `corr` | `corr` (ρ = 0.8) | `corr` (ρ = 0.999) | correlated-error posterior; led TZ from ep15k in the Othello runs |

Both use the OLIVAW recipe: 64ch x 10 blocks, sims 100 -> 200 -> 400 over the
run, 6 optimizer steps of batch 1024 per game, 600k buffer, LR steps
3e-3 -> 1e-3 (20%) -> 1e-4 (55%).

## Layout

```
src/                     Rust crate (compiled into thompsonzero._core)
  games/{connect4,othello,boop}.rs   bitboard games behind the `Game` trait
  belief.rs              Dirichlet maths, accumulators, evidence rules
  tree.rs                Thompson-sampling tree, backup, solver, targets
  engine.rs              batched multi-threaded self-play / arena driver
  python.rs              PyO3 bindings (Engine, GameState)
python/thompsonzero/
  config.py              Config + per-game presets
  network.py             DirichletNet
  loss.py                the training loss
  replay.py              replay buffer resident on the GPU
  inference.py           GPU evaluator + pipelined self-play thread
  train.py               training loop, checkpoints, benchmarks
  arena.py               matches + Bradley-Terry ladder
  cli.py                 `thompsonzero train | ladder | bench`
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
thompsonzero bench --game othello                      # throughput per stage
```

Any `Config` field can be overridden with `--set key=value` (JSON values).
A run directory holds `config.json`, `train.log`, `metrics.jsonl` (one JSON
record per log interval), `latest.pt` (resumable), `buffer.pt` (the replay
buffer, so a resume does not start from an empty buffer) and `bench_<ep>.pt`
frozen networks.  The ladder plays each new benchmark against the previous
`--opponents` at `--sims` simulations and refits Bradley-Terry over all stored
W/D/L.

## Performance design

* **Rust engine, all cores.** Thousands of concurrent games, each with its
  own arena-allocated tree (nodes store their game state, so descent never
  replays moves).  `collect`/`apply` run in parallel over games with rayon and
  release the GIL.  On a 4-core container the engine alone does ~200k
  simulations/s with a constant network, so a 16-core 9950X has headroom of
  several hundred thousand network evaluations per second: the GPU is the
  bottleneck by design.
* **Pipelined GPU inference.** Games are split into `groups`; while one
  group's batch is on the GPU (bf16, dedicated CUDA stream, `torch.compile`d),
  the CPU searches the next group.  Batch size = `num_games / groups * wave`.
* **Self-play and training run concurrently.** Self-play has its own bf16
  copy of the network; the trainer pushes weights into it every
  `sync_every_steps` steps via a staging buffer, without either side
  stalling.  The replay ratio is fixed (`train_steps_per_episode`) and
  self-play blocks if it gets `max_pending` batches ahead.
* **GPU-resident replay buffer.** Samples are copied to the GPU once; batches
  are gathered there, so a training step moves nothing over PCIe.  The
  forward/loss is compiled and runs in bf16 autocast with fp32 loss maths.

Tuning: run `thompsonzero bench`.  If the GPU is underfed (small batches),
raise `num_games` (memory: about 0.5 KB per tree node for Othello, ~2 KB for Boop) or `wave` (more
virtual-loss distortion per batch).  If training cannot keep up, self-play
simply waits.  `RAYON_NUM_THREADS` caps the engine's threads.

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
Boop games against the Python Boop of `boop_thompson_worker.py`.

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
