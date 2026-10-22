# Checkpoints

`az50k_128/`: the AlphaZero baseline (scalar value head, 128 channels x 10
blocks, 50k episodes; run as `runs/az50k_128`).  `bench_<episode>.pt` are the
benchmark networks every ~2500 episodes (load with
`thompsonzero.train.load_net`), `latest.pt` the final training state,
`config.json` its settings, and `train.log` / `metrics.jsonl` / `edax.json`
its logs.  The replay buffer is not included (751 MB).
