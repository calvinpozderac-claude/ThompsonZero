# Search lab

Offline comparisons of search variants with fixed AZ networks, using
`thompsonzero._core.LabEngine` (src/lab.rs).  Run from the repository root:

```
.venv/bin/python lab/stage1.py scalar,wdl   # screen ~80 variants vs PUCT
.venv/bin/python lab/stage2.py              # re-test leaders, combinations, eager at equal evals
.venv/bin/python lab/stage3.py              # tune around the best combination
.venv/bin/python lab/final.py               # round robin on held-out openings, Bradley-Terry
.venv/bin/python lab/calib.py <net>         # uncertainty calibration vs deep search
.venv/bin/python lab/calib_analyze.py <net>
.venv/bin/python lab/summary.py [split] [sims]
```

Results append to `runs/lab/results.jsonl` (matches are cached by their
settings).  Findings, with the 50k-episode scalar and WDL nets, 200 games
per match, 200 held-out openings x 2 colours in the final:

| Bradley-Terry Elo vs PUCT       | scalar 400 | wdl 400 | scalar 100 | wdl 100 |
|---------------------------------|-----------:|--------:|-----------:|--------:|
| max-Q pick + mixmax 0.25        | +92        | +92     | +140       | +171    |
| + root eager eval, fpu 0        | +100       | +78     | +170       | +163    |
| max-Q pick (frac 0.2)           | +53        | +57     | +109       | +153    |
| PUCT + fitted-sigma TS noise    | +9         | +1      | -18        | -9      |

Pure Thompson selection (any sigma source), Bayes (Clark max) backups,
softmax backups with t <= 0.2, Gumbel's non-root rule and eager evaluation of
every child (at equal network evaluations) all lost to PUCT.
