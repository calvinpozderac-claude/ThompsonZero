"""Screen AZ training-target variants: each trains to `stop_at` episodes (on
the 50k-episode schedule) and joins its own league, seeded with the AZ
baseline's games so only the new run's matches are played.
Usage: target_screen.py [name ...]   (default: every variant, in order)"""
import json, os, subprocess, sys, time

TZ = '.venv/bin/thompsonzero'
BASE_LEAGUE = 'runs/league_azmm/league.json'
STOP = 20000
VARIANTS = {
    'tqblend': dict(q_blend=0.5),
    'tkids': dict(double_aux=8, aux_pow=1.0),
    'tqhead': dict(q_head=1.0),
    'town': dict(own_head=1.0),
    'tdeep': dict(double_aux=8, aux_depth=3, aux_pow=1.0),
    'tstv': dict(stv_head=1.0),
    'tminimax': dict(double_aux=8, aux_pow=1.0, q_lambda=1.0, q_min_visits=4),
    'treply': dict(reply_head=0.15),
    'tvar': dict(var_head=1.0),
    # follow-ups: the two winning heads together; root-children doubling with
    # the replay window counted in played positions; all three
    'townreply': dict(own_head=1.0, reply_head=0.15),
    'tkidsw': dict(double_aux=8, aux_pow=1.0, window_main=True, buffer_size=5_200_000),
    'tall': dict(own_head=1.0, reply_head=0.15, double_aux=8, aux_pow=1.0, window_main=True,
                 buffer_size=5_200_000),
}


def seed_league(path):
    d = json.load(open(BASE_LEAGUE))
    keep = {p for p in d['players'] if p.startswith('az@')}
    d['players'] = {p: v for p, v in d['players'].items() if p in keep}
    d['pairs'] = {k: v for k, v in d['pairs'].items() if set(k.split('|')) <= keep}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(d, open(path, 'w'))


for name in sys.argv[1:] or list(VARIANTS):
    run, league = f'runs/{name}', f'runs/league_{name}'
    if os.path.exists(f'{run}/bench_{STOP}.pt') or os.path.exists(f'{run}/latest.pt'):
        print(f'{name}: already run, skipping', flush=True)
        continue
    os.makedirs(run, exist_ok=True)
    seed_league(f'{league}/league.json')
    sets = sum((['--set', f'{k}={v}'] for k, v in
                dict(VARIANTS[name], stop_at=STOP, benchmark_every=2500).items()), [])
    print(f'{time.strftime("%H:%M")} {name}: training {VARIANTS[name]}', flush=True)
    with open(f'{run}/stdout.log', 'w') as out:
        train = subprocess.Popen([TZ, 'train', '--game', 'othello', '--rule', 'az',
                                  '--run-dir', run, *sets], stdout=out, stderr=subprocess.STDOUT)
    time.sleep(5)
    with open(f'{league}/watch.log', 'a') as out:
        subprocess.Popen([TZ, 'league', '--dir', league, '--player', 'az=runs/az50k_128',
                          '--player', f'{name}={run}', '--anchor', 'az@50k', '--watch'],
                         stdout=out, stderr=subprocess.STDOUT)
    train.wait()
    print(f'{time.strftime("%H:%M")} {name}: training done (exit {train.returncode})', flush=True)
