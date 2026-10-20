import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *

V = []
def add(name, b=None, sims_b=None, **o): V.append((name, o, b if b is not None else PUCT, sims_b))
TS = dict(sel='ts')
PQ = dict(pick='q', pick_frac=0.2)
# A. re-test leaders on fresh openings
for fr in (0.05, 0.1, 0.2, 0.3, 0.5): add(f'pick q frac={fr}', pick='q', pick_frac=fr)
add('puct eager=0', eager=0)
add('puct fpu=0.0', fpu=0.0)
add('backup mixmax l=0.25', backup='mixmax', bparam=0.25)
add('puct+ucb emp c=0.25', **TS, kappa=1.0, sigma='emp', ts_c=0.25, z='one')
add('puct+ts u c=0.1', **TS, kappa=1.0, sigma='u', ts_c=0.1)
# B. fitted sigma law
for c in (0.1, 0.25): add(f'puct+ts fit c={c}', **TS, kappa=1.0, sigma='fit', ts_c=c)
add('puct+ucb fit c=0.25', **TS, kappa=1.0, sigma='fit', ts_c=0.25, z='one')
# C. combinations on the max-q pick
add('pq + fpu0', **PQ, fpu=0.0)
add('pq + eager0', **PQ, eager=0)
add('pq + eager0 + fpu0', **PQ, eager=0, fpu=0.0)
add('pq + mixmax0.25', **PQ, backup='mixmax', bparam=0.25)
add('pq + ucb emp0.25', **PQ, **TS, kappa=1.0, sigma='emp', ts_c=0.25, z='one')
add('pq + ts u0.1', **PQ, **TS, kappa=1.0, sigma='u', ts_c=0.1)
add('pq + c1.0', **PQ, c_puct=1.0)
add('pq + eager0 + fpu0 + mixmax0.25', **PQ, eager=0, fpu=0.0, backup='mixmax', bparam=0.25)
add('pq + eager0 + fpu0 + ucb emp0.25', **PQ, eager=0, fpu=0.0, **TS, kappa=1.0, sigma='emp', ts_c=0.25, z='one')
# D. eager at (roughly) equal network evaluations: PUCT gets more sims
add('eager=99 vs puct@3200', sims_b=3200, eager=99)
add('eager=all min=2 vs puct@1400', sims_b=1400, eager=99, eager_min=2)
add('pq+eager=99 vs pq@3200', b=PQ, sims_b=3200, **PQ, eager=99)
print(len(V), 'variants', flush=True)
for name, o, b, sb in V:
    for net in ('scalar', 'wdl'):
        match(name, o, b=b, net=net, split='tune', sims_b=sb)
