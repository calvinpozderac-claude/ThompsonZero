import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *

V = []
def add(name, **o): V.append((name, o))
# A. PUCT constants
for c in (1.0, 2.0, 2.5, 3.0): add(f'puct c={c}', c_puct=c)
for f in (0.0, 0.1, 0.4): add(f'puct fpu={f}', fpu=f)
# B. backups under PUCT
for p in (1.5, 2.0, 3.0): add(f'backup power p={p}', backup='power', bparam=p)
for l in (0.1, 0.25, 0.5): add(f'backup mixmax l={l}', backup='mixmax', bparam=l)
for t in (0.05, 0.1, 0.2, 0.5): add(f'backup softmax t={t}', backup='softmax', bparam=t)
for s in (0.25, 0.5, 1.0): add(f'backup bayes leaf_s={s}', backup='bayes', leaf_s=s)
# C. final move
for fr in (0.1, 0.3, 0.5): add(f'pick q frac={fr}', pick='q', pick_frac=fr)
# D. pure Thompson (kappa 0), each sigma source
TS = dict(sel='ts')
for c in (0.25, 0.5, 1.0): add(f'ts u c={c}', **TS, sigma='u', ts_c=c)
for sg in ('count', 'netvar', 'emp'):
    for c in (0.25, 0.5, 1.0): add(f'ts {sg} c={c}', **TS, sigma=sg, ts_c=c)
for c in (0.5, 1.0): add(f'ts bayes c={c}', **TS, sigma='bayes', ts_c=c, leaf_s=0.5)
# E. PUCT + Thompson noise (kappa 1) and half-bonus
for sg in ('u', 'count', 'netvar', 'emp', 'bayes'):
    for c in (0.1, 0.25, 0.5): add(f'puct+ts {sg} c={c}', **TS, kappa=1.0, sigma=sg, ts_c=c, leaf_s=0.5)
add('ts k=0.5 u c=0.5', **TS, kappa=0.5, sigma='u', ts_c=0.5)
# F. optimistic draws and deterministic variance bonuses
for z in ('abs', 'pos'):
    for c in (0.5, 1.0, 1.5): add(f'ts u z={z} c={c}', **TS, sigma='u', ts_c=c, z=z)
for sg in ('emp', 'netvar'):
    for c in (0.25, 0.5): add(f'puct+ucb {sg} c={c}', **TS, kappa=1.0, sigma=sg, ts_c=c, z='one')
# G. where / when Thompson applies
for fr in (0.25, 0.5, 0.75): add(f'hybrid ts_from={fr}', **TS, sigma='u', ts_c=0.5, ts_from=fr)
for dp in ('root', 'interior'): add(f'ts u c=0.5 at {dp}', **TS, sigma='u', ts_c=0.5, depths=dp)
# H. eager child evaluation
for e in (0, 1, 99): add(f'puct eager={e}', eager=e)
for m in (2, 8): add(f'puct eager=all min={m}', eager=99, eager_min=m)
add('ts u c=0.5 eager=all', **TS, sigma='u', ts_c=0.5, eager=99)
add('ts u c=0.5 eager=1', **TS, sigma='u', ts_c=0.5, eager=1)
# I. Gumbel MuZero's rule
for cs in (0.1, 0.3, 1.0): add(f'gumbel c_scale={cs}', sel='gumbel', c_scale=cs, pick='improved')
add('gumbel c_scale=0.3 pick visits', sel='gumbel', c_scale=0.3)

print(len(V), 'variants', flush=True)
for net in sys.argv[1].split(','):
    for name, o in V:
        match(name, o, net=net)
