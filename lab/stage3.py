"""Stage 3: tune around the best stage-2 combination (tune openings)."""
import sys; sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *

BEST = dict(pick='q', pick_frac=0.2, eager=0, fpu=0.0, backup='mixmax', bparam=0.25)
def var(**ch):
    o = dict(BEST, **ch)
    return {k: v for k, v in o.items() if v is not None}
V = [
    ('B lam=0.1', var(bparam=0.1)), ('B lam=0.5', var(bparam=0.5)),
    ('B frac=0.1', var(pick_frac=0.1)), ('B frac=0.3', var(pick_frac=0.3)),
    ('B no eager', var(eager=None)), ('B no fpu0', var(fpu=None)),
    ('B c=1.0', var(c_puct=1.0)), ('B c=1.25', var(c_puct=1.25)),
    ('B softmax t=0.5', var(backup='softmax', bparam=0.5)),
    ('B + ts fit c=0.1', var(sel='ts', kappa=1.0, sigma='fit', ts_c=0.1)),
    ('B + ts netvar c=0.1', var(sel='ts', kappa=1.0, sigma='netvar', ts_c=0.1)),
]
print(len(V), 'variants', flush=True)
for name, o in V:
    for net in ('scalar', 'wdl'):
        match(name, o, net=net, split='tune')
# low-sim check of the base combination
for net in ('scalar', 'wdl'):
    match('B @100', BEST, net=net, split='tune', sims=100)
