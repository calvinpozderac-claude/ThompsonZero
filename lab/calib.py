"""Calibration study: how well do per-edge uncertainty estimates from a
search predict the edge value's actual error (vs a deep search)?
Usage: calib.py <net> [positions]  -> runs/lab/calib_<net>.pkl"""
import pickle, sys
sys.path.insert(0, __import__('os').path.dirname(__file__))
from labrun import *
from thompsonzero import _core

def evaluate(ev, obs):
    out = ev.evaluate(obs)
    return [np.ascontiguousarray(x, dtype=np.float32) for x in out]

def run_searches(net, states, sims, opts, slots=512, seed=0):
    """One reporting search per state; returns [(rows, root)] in order."""
    ev = evaluator(net)
    slots = min(slots, len(states))
    eng = _core.LabEngine('othello', slots=slots, sims=sims, seed=seed, report=True, **opts)
    out = [None] * len(states)
    queue = list(range(len(states)))[::-1]
    slot_of, free = {}, list(range(slots))
    A = eng.num_actions
    while queue or slot_of:
        while queue and free:
            i, s = queue.pop(), free.pop()
            eng.search(s, states[i]); slot_of[s] = i
        obs, _ = eng.collect(0)
        if len(obs):
            eng.apply(0, *evaluate(ev, obs))
        else:
            eng.apply(0, np.zeros(0, np.float32), np.zeros((0, A), np.float32))
        eng.moves()
        for s, rows, root in eng.reports():
            out[slot_of.pop(s)] = (rows, np.array(root))
            free.append(s)
    return out

def positions(net, n, seed=0):
    """Positions from the calibration openings plus 0-44 plies sampled
    from the network's policy; non-terminal with >= 2 legal moves."""
    rng = np.random.default_rng(seed)
    ev = evaluator(net)
    live = []
    for op in openings('calib'):
        st = _core.GameState('othello')
        for a in op: st.apply(a)
        live.append([st, int(rng.integers(0, 45))])
    while any(k > 0 and not st.is_terminal() for st, k in live):
        act = [x for x in live if x[1] > 0 and not x[0].is_terminal()]
        obs = np.stack([x[0].observation() for x in act])
        logits = evaluate(ev, obs)[1]
        for x, lg in zip(act, logits):
            legal = x[0].legal_actions()
            p = np.exp(lg[legal] - lg[legal].max()); p /= p.sum()
            x[0].apply(int(rng.choice(legal, p=p))); x[1] -= 1
    out = [st for st, _ in live if not st.is_terminal() and len(st.legal_actions()) >= 2]
    return out[:n]

def child_truth(net, states, sims):
    """Deep-search value of every root edge's child, from the root mover's
    view: dict (position, action) -> value."""
    jobs, truth = [], {}
    for i, st in enumerate(states):
        me = st.current_player()
        for a in st.legal_actions():
            c = st.clone(); c.apply(a)
            if c.is_terminal():
                truth[i, a] = float(c.returns()[me])
            else:
                jobs.append((i, a, c, 1.0 if c.current_player() == me else -1.0))
    res = run_searches(net, [j[2] for j in jobs], sims, {})
    for (i, a, _, sgn), (rows, root) in zip(jobs, res):
        truth[i, a] = sgn * float(root[2])
    return truth

if __name__ == '__main__':
    net = sys.argv[1]
    npos = int(sys.argv[2]) if len(sys.argv) > 2 else 400
    t0 = time.time()
    states = positions(net, npos)
    print(net, len(states), 'positions', f'{time.time()-t0:.0f}s', flush=True)
    data = {'n': len(states)}
    data['truth'] = child_truth(net, states, 3200)
    print('truth 3200', f'{time.time()-t0:.0f}s', flush=True)
    data['truth2'] = child_truth(net, states, 1600)
    print('truth 1600', f'{time.time()-t0:.0f}s', flush=True)
    data['roottruth'] = [r[1][2] for r in run_searches(net, states, 3200, {})]
    runs = {
        ('puct', 50): {}, ('puct', 100): {}, ('puct', 400): {}, ('puct', 1600): {},
        ('eager0', 400): dict(eager=0),
        ('power2', 400): dict(backup='power', bparam=2.0),
        ('mixmax.25', 400): dict(backup='mixmax', bparam=0.25),
        ('softmax.1', 400): dict(backup='softmax', bparam=0.1),
        ('bayes.5', 400): dict(backup='bayes', leaf_s=0.5),
        ('bayes1', 400): dict(backup='bayes', leaf_s=1.0),
    }
    for (name, sims), o in runs.items():
        data[name, sims] = run_searches(net, states, sims, o)
        print(name, sims, f'{time.time()-t0:.0f}s', flush=True)
    with open(f'runs/lab/calib_{net}.pkl', 'wb') as f:
        pickle.dump(data, f)
