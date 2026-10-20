"""Analyse runs/lab/calib_<net>.pkl (see calib.py)."""
import pickle, sys
import numpy as np

def spearman(x, y):
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return np.corrcoef(rx, ry)[0, 1]

A, P, N, Q, MEAN, W2N, VSN, BM, BS2, TERM, CV = range(11)

def table(data, run):
    """Rows over (position, edge): position, action, prior, n, q, mean,
    w2n, vsn, bm, bs2, term, cv, N, root var, root mu, truth, truth2."""
    out = []
    for i, (rows, root) in enumerate(data[run]):
        for r in rows:
            a = int(r[A])
            out.append([i, a, r[P], r[N], r[Q], r[MEAN], r[W2N], r[VSN], r[BM], r[BS2], r[TERM], r[CV],
                        root[3], root[1], root[2], data['truth'][i, a], data['truth2'][i, a]])
    t = np.array(out, dtype=np.float64)
    names = 'pos act P n q mean w2n vsn bm bs2 term cv N rvar rmu truth truth2'.split()
    return {k: t[:, j] for j, k in enumerate(names)}

def sources(t, n0=1.0):
    n, Pr, Nt = t['n'], t['P'], t['N']
    m2 = np.maximum(n * (t['w2n'] - t['mean'] ** 2), 0)
    return {
        'u (P sqrtN/(1+n))': Pr * np.sqrt(1 + Nt) / (1 + n),
        'sqrtP sqrtN/(1+n)': np.sqrt(Pr) * np.sqrt(1 + Nt) / (1 + n),
        'sqrtN/(1+n)': np.sqrt(1 + Nt) / (1 + n),
        '1/(1+n)': 1 / (1 + n),
        'count 1/sqrt(1+n)': 1 / np.sqrt(1 + n),
        'netvar': np.sqrt(np.maximum(t['vsn'], 1e-4) / (1 + n)),
        'emp (n0=1)': np.sqrt((n0 * t['rvar'] + m2) / (n0 + n) / (1 + n)),
        'emp (n0=0.1)': np.sqrt((0.1 * t['rvar'] + m2) / (0.1 + n) / (1 + n)),
        'bayes (leaf_s=1)': np.sqrt(np.maximum(t['bs2'], 1e-8)),
    }

def nll(e, s, tau2):
    v = s * s + tau2
    return 0.5 * np.mean(np.log(2 * np.pi * v) + e * e / v)

def fit_scale(e, s, tau2):
    grid = np.exp(np.linspace(-6, 4, 400))
    vals = [nll(e, a * s, tau2) for a in grid]
    return grid[int(np.argmin(vals))]

def cv_nll(e, s, pos, tau2):
    """Two-fold (even/odd positions) cross-validated NLL at a fitted scale."""
    tot = 0
    for k in (0, 1):
        tr, te = pos % 2 == k, pos % 2 != k
        a = fit_scale(e[tr], s[tr], tau2)
        tot += nll(e[te], a * s[te], tau2) * te.sum()
    return tot / len(e)

def fit_powerlaw(t, e, tau2, idx):
    """log sigma = theta . features, fitted by Gaussian ML (torch L-BFGS)."""
    import torch
    n, Pr, Nt = t['n'], t['P'], t['N']
    m2 = np.maximum(n * (t['w2n'] - t['mean'] ** 2), 0)
    X = np.stack([np.ones_like(n), np.log1p(n), np.log(Pr), np.log1p(Nt),
                  np.log(np.maximum(t['vsn'], 1e-4)), np.log((0.1 * t['rvar'] + m2) / (0.1 + n))], 1)
    def f(th, X, e):
        s2 = np.exp(2 * X @ th) + tau2
        return 0.5 * np.mean(np.log(2 * np.pi * s2) + e * e / s2)
    def minimize(X, e):
        Xt, et = torch.tensor(X), torch.tensor(e)
        th = torch.zeros(X.shape[1], dtype=torch.float64, requires_grad=True)
        opt = torch.optim.LBFGS([th], max_iter=500, line_search_fn='strong_wolfe')
        def closure():
            opt.zero_grad()
            s2 = torch.exp(2 * Xt @ th) + tau2
            loss = 0.5 * torch.mean(torch.log(s2) + et * et / s2)
            loss.backward()
            return loss
        opt.step(closure)
        return th.detach().numpy()
    tot = 0
    for k in (0, 1):
        tr, te = idx % 2 == k, idx % 2 != k
        th = minimize(X[tr], e[tr])
        tot += f(th, X[te], e[te]) * te.sum()
    th = minimize(X, e)
    return tot / len(e), th

def analyse(net, run=('puct', 400)):
    data = pickle.load(open(f'runs/lab/calib_{net}.pkl', 'rb'))
    t = table(data, run)
    keep = (t['n'] >= 1) & np.isnan(t['term'])
    d = {k: v[keep] for k, v in t.items()}
    e = d['truth'] - d['q']
    noise = d['truth'] - d['truth2']
    tau2 = max(np.var(noise) / 2, 1e-6)   # truth(3200)'s own error, roughly
    print(f'\n=== {net} {run}: {len(e)} visited edges in {data["n"]} positions; '
          f'rms err {np.sqrt(np.mean(e**2)):.3f}; truth noise rms ~{np.sqrt(tau2):.3f}')
    print(f'{"source":<22} {"cv NLL":>8} {"spearman":>9}   ratio rms(err)/sigma by visits 1 | 2-3 | 4-10 | 11-30 | 31-100 | >100')
    buckets = [(1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 1e9)]
    res = {}
    for name, s in sources(d).items():
        c = cv_nll(e, s, d['pos'], tau2)
        rho = spearman(s, np.abs(e))
        a = fit_scale(e, s, tau2)
        ratios = []
        for lo, hi in buckets:
            m = (d['n'] >= lo) & (d['n'] <= hi)
            ratios.append(np.sqrt(np.mean(e[m] ** 2)) / np.sqrt(np.mean((a * s[m]) ** 2 + tau2)) if m.sum() > 20 else np.nan)
        res[name] = c
        print(f'{name:<22} {c:8.3f} {rho:9.3f}   ' + ' '.join(f'{x:5.2f}' for x in ratios) + f'   (scale {a:.3f})')
    c, th = fit_powerlaw(d, e, tau2, d['pos'])
    print(f'{"fitted power law":<22} {c:8.3f}            theta [1, log1p n, log P, log1p N, log vsn, log emp s2] = {np.round(th, 2)}')
    return data, t

def decisions(net, run=('puct', 400), scale_src='emp (n0=1)'):
    """How well does each belief pick / identify the best root move?"""
    data = pickle.load(open(f'runs/lab/calib_{net}.pkl', 'rb'))
    t = table(data, run)
    rng = np.random.default_rng(0)
    vis = (t['n'] >= 1)
    e = (t['truth'] - t['q'])[vis & np.isnan(t['term'])]
    srcs = sources(t)
    scales = {k: fit_scale(e, s[vis & np.isnan(t['term'])], 0.0) for k, s in srcs.items()}
    regret = {'most visited': [], 'max q (n>=10% max)': [], 'max q (any visited)': [], 'max prior': []}
    for k in srcs: regret[f'max P(best) {k}'] = []
    for k in ('u (P sqrtN/(1+n))', 'count 1/sqrt(1+n)', 'netvar', 'emp (n0=1)'):
        for b in (0.5, 1.0, 2.0): regret[f'LCB q-{b}sd {k}'] = []
    ll = {k: [] for k in ['visit share', 'prior'] + list(srcs)}
    missed = 0
    for i in range(data['n']):
        m = t['pos'] == i
        n, q, tr, Pr, term = t['n'][m], t['q'][m], t['truth'][m], t['P'][m], t['term'][m]
        best_all = tr.max()
        v = n >= 1
        if tr[~v].size and tr[~v].max() > tr[v].max() + 0.05:
            missed += 1
        regret['most visited'].append(best_all - tr[np.argmax(n + 1e-6 * np.nan_to_num(q, nan=-9))])
        regret['max q (n>=10% max)'].append(best_all - tr[np.argmax(np.where(n >= 0.1 * n.max(), q, -9))])
        regret['max q (any visited)'].append(best_all - tr[np.argmax(np.where(v, q, -9))])
        regret['max prior'].append(best_all - tr[np.argmax(Pr)])
        tb = np.argmax(np.where(v, tr, -9))  # truly best visited move
        ll['visit share'].append(-np.log(max(n[tb] / n.sum(), 1e-4)))
        ll['prior'].append(-np.log(max(Pr[tb] / Pr[v].sum(), 1e-4)))
        for k, s in srcs.items():
            sig = np.where(np.isnan(term), scales[k] * s[m], 0.0)
            qq = np.where(np.isnan(term), q, term)
            draws = qq[v][None, :] + sig[v][None, :] * rng.standard_normal((4000, v.sum()))
            pb = np.bincount(draws.argmax(1), minlength=v.sum()) / 4000
            idx = np.flatnonzero(v)
            ll[k].append(-np.log(max(pb[list(idx).index(tb)], 1e-4)))
            regret[f'max P(best) {k}'].append(best_all - tr[idx[np.argmax(pb)]])
            for b in (0.5, 1.0, 2.0):
                if f'LCB q-{b}sd {k}' in regret:
                    regret[f'LCB q-{b}sd {k}'].append(best_all - tr[idx[np.argmax((qq - b * sig)[v])]])
    print(f'\n--- {net} {run}: root decisions over {data["n"]} positions '
          f'(true best move unvisited and >0.05 better in {missed})')
    for k, r in regret.items():
        r = np.array(r)
        print(f'  regret {k:<38} mean {r.mean():.4f}  P(regret>0.05) {np.mean(r > 0.05):.3f}')
    for k, x in ll.items():
        print(f'  best-move log-loss {k:<28} {np.mean(x):.3f}')

def backups(net):
    data = pickle.load(open(f'runs/lab/calib_{net}.pkl', 'rb'))
    rt = np.array(data['roottruth'])
    print(f'\n--- {net}: value accuracy at 400 sims by backup (edges visited >= 1; and root)')
    for run in [('puct', 400), ('power2', 400), ('mixmax.25', 400), ('softmax.1', 400), ('bayes.5', 400), ('bayes1', 400), ('eager0', 400)]:
        t = table(data, run)
        m = (t['n'] >= 1) & np.isnan(t['term'])
        e = t['truth'][m] - t['q'][m]
        w = t['n'][m]
        roots = np.array([r[1][2] for r in data[run]])
        print(f'  {run[0]:<10} edge rms {np.sqrt(np.mean(e**2)):.4f}  visit-weighted rms {np.sqrt(np.sum(w*e**2)/w.sum()):.4f}  '
              f'bias {np.mean(e):+.4f}   root rms {np.sqrt(np.mean((roots-rt)**2)):.4f}  root bias {np.mean(rt-roots):+.4f}')

def eager_info(net):
    """Unvisited and lightly visited moves: what does a child evaluation add?"""
    data = pickle.load(open(f'runs/lab/calib_{net}.pkl', 'rb'))
    t = table(data, ('eager0', 400))
    nt = np.isnan(t['term'])
    cv = t['cv'][nt]; tr = t['truth'][nt]; n = t['n'][nt]; rmu = t['rmu'][nt]
    print(f'\n--- {net}: one network evaluation of a child vs the truth, by the visits PUCT gave it')
    print(f'  all children: rms(child eval - truth) {np.sqrt(np.mean((cv-tr)**2)):.4f}; '
          f'rms(FPU - truth) {np.sqrt(np.mean((rmu-0.2-tr)**2)):.4f}')
    tp = table(data, ('puct', 400))
    for lo, hi in [(1, 1), (2, 3), (4, 10), (11, 30), (31, 100), (101, 1e9)]:
        m = (tp['n'] >= lo) & (tp['n'] <= hi) & np.isnan(tp['term'])
        print(f'  PUCT-400 edges with n in [{lo},{int(min(hi,9999))}]: rms(q - truth) {np.sqrt(np.mean((tp["q"][m]-tp["truth"][m])**2)):.4f}  ({m.sum()} edges)')

if __name__ == '__main__':
    for net in sys.argv[1:]:
        for run in [('puct', 100), ('puct', 400), ('puct', 1600)]:
            analyse(net, run)
        decisions(net, ('puct', 100)); decisions(net, ('puct', 400))
        backups(net)
        eager_info(net)
