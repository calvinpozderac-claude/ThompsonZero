"""Training loss (the recipe of every run of record).

    klv   KL( search state belief  || V(s) )            distillation
    kla   KL( search edge beliefs  || Q(s,a) )          over every searched edge
    cev   -log E[V(s)]_z                                game outcome z
    cea   -log E[Q(s,a*)]_z                             the move actually played
    cons  KL( flip V(s') || Q(s,a*) )                   one-ply consistency,
                                                        V(s') stop-gradient

All KLs are the forward KL(target || prediction) between Dirichlets.  With
`split`, each KL term instead trains the mean and the concentration apart
(`split_kl`).
"""
import torch

from .network import ALPHA_FLOOR, alphas, cal_conc, conc

TERMINAL_CONC, TERMINAL_EPS = 100.0, 0.05
TARGET_EPS = 0.05          # = belief.rs TARGET_EPS
_FLIP = [2, 1, 0]


def safe_target(a):
    """A network belief made safe as a KL target, as belief.rs does for
    search targets: total capped at TERMINAL_CONC, then every component
    floored at TARGET_EPS.  Without the floor a saturated head gives target
    components near 0, where digamma ~ -1/a and the KL (and its gradient)
    blow up."""
    tot = a.sum(-1, keepdim=True)
    a = a * (TERMINAL_CONC / tot).clamp(max=1.0)
    return a.clamp_min(TARGET_EPS)


def dirichlet_kl(a, b):
    """KL( Dir(a) || Dir(b) ) over the last axis."""
    a0, b0 = a.sum(-1), b.sum(-1)
    return (torch.lgamma(a0) - torch.lgamma(a).sum(-1) - torch.lgamma(b0) + torch.lgamma(b).sum(-1)
            + ((a - b) * (torch.digamma(a) - torch.digamma(a0).unsqueeze(-1))).sum(-1))


def split_kl(t, raw, mean_w=1.0):
    """A KL(target || Dir) term with the mean and the concentration learned
    apart: cross-entropy to the target's mean trains the mean, and the KL
    with that mean detached trains the concentration.  The Dirichlet KL's
    pull on the mean grows with the predicted concentration, so with
    confident targets it would otherwise drown out the game-outcome terms.

    Target components are floored at TARGET_EPS, as `safe_target` does: a
    search target can carry ~1e-9 on an outcome its leaves all ruled out,
    where digamma ~ -1/a makes the KL ~1e9.  The coupled KL survives that by
    driving the predicted mean there to 0 too; with the mean detached the
    only way out would be collapsing the concentration.

    `mean_w` scales the cross-entropy: how hard the bootstrapped search
    means pull on the network's means, next to the game-outcome terms."""
    t = t.clamp_min(TARGET_EPS)
    raw = raw.float()
    logp = torch.log_softmax(raw[..., :3], dim=-1)
    ce = -(t / t.sum(-1, keepdim=True) * logp).sum(-1)
    a = (conc(raw[..., 3:]) * logp.exp().detach()).clamp_min(ALPHA_FLOOR)
    return mean_w * ce + dirichlet_kl(t, a)


def cal_kl(t, raw, c):
    """The calibrated channel's loss: KL from the point target (the search
    target's mean at concentration `c`) to Dir(cal_conc * mean), with the
    mean detached.  `c` far above any calibrated value lets the channel
    learn the concentration its own mean error supports."""
    pt = (c * t / t.sum(-1, keepdim=True)).clamp_min(TARGET_EPS)
    p = torch.softmax(raw.float()[..., :3], dim=-1).detach()
    return dirichlet_kl(pt, (cal_conc(raw) * p).clamp_min(ALPHA_FLOOR))


def _wmean(x, w):
    # where(), not x * w: a masked-out row must not leak an inf/nan
    return torch.where(w > 0, x * w, 0.0).sum() / w.sum().clamp_min(1.0)


def compute(v_raw, a_raw, v2_raw, b, weights, split=False, mean_w=1.0, cal_c=0.0):
    """`v_raw`/`a_raw` for the batch, `v2_raw` for each sample's successor
    (rows without one are ignored).  `b` is a batch from ReplayBuffer.sample.
    Returns (total, dict of detached scalars)."""
    kl = (lambda t, raw: split_kl(t, raw, mean_w)) if split else (lambda t, raw: dirichlet_kl(t, alphas(raw)))
    B = v_raw.shape[0]
    rows = torch.arange(B, device=v_raw.device)
    v_alpha = alphas(v_raw)
    a_alpha = alphas(a_raw)
    v_logp = torch.log_softmax(v_raw.float()[:, :3], dim=-1)
    a_logp = torch.log_softmax(a_raw.float()[..., :3], dim=-1)

    played = b['played']
    has_move = (played >= 0).float()
    pidx = played.clamp_min(0)
    z = b['z'].unsqueeze(1)
    zw = b['z_w']
    pw = has_move * zw

    parts = {}
    parts['klv'] = kl(b['v'], v_raw).mean()
    ev_mask = b['ev_mask'].float()
    parts['kla'] = _wmean(kl(b['ev'], a_raw), ev_mask)
    parts['cev'] = _wmean(-v_logp.gather(1, z).squeeze(1), zw)
    parts['cea'] = _wmean(-a_logp[rows, pidx].gather(1, z).squeeze(1), pw)

    # consistency target: the successor's flipped belief, or the terminal spike
    nt = b['next_term']
    spike = torch.full((B, 3), TERMINAL_EPS, device=v_raw.device)
    spike[rows, nt.clamp_min(0)] = TERMINAL_CONC
    v2 = safe_target(alphas(v2_raw).detach()[:, _FLIP])
    tgt = torch.where(b['has_next'].unsqueeze(1), v2, spike)
    cw = has_move * (b['has_next'] | (nt >= 0)).float()
    parts['cons'] = _wmean(kl(tgt, a_raw[rows, pidx]), cw)

    total = sum(w * parts[k] for k, w in weights.items() if w)
    if cal_c > 0:
        parts['calv'] = cal_kl(b['v'], v_raw, cal_c).mean()
        parts['cala'] = _wmean(cal_kl(b['ev'], a_raw, cal_c), ev_mask)
        total = total + parts['calv'] + parts['cala']
    with torch.no_grad():
        parts = {k: v.detach() for k, v in parts.items()}
        parts['loss'] = total.detach()
        # mean concentration, predicted vs target (unsolved samples only:
        # solver spikes would otherwise dominate the means)
        uns = (~b['solved']).float()
        parts['v_conc'] = _wmean(v_alpha.sum(-1), uns)
        parts['v_conc_tgt'] = _wmean(b['v'].sum(-1), uns)
        m = ev_mask * uns.unsqueeze(1)
        parts['a_conc'] = _wmean(a_alpha.sum(-1), m)
        parts['a_conc_tgt'] = _wmean(b['ev'].sum(-1), m)
        if cal_c > 0:
            parts['v_cal'] = _wmean(cal_conc(v_raw).squeeze(-1), uns)
            parts['a_cal'] = _wmean(cal_conc(a_raw).squeeze(-1), m)
    return total, parts

