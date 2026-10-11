"""Training loss (the recipe of every run of record).

    klv   KL( search state belief  || V(s) )            distillation
    kla   KL( search edge beliefs  || Q(s,a) )          over every searched edge
    cev   -log E[V(s)]_z                                game outcome z
    cea   -log E[Q(s,a*)]_z                             the move actually played
    cons  KL( flip V(s') || Q(s,a*) )                   one-ply consistency,
                                                        V(s') stop-gradient

All KLs are the forward KL(target || prediction) between Dirichlets.
"""
import torch

from .network import alphas

TERMINAL_CONC, TERMINAL_EPS = 100.0, 0.05
_FLIP = [2, 1, 0]


def dirichlet_kl(a, b):
    """KL( Dir(a) || Dir(b) ) over the last axis."""
    a0, b0 = a.sum(-1), b.sum(-1)
    return (torch.lgamma(a0) - torch.lgamma(a).sum(-1) - torch.lgamma(b0) + torch.lgamma(b).sum(-1)
            + ((a - b) * (torch.digamma(a) - torch.digamma(a0).unsqueeze(-1))).sum(-1))


def _wmean(x, w):
    # where(), not x * w: a masked-out row must not leak an inf/nan
    return torch.where(w > 0, x * w, 0.0).sum() / w.sum().clamp_min(1.0)


def compute(v_raw, a_raw, v2_raw, b, weights):
    """`v_raw`/`a_raw` for the batch, `v2_raw` for each sample's successor
    (rows without one are ignored).  `b` is a batch from ReplayBuffer.sample.
    Returns (total, dict of detached scalars)."""
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
    q_played = a_alpha[rows, pidx]

    parts = {}
    parts['klv'] = dirichlet_kl(b['v'], v_alpha).mean()
    ev_mask = b['ev_mask'].float()
    parts['kla'] = _wmean(dirichlet_kl(b['ev'], a_alpha), ev_mask)
    parts['cev'] = _wmean(-v_logp.gather(1, z).squeeze(1), zw)
    parts['cea'] = _wmean(-a_logp[rows, pidx].gather(1, z).squeeze(1), pw)

    # consistency target: the successor's flipped belief, or the terminal spike
    nt = b['next_term']
    spike = torch.full((B, 3), TERMINAL_EPS, device=v_raw.device)
    spike[rows, nt.clamp_min(0)] = TERMINAL_CONC
    v2 = alphas(v2_raw).detach()[:, _FLIP]
    tgt = torch.where(b['has_next'].unsqueeze(1), v2, spike)
    cw = has_move * (b['has_next'] | (nt >= 0)).float()
    parts['cons'] = _wmean(dirichlet_kl(tgt, q_played), cw)

    total = sum(w * parts[k] for k, w in weights.items() if w)
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
    return total, parts

