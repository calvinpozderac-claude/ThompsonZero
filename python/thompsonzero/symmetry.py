"""Board-symmetry augmentation of training batches (Othello).

Othello's rules are invariant under the eight symmetries of the square, so
every training row can be replaced by a random one of its eight images, as
AlphaGo Zero did.  Everything that refers to a square moves together:
observations, per-action targets and masks, and the index of the move played.
The pass action (id 64) is fixed by every symmetry.

Works on both replay layouts (`replay.py`): AlphaZero rows (obs, pi) and
ThompsonZero rows (obs, next_obs, legal, ev, ev_mask, played).
"""
import torch

_SQUARES = {'othello': 8}
_SPATIAL = ('obs', 'next_obs')                 # (n, C, s, s)
_PER_ACTION = ('pi', 'legal', 'ev', 'ev_mask')  # (n, A, ...)


def _d4(x, k):
    """Symmetry k (0..7) of the trailing two (square) axes."""
    if k >= 4:
        x = x.flip(-1)
    return torch.rot90(x, k % 4, dims=(-2, -1))


def tables(s, device=None):
    """(src, dst), both (8, s*s): under symmetry k, square p of the image
    shows square src[k, p] of the original, and square i goes to dst[k, i]."""
    idx = torch.arange(s * s, device=device).reshape(s, s)
    src = torch.stack([_d4(idx, k).reshape(-1) for k in range(8)])
    dst = torch.argsort(src, dim=1)
    return src, dst


def augment(b, game, k=None):
    """Apply an independent random symmetry to every row of batch `b` (or
    symmetry k[i] to row i)."""
    if game not in _SQUARES:
        raise ValueError(f'no symmetry augmentation for {game!r}')
    s = _SQUARES[game]
    ss = s * s
    obs = b['obs']
    n, dev = obs.shape[0], obs.device
    if k is None:
        k = torch.randint(0, 8, (n,), device=dev)
    src, dst = tables(s, dev)
    src, dst = src[k], dst[k]                      # (n, s*s) per row
    out = dict(b)
    for key in _SPATIAL:
        if key in b:
            x = b[key]
            flat = x.reshape(n, x.shape[1], ss)
            out[key] = flat.gather(2, src[:, None, :].expand_as(flat)).reshape(x.shape)
    for key in _PER_ACTION:
        if key in b:
            x = b[key]
            board = x[:, :ss]
            idx = src.reshape(n, ss, *([1] * (x.dim() - 2))).expand_as(board)
            out[key] = torch.cat([board.gather(1, idx), x[:, ss:]], dim=1)
    if 'played' in b:
        p = b['played']
        on_board = (p >= 0) & (p < ss)
        moved = dst.gather(1, p.clamp(0, ss - 1).long()[:, None]).squeeze(1).to(p.dtype)
        out['played'] = torch.where(on_board, moved, p)
    return out
