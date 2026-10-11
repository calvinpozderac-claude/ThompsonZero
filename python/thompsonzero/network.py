"""The network: a residual trunk with two Dirichlet heads.

Both heads emit four numbers per belief, (logit_win, logit_draw, logit_loss,
raw concentration), from the mover's perspective:

    V(s)   = Dir(conc(raw) * softmax(logits))           state head, (B, 4)
    Q(s,a) = Dir(conc(raw_a) * softmax(logits_a))       action head, (B, A, 4)

There is no policy head: search selects by Thompson sampling Q.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

ALPHA_FLOOR = 1e-9
HEAD_INIT_STD = 0.01      # breaks symmetry; small enough not to be an opinion
HEAD_CONC_BIAS = 1.4      # softplus(1.4) = 1.62: weak initial confidence


def conc(raw):
    """Raw output -> concentration > 0, unbounded above.

    softplus computed accurately in both tails: exp(x) below -10 (where
    log1p(exp(x)) underflows to an exact 0 in fp32 and lgamma(0) would NaN the
    loss), identity above 20.  No clamp on the output, so the gradient never
    dies."""
    tail = torch.exp(raw.clamp(max=-10.0))
    body = F.softplus(raw.clamp(min=-10.0, max=20.0))
    return torch.where(raw > 20.0, raw, torch.where(raw < -10.0, tail, body))


def alphas(raw):
    """(..., 4) raw head output -> (..., 3) Dirichlet parameters (fp32)."""
    raw = raw.float()
    p = torch.softmax(raw[..., :3], dim=-1)
    return (conc(raw[..., 3:]) * p).clamp_min(ALPHA_FLOOR)


def activate(raw):
    """(..., 4) raw -> (..., 4) (p_win, p_draw, p_loss, concentration), the
    format the Rust engine consumes."""
    raw = raw.float()
    return torch.cat([torch.softmax(raw[..., :3], dim=-1), conc(raw[..., 3:])], dim=-1)


def _norm(ch):
    g = min(8, ch)
    while ch % g:
        g -= 1
    return nn.GroupNorm(g, ch)


class SEBlock(nn.Module):
    def __init__(self, ch, reduction=4):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.fc = nn.Sequential(nn.Linear(ch, mid), nn.ReLU(inplace=True), nn.Linear(mid, 2 * ch))

    def forward(self, x):
        scale, bias = self.fc(x.mean(dim=(2, 3))).chunk(2, dim=1)
        return x * torch.sigmoid(scale)[:, :, None, None] + bias[:, :, None, None]


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _norm(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), _norm(ch))
        self.se = SEBlock(ch)

    def forward(self, x):
        return F.relu(self.se(self.body(x)) + x)


class DirichletNet(nn.Module):
    """forward(x) -> (v_raw (B, 4), a_raw (B, A, 4))."""

    def __init__(self, obs_shape, num_actions, channels=64, blocks=10, head_channels=8):
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape, self.num_actions = tuple(obs_shape), num_actions
        self.stem = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1, bias=False),
                                  _norm(channels), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(channels, head_channels, 1, bias=False),
                                  _norm(head_channels), nn.ReLU(inplace=True), nn.Flatten())
        flat = head_channels * h * w
        self.v_out = nn.Linear(flat, 4)
        self.a_out = nn.Linear(flat, num_actions * 4)
        # near-uniform beliefs with weak confidence, so search dominates early
        with torch.no_grad():
            for lin in (self.v_out, self.a_out):
                nn.init.normal_(lin.weight, std=HEAD_INIT_STD)
                lin.bias.view(-1, 4).zero_()
                lin.bias.view(-1, 4)[:, 3] = HEAD_CONC_BIAS

    def forward(self, x):
        h = self.head(self.trunk(self.stem(x)))
        return self.v_out(h), self.a_out(h).view(-1, self.num_actions, 4)


def build(cfg, obs_shape, num_actions):
    return DirichletNet(obs_shape, num_actions, cfg.channels, cfg.blocks, cfg.head_channels)
