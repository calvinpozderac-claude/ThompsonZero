"""The networks.

ThompsonZero (`DirichletNet`): a residual trunk with two Dirichlet heads.

Both heads emit four numbers per belief, (logit_win, logit_draw, logit_loss,
raw concentration), from the mover's perspective:

    V(s)   = Dir(conc(raw) * softmax(logits))           state head, (B, 4)
    Q(s,a) = Dir(conc(raw_a) * softmax(logits_a))       action head, (B, A, 4)

There is no policy head: search selects by Thompson sampling Q.

With `cal`, each belief gets a fifth number, a CALIBRATED log-concentration
read off the trunk features with the gradient stopped (so it cannot change
what the trunk or the means learn), trained on high-concentration point
targets (`loss.cal_kl`).  `use_cal` makes the search see it in place of the
model's own concentration.

The trunk is either the research one (`trunk='se'`: GroupNorm residual blocks
with squeeze-excitation) or AlphaGo Zero's (`trunk='az'`: BatchNorm, no SE),
the same trunk `AZNet` uses, so a TZ net can differ from the AlphaZero
baseline in its heads only.

AlphaZero (`AZNet`): the AlphaGo Zero network, as OLIVAW uses it -- a
convolutional block, a tower of plain residual blocks (BatchNorm, no
squeeze-excitation), a policy head (logits over every action id) and a tanh
value head.

`postprocess` turns a forward pass into the arrays the matching Rust engine
consumes, and `empty_outputs` is the same for a batch of zero rows.
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


CAL_INIT = 2.3           # exp(2.3) = 10
CAL_RANGE = (-5.0, 12.0)  # exp: 0.007 .. 163k


def cal_conc(raw):
    """The calibrated channel (raw[..., 4]) -> concentration (exp: it spans
    orders of magnitude between openings and endgames)."""
    return torch.exp(raw[..., 4:5].float().clamp(*CAL_RANGE))


def alphas(raw):
    """(..., 4 or 5) raw head output -> (..., 3) Dirichlet parameters (fp32)
    with the model's own concentration."""
    raw = raw.float()
    p = torch.softmax(raw[..., :3], dim=-1)
    return (conc(raw[..., 3:4]) * p).clamp_min(ALPHA_FLOOR)


def activate(raw, cal=False):
    """(..., 4 or 5) raw -> (..., 4) (p_win, p_draw, p_loss, concentration),
    the format the Rust engine consumes; `cal`: the calibrated concentration."""
    raw = raw.float()
    c = cal_conc(raw) if cal else conc(raw[..., 3:4])
    return torch.cat([torch.softmax(raw[..., :3], dim=-1), c], dim=-1)


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

    def __init__(self, obs_shape, num_actions, channels=64, blocks=10, head_channels=8, trunk='se',
                 cal=False):
        super().__init__()
        self.use_cal = False
        c, h, w = obs_shape
        self.obs_shape, self.num_actions = tuple(obs_shape), num_actions
        if trunk == 'se':
            norm, block = _norm, ResBlock
        elif trunk == 'az':
            norm, block = nn.BatchNorm2d, AZResBlock
        else:
            raise ValueError(f'unknown trunk {trunk!r}; expected se | az')
        self.stem = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1, bias=False),
                                  norm(channels), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[block(channels) for _ in range(blocks)])
        self.head = nn.Sequential(nn.Conv2d(channels, head_channels, 1, bias=False),
                                  norm(head_channels), nn.ReLU(inplace=True), nn.Flatten())
        flat = head_channels * h * w
        self.v_out = nn.Linear(flat, 4)
        self.a_out = nn.Linear(flat, num_actions * 4)
        # near-uniform beliefs with weak confidence, so search dominates early
        with torch.no_grad():
            for lin in (self.v_out, self.a_out):
                nn.init.normal_(lin.weight, std=HEAD_INIT_STD)
                lin.bias.view(-1, 4).zero_()
                lin.bias.view(-1, 4)[:, 3] = HEAD_CONC_BIAS
        self.cal = cal
        if cal:
            self.v_cal = nn.Linear(flat, 1)
            self.a_cal = nn.Linear(flat, num_actions)
            with torch.no_grad():
                for lin in (self.v_cal, self.a_cal):
                    nn.init.normal_(lin.weight, std=HEAD_INIT_STD)
                    lin.bias.fill_(CAL_INIT)

    def forward(self, x):
        h = self.head(self.trunk(self.stem(x)))
        v, a = self.v_out(h), self.a_out(h).view(-1, self.num_actions, 4)
        if self.cal:
            hd = h.detach()
            v = torch.cat([v, self.v_cal(hd)], dim=-1)
            a = torch.cat([a, self.a_cal(hd).unsqueeze(-1)], dim=-1)
        return v, a

    def postprocess(self, v, a):
        """-> (v (B, 4), a (B, A, 4)) activated, for `Engine.apply`."""
        cal = self.cal and self.use_cal
        return activate(v, cal), activate(a, cal)

    def empty_outputs(self):
        return (torch.zeros((0, 4)).numpy(), torch.zeros((0, self.num_actions, 4)).numpy())


class AZResBlock(nn.Module):
    """AlphaGo Zero residual block: conv-BN-ReLU-conv-BN, skip, ReLU."""

    def __init__(self, ch):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False), nn.BatchNorm2d(ch))

    def forward(self, x):
        return F.relu(self.body(x) + x)


class AZNet(nn.Module):
    """forward(x) -> (v, logits (B, A)) for the mover: v (B,) in [-1, 1],
    or with `wdl` (B, 3) win/draw/loss logits, read by the search as
    p_win - p_loss (`postprocess`)."""

    def __init__(self, obs_shape, num_actions, channels=128, blocks=10, value_hidden=256,
                 wdl=False):
        super().__init__()
        c, h, w = obs_shape
        self.obs_shape, self.num_actions = tuple(obs_shape), num_actions
        self.wdl = wdl
        self.stem = nn.Sequential(nn.Conv2d(c, channels, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(channels), nn.ReLU(inplace=True))
        self.trunk = nn.Sequential(*[AZResBlock(channels) for _ in range(blocks)])
        self.policy = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False), nn.BatchNorm2d(2), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(2 * h * w, num_actions))
        self.value = nn.Sequential(
            nn.Conv2d(channels, 1, 1, bias=False), nn.BatchNorm2d(1), nn.ReLU(inplace=True),
            nn.Flatten(), nn.Linear(h * w, value_hidden), nn.ReLU(inplace=True),
            *((nn.Linear(value_hidden, 3),) if wdl else (nn.Linear(value_hidden, 1), nn.Tanh())))

    def forward(self, x):
        t = self.trunk(self.stem(x))
        v = self.value(t)
        return (v if self.wdl else v.squeeze(1)), self.policy(t)

    def postprocess(self, v, logits):
        """-> (v (B,), logits (B, A)) as fp32, for `AzEngine.apply`."""
        if self.wdl:
            p = torch.softmax(v.float(), dim=-1)
            return p[:, 0] - p[:, 2], logits.float()
        return v.float(), logits.float()

    def empty_outputs(self):
        return (torch.zeros((0,)).numpy(), torch.zeros((0, self.num_actions)).numpy())


def build(cfg, obs_shape, num_actions):
    if cfg.algo == 'az':
        if cfg.value_head not in ('scalar', 'wdl'):
            raise ValueError(f'unknown value_head {cfg.value_head!r}')
        return AZNet(obs_shape, num_actions, cfg.channels, cfg.blocks,
                     wdl=cfg.value_head == 'wdl')
    net = DirichletNet(obs_shape, num_actions, cfg.channels, cfg.blocks, cfg.head_channels,
                       cfg.trunk, cal=cfg.cal_conc > 0)
    net.use_cal = cfg.cal_search
    return net
