"""Replay buffer that lives on the training device.

Samples arrive from the engine as dense numpy arrays (one row per position,
action-indexed targets); they are copied once into preallocated device tensors
and minibatches are gathered on the device, so a training step moves no data
over PCIe.

ThompsonZero row layout (A = number of actions):
    obs, next_obs  (C,H,W)   network input before / after the played move
    has_next       bool      next_obs is valid (the move did not end the game)
    legal          (A,)      legal-action mask
    v              (3,)      search state belief (target of klv)
    ev, ev_mask    (A,3),(A,) search edge beliefs (targets of kla)
    played         int       action played, -1 for aux samples
    next_term      int       outcome of the played move if it ended the game
    z, z_w         int,float game outcome (0 win, 1 draw, 2 loss) and its weight
    solved         bool      exact solver label

AlphaZero row layout (`AZ_SCHEMA`):
    obs            (C,H,W)   network input
    pi             (A,)      search visit distribution (all zero: no target)
    z, z_w         float     value target (game result, or q for doubles)
                             and its weight
    pi_w, q, main  float     policy weight, search value, 1 for played positions
    stv, stv_w     float     short-term value target and weight
    reply_w, own_w float     weights of the optional rows below
    reply          (A,)      the opponent's next search policy   (`az_schema(reply=True)`)
    own            (H*W,)    final ownership for the mover       (`az_schema(own=True)`)
"""
import numpy as np
import torch

# field -> (shape, storage dtype on device, as long?); 'obs' / 'A' in a shape
# stand for the observation shape / the number of actions
TZ_SCHEMA = {
    'obs': ('obs', torch.float16, False), 'next_obs': ('obs', torch.float16, False),
    'has_next': ((), torch.bool, False), 'legal': (('A',), torch.bool, False),
    'v': ((3,), torch.float32, False), 'ev': (('A', 3), torch.float16, False),
    'ev_mask': (('A',), torch.bool, False), 'played': ((), torch.int16, True),
    'next_term': ((), torch.int8, True), 'z': ((), torch.int8, True),
    'z_w': ((), torch.float16, False), 'solved': ((), torch.bool, False),
}
AZ_SCHEMA = {
    'obs': ('obs', torch.float16, False), 'pi': (('A',), torch.float16, False),
    'z': ((), torch.float16, False), 'z_w': ((), torch.float16, False),
    'pi_w': ((), torch.float16, False), 'q': ((), torch.float16, False),
    'main': ((), torch.float16, False), 'stv': ((), torch.float16, False),
    'stv_w': ((), torch.float16, False), 'reply_w': ((), torch.float16, False),
    'own_w': ((), torch.float16, False),
}


def az_schema(reply=False, own=False, squares=64):
    s = dict(AZ_SCHEMA)
    if reply:
        s['reply'] = (('A',), torch.float16, False)
    if own:
        s['own'] = ((squares,), torch.float16, False)
    return s


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, num_actions, device, schema=TZ_SCHEMA):
        self.capacity, self.device = capacity, device
        self.long = {k for k, (_, _, lg) in schema.items() if lg}

        def shape(s):
            return tuple(obs_shape) if s == 'obs' else tuple(num_actions if d == 'A' else d for d in s)
        self.data = {k: torch.zeros((capacity, *shape(s)), dtype=dt, device=device)
                     for k, (s, dt, _) in schema.items()}
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, samples):
        """Append a drained engine batch; returns the row indices written."""
        n = len(samples['obs'])
        if n == 0:
            return torch.zeros(0, dtype=torch.long, device=self.device)
        if n > self.capacity:
            samples = {k: v[-self.capacity:] for k, v in samples.items()}
            n = self.capacity
        idx = (self.pos + np.arange(n)) % self.capacity
        idx_t = torch.from_numpy(idx).to(self.device)
        for k, buf in self.data.items():
            src = torch.from_numpy(np.ascontiguousarray(samples[k]))
            buf.index_copy_(0, idx_t, src.to(self.device, buf.dtype, non_blocking=True))
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)
        return idx_t

    def sample(self, batch_size, generator=None, window=None):
        """A uniform batch from the newest `window` rows (default: all)."""
        k = self.size if not window else min(window, self.size)
        back = torch.randint(0, k, (batch_size,), device=self.device, generator=generator)
        return self.gather((self.pos - 1 - back) % self.capacity)

    def gather(self, idx):
        """The rows `idx` (a device tensor) as a training batch."""
        out = {}
        for k, buf in self.data.items():
            x = buf.index_select(0, idx)
            if k in self.long:
                x = x.long()
            elif x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            out[k] = x
        return out

    def placeholder_batch(self, batch_size):
        """A well-formed batch of the right shapes and dtypes (finite
        targets, no successors), for compiling the training step."""
        b = {k: v[:1].expand(batch_size, *v.shape[1:]).clone() for k, v in self.data.items()}
        for k, fill in (('v', 1.0), ('ev', 1.0), ('played', -1), ('next_term', -1), ('z_w', 1.0),
                        ('pi_w', 1.0), ('main', 1.0), ('stv_w', 1.0), ('reply_w', 1.0), ('own_w', 1.0)):
            if k in b:
                b[k].fill_(fill)
        for k in ('pi', 'reply'):
            if k in b:
                b[k].fill_(1.0 / b[k].shape[1])
        return {k: (v.long() if k in self.long else v.float() if v.is_floating_point() else v)
                for k, v in b.items()}

    def state_dict(self):
        return {'pos': self.pos, 'size': self.size,
                'data': {k: v[:self.size].cpu() for k, v in self.data.items()}}

    def load_state_dict(self, sd):
        n = min(sd['size'], self.capacity)
        for k, v in sd['data'].items():
            self.data[k][:n].copy_(v[:n].to(self.device))
        self.size = n
        self.pos = sd['pos'] % self.capacity if n == self.capacity else n
