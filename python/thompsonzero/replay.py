"""Replay buffer that lives on the training device.

Samples arrive from the engine as dense numpy arrays (one row per position,
action-indexed targets); they are copied once into preallocated device tensors
and minibatches are gathered on the device, so a training step moves no data
over PCIe.

Row layout (A = number of actions):
    obs, next_obs  (C,H,W)   network input before / after the played move
    has_next       bool      next_obs is valid (the move did not end the game)
    legal          (A,)      legal-action mask
    v              (3,)      search state belief (target of klv)
    ev, ev_mask    (A,3),(A,) search edge beliefs (targets of kla)
    played         int       action played, -1 for aux samples
    next_term      int       outcome of the played move if it ended the game
    z, z_w         int,float game outcome (0 win, 1 draw, 2 loss) and its weight
    solved         bool      exact solver label
"""
import numpy as np
import torch

_STORE = {  # field -> storage dtype on device
    'obs': torch.float16, 'next_obs': torch.float16, 'has_next': torch.bool,
    'legal': torch.bool, 'v': torch.float32, 'ev': torch.float16, 'ev_mask': torch.bool,
    'played': torch.int16, 'next_term': torch.int8, 'z': torch.int8,
    'z_w': torch.float16, 'solved': torch.bool,
}
_LONG = ('played', 'next_term', 'z')


class ReplayBuffer:
    def __init__(self, capacity, obs_shape, num_actions, device):
        self.capacity, self.device = capacity, device
        shapes = {'obs': obs_shape, 'next_obs': obs_shape, 'legal': (num_actions,),
                  'v': (3,), 'ev': (num_actions, 3), 'ev_mask': (num_actions,)}
        self.data = {k: torch.zeros((capacity, *shapes.get(k, ())), dtype=dt, device=device)
                     for k, dt in _STORE.items()}
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, samples):
        n = len(samples['obs'])
        if n == 0:
            return
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

    def sample(self, batch_size, generator=None):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device, generator=generator)
        out = {}
        for k, buf in self.data.items():
            x = buf.index_select(0, idx)
            if k in _LONG:
                x = x.long()
            elif x.dtype in (torch.float16, torch.bfloat16):
                x = x.float()
            out[k] = x
        return out

    def placeholder_batch(self, batch_size):
        """A well-formed batch of the right shapes and dtypes (finite
        targets, no successors), for compiling the training step."""
        b = {k: v[:1].expand(batch_size, *v.shape[1:]).clone() for k, v in self.data.items()}
        b['v'].fill_(1.0)
        b['ev'].fill_(1.0)
        b['played'].fill_(-1)
        b['next_term'].fill_(-1)
        return {k: (v.long() if k in _LONG else v.float() if v.is_floating_point() else v)
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
