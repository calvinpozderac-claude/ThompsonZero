"""Network evaluation for the engine, and the self-play thread that drives it.

Self-play uses its own copy of the network (bf16 on CUDA, on a dedicated CUDA
stream) so it never waits for, or races with, the optimizer.  The trainer
pushes fresh weights into it every few steps with `Evaluator.load_from`.

The engine's games are split into groups; while one group's batch is on the
GPU the CPU already searches the next group:

    for g in 0, 1, 0, 1, ...:
        finish group g's pending batch -> engine.apply(g, ...)
        engine.collect(g)                       (all cores, GIL released)
        launch group g's forward (asynchronous)
"""
import copy
import queue
import threading

import numpy as np
import torch

from .network import activate


def _bucket(n):
    """Rows to run for an n-row batch.  The compiled graph has a symbolic
    batch size, so this only rounds up to a multiple of 128 (bounded waste,
    and a bounded number of shapes for cuDNN's autotuner to benchmark)."""
    return max(64, -(-n // 128) * 128)


class Evaluator:
    """Owns an inference copy of a network and evaluates engine batches."""

    def __init__(self, net, device, amp=True, compile=False):
        self.device = torch.device(device)
        self.cuda = self.device.type == 'cuda'
        self.dtype = torch.bfloat16 if (amp and self.cuda) else torch.float32
        self.net = copy.deepcopy(net).to(self.device, self.dtype).eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        self.forward = self.net
        self.stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.lock = threading.Lock()
        self._staged = None        # weight staging buffers (see load_from)
        self._read_done = None
        if compile:
            self._compile()

    def _compile(self):
        """Compile with a symbolic batch size and warm it up HERE, on the
        calling thread, so self-play never compiles concurrently with the
        trainer.  Falls back to eager if the toolchain is missing."""
        self.forward = torch.compile(self.net, dynamic=True)
        try:
            for n in (64, 128):
                self.evaluate(np.zeros((n, *self.net.obs_shape), np.float32), self.net.num_actions)
        except Exception as e:  # noqa: BLE001 -- any compiler failure
            print(f'torch.compile unavailable for inference ({type(e).__name__}: {e}); using eager')
            self.forward = self.net

    @torch.no_grad()
    def load_from(self, net):
        """Copy weights from the training `net` without stalling either side.

        CUDA: the trainer's stream snapshots the weights into staging buffers
        (ordered after the optimizer step that produced them); the inference
        stream waits for that snapshot and copies it in, so the next optimizer
        step can never overwrite a weight mid-copy, and the trainer never waits
        for in-flight inference."""
        src = net.state_dict()
        dst = self.net.state_dict()
        with self.lock:
            if not self.cuda:
                for k, v in dst.items():
                    v.copy_(src[k])
                return
            cur = torch.cuda.current_stream(self.device)
            if self._staged is None:
                self._staged = {k: torch.empty_like(v) for k, v in dst.items()}
            if self._read_done is not None:
                cur.wait_event(self._read_done)   # previous snapshot consumed
            for k, buf in self._staged.items():
                buf.copy_(src[k])
            written = torch.cuda.Event()
            written.record(cur)
            self.stream.wait_event(written)
            with torch.cuda.stream(self.stream):
                for k, v in dst.items():
                    v.copy_(self._staged[k])
                self._read_done = torch.cuda.Event()
                self._read_done.record(self.stream)

    @torch.no_grad()
    def submit(self, obs):
        """Start evaluating `obs` (numpy (n, C, H, W)); returns a handle."""
        n = len(obs)
        if n == 0:
            return None
        x = torch.from_numpy(obs)
        pad = _bucket(n) - n
        if pad:
            x = torch.cat([x, x.new_zeros((pad, *x.shape[1:]))])
        with self.lock:
            if not self.cuda:
                v, a = self.forward(x.to(self.dtype))
                return n, activate(v[:n]).numpy(), activate(a[:n]).numpy(), None
            with torch.cuda.stream(self.stream):
                x = x.pin_memory().to(self.device, self.dtype, non_blocking=True)
                v, a = self.forward(x)
                v = activate(v[:n]).to('cpu', non_blocking=True)
                a = activate(a[:n]).to('cpu', non_blocking=True)
                done = torch.cuda.Event()
                done.record(self.stream)
        return n, v, a, done

    @staticmethod
    def result(handle, num_actions):
        """Wait for a handle; returns (v (n, 4), a (n, A, 4)) float32 numpy."""
        if handle is None:
            return np.zeros((0, 4), np.float32), np.zeros((0, num_actions, 4), np.float32)
        n, v, a, done = handle
        if done is not None:
            done.synchronize()
            v, a = v.numpy(), a.numpy()
        return v, a

    def evaluate(self, obs, num_actions):
        return self.result(self.submit(obs), num_actions)


def evaluate_split(evaluators, obs, nets, num_actions):
    """Arena batches: row i is evaluated by evaluators[nets[i]]."""
    v = np.zeros((len(obs), 4), np.float32)
    a = np.zeros((len(obs), num_actions, 4), np.float32)
    handles = [(k, nets == k, ev.submit(obs[nets == k])) for k, ev in enumerate(evaluators)]
    for k, mask, h in handles:
        if mask.any():
            v[mask], a[mask] = Evaluator.result(h, num_actions)
    return v, a


class SelfPlay(threading.Thread):
    """Runs the engine forever, handing finished samples to `out`.

    `out` is a bounded queue of (samples, games finished) items, so self-play
    blocks rather than running ahead of the trainer."""

    def __init__(self, engine, evaluator, max_pending=4):
        super().__init__(daemon=True, name='selfplay')
        self.engine, self.evaluator = engine, evaluator
        self.out = queue.Queue(maxsize=max_pending)
        self.sims = None             # set by the trainer; applied between waves
        self.stats = None            # engine counters, refreshed every wave
        self.error = None
        self._stop = threading.Event()
        self._games = 0

    def stop(self):
        self._stop.set()

    def run(self):
        try:
            self._loop()
        except BaseException as e:   # surfaced by the trainer
            self.error = e

    def _loop(self):
        eng, ev = self.engine, self.evaluator
        A = eng.num_actions
        collected = [False] * eng.groups
        pending = [None] * eng.groups
        while not self._stop.is_set():
            for g in range(eng.groups):
                if collected[g]:  # even with no rows: finished searches play their move
                    eng.apply(g, *Evaluator.result(pending[g], A))
                if self.sims is not None:
                    eng.set_sims(self.sims)
                obs, _nets = eng.collect(g)
                pending[g] = ev.submit(obs)
                collected[g] = True
            self.stats = eng.stats()
            games = self.stats['games']
            if games > self._games:
                item = (eng.drain(), games - self._games)
                self._games = games
                while not self._stop.is_set():
                    try:
                        self.out.put(item, timeout=0.5)
                        break
                    except queue.Full:
                        pass
