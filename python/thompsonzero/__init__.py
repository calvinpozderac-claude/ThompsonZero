"""ThompsonZero: Thompson-sampling search over Dirichlet value beliefs.

The search engine and games are Rust (`thompsonzero._core`); the network,
loss and training loop are PyTorch.  Entry points: `thompsonzero.train.train`
and the `thompsonzero` command (see `cli.py`).
"""
from ._core import Engine, GameState
from .config import Config, preset

__all__ = ['Engine', 'GameState', 'Config', 'preset']
