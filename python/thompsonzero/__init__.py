"""ThompsonZero: Thompson-sampling search over Dirichlet value beliefs, and an
AlphaZero baseline (`az.py`).

The search engines and games are Rust (`thompsonzero._core`); the networks,
losses and training loop are PyTorch.  Entry points: `thompsonzero.train.train`
and the `thompsonzero` command (see `cli.py`).
"""
from ._core import AzEngine, Engine, GameState
from .config import Config, preset

__all__ = ['AzEngine', 'Engine', 'GameState', 'Config', 'preset']
