"""Offline search lab (src/lab.rs): play-time comparisons of selection
rules, value backups and uncertainty estimates with a fixed network.

    side = LabSide('othello', sel='ts', sigma='emp', kappa=1.0)
    w, d, l = play_pair([(side, ev), (LabSide('othello'), ev)], 200, 400, openings=ops)

A `LabSide` stands in for a config in `arena.play_games` (see
`arena.search_engine`); it keeps its engines so `evals` can report the
network evaluations its searches used.  Options are `src/lab.rs`'s
`LabCfg` fields (`LabEngine` keyword arguments).
"""
from . import _core


class LabSide:
    """`sims`, if given, overrides the match's simulation count (for
    comparisons at equal network evaluations)."""

    def __init__(self, game, sims=None, **opts):
        self.game = game
        self.sims = sims
        self.opts = opts
        self.engines = []
        self.searches = 0

    def make_engine(self, slots, sims, seed=0):
        eng = _Counted(self, _core.LabEngine(self.game, slots=slots, sims=self.sims or sims,
                                             seed=seed, **self.opts))
        self.engines.append(eng)
        return eng

    def evals(self):
        """Network evaluations used so far, and searches run."""
        return sum(e.evals() for e in self.engines), self.searches

    def __repr__(self):
        return 'LabSide(' + ', '.join(f'{k}={v!r}' for k, v in self.opts.items()) + ')'


class _Counted:
    """A LabEngine that counts the searches it was asked for."""

    def __init__(self, side, eng):
        self._side, self._eng = side, eng

    def search(self, slot, state):
        self._side.searches += 1
        return self._eng.search(slot, state)

    def __getattr__(self, name):
        return getattr(self._eng, name)
