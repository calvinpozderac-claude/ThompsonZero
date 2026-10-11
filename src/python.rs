//! Python bindings: `thompsonzero._core.Engine` and `GameState`.
//!
//! The heavy calls release the GIL, so the trainer thread keeps running while
//! the engine searches on every core.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{IntoPyArray, PyArray1, PyArray3, PyArray4, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::belief::{Rule, TARGET_EPS};
use crate::engine::{Config, Engine, Sample};
use crate::games::{Boop, Connect4, Game, Othello};

/// Run `$body` with `$g` bound to the concrete game type of `$name`.
macro_rules! with_game {
    ($name:expr, $g:ident => $body:expr) => {
        match $name {
            "connect_four" | "connect4" => {
                type $g = Connect4;
                $body
            }
            "othello" => {
                type $g = Othello;
                $body
            }
            "boop" => {
                type $g = Boop;
                $body
            }
            other => Err(PyValueError::new_err(format!(
                "unknown game {other:?}; expected connect_four | othello | boop"
            ))),
        }
    };
}

enum AnyEngine {
    C4(Engine<Connect4>),
    Othello(Engine<Othello>),
    Boop(Engine<Boop>),
}

macro_rules! each {
    ($any:expr, $e:ident => $body:expr) => {
        match $any {
            AnyEngine::C4($e) => $body,
            AnyEngine::Othello($e) => $body,
            AnyEngine::Boop($e) => $body,
        }
    };
}

trait Wrap: Game + Sized {
    fn wrap(e: Engine<Self>) -> AnyEngine;
    fn wrap_state(s: Self) -> AnyState;
}
impl Wrap for Connect4 {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::C4(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::C4(s)
    }
}
impl Wrap for Othello {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::Othello(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::Othello(s)
    }
}
impl Wrap for Boop {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::Boop(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::Boop(s)
    }
}

/// Batched self-play / arena engine (see `src/engine.rs`).
#[pyclass(name = "Engine", module = "thompsonzero._core")]
pub struct PyEngine {
    inner: AnyEngine,
    #[pyo3(get)]
    game: String,
    #[pyo3(get)]
    obs_shape: (usize, usize, usize),
    #[pyo3(get)]
    num_actions: usize,
    #[pyo3(get)]
    max_plies: usize,
}

/// (observations, network index per row) from `Engine.collect`.
type Batch<'py> = (Bound<'py, PyArray4<f32>>, Bound<'py, PyArray1<u8>>);

fn obs_shape<G: Game>() -> (usize, usize, usize) {
    (G::OBS_SHAPE[0], G::OBS_SHAPE[1], G::OBS_SHAPE[2])
}

#[pymethods]
impl PyEngine {
    #[new]
    #[pyo3(signature = (game, *, num_games=512, groups=2, seed=0, sims=200, wave=4,
        search="additive_mle", target="avgconc", rho_search=0.8, rho_target=0.999,
        sample_leaf=false, virtual_loss=1.0, descent_temp=1.0, temp_threshold=20,
        max_plies=None, double_aux=1, double_min_visits=8.0, solved_aux=true,
        arena_games=None, opening_plies=2))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        game: &str,
        num_games: usize,
        groups: usize,
        seed: u64,
        sims: usize,
        wave: usize,
        search: &str,
        target: &str,
        rho_search: f64,
        rho_target: f64,
        sample_leaf: bool,
        virtual_loss: f64,
        descent_temp: f64,
        temp_threshold: usize,
        max_plies: Option<usize>,
        double_aux: usize,
        double_min_visits: f64,
        solved_aux: bool,
        arena_games: Option<usize>,
        opening_plies: usize,
    ) -> PyResult<Self> {
        if groups == 0 || num_games < groups {
            return Err(PyValueError::new_err("need num_games >= groups >= 1"));
        }
        if wave == 0 {
            return Err(PyValueError::new_err("wave must be >= 1"));
        }
        let mut cfg = Config::new(
            Rule::parse(search, rho_search).map_err(PyValueError::new_err)?,
            Rule::parse(target, rho_target).map_err(PyValueError::new_err)?,
        );
        cfg.num_games = num_games;
        cfg.groups = groups;
        cfg.seed = seed;
        cfg.sims = sims;
        cfg.wave = wave;
        cfg.search.sample_leaf = sample_leaf;
        cfg.search.virtual_loss = virtual_loss;
        cfg.descent_temp = descent_temp;
        cfg.temp_threshold = temp_threshold;
        cfg.double_aux = double_aux;
        cfg.double_min_visits = double_min_visits;
        cfg.solved_aux = solved_aux;
        cfg.arena_games = arena_games;
        cfg.opening_plies = opening_plies;
        with_game!(game, G => {
            cfg.max_plies = max_plies.unwrap_or(G::MAX_PLIES);
            Ok(PyEngine {
                max_plies: cfg.max_plies,
                inner: G::wrap(Engine::<G>::new(cfg)),
                game: G::NAME.to_string(),
                obs_shape: obs_shape::<G>(),
                num_actions: G::NUM_ACTIONS,
            })
        })
    }

    /// Search every game in `group` up to its next network evaluation.
    /// Returns (obs float32 (rows, C, H, W), net index uint8 (rows,)).
    fn collect<'py>(&mut self, py: Python<'py>, group: usize) -> PyResult<Batch<'py>> {
        if group >= self.groups() {
            return Err(PyValueError::new_err("group out of range"));
        }
        let inner = &mut self.inner;
        let (obs, nets) = py.allow_threads(|| each!(inner, e => e.collect(group)));
        let (c, h, w) = self.obs_shape;
        let obs = Array4::from_shape_vec((nets.len(), c, h, w), obs).unwrap();
        Ok((obs.into_pyarray(py), Array1::from_vec(nets).into_pyarray(py)))
    }

    /// Feed back the activated evaluations for the last `collect(group)`:
    /// v float32 (rows, 4) and a float32 (rows, num_actions, 4), each row
    /// (p_win, p_draw, p_loss, concentration) from the mover's perspective.
    fn apply(
        &mut self,
        py: Python<'_>,
        group: usize,
        v: PyReadonlyArray2<f32>,
        a: PyReadonlyArray3<f32>,
    ) -> PyResult<()> {
        if group >= self.groups() {
            return Err(PyValueError::new_err("group out of range"));
        }
        let rows = self.pending_rows(group);
        if v.shape() != [rows, 4] || a.shape() != [rows, self.num_actions, 4] {
            return Err(PyValueError::new_err(format!(
                "expected v ({rows}, 4) and a ({rows}, {}, 4), got {:?} and {:?}",
                self.num_actions,
                v.shape(),
                a.shape()
            )));
        }
        let (v, a) = (v.as_slice()?, a.as_slice()?);
        let inner = &mut self.inner;
        py.allow_threads(|| each!(inner, e => e.apply(group, v, a)));
        Ok(())
    }

    /// Every sample from games finished since the last call, as a dict of
    /// dense numpy arrays (see `replay.py`).
    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = each!(&mut self.inner, e => e.drain());
        samples_to_dict(py, &samples, self.obs_shape, self.num_actions)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = each!(&self.inner, e => e.stats().clone());
        let d = PyDict::new(py);
        for (k, v) in [
            ("games", s.games),
            ("plies", s.plies),
            ("draws", s.draws),
            ("cutoffs", s.cutoffs),
            ("sims", s.sims),
            ("evals", s.evals),
            ("samples", s.samples),
            ("solved_aux", s.solved_aux),
            ("double_aux", s.double_aux),
            ("wins", s.wins),
            ("losses", s.losses),
        ] {
            d.set_item(k, v)?;
        }
        Ok(d)
    }

    /// Simulations per move for games started from now on.
    fn set_sims(&mut self, sims: usize) {
        each!(&mut self.inner, e => e.set_sims(sims))
    }

    /// Arena mode: every game has been played.
    #[getter]
    fn done(&self) -> bool {
        each!(&self.inner, e => e.done())
    }

    #[getter]
    fn groups(&self) -> usize {
        each!(&self.inner, e => e.cfg.groups)
    }
}

impl PyEngine {
    fn pending_rows(&self, group: usize) -> usize {
        each!(&self.inner, e => e.pending_rows(group))
    }
}

fn samples_to_dict<'py>(
    py: Python<'py>,
    s: &[Sample],
    shape: (usize, usize, usize),
    na: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = s.len();
    let (c, h, w) = shape;
    let d = c * h * w;
    let mut obs = Array2::<f32>::zeros((n, d));
    let mut next_obs = Array2::<f32>::zeros((n, d));
    let mut legal = Array2::<bool>::from_elem((n, na), false);
    let mut ev = Array3::<f32>::from_elem((n, na, 3), TARGET_EPS as f32);
    let mut ev_mask = Array2::<bool>::from_elem((n, na), false);
    let mut v = Array2::<f32>::zeros((n, 3));
    let mut played = Array1::<i64>::from_elem(n, -1);
    let mut has_next = Array1::<bool>::from_elem(n, false);
    let mut next_term = Array1::<i64>::from_elem(n, -1);
    let mut z = Array1::<i64>::zeros(n);
    let mut z_w = Array1::<f32>::zeros(n);
    let mut solved = Array1::<bool>::from_elem(n, false);
    for (i, x) in s.iter().enumerate() {
        obs.row_mut(i).as_slice_mut().unwrap().copy_from_slice(&x.obs);
        if let Some(o) = &x.next_obs {
            next_obs.row_mut(i).as_slice_mut().unwrap().copy_from_slice(o);
            has_next[i] = true;
        }
        for &a in &x.legal {
            legal[[i, a as usize]] = true;
        }
        for &(a, t) in &x.edges {
            ev_mask[[i, a as usize]] = true;
            for j in 0..3 {
                ev[[i, a as usize, j]] = t[j];
            }
        }
        for j in 0..3 {
            v[[i, j]] = x.v[j];
        }
        played[i] = x.played as i64;
        next_term[i] = x.next_term as i64;
        z[i] = x.z as i64;
        z_w[i] = x.z_w;
        solved[i] = x.solved;
    }
    let out = PyDict::new(py);
    out.set_item("obs", obs.into_shape_with_order((n, c, h, w)).unwrap().into_pyarray(py))?;
    out.set_item(
        "next_obs",
        next_obs.into_shape_with_order((n, c, h, w)).unwrap().into_pyarray(py),
    )?;
    out.set_item("has_next", has_next.into_pyarray(py))?;
    out.set_item("legal", legal.into_pyarray(py))?;
    out.set_item("ev", ev.into_pyarray(py))?;
    out.set_item("ev_mask", ev_mask.into_pyarray(py))?;
    out.set_item("v", v.into_pyarray(py))?;
    out.set_item("played", played.into_pyarray(py))?;
    out.set_item("next_term", next_term.into_pyarray(py))?;
    out.set_item("z", z.into_pyarray(py))?;
    out.set_item("z_w", z_w.into_pyarray(py))?;
    out.set_item("solved", solved.into_pyarray(py))?;
    Ok(out)
}

// ════════════════════════════════════════════════════════════════════════════
//  A single game state, for tests, debugging and play front-ends
// ════════════════════════════════════════════════════════════════════════════

#[derive(Clone)]
enum AnyState {
    C4(Connect4),
    Othello(Othello),
    Boop(Boop),
}

macro_rules! each_state {
    ($any:expr, $s:ident => $body:expr) => {
        match $any {
            AnyState::C4($s) => $body,
            AnyState::Othello($s) => $body,
            AnyState::Boop($s) => $body,
        }
    };
}

#[pyclass(name = "GameState", module = "thompsonzero._core")]
#[derive(Clone)]
pub struct PyGameState {
    inner: AnyState,
}

#[pymethods]
impl PyGameState {
    #[new]
    fn new(game: &str) -> PyResult<Self> {
        with_game!(game, G => Ok(PyGameState { inner: G::wrap_state(G::new()) }))
    }

    fn legal_actions(&self) -> Vec<u16> {
        let mut v = Vec::new();
        each_state!(&self.inner, s => s.legal_actions(&mut v));
        v
    }

    fn apply(&mut self, action: u16) -> PyResult<()> {
        if !self.legal_actions().contains(&action) {
            return Err(PyValueError::new_err(format!("illegal action {action}")));
        }
        each_state!(&mut self.inner, s => s.apply(action));
        Ok(())
    }

    fn current_player(&self) -> usize {
        each_state!(&self.inner, s => s.current_player())
    }

    fn is_terminal(&self) -> bool {
        each_state!(&self.inner, s => s.is_terminal())
    }

    /// (player 0, player 1) results, or None while the game is running.
    fn returns(&self) -> Option<(f32, f32)> {
        each_state!(&self.inner, s => s.outcome()).map(|r| (r as f32, -r as f32))
    }

    /// Network input from the mover's perspective, float32 (C, H, W).
    fn observation<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray3<f32>> {
        let (shape, data) = each_state!(&self.inner, s => {
            let mut o = vec![0f32; obs_size(s)];
            s.encode(&mut o);
            (obs_shape_of(s), o)
        });
        Array3::from_shape_vec(shape, data).unwrap().into_pyarray(py)
    }

    fn clone(&self) -> Self {
        Clone::clone(self)
    }

    #[getter]
    fn num_actions(&self) -> usize {
        each_state!(&self.inner, s => num_actions_of(s))
    }

    #[getter]
    fn obs_shape(&self) -> (usize, usize, usize) {
        each_state!(&self.inner, s => obs_shape_of(s))
    }

    #[getter]
    fn game(&self) -> &'static str {
        each_state!(&self.inner, s => name_of(s))
    }

    fn __str__(&self) -> String {
        each_state!(&self.inner, s => s.render())
    }
}

fn obs_size<G: Game>(_: &G) -> usize {
    G::OBS_SIZE
}

fn obs_shape_of<G: Game>(_: &G) -> (usize, usize, usize) {
    obs_shape::<G>()
}

fn num_actions_of<G: Game>(_: &G) -> usize {
    G::NUM_ACTIONS
}

fn name_of<G: Game>(_: &G) -> &'static str {
    G::NAME
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyGameState>()?;
    Ok(())
}
