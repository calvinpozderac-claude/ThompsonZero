//! Python bindings: `thompsonzero._core.Engine` and `GameState`.
//!
//! The heavy calls release the GIL, so the trainer thread keeps running while
//! the engine searches on every core.

use numpy::ndarray::{Array1, Array2, Array3, Array4};
use numpy::{
    IntoPyArray, PyArray1, PyArray2, PyArray3, PyArray4, PyReadonlyArray1, PyReadonlyArray2, PyReadonlyArray3,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::az::{AzConfig, AzEngine, AzSample};
use crate::belief::{Rule, RAW, TARGET_EPS};
use crate::engine::{Config, Engine, Sample, Stats};
use crate::games::{Boop, Connect4, Game, Othello};
use crate::lab::{self, LabCfg, LabEngine};
use crate::puct::{PuctCfg, Select};
use crate::tree::Report;

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

enum AnyAz {
    C4(AzEngine<Connect4>),
    Othello(AzEngine<Othello>),
    Boop(AzEngine<Boop>),
}

enum AnyLab {
    C4(LabEngine<Connect4>),
    Othello(LabEngine<Othello>),
    Boop(LabEngine<Boop>),
}

macro_rules! each_lab {
    ($any:expr, $e:ident => $body:expr) => {
        match $any {
            AnyLab::C4($e) => $body,
            AnyLab::Othello($e) => $body,
            AnyLab::Boop($e) => $body,
        }
    };
}

macro_rules! each_az {
    ($any:expr, $e:ident => $body:expr) => {
        match $any {
            AnyAz::C4($e) => $body,
            AnyAz::Othello($e) => $body,
            AnyAz::Boop($e) => $body,
        }
    };
}

trait Wrap: Game + Sized {
    fn wrap(e: Engine<Self>) -> AnyEngine;
    fn wrap_az(e: AzEngine<Self>) -> AnyAz;
    fn wrap_lab(e: LabEngine<Self>) -> AnyLab;
    fn wrap_state(s: Self) -> AnyState;
    fn unwrap_state(s: &AnyState) -> Option<Self>;
}
impl Wrap for Connect4 {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::C4(e)
    }
    fn wrap_az(e: AzEngine<Self>) -> AnyAz {
        AnyAz::C4(e)
    }
    fn wrap_lab(e: LabEngine<Self>) -> AnyLab {
        AnyLab::C4(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::C4(s)
    }
    fn unwrap_state(s: &AnyState) -> Option<Self> {
        match s {
            AnyState::C4(x) => Some(x.clone()),
            _ => None,
        }
    }
}
impl Wrap for Othello {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::Othello(e)
    }
    fn wrap_az(e: AzEngine<Self>) -> AnyAz {
        AnyAz::Othello(e)
    }
    fn wrap_lab(e: LabEngine<Self>) -> AnyLab {
        AnyLab::Othello(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::Othello(s)
    }
    fn unwrap_state(s: &AnyState) -> Option<Self> {
        match s {
            AnyState::Othello(x) => Some(x.clone()),
            _ => None,
        }
    }
}
impl Wrap for Boop {
    fn wrap(e: Engine<Self>) -> AnyEngine {
        AnyEngine::Boop(e)
    }
    fn wrap_az(e: AzEngine<Self>) -> AnyAz {
        AnyAz::Boop(e)
    }
    fn wrap_lab(e: LabEngine<Self>) -> AnyLab {
        AnyLab::Boop(e)
    }
    fn wrap_state(s: Self) -> AnyState {
        AnyState::Boop(s)
    }
    fn unwrap_state(s: &AnyState) -> Option<Self> {
        match s {
            AnyState::Boop(x) => Some(x.clone()),
            _ => None,
        }
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
        sample_leaf=false, best_value=false, virtual_loss=1.0, descent_temp=1.0, temp_threshold=20,
        root_temp=1.0, max_plies=None, double_aux=1, double_min_visits=8.0, solved_aux=true,
        arena_games=None, opening_plies=2, service=false, report=false))]
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
        best_value: bool,
        virtual_loss: f64,
        descent_temp: f64,
        temp_threshold: usize,
        root_temp: f64,
        max_plies: Option<usize>,
        double_aux: usize,
        double_min_visits: f64,
        solved_aux: bool,
        arena_games: Option<usize>,
        opening_plies: usize,
        service: bool,
        report: bool,
    ) -> PyResult<Self> {
        if groups == 0 || num_games < groups {
            return Err(PyValueError::new_err("need num_games >= groups >= 1"));
        }
        if wave == 0 {
            return Err(PyValueError::new_err("wave must be >= 1"));
        }
        if service && arena_games.is_some() {
            return Err(PyValueError::new_err("a search service plays no arena games"));
        }
        if report && !service {
            return Err(PyValueError::new_err("reports come from a search service"));
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
        cfg.search.best_value = best_value;
        cfg.search.virtual_loss = virtual_loss;
        cfg.descent_temp = descent_temp;
        cfg.temp_threshold = temp_threshold;
        cfg.root_temp = root_temp;
        cfg.double_aux = double_aux;
        cfg.double_min_visits = double_min_visits;
        cfg.solved_aux = solved_aux;
        cfg.arena_games = arena_games;
        cfg.opening_plies = opening_plies;
        cfg.service = service;
        cfg.report = report;
        cfg.search.track_all = report;
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
        stats_to_dict(py, &each!(&self.inner, e => e.stats().clone()))
    }

    /// Simulations per move for games started from now on.
    fn set_sims(&mut self, sims: usize) {
        each!(&mut self.inner, e => e.set_sims(sims))
    }

    /// Arena mode: every game has been played.  Search service: every
    /// search has finished.
    #[getter]
    fn done(&self) -> bool {
        each!(&self.inner, e => e.done())
    }

    /// Search service: search `state` in the idle `slot`; its move comes
    /// back from `moves` after enough collect/apply rounds.
    fn search(&mut self, slot: usize, state: &PyGameState) -> PyResult<()> {
        each!(&mut self.inner, e => search_in(e, slot, &state.inner))
    }

    /// Search service: [(slot, move)] of the searches finished since the
    /// last call.
    fn moves(&mut self) -> Vec<(usize, u16)> {
        each!(&mut self.inner, e => e.moves())
    }

    /// Search service with `report=True`: [(slot, report)] of the searches
    /// finished since the last call.  A report is a dict: `v_alpha` (3,) and
    /// `node` (RAW,) for the root; `actions`, `term`, `prior` (E, 3), `sel`
    /// (E, 3) and `acc` (E, RAW) per root edge (`Acc::raw` in belief.rs).
    fn reports<'py>(&mut self, py: Python<'py>) -> PyResult<Vec<(usize, Bound<'py, PyDict>)>> {
        let reps = each!(&mut self.inner, e => e.reports());
        reps.iter().map(|(i, r)| Ok((*i, report_to_dict(py, r)?))).collect()
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

fn state_for<G: Wrap>(s: &AnyState) -> PyResult<G> {
    G::unwrap_state(s).ok_or_else(|| PyValueError::new_err(format!("expected a {} state", G::NAME)))
}

fn search_in<G: Wrap>(e: &mut Engine<G>, slot: usize, s: &AnyState) -> PyResult<()> {
    e.search(slot, state_for(s)?).map_err(PyValueError::new_err)
}

fn search_in_az<G: Wrap>(e: &mut AzEngine<G>, slot: usize, s: &AnyState) -> PyResult<()> {
    e.search(slot, state_for(s)?).map_err(PyValueError::new_err)
}

fn stats_to_dict<'py>(py: Python<'py>, s: &Stats) -> PyResult<Bound<'py, PyDict>> {
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

// ════════════════════════════════════════════════════════════════════════════
//  The AlphaZero engine
// ════════════════════════════════════════════════════════════════════════════

/// Batched AlphaZero self-play / arena engine (see `src/az.rs`).
#[pyclass(name = "AzEngine", module = "thompsonzero._core")]
pub struct PyAzEngine {
    inner: AnyAz,
    #[pyo3(get)]
    game: String,
    #[pyo3(get)]
    obs_shape: (usize, usize, usize),
    #[pyo3(get)]
    num_actions: usize,
    #[pyo3(get)]
    max_plies: usize,
}

#[pymethods]
impl PyAzEngine {
    #[new]
    #[pyo3(signature = (game, *, num_games=512, groups=2, seed=0, sims=200, wave=4,
        c_puct=1.5, fpu_reduction=0.2, virtual_loss=1.0, noise_alpha=10.0, noise_frac=0.25,
        temp_threshold=20, max_plies=None, reuse_tree=true, double_aux=1, double_min_visits=8,
        arena_games=None, external=false, openings=None, opening_plies=2, service=false,
        select="puct", ts_c=None, ts_from=0.0, mixmax=0.0, pick_frac=0.0, aux_depth=1, aux_pow=0.0,
        q_lambda=0.0, q_min_visits=1, stv_lambda=0.8))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        game: &str,
        num_games: usize,
        groups: usize,
        seed: u64,
        sims: usize,
        wave: usize,
        c_puct: f32,
        fpu_reduction: f32,
        virtual_loss: f32,
        noise_alpha: f32,
        noise_frac: f32,
        temp_threshold: usize,
        max_plies: Option<usize>,
        reuse_tree: bool,
        double_aux: usize,
        double_min_visits: u32,
        arena_games: Option<usize>,
        external: bool,
        openings: Option<Vec<Vec<u16>>>,
        opening_plies: usize,
        service: bool,
        select: &str,
        ts_c: Option<f32>,
        ts_from: f32,
        mixmax: f32,
        pick_frac: f32,
        aux_depth: usize,
        aux_pow: f32,
        q_lambda: f32,
        q_min_visits: u32,
        stv_lambda: f32,
    ) -> PyResult<Self> {
        if groups == 0 || num_games < groups {
            return Err(PyValueError::new_err("need num_games >= groups >= 1"));
        }
        if wave == 0 {
            return Err(PyValueError::new_err("wave must be >= 1"));
        }
        if external && arena_games.is_none() {
            return Err(PyValueError::new_err("external play needs arena_games"));
        }
        if service && arena_games.is_some() {
            return Err(PyValueError::new_err("a search service plays no arena games"));
        }
        let select = match select {
            "puct" => Select::Puct,
            "dirichlet" => Select::Dirichlet,
            "gaussian" => Select::Gaussian,
            _ => return Err(PyValueError::new_err(format!("unknown select {select:?}"))),
        };
        let mut cfg = AzConfig {
            num_games,
            groups,
            seed,
            sims,
            wave,
            puct: PuctCfg {
                c_puct,
                fpu_reduction,
                virtual_loss,
                select,
                ts_c: ts_c.unwrap_or(c_puct),
                mixmax,
                pick_frac,
            },
            ts_from,
            noise_alpha,
            noise_frac,
            temp_threshold,
            max_plies: 0,
            reuse_tree,
            double_aux,
            double_min_visits,
            aux_depth,
            aux_pow,
            q_lambda,
            q_min_visits,
            stv_lambda,
            arena_games,
            external,
            openings: openings.unwrap_or_default(),
            opening_plies,
            service,
        };
        with_game!(game, G => {
            // an illegal opening would panic deep inside a worker thread
            let mut legal = Vec::new();
            for (i, seq) in cfg.openings.iter().enumerate() {
                let mut st = G::new();
                for &a in seq {
                    st.legal_actions(&mut legal);
                    if !legal.contains(&a) {
                        return Err(PyValueError::new_err(format!("opening {i}: illegal move {a}")));
                    }
                    st.apply(a);
                }
                if st.is_terminal() {
                    return Err(PyValueError::new_err(format!("opening {i} ends the game")));
                }
            }
            cfg.max_plies = max_plies.unwrap_or(G::MAX_PLIES);
            Ok(PyAzEngine {
                max_plies: cfg.max_plies,
                inner: G::wrap_az(AzEngine::<G>::new(cfg)),
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
        let (obs, nets) = py.allow_threads(|| each_az!(inner, e => e.collect(group)));
        let (c, h, w) = self.obs_shape;
        let obs = Array4::from_shape_vec((nets.len(), c, h, w), obs).unwrap();
        Ok((obs.into_pyarray(py), Array1::from_vec(nets).into_pyarray(py)))
    }

    /// Feed back the evaluations for the last `collect(group)`: v float32
    /// (rows,) the mover's value in [-1, 1], logits float32 (rows,
    /// num_actions) raw policy logits, and optionally d float32 (rows,) the
    /// draw probability of a win/draw/loss head (Thompson selection reads it).
    #[pyo3(signature = (group, v, logits, d=None))]
    fn apply(
        &mut self,
        py: Python<'_>,
        group: usize,
        v: PyReadonlyArray1<f32>,
        logits: PyReadonlyArray2<f32>,
        d: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<()> {
        if group >= self.groups() {
            return Err(PyValueError::new_err("group out of range"));
        }
        let rows = self.pending_rows(group);
        if v.shape() != [rows] || logits.shape() != [rows, self.num_actions] {
            return Err(PyValueError::new_err(format!(
                "expected v ({rows},) and logits ({rows}, {}), got {:?} and {:?}",
                self.num_actions,
                v.shape(),
                logits.shape()
            )));
        }
        if d.as_ref().is_some_and(|d| d.shape() != [rows]) {
            return Err(PyValueError::new_err(format!("expected d ({rows},)")));
        }
        let (v, logits) = (v.as_slice()?, logits.as_slice()?);
        let d = match &d {
            Some(d) => Some(d.as_slice()?),
            None => None,
        };
        let inner = &mut self.inner;
        py.allow_threads(|| each_az!(inner, e => e.apply(group, v, d, logits)));
        Ok(())
    }

    /// Every sample from games finished since the last call, as a dict of
    /// dense numpy arrays: obs (n, C, H, W), pi (n, A), z (n,), z_w (n,),
    /// pi_w, q, main, stv, stv_w (n,), reply (n, A), reply_w (n,), own (n,
    /// H * W), own_w (n,) -- see `AzSample`.
    fn drain<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let samples = each_az!(&mut self.inner, e => e.drain());
        az_samples_to_dict(py, &samples, self.obs_shape, self.num_actions)
    }

    /// Arena results since the last call: [(game, net 0's player, result
    /// for net 0)].
    fn results(&mut self) -> Vec<(usize, u8, i8)> {
        each_az!(&mut self.inner, e => e.take_results())
            .into_iter()
            .map(|r| (r.game, r.seat0, r.result))
            .collect()
    }

    /// External mode: [(slot, GameState)] for every game where the external
    /// opponent is to move.
    fn waiting(&self) -> Vec<(usize, PyGameState)> {
        each_az!(&self.inner, e => e
            .waiting()
            .into_iter()
            .map(|(i, s)| (i, PyGameState { inner: Wrap::wrap_state(s) }))
            .collect())
    }

    /// External mode: play the opponent's move in `slot`.
    fn play_external(&mut self, slot: usize, action: u16) -> PyResult<()> {
        each_az!(&mut self.inner, e => e.play_external(slot, action)).map_err(PyValueError::new_err)
    }

    fn stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        stats_to_dict(py, &each_az!(&self.inner, e => e.stats().clone()))
    }

    /// Simulations per move for searches started from now on.
    fn set_sims(&mut self, sims: usize) {
        each_az!(&mut self.inner, e => e.set_sims(sims))
    }

    /// Arena mode: every game has been played.  Search service: every
    /// search has finished.
    #[getter]
    fn done(&self) -> bool {
        each_az!(&self.inner, e => e.done())
    }

    /// Search service: search `state` in the idle `slot`; its move comes
    /// back from `moves` after enough collect/apply rounds.
    fn search(&mut self, slot: usize, state: &PyGameState) -> PyResult<()> {
        each_az!(&mut self.inner, e => search_in_az(e, slot, &state.inner))
    }

    /// Search service: [(slot, move)] of the searches finished since the
    /// last call.
    fn moves(&mut self) -> Vec<(usize, u16)> {
        each_az!(&mut self.inner, e => e.moves())
    }

    #[getter]
    fn groups(&self) -> usize {
        each_az!(&self.inner, e => e.cfg.groups)
    }
}

impl PyAzEngine {
    fn pending_rows(&self, group: usize) -> usize {
        each_az!(&self.inner, e => e.pending_rows(group))
    }
}

fn az_samples_to_dict<'py>(
    py: Python<'py>,
    s: &[AzSample],
    shape: (usize, usize, usize),
    na: usize,
) -> PyResult<Bound<'py, PyDict>> {
    let n = s.len();
    let (c, h, w) = shape;
    let mut obs = Array2::<f32>::zeros((n, c * h * w));
    let mut pi = Array2::<f32>::zeros((n, na));
    let mut reply = Array2::<f32>::zeros((n, na));
    let mut own = Array2::<f32>::zeros((n, h * w));
    let mut cols: [Array1<f32>; 9] = std::array::from_fn(|_| Array1::<f32>::zeros(n));
    for (i, x) in s.iter().enumerate() {
        obs.row_mut(i).as_slice_mut().unwrap().copy_from_slice(&x.obs);
        for &(a, p) in &x.pi {
            pi[[i, a as usize]] = p;
        }
        for &(a, p) in &x.reply {
            reply[[i, a as usize]] = p;
        }
        if !x.own.is_empty() {
            own.row_mut(i).as_slice_mut().unwrap().copy_from_slice(&x.own);
        }
        let row = [x.z, x.z_w, x.pi_w, x.q, x.main as u8 as f32, x.stv, x.stv_w, x.reply_w, x.own_w];
        for (col, v) in cols.iter_mut().zip(row) {
            col[i] = v;
        }
    }
    let out = PyDict::new(py);
    out.set_item("obs", obs.into_shape_with_order((n, c, h, w)).unwrap().into_pyarray(py))?;
    out.set_item("pi", pi.into_pyarray(py))?;
    out.set_item("reply", reply.into_pyarray(py))?;
    out.set_item("own", own.into_pyarray(py))?;
    let names = ["z", "z_w", "pi_w", "q", "main", "stv", "stv_w", "reply_w", "own_w"];
    for (name, col) in names.into_iter().zip(cols) {
        out.set_item(name, col.into_pyarray(py))?;
    }
    Ok(out)
}

fn report_to_dict<'py>(py: Python<'py>, r: &Report) -> PyResult<Bound<'py, PyDict>> {
    let n = r.edges.len();
    let mut prior = Array2::<f64>::zeros((n, 3));
    let mut sel = Array2::<f64>::zeros((n, 3));
    let mut acc = Array2::<f64>::zeros((n, RAW));
    for (i, e) in r.edges.iter().enumerate() {
        for j in 0..3 {
            prior[[i, j]] = e.prior[j];
            sel[[i, j]] = e.sel[j];
        }
        acc.row_mut(i).as_slice_mut().unwrap().copy_from_slice(&e.acc);
    }
    let actions: Vec<u16> = r.edges.iter().map(|e| e.action).collect();
    let term: Vec<i8> = r.edges.iter().map(|e| e.term).collect();
    let out = PyDict::new(py);
    out.set_item("v_alpha", Array1::from_vec(r.v_alpha.to_vec()).into_pyarray(py))?;
    out.set_item("node", Array1::from_vec(r.node.to_vec()).into_pyarray(py))?;
    out.set_item("actions", Array1::from_vec(actions).into_pyarray(py))?;
    out.set_item("term", Array1::from_vec(term).into_pyarray(py))?;
    out.set_item("prior", prior.into_pyarray(py))?;
    out.set_item("sel", sel.into_pyarray(py))?;
    out.set_item("acc", acc.into_pyarray(py))?;
    Ok(out)
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

    /// Othello only: the position in Edax's `setboard` format.
    fn edax_board(&self) -> PyResult<String> {
        match &self.inner {
            AnyState::Othello(s) => Ok(s.edax_board()),
            _ => Err(PyValueError::new_err("edax_board is only defined for othello")),
        }
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

// ════════════════════════════════════════════════════════════════════════════
// Search lab
// ════════════════════════════════════════════════════════════════════════════

fn search_in_lab<G: Wrap>(e: &mut LabEngine<G>, slot: usize, s: &AnyState) -> PyResult<()> {
    e.search(slot, state_for(s)?).map_err(PyValueError::new_err)
}

/// `LabCfg` from keyword options (see `src/lab.rs`); unknown keys or values
/// are errors.
fn lab_cfg(opts: Option<&Bound<'_, PyDict>>) -> PyResult<LabCfg> {
    let mut c = LabCfg::default();
    let Some(opts) = opts else { return Ok(c) };
    fn pick<T: Copy>(v: &str, key: &str, table: &[(&str, T)]) -> PyResult<T> {
        table
            .iter()
            .find(|(k, _)| *k == v)
            .map(|x| x.1)
            .ok_or_else(|| PyValueError::new_err(format!("bad {key} {v:?}")))
    }
    for (k, v) in opts.iter() {
        let key: String = k.extract()?;
        let f = || -> PyResult<f32> { v.extract::<f32>() };
        let st = || -> PyResult<String> { v.extract::<String>() };
        match key.as_str() {
            "sel" => {
                c.sel = pick(&st()?, &key, &[("puct", lab::Sel::Puct), ("ts", lab::Sel::Ts), ("gumbel", lab::Sel::Gumbel)])?
            }
            "sigma" => {
                c.sigma = pick(
                    &st()?,
                    &key,
                    &[
                        ("u", lab::Sigma::U),
                        ("count", lab::Sigma::Count),
                        ("netvar", lab::Sigma::NetVar),
                        ("emp", lab::Sigma::Emp),
                        ("bayes", lab::Sigma::Bayes),
                        ("fit", lab::Sigma::Fit),
                    ],
                )?
            }
            "z" => {
                c.zmode = pick(
                    &st()?,
                    &key,
                    &[
                        ("normal", lab::ZMode::Normal),
                        ("abs", lab::ZMode::Abs),
                        ("pos", lab::ZMode::Pos),
                        ("one", lab::ZMode::One),
                    ],
                )?
            }
            "backup" => {
                c.backup = pick(
                    &st()?,
                    &key,
                    &[
                        ("mean", lab::Backup::Mean),
                        ("power", lab::Backup::Power),
                        ("mixmax", lab::Backup::MixMax),
                        ("softmax", lab::Backup::SoftMax),
                        ("bayes", lab::Backup::Bayes),
                    ],
                )?
            }
            "pick" => {
                c.pick = pick(
                    &st()?,
                    &key,
                    &[("visits", lab::Pick::Visits), ("q", lab::Pick::Q), ("improved", lab::Pick::Improved)],
                )?
            }
            "depths" => {
                c.depths = pick(
                    &st()?,
                    &key,
                    &[("all", lab::Depths::All), ("root", lab::Depths::Root), ("interior", lab::Depths::Interior)],
                )?
            }
            "c_puct" => c.c_puct = f()?,
            "fpu" => c.fpu = f()?,
            "virtual_loss" => c.virtual_loss = f()?,
            "kappa" => c.kappa = f()?,
            "ts_c" => c.ts_c = f()?,
            "pexp" => c.pexp = f()?,
            "nexp" => c.nexp = f()?,
            "n0" => c.n0 = f()?,
            "fit_a" => c.fit[0] = f()?,
            "fit_b" => c.fit[1] = f()?,
            "fit_c" => c.fit[2] = f()?,
            "ts_from" => c.ts_from = f()?,
            "bparam" => c.bparam = f()?,
            "leaf_s" => c.leaf_s = f()?,
            "c_visit" => c.c_visit = f()?,
            "c_scale" => c.c_scale = f()?,
            "pick_frac" => c.pick_frac = f()?,
            "eager" => c.eager = v.extract::<i32>()?,
            "eager_min" => c.eager_min = v.extract::<u32>()?,
            "report" => c.report = v.extract::<bool>()?,
            _ => return Err(PyValueError::new_err(format!("unknown lab option {key:?}"))),
        }
    }
    Ok(c)
}

/// Offline search lab as a search service (see `src/lab.rs`): `search(slot,
/// state)`, then collect/apply waves until `moves()` returns the move.
#[pyclass(name = "LabEngine", module = "thompsonzero._core")]
pub struct PyLabEngine {
    inner: AnyLab,
    #[pyo3(get)]
    num_actions: usize,
    obs_shape: (usize, usize, usize),
}

#[pymethods]
impl PyLabEngine {
    #[new]
    #[pyo3(signature = (game, *, slots, sims, wave=4, seed=0, **opts))]
    fn new(game: &str, slots: usize, sims: usize, wave: usize, seed: u64, opts: Option<&Bound<'_, PyDict>>) -> PyResult<Self> {
        let cfg = lab_cfg(opts)?;
        if wave == 0 || slots == 0 {
            return Err(PyValueError::new_err("need wave >= 1 and slots >= 1"));
        }
        with_game!(game, G => Ok(PyLabEngine {
            inner: G::wrap_lab(LabEngine::<G>::new(cfg, slots, sims, wave, seed)),
            num_actions: G::NUM_ACTIONS,
            obs_shape: obs_shape::<G>(),
        }))
    }

    fn search(&mut self, slot: usize, state: &PyGameState) -> PyResult<()> {
        each_lab!(&mut self.inner, e => search_in_lab(e, slot, &state.inner))
    }

    /// Simulate every active search up to its next evaluations; the group
    /// argument is accepted for AzEngine compatibility.
    #[pyo3(signature = (group=0))]
    fn collect<'py>(&mut self, py: Python<'py>, group: usize) -> PyResult<Batch<'py>> {
        let _ = group;
        let inner = &mut self.inner;
        let obs = py.allow_threads(|| each_lab!(inner, e => e.collect()));
        let (c, h, w) = self.obs_shape;
        let rows = obs.len() / (c * h * w);
        let obs = Array4::from_shape_vec((rows, c, h, w), obs).unwrap();
        Ok((obs.into_pyarray(py), Array1::<u8>::zeros(rows).into_pyarray(py)))
    }

    #[pyo3(signature = (group, v, logits, d=None))]
    fn apply(
        &mut self,
        py: Python<'_>,
        group: usize,
        v: PyReadonlyArray1<f32>,
        logits: PyReadonlyArray2<f32>,
        d: Option<PyReadonlyArray1<f32>>,
    ) -> PyResult<()> {
        let _ = group;
        let rows = each_lab!(&self.inner, e => e.pending_rows());
        if v.shape() != [rows] || logits.shape() != [rows, self.num_actions] {
            return Err(PyValueError::new_err(format!("expected {rows} rows")));
        }
        if d.as_ref().is_some_and(|d| d.shape() != [rows]) {
            return Err(PyValueError::new_err(format!("expected d ({rows},)")));
        }
        let (v, logits) = (v.as_slice()?, logits.as_slice()?);
        let d = match &d {
            Some(d) => Some(d.as_slice()?),
            None => None,
        };
        let inner = &mut self.inner;
        py.allow_threads(|| each_lab!(inner, e => e.apply(v, d, logits)));
        Ok(())
    }

    fn moves(&mut self) -> Vec<(usize, u16)> {
        each_lab!(&mut self.inner, e => e.moves())
    }

    /// Finished searches' root reports: [(slot, rows (edges, 11) float32,
    /// [v, var, mu, N])] (columns in `Tree::report`).
    fn reports<'py>(&mut self, py: Python<'py>) -> Vec<(usize, Bound<'py, PyArray2<f32>>, [f32; 4])> {
        each_lab!(&mut self.inner, e => e.reports())
            .into_iter()
            .map(|r| {
                let n = r.rows.len() / lab::REPORT_COLS;
                let a = Array2::from_shape_vec((n, lab::REPORT_COLS), r.rows).unwrap();
                (r.slot, a.into_pyarray(py), r.root)
            })
            .collect()
    }

    /// Network evaluations so far.
    fn evals(&self) -> u64 {
        each_lab!(&self.inner, e => e.evals())
    }

    fn set_sims(&mut self, sims: usize) {
        each_lab!(&mut self.inner, e => e.sims = sims)
    }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyEngine>()?;
    m.add_class::<PyAzEngine>()?;
    m.add_class::<PyLabEngine>()?;
    m.add_class::<PyGameState>()?;
    Ok(())
}
