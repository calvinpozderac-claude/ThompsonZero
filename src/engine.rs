//! Batched, multi-threaded self-play and arena play.
//!
//! The engine owns many concurrent games ("slots"), each with its own tree.
//! It is driven from Python in waves:
//!
//! ```text
//!   obs, nets = engine.collect(group)    # tree work on every core (no GIL)
//!   ...network forward on the GPU...
//!   engine.apply(group, v, a)            # expand, back up, play moves
//! ```
//!
//! Slots are split into `groups` so that while one group's batch is on the
//! GPU the CPU is already selecting leaves for the next.  Each `collect` runs
//! up to `wave` simulations per game (virtual loss diversifies them), so one
//! network batch holds up to `games / groups * wave` positions.
//!
//! In ARENA mode two networks play each other: every request is tagged with
//! the network of the player whose move is being searched, the move is the
//! greedy posterior pick, trees are not reused across moves (each side
//! searches with its own network only), and results are tallied for net 0.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::belief::{outcome_index, Rule};
use crate::games::Game;
use crate::tree::{Eval, SearchCfg, Selected, Tree};

#[derive(Clone, Debug)]
pub struct Config {
    pub num_games: usize,
    pub groups: usize,
    pub seed: u64,
    /// Simulations per move (read when a game starts).
    pub sims: usize,
    /// Simulations per game per `collect`.
    pub wave: usize,
    pub search: SearchCfg,
    /// Sharpness of in-tree descent: beliefs are sampled from Dir(temp * α).
    pub descent_temp: f64,
    /// Self-play: Thompson-pick the move for this many plies, greedy after.
    pub temp_threshold: usize,
    pub max_plies: usize,
    /// Self-play: also emit targets for this many most-visited unplayed root
    /// children per move (OLIVAW's "doubling"), if visited at least
    /// `double_min_visits` times.
    pub double_aux: usize,
    pub double_min_visits: f64,
    /// Self-play: emit an exact-label sample for every node the solver proves.
    pub solved_aux: bool,
    /// `Some(n)`: arena mode, play `n` games then stop.
    pub arena_games: Option<usize>,
    /// Arena: uniformly random opening plies for variety.
    pub opening_plies: usize,
}

impl Config {
    pub fn new(search: Rule, target: Rule) -> Config {
        Config {
            num_games: 512,
            groups: 2,
            seed: 0,
            sims: 200,
            wave: 4,
            search: SearchCfg {
                search,
                target,
                sample_leaf: false,
                virtual_loss: 1.0,
            },
            descent_temp: 1.0,
            temp_threshold: 20,
            max_plies: usize::MAX,
            double_aux: 1,
            double_min_visits: 8.0,
            solved_aux: true,
            arena_games: None,
            opening_plies: 2,
        }
    }
}

/// One training sample (see `python/thompsonzero/replay.py` for the layout).
#[derive(Clone, Debug)]
pub struct Sample {
    pub obs: Vec<f32>,
    pub legal: Vec<u16>,
    /// State target.
    pub v: [f32; 3],
    /// (action, target) for every edge search touched.
    pub edges: Vec<(u16, [f32; 3])>,
    /// Action played from here, or -1 (aux samples).
    pub played: i32,
    /// Position after `played`, unless that move ended the game...
    pub next_obs: Option<Vec<f32>>,
    /// ...in which case its outcome for this sample's mover (0/1/2), else -1.
    pub next_term: i8,
    /// Game outcome for this sample's mover (0 win, 1 draw, 2 loss).
    pub z: u8,
    /// 0 when the outcome was never observed (cut-off games, doubles).
    pub z_w: f32,
    /// Exact solver label.
    pub solved: bool,
    player: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Stats {
    pub games: u64,
    pub plies: u64,
    pub draws: u64,
    pub cutoffs: u64,
    pub sims: u64,
    pub evals: u64,
    pub samples: u64,
    pub solved_aux: u64,
    pub double_aux: u64,
    /// Arena: results from net 0's point of view.
    pub wins: u64,
    pub losses: u64,
}

impl Stats {
    fn merge(&mut self, o: &Stats) {
        self.games += o.games;
        self.plies += o.plies;
        self.draws += o.draws;
        self.cutoffs += o.cutoffs;
        self.sims += o.sims;
        self.evals += o.evals;
        self.samples += o.samples;
        self.solved_aux += o.solved_aux;
        self.double_aux += o.double_aux;
        self.wins += o.wins;
        self.losses += o.losses;
    }
}

enum Request<G> {
    Root(G),
    Child { edge: u32, state: G },
}

struct Slot<G> {
    active: bool,
    state: G,
    tree: Tree<G>,
    sims: usize,
    done: usize,
    ply: usize,
    /// Arena: which net plays player 0.
    seat0: u8,
    rng: SmallRng,
    hist: Vec<Sample>,
    aux: Vec<Sample>,
    requests: Vec<Request<G>>,
    /// Path range in `paths` of each simulation awaiting an evaluation.
    pending: Vec<(usize, usize)>,
    paths: Vec<(u32, u32)>,
    path: Vec<(u32, u32)>,
    solved: Vec<u32>,
    finished: Vec<Sample>,
    stats: Stats,
}

pub struct Engine<G: Game> {
    pub cfg: Config,
    slots: Vec<Slot<G>>,
    started: AtomicUsize,
    /// Rows each slot contributed to the last `collect` of its group.
    counts: Vec<usize>,
    finished: Vec<Sample>,
    stats: Stats,
}

impl<G: Game> Engine<G> {
    pub fn new(cfg: Config) -> Engine<G> {
        assert!(cfg.num_games >= cfg.groups && cfg.groups >= 1);
        let started = AtomicUsize::new(0);
        let mut slots: Vec<Slot<G>> = (0..cfg.num_games)
            .map(|i| Slot {
                active: false,
                state: G::new(),
                tree: Tree::default(),
                sims: cfg.sims,
                done: 0,
                ply: 0,
                seat0: 0,
                rng: SmallRng::seed_from_u64(cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64),
                hist: Vec::new(),
                aux: Vec::new(),
                requests: Vec::new(),
                pending: Vec::new(),
                paths: Vec::new(),
                path: Vec::new(),
                solved: Vec::new(),
                finished: Vec::new(),
                stats: Stats::default(),
            })
            .collect();
        for s in &mut slots {
            s.new_game(&cfg, &started);
        }
        Engine {
            counts: vec![0; cfg.num_games],
            cfg,
            slots,
            started,
            finished: Vec::new(),
            stats: Stats::default(),
        }
    }

    fn group_range(&self, group: usize) -> std::ops::Range<usize> {
        let n = self.slots.len();
        let g = self.cfg.groups;
        assert!(group < g, "group {group} out of range");
        (group * n / g)..((group + 1) * n / g)
    }

    /// Run simulations up to the next network evaluation for every game in
    /// `group`.  Returns the observations to evaluate (row-major, `OBS_SIZE`
    /// floats each) and the network index for each row.
    pub fn collect(&mut self, group: usize) -> (Vec<f32>, Vec<u8>) {
        let r = self.group_range(group);
        let cfg = &self.cfg;
        self.slots[r.clone()].par_iter_mut().for_each(|s| s.collect(cfg));
        let total: usize = self.slots[r.clone()].iter().map(|s| s.requests.len()).sum();
        let mut obs = vec![0f32; total * G::OBS_SIZE];
        let mut nets = Vec::with_capacity(total);
        let mut row = 0;
        for i in r {
            let s = &self.slots[i];
            self.counts[i] = s.requests.len();
            let net = if cfg.arena_games.is_some() {
                s.seat0 ^ s.state.current_player() as u8
            } else {
                0
            };
            for q in &s.requests {
                let st = match q {
                    Request::Root(st) | Request::Child { state: st, .. } => st,
                };
                st.encode(&mut obs[row * G::OBS_SIZE..(row + 1) * G::OBS_SIZE]);
                nets.push(net);
                row += 1;
            }
        }
        (obs, nets)
    }

    /// Feed back the evaluations for the rows of the last `collect(group)`:
    /// `v` is (rows, 4), `a` is (rows, NUM_ACTIONS, 4), both activated.
    pub fn apply(&mut self, group: usize, v: &[f32], a: &[f32]) {
        let r = self.group_range(group);
        let rows: usize = self.counts[r.clone()].iter().sum();
        assert_eq!(v.len(), rows * 4, "v has the wrong number of rows");
        assert_eq!(a.len(), rows * G::NUM_ACTIONS * 4, "a has the wrong number of rows");
        let mut offsets = Vec::with_capacity(r.len());
        let mut off = 0;
        for i in r.clone() {
            offsets.push(off);
            off += self.counts[i];
        }
        let (cfg, started) = (&self.cfg, &self.started);
        self.slots[r]
            .par_iter_mut()
            .zip(offsets)
            .for_each(|(s, off)| s.apply(cfg, started, &v[off * 4..], &a[off * G::NUM_ACTIONS * 4..]));
        for s in &mut self.slots {
            self.finished.append(&mut s.finished);
            self.stats.merge(&std::mem::take(&mut s.stats));
        }
    }

    /// Rows returned by the last `collect(group)`.
    pub fn pending_rows(&self, group: usize) -> usize {
        self.counts[self.group_range(group)].iter().sum()
    }

    /// Take every sample from games finished since the last call.
    pub fn drain(&mut self) -> Vec<Sample> {
        std::mem::take(&mut self.finished)
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn set_sims(&mut self, sims: usize) {
        self.cfg.sims = sims;
    }

    /// Arena: all games played.
    pub fn done(&self) -> bool {
        self.slots.iter().all(|s| !s.active)
    }
}

impl<G: Game> Slot<G> {
    fn new_game(&mut self, cfg: &Config, started: &AtomicUsize) {
        self.tree.clear();
        self.state = G::new();
        self.sims = cfg.sims;
        self.done = 0;
        self.ply = 0;
        self.hist.clear();
        self.aux.clear();
        let Some(total) = cfg.arena_games else {
            self.active = true;
            return;
        };
        let k = started.fetch_add(1, Ordering::Relaxed);
        self.active = k < total;
        self.seat0 = (k % 2) as u8; // alternate colours
        let mut legal = Vec::new();
        'retry: loop {
            self.state = G::new();
            for _ in 0..cfg.opening_plies {
                self.state.legal_actions(&mut legal);
                self.state.apply(legal[self.rng.random_range(0..legal.len())]);
                if self.state.is_terminal() {
                    continue 'retry;
                }
            }
            break;
        }
        self.ply = cfg.opening_plies;
    }

    fn collect(&mut self, cfg: &Config) {
        self.requests.clear();
        self.pending.clear();
        self.paths.clear();
        if !self.active {
            return;
        }
        if self.tree.is_empty() {
            self.requests.push(Request::Root(self.state.clone()));
            return;
        }
        if self.tree.solved(0).is_some() {
            return;
        }
        let sc = &cfg.search;
        for _ in 0..cfg.wave.min(self.sims.saturating_sub(self.done)) {
            match self.tree.select(sc, &mut self.rng, cfg.descent_temp, &mut self.path) {
                Selected::Terminal => {
                    self.solved.clear();
                    self.tree.propagate_solved(&self.path, sc, &mut self.solved);
                    if cfg.solved_aux && cfg.arena_games.is_none() {
                        for i in 0..self.solved.len() {
                            let n = self.solved[i];
                            let out = self.tree.solved(n).unwrap();
                            let mut t = self.sample(n, cfg);
                            t.solved = true;
                            t.z = out as u8;
                            self.aux.push(t);
                            self.stats.solved_aux += 1;
                        }
                    }
                    self.done += 1;
                    self.stats.sims += 1;
                    if self.tree.solved(0).is_some() {
                        break;
                    }
                }
                Selected::Leaf(state) => {
                    // several simulations of a wave may reach the same edge:
                    // one evaluation serves them all
                    let edge = self.path.last().unwrap().1;
                    if !self
                        .requests
                        .iter()
                        .any(|q| matches!(q, Request::Child { edge: e, .. } if *e == edge))
                    {
                        self.requests.push(Request::Child { edge, state });
                    }
                    let start = self.paths.len();
                    self.paths.extend_from_slice(&self.path);
                    self.pending.push((start, self.paths.len()));
                }
            }
        }
    }

    fn apply(&mut self, cfg: &Config, started: &AtomicUsize, v: &[f32], a: &[f32]) {
        if !self.active {
            return;
        }
        let sc = &cfg.search;
        let na = G::NUM_ACTIONS * 4;
        for (i, q) in self.requests.drain(..).enumerate() {
            let ev = Eval {
                v: &v[i * 4..i * 4 + 4],
                a: &a[i * na..(i + 1) * na],
            };
            match q {
                Request::Root(st) => {
                    self.tree.add_node(st, ev, sc, &mut self.rng);
                }
                Request::Child { edge, state } => {
                    self.tree.expand(edge, state, ev, sc, &mut self.rng);
                }
            }
            self.stats.evals += 1;
        }
        for &(lo, hi) in &self.pending {
            self.tree.backup_leaf(&self.paths[lo..hi], sc, &mut self.rng);
        }
        self.done += self.pending.len();
        self.stats.sims += self.pending.len() as u64;
        if self.tree.is_empty() || (self.done < self.sims && self.tree.solved(0).is_none()) {
            return;
        }
        self.play_move(cfg);
        if self.state.is_terminal() || self.ply >= cfg.max_plies {
            self.finish(cfg);
            self.new_game(cfg, started);
        }
    }

    /// The searched root as a training sample (`played`/`next_*`/`z` are
    /// filled in later).
    fn sample(&self, n: u32, cfg: &Config) -> Sample {
        let node = &self.tree.nodes[n as usize];
        let t = self.tree.target(n, &cfg.search);
        let mut obs = vec![0f32; G::OBS_SIZE];
        node.state.encode(&mut obs);
        let f = |d: [f64; 3]| [d[0] as f32, d[1] as f32, d[2] as f32];
        Sample {
            obs,
            legal: self.tree.edges_of(n).iter().map(|e| e.action).collect(),
            v: f(t.v),
            edges: t.edges.into_iter().map(|(a, d)| (a, f(d))).collect(),
            played: -1,
            next_obs: None,
            next_term: -1,
            z: 1,
            z_w: 1.0,
            solved: false,
            player: node.player,
        }
    }

    fn play_move(&mut self, cfg: &Config) {
        let selfplay = cfg.arena_games.is_none();
        let thompson = selfplay && self.ply < cfg.temp_threshold;
        let idx = self.tree.pick(&mut self.rng, thompson);
        let action = self.tree.edges_of(0)[idx].action;
        let mover = self.state.current_player();
        if selfplay {
            let mut t = self.sample(0, cfg);
            t.played = action as i32;
            self.emit_doubles(idx, cfg);
            self.state.apply(action);
            match self.state.result_for(mover) {
                Some(r) => t.next_term = outcome_index(r) as i8,
                None => {
                    let mut o = vec![0f32; G::OBS_SIZE];
                    self.state.encode(&mut o);
                    t.next_obs = Some(o);
                }
            }
            self.hist.push(t);
            self.tree.advance(idx);
        } else {
            self.state.apply(action);
            self.tree.clear();
        }
        self.ply += 1;
        self.done = 0;
    }

    /// OLIVAW's doubling: the most-visited unplayed children are free,
    /// well-resolved targets.  They have no game outcome (z_w = 0), so they
    /// train only the search-belief terms.
    fn emit_doubles(&mut self, played: usize, cfg: &Config) {
        if cfg.double_aux == 0 {
            return;
        }
        let mut order: Vec<(f64, usize)> = self.tree.visits(0).enumerate().map(|(i, n)| (n, i)).collect();
        order.sort_by(|x, y| y.0.total_cmp(&x.0));
        let mut got = 0;
        for (n, i) in order {
            if n < cfg.double_min_visits || got == cfg.double_aux {
                break;
            }
            let child = self.tree.edges_of(0)[i].child;
            if i == played || child == crate::tree::NONE {
                continue;
            }
            let mut t = self.sample(child, cfg);
            t.z_w = 0.0;
            self.aux.push(t);
            self.stats.double_aux += 1;
            got += 1;
        }
    }

    fn finish(&mut self, cfg: &Config) {
        self.stats.games += 1;
        self.stats.plies += self.ply as u64;
        match self.state.outcome() {
            Some(0) => self.stats.draws += 1,
            None => self.stats.cutoffs += 1,
            _ => {}
        }
        if let Some(r) = self.state.outcome() {
            // net that played player 0 is seat0; report for net 0
            let r0 = if self.seat0 == 0 { r } else { -r };
            if cfg.arena_games.is_some() {
                self.stats.wins += (r0 > 0) as u64;
                self.stats.losses += (r0 < 0) as u64;
            }
        }
        if cfg.arena_games.is_some() {
            return;
        }
        let outcome = self.state.outcome();
        for t in &mut self.hist {
            match outcome {
                Some(_) => t.z = outcome_index(self.state.result_for(t.player as usize).unwrap()) as u8,
                None => t.z_w = 0.0,
            }
        }
        self.stats.samples += (self.hist.len() + self.aux.len()) as u64;
        self.finished.append(&mut self.hist);
        self.finished.append(&mut self.aux);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{Boop, Connect4, Othello};

    fn uniform<G: Game>(rows: usize) -> (Vec<f32>, Vec<f32>) {
        let one = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.5];
        let v = (0..rows).flat_map(|_| one).collect();
        let a = (0..rows * G::NUM_ACTIONS).flat_map(|_| one).collect();
        (v, a)
    }

    fn run<G: Game>(cfg: Config, waves: usize) -> Engine<G> {
        let mut e = Engine::<G>::new(cfg);
        for _ in 0..waves {
            for g in 0..e.cfg.groups {
                let (obs, nets) = e.collect(g);
                assert_eq!(obs.len(), nets.len() * G::OBS_SIZE);
                let (v, a) = uniform::<G>(nets.len());
                e.apply(g, &v, &a);
            }
        }
        e
    }

    fn selfplay_cfg(rule: &str) -> Config {
        let r = Rule::parse(rule, 0.8).unwrap();
        let mut c = Config::new(r, r);
        c.num_games = 8;
        c.sims = 16;
        c
    }

    #[test]
    fn selfplay_produces_consistent_samples() {
        for rule in ["additive_mle", "corr"] {
            let mut cfg = selfplay_cfg(rule);
            cfg.search.target = Rule::parse(if rule == "corr" { "corr" } else { "avgconc" }, 0.999).unwrap();
            let mut e = run::<Connect4>(cfg, 400);
            let st = e.stats().clone();
            assert!(st.games > 0, "{rule}: no games finished");
            let samples = e.drain();
            assert_eq!(samples.len() as u64, st.samples);
            for s in &samples {
                assert_eq!(s.obs.len(), Connect4::OBS_SIZE);
                assert!(s.v.iter().all(|x| x.is_finite() && *x > 0.0));
                assert!(s.edges.iter().all(|(a, _)| s.legal.contains(a)));
                if s.played >= 0 {
                    assert!(s.legal.contains(&(s.played as u16)));
                    assert!(s.next_obs.is_some() != (s.next_term >= 0));
                }
            }
        }
    }

    #[test]
    fn arena_plays_exactly_n_games_and_alternates_seats() {
        for (rule, n) in [("additive_mle", 10usize), ("corr", 7)] {
            let mut cfg = selfplay_cfg(rule);
            cfg.arena_games = Some(n);
            let mut e = Engine::<Othello>::new(cfg);
            let mut guard = 0;
            while !e.done() {
                for g in 0..2 {
                    let (_obs, nets) = e.collect(g);
                    let (v, a) = uniform::<Othello>(nets.len());
                    e.apply(g, &v, &a);
                }
                guard += 1;
                assert!(guard < 100_000);
            }
            assert_eq!(e.stats().games as usize, n);
            assert!(e.drain().is_empty());
        }
    }

    #[test]
    fn zero_sims_plays_from_the_prior() {
        let mut cfg = selfplay_cfg("additive");
        cfg.sims = 0;
        cfg.arena_games = Some(4);
        let mut e = Engine::<Boop>::new(cfg);
        while !e.done() {
            for g in 0..2 {
                let (_o, nets) = e.collect(g);
                let (v, a) = uniform::<Boop>(nets.len());
                e.apply(g, &v, &a);
            }
        }
        assert_eq!(e.stats().games, 4);
        assert_eq!(e.stats().sims, 0);
    }
}
