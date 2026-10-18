//! Batched, multi-threaded AlphaZero self-play and arena play.
//!
//! The same wave protocol as [`crate::engine`], over the PUCT tree of
//! [`crate::puct`]:
//!
//! ```text
//!   obs, nets = engine.collect(group)    # tree work on every core (no GIL)
//!   ...network forward on the GPU...
//!   engine.apply(group, v, logits)       # expand, back up, play moves
//! ```
//!
//! `v` is one value per row (the mover's, in [-1, 1]) and `logits` one policy
//! logit per action id; the engine takes the softmax over the legal moves.
//!
//! SELF-PLAY follows AlphaGo Zero / OLIVAW: Dirichlet noise on the root
//! priors, moves sampled in proportion to visits for the first
//! `temp_threshold` plies and the most visited after, the subtree kept
//! between moves, and every searched position emitted as a sample
//! (visit-count policy target, game-outcome value target).  OLIVAW's
//! "doubling" adds the most visited unplayed root children with their own
//! visit distribution and their search value q as the value target.  Forced
//! moves (a single legal action) are played without search and emit nothing.
//!
//! ARENA mode plays `arena_games` games: game k starts from opening
//! `openings[(k / 2) % len]` (or `opening_plies` random plies if there are
//! none) and net 0 plays black in even games, white in odd ones, so every
//! opening is played from both seats.  Moves are the most visited child of a
//! fresh search (no noise, no tree reuse, so `sims` bounds the tree size).
//! With `external`, net 0's opponent is not a network: its turns are listed
//! by `waiting` and answered by `play_external`.
//!
//! As a SEARCH SERVICE the engine has no games of its own: `search(slot,
//! state)` hands a slot a position and the most visited move of a fresh
//! search with net 0 comes back from `moves` (see `crate::engine`).

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::{rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::engine::Stats;
use crate::games::Game;
use crate::puct::{PuctCfg, Selected, Tree, NONE};

#[derive(Clone, Debug)]
pub struct AzConfig {
    pub num_games: usize,
    pub groups: usize,
    pub seed: u64,
    /// Simulations per move (read when a move's search starts).
    pub sims: usize,
    /// Simulations per game per `collect`.
    pub wave: usize,
    pub puct: PuctCfg,
    /// Self-play root noise: Dir(alpha) with alpha = min(1, noise_alpha / B)
    /// over B legal moves, mixed in with weight `noise_frac`.
    pub noise_alpha: f32,
    pub noise_frac: f32,
    /// Self-play: sample moves by visit count for this many plies.
    pub temp_threshold: usize,
    pub max_plies: usize,
    /// Self-play: keep the played child's subtree for the next move.
    pub reuse_tree: bool,
    /// Self-play: emit up to this many most-visited unplayed root children
    /// per move, if visited at least `double_min_visits` times.
    pub double_aux: usize,
    pub double_min_visits: u32,
    /// `Some(n)`: arena mode, play `n` games then stop.
    pub arena_games: Option<usize>,
    /// Arena: net 0's opponent moves come from `play_external`.
    pub external: bool,
    /// Arena: opening move sequences (see the module docs).
    pub openings: Vec<Vec<u16>>,
    /// Arena without openings: uniformly random opening plies.
    pub opening_plies: usize,
    /// Search service (see the module docs).
    pub service: bool,
}

impl AzConfig {
    /// Self-play (as opposed to arena play or a search service).
    pub fn selfplay(&self) -> bool {
        self.arena_games.is_none() && !self.service
    }
}

impl Default for AzConfig {
    fn default() -> Self {
        AzConfig {
            num_games: 512,
            groups: 2,
            seed: 0,
            sims: 200,
            wave: 4,
            puct: PuctCfg {
                c_puct: 1.5,
                fpu_reduction: 0.2,
                virtual_loss: 1.0,
            },
            noise_alpha: 10.0,
            noise_frac: 0.25,
            temp_threshold: 20,
            max_plies: usize::MAX,
            reuse_tree: true,
            double_aux: 1,
            double_min_visits: 8,
            arena_games: None,
            external: false,
            openings: Vec::new(),
            opening_plies: 2,
            service: false,
        }
    }
}

/// One training sample.
#[derive(Clone, Debug)]
pub struct AzSample {
    pub obs: Vec<f32>,
    /// Visit distribution over the searched moves as (action, probability).
    pub pi: Vec<(u16, f32)>,
    /// Value target for the mover: the game result, or q for doubles.
    pub z: f32,
    /// 0 when the game was cut off before its outcome was observed.
    pub z_w: f32,
    player: u8,
}

/// An arena game's result for net 0 (+1 / 0 / -1); `seat0` is net 0's player.
#[derive(Clone, Copy, Debug)]
pub struct GameResult {
    pub game: usize,
    pub seat0: u8,
    pub result: i8,
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
    /// Arena: index of this game and which player net 0 is.
    game: usize,
    seat0: u8,
    /// Root noise applied to the current root.
    noised: bool,
    /// External opponent to move.
    waiting: bool,
    /// Search service: the move found, until `moves` takes it.
    moved: Option<u16>,
    rng: SmallRng,
    hist: Vec<AzSample>,
    aux: Vec<AzSample>,
    requests: Vec<Request<G>>,
    /// Path range in `paths` of each simulation awaiting an evaluation.
    pending: Vec<(usize, usize)>,
    paths: Vec<(u32, u32)>,
    path: Vec<(u32, u32)>,
    finished: Vec<AzSample>,
    results: Vec<GameResult>,
    stats: Stats,
}

pub struct AzEngine<G: Game> {
    pub cfg: AzConfig,
    slots: Vec<Slot<G>>,
    started: AtomicUsize,
    /// Rows each slot contributed to the last `collect` of its group.
    counts: Vec<usize>,
    finished: Vec<AzSample>,
    results: Vec<GameResult>,
    stats: Stats,
}

impl<G: Game> AzEngine<G> {
    pub fn new(cfg: AzConfig) -> AzEngine<G> {
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
                game: 0,
                seat0: 0,
                noised: false,
                waiting: false,
                moved: None,
                rng: SmallRng::seed_from_u64(cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64),
                hist: Vec::new(),
                aux: Vec::new(),
                requests: Vec::new(),
                pending: Vec::new(),
                paths: Vec::new(),
                path: Vec::new(),
                finished: Vec::new(),
                results: Vec::new(),
                stats: Stats::default(),
            })
            .collect();
        for s in &mut slots {
            s.new_game(&cfg, &started);
        }
        AzEngine {
            counts: vec![0; cfg.num_games],
            cfg,
            slots,
            started,
            finished: Vec::new(),
            results: Vec::new(),
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
        let slots = &mut self.slots[r.clone()];
        slots.par_iter_mut().for_each(|s| s.collect(cfg));
        let mut nets = Vec::new();
        for (s, count) in slots.iter().zip(&mut self.counts[r]) {
            *count = s.requests.len();
            let net = if cfg.arena_games.is_some() {
                s.seat0 ^ s.state.current_player() as u8
            } else {
                0
            };
            nets.extend(std::iter::repeat_n(net, *count));
        }
        // encode in parallel, each slot straight into its own rows
        let mut obs = vec![0f32; nets.len() * G::OBS_SIZE];
        let mut chunks = Vec::with_capacity(slots.len());
        let mut rest = obs.as_mut_slice();
        for s in slots.iter() {
            let (head, tail) = rest.split_at_mut(s.requests.len() * G::OBS_SIZE);
            chunks.push(head);
            rest = tail;
        }
        slots.par_iter().zip(chunks).for_each(|(s, out)| {
            for (q, row) in s.requests.iter().zip(out.chunks_exact_mut(G::OBS_SIZE)) {
                match q {
                    Request::Root(st) | Request::Child { state: st, .. } => st.encode(row),
                }
            }
        });
        (obs, nets)
    }

    /// Feed back the evaluations for the rows of the last `collect(group)`:
    /// `v` has one value per row, `logits` NUM_ACTIONS per row.
    pub fn apply(&mut self, group: usize, v: &[f32], logits: &[f32]) {
        let r = self.group_range(group);
        let rows: usize = self.counts[r.clone()].iter().sum();
        assert_eq!(v.len(), rows, "v has the wrong number of rows");
        assert_eq!(
            logits.len(),
            rows * G::NUM_ACTIONS,
            "logits has the wrong number of rows"
        );
        let mut offsets = Vec::with_capacity(r.len());
        let mut off = 0;
        for i in r.clone() {
            offsets.push(off);
            off += self.counts[i];
        }
        let (cfg, started) = (&self.cfg, &self.started);
        self.slots[r.clone()]
            .par_iter_mut()
            .zip(offsets)
            .for_each(|(s, off)| s.apply(cfg, started, &v[off..], &logits[off * G::NUM_ACTIONS..]));
        // consumed: a repeated apply now expects (and gets) zero rows
        self.counts[r.clone()].fill(0);
        for i in r {
            self.gather(i);
        }
    }

    fn gather(&mut self, i: usize) {
        let s = &mut self.slots[i];
        self.finished.append(&mut s.finished);
        self.results.append(&mut s.results);
        self.stats.merge(&std::mem::take(&mut s.stats));
    }

    /// Rows returned by the last `collect(group)`.
    pub fn pending_rows(&self, group: usize) -> usize {
        self.counts[self.group_range(group)].iter().sum()
    }

    /// Take every sample from games finished since the last call.
    pub fn drain(&mut self) -> Vec<AzSample> {
        std::mem::take(&mut self.finished)
    }

    /// Take the arena results recorded since the last call.
    pub fn take_results(&mut self) -> Vec<GameResult> {
        std::mem::take(&mut self.results)
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

    /// External mode: (slot, position) of every game where the external
    /// opponent is to move.
    pub fn waiting(&self) -> Vec<(usize, G)> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.active && s.waiting)
            .map(|(i, s)| (i, s.state.clone()))
            .collect()
    }

    /// External mode: play the opponent's `action` in `slot`.
    pub fn play_external(&mut self, slot: usize, action: u16) -> Result<(), String> {
        let s = self.slots.get_mut(slot).ok_or("slot out of range")?;
        if !(s.active && s.waiting) {
            return Err(format!("slot {slot} is not waiting for an external move"));
        }
        let mut legal = Vec::new();
        s.state.legal_actions(&mut legal);
        if !legal.contains(&action) {
            return Err(format!("illegal external move {action} in slot {slot}"));
        }
        s.state.apply(action);
        s.ply += 1;
        s.waiting = false;
        s.after_move(&self.cfg, &self.started);
        self.gather(slot);
        Ok(())
    }

    /// Search service: search `state` in `slot` (which must be idle).
    pub fn search(&mut self, slot: usize, state: G) -> Result<(), String> {
        if !self.cfg.service {
            return Err("not a search service".into());
        }
        let s = self.slots.get_mut(slot).ok_or("slot out of range")?;
        if s.active || s.moved.is_some() {
            return Err(format!("slot {slot} is busy"));
        }
        if state.is_terminal() {
            return Err("cannot search a finished game".into());
        }
        s.tree.clear();
        s.state = state;
        s.sims = self.cfg.sims;
        s.done = 0;
        s.active = true;
        Ok(())
    }

    /// Search service: (slot, move) of every search finished since the
    /// last call; those slots are idle again.
    pub fn moves(&mut self) -> Vec<(usize, u16)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.moved.take().map(|a| (i, a)))
            .collect()
    }
}

impl<G: Game> Slot<G> {
    fn new_game(&mut self, cfg: &AzConfig, started: &AtomicUsize) {
        self.tree.clear();
        self.state = G::new();
        self.sims = cfg.sims;
        self.done = 0;
        self.ply = 0;
        self.noised = false;
        self.waiting = false;
        self.hist.clear();
        self.aux.clear();
        if cfg.service {
            self.active = false; // until `search`
            return;
        }
        let Some(total) = cfg.arena_games else {
            self.active = true;
            return;
        };
        let k = started.fetch_add(1, Ordering::Relaxed);
        self.active = k < total;
        self.game = k;
        self.seat0 = (k % 2) as u8; // every opening from both seats
        if !cfg.openings.is_empty() {
            for &a in &cfg.openings[(k / 2) % cfg.openings.len()] {
                self.state.apply(a);
            }
            self.ply = cfg.openings[(k / 2) % cfg.openings.len()].len();
        } else {
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
        self.waiting = self.active && cfg.external && self.state.current_player() as u8 != self.seat0;
    }

    fn collect(&mut self, cfg: &AzConfig) {
        self.requests.clear();
        self.pending.clear();
        self.paths.clear();
        if !self.active || self.waiting {
            return;
        }
        if self.tree.is_empty() {
            self.requests.push(Request::Root(self.state.clone()));
            return;
        }
        if self.tree.edges_of(0).len() == 1 {
            self.done = self.sims; // forced move: nothing to search
            return;
        }
        if !self.noised {
            if cfg.selfplay() {
                self.tree.add_root_noise(&mut self.rng, cfg.noise_alpha, cfg.noise_frac);
            }
            self.noised = true;
        }
        for _ in 0..cfg.wave.min(self.sims.saturating_sub(self.done)) {
            match self.tree.select(&cfg.puct, &mut self.path) {
                Selected::Terminal => {
                    self.done += 1;
                    self.stats.sims += 1;
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

    fn apply(&mut self, cfg: &AzConfig, started: &AtomicUsize, v: &[f32], logits: &[f32]) {
        if !self.active {
            return;
        }
        let na = G::NUM_ACTIONS;
        for (i, q) in self.requests.drain(..).enumerate() {
            let l = &logits[i * na..(i + 1) * na];
            match q {
                Request::Root(st) => {
                    self.tree.add_node(st, v[i], l);
                }
                Request::Child { edge, state } => {
                    self.tree.expand(edge, state, v[i], l);
                }
            }
            self.stats.evals += 1;
        }
        for &(lo, hi) in &self.pending {
            self.tree.backup_leaf(&self.paths[lo..hi], &cfg.puct);
        }
        self.done += self.pending.len();
        self.stats.sims += self.pending.len() as u64;
        self.pending.clear();
        self.paths.clear();
        if self.waiting || self.tree.is_empty() || self.done < self.sims {
            return;
        }
        if cfg.service {
            let idx = self.tree.pick(&mut self.rng, false);
            self.moved = Some(self.tree.edges_of(0)[idx].action);
            self.tree.clear();
            self.active = false;
            return;
        }
        self.play_move(cfg);
        self.after_move(cfg, started);
    }

    /// End the game if it is over, else hand the move to the external
    /// opponent if it is theirs.
    fn after_move(&mut self, cfg: &AzConfig, started: &AtomicUsize) {
        if self.state.is_terminal() || self.ply >= cfg.max_plies {
            self.finish(cfg);
            self.new_game(cfg, started);
        } else if cfg.external {
            self.waiting = self.state.current_player() as u8 != self.seat0;
        }
    }

    fn play_move(&mut self, cfg: &AzConfig) {
        let selfplay = cfg.selfplay();
        let forced = self.tree.edges_of(0).len() == 1;
        let idx = self.tree.pick(&mut self.rng, selfplay && self.ply < cfg.temp_threshold);
        let action = self.tree.edges_of(0)[idx].action;
        if selfplay && !forced {
            let mut obs = vec![0f32; G::OBS_SIZE];
            self.state.encode(&mut obs);
            let mut pi = self.tree.policy(0);
            if pi.is_empty() {
                pi.push((action, 1.0)); // sims = 0: the move played
            }
            self.hist.push(AzSample {
                obs,
                pi,
                z: 0.0,
                z_w: 1.0,
                player: self.state.current_player() as u8,
            });
            self.emit_doubles(idx, cfg);
        }
        self.state.apply(action);
        if selfplay && cfg.reuse_tree {
            self.tree.advance(idx);
        } else {
            self.tree.clear();
        }
        self.noised = false;
        self.ply += 1;
        self.done = 0;
    }

    /// OLIVAW's doubling: the most visited unplayed children are free,
    /// well-resolved targets, with their search value q in place of z.
    fn emit_doubles(&mut self, played: usize, cfg: &AzConfig) {
        if cfg.double_aux == 0 {
            return;
        }
        let edges = self.tree.edges_of(0);
        let mut order: Vec<(u32, usize)> = edges.iter().enumerate().map(|(i, e)| (e.n, i)).collect();
        order.sort_by_key(|x| std::cmp::Reverse(x.0));
        let mut got = 0;
        for (n, i) in order {
            if n < cfg.double_min_visits || got == cfg.double_aux {
                break;
            }
            let child = self.tree.edges_of(0)[i].child;
            if i == played || child == NONE {
                continue;
            }
            let pi = self.tree.policy(child);
            if pi.is_empty() {
                continue;
            }
            let node = &self.tree.nodes[child as usize];
            let mut obs = vec![0f32; G::OBS_SIZE];
            node.state.encode(&mut obs);
            self.aux.push(AzSample {
                obs,
                pi,
                z: self.tree.mean_value(child),
                z_w: 1.0,
                player: node.player,
            });
            self.stats.double_aux += 1;
            got += 1;
        }
    }

    fn finish(&mut self, cfg: &AzConfig) {
        self.stats.games += 1;
        self.stats.plies += self.ply as u64;
        let outcome = self.state.outcome();
        match outcome {
            Some(0) => self.stats.draws += 1,
            None => self.stats.cutoffs += 1,
            _ => {}
        }
        if cfg.arena_games.is_some() {
            // net 0 played player `seat0`; report for net 0 (cut off = 0)
            let r = outcome.unwrap_or(0);
            let r0 = if self.seat0 == 0 { r } else { -r };
            self.stats.wins += (r0 > 0) as u64;
            self.stats.losses += (r0 < 0) as u64;
            self.results.push(GameResult {
                game: self.game,
                seat0: self.seat0,
                result: r0,
            });
            return;
        }
        for t in &mut self.hist {
            match self.state.result_for(t.player as usize) {
                Some(r) => t.z = r as f32,
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
    use crate::games::{Connect4, Othello};

    fn uniform<G: Game>(rows: usize) -> (Vec<f32>, Vec<f32>) {
        (vec![0.0; rows], vec![0.0; rows * G::NUM_ACTIONS])
    }

    fn cfg() -> AzConfig {
        AzConfig {
            num_games: 8,
            sims: 16,
            ..AzConfig::default()
        }
    }

    #[test]
    fn selfplay_produces_consistent_samples() {
        let mut e = AzEngine::<Othello>::new(cfg());
        let mut guard = 0;
        while e.stats().games < 4 {
            for g in 0..e.cfg.groups {
                let (obs, nets) = e.collect(g);
                assert_eq!(obs.len(), nets.len() * Othello::OBS_SIZE);
                let (v, l) = uniform::<Othello>(nets.len());
                e.apply(g, &v, &l);
            }
            guard += 1;
            assert!(guard < 100_000);
        }
        let st = e.stats().clone();
        let samples = e.drain();
        assert_eq!(samples.len() as u64, st.samples);
        assert!(st.double_aux > 0);
        for s in &samples {
            assert_eq!(s.obs.len(), Othello::OBS_SIZE);
            let p: f32 = s.pi.iter().map(|x| x.1).sum();
            assert!((p - 1.0).abs() < 1e-4);
            assert!((-1.0..=1.0).contains(&s.z));
        }
    }

    #[test]
    fn arena_plays_openings_from_both_seats() {
        let mut c = cfg();
        c.arena_games = Some(6);
        c.openings = vec![vec![3, 3], vec![0, 6], vec![2, 4]];
        let mut e = AzEngine::<Connect4>::new(c);
        while !e.done() {
            for g in 0..2 {
                let (_obs, nets) = e.collect(g);
                let (v, l) = uniform::<Connect4>(nets.len());
                e.apply(g, &v, &l);
            }
        }
        assert_eq!(e.stats().games, 6);
        let mut r = e.take_results();
        r.sort_by_key(|x| x.game);
        let seats: Vec<u8> = r.iter().map(|x| x.seat0).collect();
        assert_eq!(seats, vec![0, 1, 0, 1, 0, 1]);
        assert!(e.drain().is_empty());
    }

    #[test]
    fn service_searches_given_positions() {
        let mut c = cfg();
        c.service = true;
        c.num_games = 3;
        c.groups = 1;
        let mut e = AzEngine::<Othello>::new(c);
        assert!(e.done());
        let mut st = Othello::new();
        st.apply(37);
        let mut legal = Vec::new();
        st.legal_actions(&mut legal);
        e.search(1, st.clone()).unwrap();
        assert!(e.search(1, st.clone()).is_err());
        let mut got = Vec::new();
        let mut guard = 0;
        while got.is_empty() {
            let (_obs, nets) = e.collect(0);
            let (v, l) = uniform::<Othello>(nets.len());
            e.apply(0, &v, &l);
            got = e.moves();
            guard += 1;
            assert!(guard < 1000);
        }
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].0, 1);
        assert!(legal.contains(&got[0].1));
        assert!(e.done() && e.drain().is_empty() && e.take_results().is_empty());
        e.search(1, st).unwrap(); // idle again
    }

    #[test]
    fn external_opponent_gets_every_other_move() {
        let mut c = cfg();
        c.arena_games = Some(4);
        c.external = true;
        c.openings = vec![vec![37, 43, 26, 19]]; // f5 d6 c4 d3: black to move
        let mut e = AzEngine::<Othello>::new(c);
        let mut legal = Vec::new();
        let mut guard = 0;
        while !e.done() {
            for g in 0..2 {
                let (_obs, nets) = e.collect(g);
                assert!(nets.iter().all(|&n| n == 0));
                let (v, l) = uniform::<Othello>(nets.len());
                e.apply(g, &v, &l);
            }
            for (slot, st) in e.waiting() {
                st.legal_actions(&mut legal);
                e.play_external(slot, legal[0]).unwrap();
            }
            guard += 1;
            assert!(guard < 100_000);
        }
        let r = e.take_results();
        assert_eq!(r.len(), 4);
        assert!(e.play_external(0, 0).is_err());
    }
}
