//! Offline search lab: an AlphaZero-style tree whose selection rule, value
//! backup and per-edge uncertainty are all configurable, run as a search
//! service (`LabEngine`) for play-time comparisons and calibration reports.
//! Nothing here is used in training; the defaults reproduce `puct`'s PUCT.
//!
//! Every edge keeps the AZ sums (visits n, value sum w) plus the sum of
//! squared backups and the sum of the leaves' predicted outcome variances
//! (1 - d - v^2: a win/draw/loss head's, or 1 - v^2 with a scalar head).
//! Each node keeps its BACKED-UP value `mu` under the chosen backup:
//!
//! ```text
//!   mean     (v + sum_a w_a) / (1 + N)                       (AlphaZero)
//!   power    (v + sum_a n_a^p q_a) / (1 + sum_a n_a^p)       (p > 1 leans to the most visited)
//!   mixmax   (1 - l) * mean + l * max_a q_a
//!   softmax  t * log((e^(v/t) + sum_a n_a e^(q_a/t)) / (1 + N))
//!   bayes    Clark's moment-matched max of the visited children's
//!            N(q_a, s_a^2), a leaf being N(v, leaf_s^2 * var)
//! ```
//!
//! with q_a the child's backed-up value seen from this node (sums for
//! `mean`, recursive otherwise).  Selection scores each edge as
//!
//! ```text
//!   x = q + kappa * U + sigma * g(z),   U = c_puct P sqrt(1 + N) / (1 + n)
//! ```
//!
//! with z ~ N(0, 1) and g(z) one of z, |z|, max(z, 0) or 1 (a deterministic
//! upper bound); `puct` is kappa = 1, sigma = 0.  sigma comes from
//!
//! ```text
//!   u        ts_c P^pexp (1 + N)^nexp / (1 + n)      (PUCT's U)
//!   count    ts_c / sqrt(1 + n)
//!   netvar   ts_c sqrt(mean leaf variance / (1 + n))
//!   emp      ts_c sqrt(s^2 / (1 + n)), s^2 the backups' sample variance
//!            shrunk towards the node's predicted variance with weight n0
//!   bayes    ts_c * the child's propagated standard deviation
//!   fit      ts_c var^fit_a s2^fit_b / (1 + n)^fit_c, var the mean leaf
//!            variance and s2 `emp`'s variance with n0 = 0.1 (a power law
//!            fitted to deep-search errors)
//! ```
//!
//! An unvisited edge's q is the node's backed-up value minus `fpu`, unless
//! its child was evaluated EAGERLY: with `eager` = d, every node at depth
//! <= d has all its children evaluated by the network as soon as it is
//! expanded (or, with `eager_min` = m, once m simulations have passed
//! through it), and an unvisited edge with an evaluated child takes the
//! child's value as its q.  The first simulation to take such an edge
//! backs up that value with no further evaluation.
//! `gumbel` instead selects as Gumbel MuZero's non-root rule (argmax of
//! pi'(a) - n_a / (1 + N), pi' = softmax(log P + sigma(completed q))).  The
//! move played is the most visited (`visits`), the best q among edges with
//! at least `pick_frac` of the top visits (`q`), or argmax pi' (`improved`).

use rand::rngs::SmallRng;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

use crate::games::Game;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sel {
    Puct,
    Ts,
    Gumbel,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sigma {
    U,
    Count,
    NetVar,
    Emp,
    Bayes,
    Fit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ZMode {
    Normal,
    Abs,
    Pos,
    One,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Backup {
    Mean,
    Power,
    MixMax,
    SoftMax,
    Bayes,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pick {
    Visits,
    Q,
    Improved,
}

/// Where Thompson selection applies (PUCT elsewhere).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Depths {
    All,
    Root,
    Interior,
}

#[derive(Clone, Copy, Debug)]
pub struct LabCfg {
    pub sel: Sel,
    pub c_puct: f32,
    pub fpu: f32,
    pub virtual_loss: f32,
    pub kappa: f32,
    pub sigma: Sigma,
    pub ts_c: f32,
    pub pexp: f32,
    pub nexp: f32,
    pub n0: f32,
    pub fit: [f32; 3],
    pub zmode: ZMode,
    /// PUCT for this fraction of each search's simulations first.
    pub ts_from: f32,
    pub depths: Depths,
    pub backup: Backup,
    /// p (power), l (mixmax) or t (softmax).
    pub bparam: f32,
    pub leaf_s: f32,
    pub c_visit: f32,
    pub c_scale: f32,
    pub pick: Pick,
    pub pick_frac: f32,
    /// Evaluate the children of nodes at depth <= this (-1: never).
    pub eager: i32,
    /// ... once this many simulations have passed through (0: on creation).
    pub eager_min: u32,
    /// Record a per-edge report of each finished root search.
    pub report: bool,
}

impl Default for LabCfg {
    fn default() -> Self {
        LabCfg {
            sel: Sel::Puct,
            c_puct: 1.5,
            fpu: 0.2,
            virtual_loss: 1.0,
            kappa: 0.0,
            sigma: Sigma::U,
            ts_c: 0.5,
            pexp: 1.0,
            nexp: 0.5,
            n0: 1.0,
            fit: [0.5, 0.4, 0.25],
            zmode: ZMode::Normal,
            ts_from: 0.0,
            depths: Depths::All,
            backup: Backup::Mean,
            bparam: 1.0,
            leaf_s: 1.0,
            c_visit: 50.0,
            c_scale: 1.0,
            pick: Pick::Visits,
            pick_frac: 0.1,
            eager: -1,
            eager_min: 0,
            report: false,
        }
    }
}

impl LabCfg {
    /// Whether nodes must carry the propagated (Bayes) distribution.
    fn bayes(&self) -> bool {
        self.backup == Backup::Bayes || (self.sel == Sel::Ts && self.sigma == Sigma::Bayes) || self.report
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub action: u16,
    pub prior: f32,
    pub n: u32,
    /// Sums of backed-up values and their squares (node mover's view).
    pub w: f32,
    pub w2: f32,
    /// Sum of the backed-up leaves' outcome variances.
    pub vs: f32,
    vloss: u16,
    /// Exact result of a terminal child for the node's mover, else NaN.
    pub term: f32,
    pub child: u32,
}

#[derive(Clone, Debug)]
pub struct Node<G> {
    pub state: G,
    pub player: u8,
    first: u32,
    len: u32,
    /// Network value for the mover and predicted outcome variance.
    pub v: f32,
    pub var: f32,
    /// Backed-up value under the configured backup.
    pub mu: f32,
    /// Propagated mean and variance (Bayes).
    pub bmu: f32,
    pub bs2: f32,
    pub depth: u16,
    /// Children queued for eager evaluation.
    peeked: bool,
}

pub enum Selected<G> {
    /// Ended at a terminal or an eagerly evaluated child; backed up.
    Done,
    Leaf(G),
}

pub struct Tree<G> {
    pub nodes: Vec<Node<G>>,
    edges: Vec<Edge>,
    legal: Vec<u16>,
    scratch: Vec<(f32, f32)>,
    /// Nodes whose children await eager evaluation.
    pub peek_queue: Vec<u32>,
}

impl<G: Game> Default for Tree<G> {
    fn default() -> Self {
        Tree {
            nodes: Vec::new(),
            edges: Vec::new(),
            legal: Vec::new(),
            scratch: Vec::new(),
            peek_queue: Vec::new(),
        }
    }
}

/// Standard normal cdf (Abramowitz & Stegun 7.1.26, |error| < 1.5e-7).
fn norm_cdf(x: f32) -> f32 {
    let z = (x as f64) / std::f64::consts::SQRT_2;
    let t = 1.0 / (1.0 + 0.3275911 * z.abs());
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592)
            * t
            * (-z * z).exp();
    let erf = if z >= 0.0 { y } else { -y };
    (0.5 * (1.0 + erf)) as f32
}

fn norm_pdf(x: f32) -> f32 {
    (-0.5 * x * x).exp() * 0.398_942_3
}

/// Moments of max(X, Y) for independent normals (Clark 1961).
pub fn clark_max(a: (f32, f32), b: (f32, f32)) -> (f32, f32) {
    let ((m1, v1), (m2, v2)) = (a, b);
    let t2 = v1 + v2;
    if t2 < 1e-10 {
        return if m1 >= m2 { a } else { b };
    }
    let t = t2.sqrt();
    let al = (m1 - m2) / t;
    let (p, d) = (norm_cdf(al), norm_pdf(al));
    let mu = m1 * p + m2 * (1.0 - p) + t * d;
    let ex2 = (m1 * m1 + v1) * p + (m2 * m2 + v2) * (1.0 - p) + (m1 + m2) * t * d;
    (mu, (ex2 - mu * mu).max(0.0))
}

impl<G: Game> Tree<G> {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.peek_queue.clear();
    }

    pub fn edges_of(&self, n: u32) -> &[Edge] {
        let nd = &self.nodes[n as usize];
        &self.edges[nd.first as usize..(nd.first + nd.len) as usize]
    }

    pub fn add_node(&mut self, state: G, depth: u16, v: f32, d: f32, logits: &[f32], cfg: &LabCfg) -> u32 {
        state.legal_actions(&mut self.legal);
        let first = self.edges.len() as u32;
        let max = self
            .legal
            .iter()
            .map(|&a| logits[a as usize])
            .fold(f32::NEG_INFINITY, f32::max);
        let mut total = 0.0;
        for &a in &self.legal {
            let p = (logits[a as usize] - max).exp();
            total += p;
            self.edges.push(Edge {
                action: a,
                prior: p,
                n: 0,
                w: 0.0,
                w2: 0.0,
                vs: 0.0,
                vloss: 0,
                term: f32::NAN,
                child: NONE,
            });
        }
        for e in &mut self.edges[first as usize..] {
            e.prior /= total;
        }
        let v = v.clamp(-1.0, 1.0);
        let var = (1.0 - d.clamp(0.0, 1.0) - v * v).max(1e-4);
        self.nodes.push(Node {
            player: state.current_player() as u8,
            state,
            first,
            len: self.legal.len() as u32,
            v,
            var,
            mu: v,
            bmu: v,
            bs2: cfg.leaf_s * cfg.leaf_s * var,
            depth,
            peeked: false,
        });
        (self.nodes.len() - 1) as u32
    }

    /// Attach the evaluated child of edge `e`; `visited`: by a simulation
    /// (not an eager evaluation), so its own children may be queued.
    pub fn expand(&mut self, e: u32, state: G, depth: u16, v: f32, d: f32, logits: &[f32], cfg: &LabCfg, visited: bool) {
        let n = self.add_node(state, depth, v, d, logits, cfg);
        self.edges[e as usize].child = n;
        if visited {
            self.on_visit(n, cfg);
        }
    }

    /// A simulation reached node `n`: queue its children for eager
    /// evaluation if due.
    pub fn on_visit(&mut self, n: u32, cfg: &LabCfg) {
        let nd = &mut self.nodes[n as usize];
        if !nd.peeked && cfg.eager >= nd.depth as i32 && cfg.eager_min == 0 {
            nd.peeked = true;
            self.peek_queue.push(n);
        }
    }

    pub fn set_term(&mut self, e: u32, r: f32) {
        self.edges[e as usize].term = r;
    }

    /// The eagerly evaluated child value of unvisited edge `e`, from the
    /// node's mover's view.
    fn peek_q(&self, player: u8, e: &Edge) -> Option<f32> {
        if !e.term.is_nan() {
            return Some(e.term);
        }
        if e.child == NONE {
            return None;
        }
        let c = &self.nodes[e.child as usize];
        Some(if c.player == player { c.v } else { -c.v })
    }

    /// Edge `e` (of a node whose mover is `player`) as seen by that mover:
    /// its backed-up value, or None while unvisited.
    fn edge_q(&self, player: u8, e: &Edge, cfg: &LabCfg) -> Option<f32> {
        if !e.term.is_nan() {
            return Some(e.term);
        }
        if e.n == 0 {
            return None;
        }
        if cfg.backup == Backup::Mean || e.child == NONE {
            return Some(e.w / e.n as f32);
        }
        let c = &self.nodes[e.child as usize];
        Some(if c.player == player { c.mu } else { -c.mu })
    }

    /// The Bayes belief N(mean, var) about edge `e`, if visited.
    fn edge_belief(&self, player: u8, e: &Edge) -> Option<(f32, f32)> {
        if !e.term.is_nan() {
            return Some((e.term, 0.0));
        }
        if e.n == 0 || e.child == NONE {
            return None;
        }
        let c = &self.nodes[e.child as usize];
        Some((if c.player == player { c.bmu } else { -c.bmu }, c.bs2))
    }

    /// Recompute node `n`'s backed-up value (its children are current).
    fn refresh(&mut self, n: u32, cfg: &LabCfg) {
        let nd = &self.nodes[n as usize];
        let (lo, hi, player, v) = (nd.first as usize, (nd.first + nd.len) as usize, nd.player, nd.v);
        if cfg.bayes() {
            let mut sc = std::mem::take(&mut self.scratch);
            sc.clear();
            sc.extend(self.edges[lo..hi].iter().filter_map(|e| self.edge_belief(player, e)));
            let nd = &self.nodes[n as usize];
            let (mut m, mut s2) = (nd.v, cfg.leaf_s * cfg.leaf_s * nd.var);
            if !sc.is_empty() {
                sc.sort_by(|a, b| b.0.total_cmp(&a.0));
                let mut acc = sc[0];
                for &x in &sc[1..] {
                    acc = clark_max(acc, x);
                }
                (m, s2) = (acc.0.clamp(-1.0, 1.0), acc.1);
            }
            self.scratch = sc;
            let nd = &mut self.nodes[n as usize];
            nd.bmu = m;
            nd.bs2 = s2;
        }
        let edges = &self.edges[lo..hi];
        let mu = match cfg.backup {
            Backup::Mean => {
                let (mut w, mut k) = (v, 1u32);
                for e in edges {
                    w += e.w;
                    k += e.n;
                }
                w / k as f32
            }
            Backup::Power => {
                let (mut w, mut k) = (v, 1.0f32);
                for e in edges {
                    if let Some(q) = self.edge_q(player, e, cfg) {
                        let x = (e.n as f32).powf(cfg.bparam);
                        w += x * q;
                        k += x;
                    }
                }
                w / k
            }
            Backup::MixMax => {
                let (mut w, mut k, mut mx) = (v, 1u32, f32::NEG_INFINITY);
                for e in edges {
                    if let Some(q) = self.edge_q(player, e, cfg) {
                        w += e.n as f32 * q;
                        k += e.n;
                        mx = mx.max(q);
                    }
                }
                let mean = w / k as f32;
                if mx.is_finite() {
                    (1.0 - cfg.bparam) * mean + cfg.bparam * mx
                } else {
                    mean
                }
            }
            Backup::SoftMax => {
                let t = cfg.bparam.max(1e-3);
                let mut terms: Vec<(f32, f32)> = vec![(1.0, v)];
                let mut k = 1u32;
                for e in edges {
                    if let Some(q) = self.edge_q(player, e, cfg) {
                        terms.push((e.n as f32, q));
                        k += e.n;
                    }
                }
                let m = terms.iter().map(|x| x.1).fold(f32::NEG_INFINITY, f32::max);
                let s: f32 = terms.iter().map(|&(c, q)| c * ((q - m) / t).exp()).sum();
                m + t * (s / k as f32).ln()
            }
            Backup::Bayes => self.nodes[n as usize].bmu,
        };
        self.nodes[n as usize].mu = mu.clamp(-1.0, 1.0);
    }

    /// Fold value `x` (for `persp`) with leaf variance `var` into the path.
    fn backup(&mut self, path: &[(u32, u32)], x: f32, var: f32, persp: u8, cfg: &LabCfg) {
        for &(n, e) in path.iter().rev() {
            let y = if self.nodes[n as usize].player == persp { x } else { -x };
            let edge = &mut self.edges[e as usize];
            edge.vloss -= 1;
            edge.n += 1;
            edge.w += y;
            edge.w2 += y * y;
            edge.vs += var;
            self.refresh(n, cfg);
        }
    }

    pub fn backup_leaf(&mut self, path: &[(u32, u32)], cfg: &LabCfg) {
        let &(_, e) = path.last().unwrap();
        let c = &self.nodes[self.edges[e as usize].child as usize];
        let (v, var, p) = (c.v, c.var, c.player);
        self.backup(path, v, var, p, cfg);
    }

    /// Descend from the root, applying virtual loss, until an unexpanded or
    /// terminal edge.  `puct`: PUCT everywhere (the `ts_from` phase).
    pub fn select(&mut self, cfg: &LabCfg, puct: bool, path: &mut Vec<(u32, u32)>, rng: &mut SmallRng) -> Selected<G> {
        path.clear();
        let mut n = 0u32;
        let mut depth = 0;
        loop {
            let ts = !puct
                && match cfg.depths {
                    Depths::All => true,
                    Depths::Root => depth == 0,
                    Depths::Interior => depth > 0,
                };
            if cfg.eager_min > 0 {
                let nd = &self.nodes[n as usize];
                if !nd.peeked && cfg.eager >= nd.depth as i32 {
                    let k: u32 = self.edges_of(n).iter().map(|e| e.n).sum();
                    if k + 1 >= cfg.eager_min {
                        self.nodes[n as usize].peeked = true;
                        self.peek_queue.push(n);
                    }
                }
            }
            let e = match cfg.sel {
                Sel::Gumbel if !puct => self.gumbel_edge(n, cfg, false),
                Sel::Ts if ts => self.score_edge(n, cfg, true, rng),
                _ => self.score_edge(n, cfg, false, rng),
            };
            self.edges[e].vloss += 1;
            path.push((n, e as u32));
            let edge = &self.edges[e];
            if !edge.term.is_nan() {
                let (t, persp) = (edge.term, self.nodes[n as usize].player);
                self.backup(path, t, 0.0, persp, cfg);
                return Selected::Done;
            }
            if edge.child != NONE && edge.n == 0 {
                // eagerly evaluated: its value is the leaf's
                let c = edge.child;
                self.on_visit(c, cfg);
                self.backup_leaf(path, cfg);
                return Selected::Done;
            }
            if edge.child != NONE {
                n = edge.child;
                depth += 1;
                continue;
            }
            let node = &self.nodes[n as usize];
            let mut st = node.state.clone();
            st.apply(edge.action);
            if let Some(r) = st.result_for(node.player as usize) {
                let persp = node.player;
                self.edges[e].term = r as f32;
                self.backup(path, r as f32, 0.0, persp, cfg);
                return Selected::Done;
            }
            return Selected::Leaf(st);
        }
    }

    /// The uncertainty of edge `e` (k = visits + in flight).
    fn sigma(&self, nd: &Node<G>, e: &Edge, k: u32, total: u32, cfg: &LabCfg) -> f32 {
        if !e.term.is_nan() {
            return 0.0;
        }
        let kf = 1.0 + k as f32;
        let nvar = if e.n == 0 && e.child != NONE {
            self.nodes[e.child as usize].var
        } else {
            nd.var
        };
        cfg.ts_c
            * match cfg.sigma {
                Sigma::U => e.prior.powf(cfg.pexp) * ((1 + total) as f32).powf(cfg.nexp) / kf,
                Sigma::Count => 1.0 / kf.sqrt(),
                Sigma::NetVar => {
                    let var = if e.n > 0 { e.vs / e.n as f32 } else { nvar };
                    (var / kf).sqrt()
                }
                Sigma::Emp => {
                    let n = e.n as f32;
                    let m2 = if e.n > 0 { (e.w2 - e.w * e.w / n).max(0.0) } else { 0.0 };
                    ((cfg.n0 * nvar + m2) / (cfg.n0 + n) / kf).sqrt()
                }
                Sigma::Fit => {
                    let n = e.n as f32;
                    let var = if e.n > 0 { e.vs / n } else { nvar };
                    let m2 = if e.n > 0 { (e.w2 - e.w * e.w / n).max(0.0) } else { 0.0 };
                    let s2 = (0.1 * nvar + m2) / (0.1 + n);
                    var.max(1e-4).powf(cfg.fit[0]) * s2.max(1e-6).powf(cfg.fit[1]) / kf.powf(cfg.fit[2])
                }
                Sigma::Bayes => {
                    if e.child != NONE {
                        self.nodes[e.child as usize].bs2.sqrt()
                    } else {
                        cfg.leaf_s * nd.var.sqrt()
                    }
                }
            }
    }

    /// Global index of node `n`'s best edge under q + kappa U + sigma g(z)
    /// (`ts`) or q + U.
    fn score_edge(&self, n: u32, cfg: &LabCfg, ts: bool, rng: &mut SmallRng) -> usize {
        let nd = &self.nodes[n as usize];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        let total: u32 = self.edges[lo..hi].iter().map(|e| e.n + e.vloss as u32).sum();
        let fpu = nd.mu - cfg.fpu;
        let scale = cfg.c_puct * ((1 + total) as f32).sqrt();
        let mut best = (f32::NEG_INFINITY, lo);
        for (i, e) in self.edges[lo..hi].iter().enumerate() {
            let k = e.n + e.vloss as u32;
            let q = if k == 0 {
                if e.term.is_nan() {
                    self.peek_q(nd.player, e).unwrap_or(fpu)
                } else {
                    e.term // found terminal by an eager evaluation
                }
            } else {
                let base = self.edge_q(nd.player, e, cfg).unwrap_or(fpu);
                (base * e.n as f32 - cfg.virtual_loss * e.vloss as f32) / k as f32
            };
            let u = scale * e.prior / (1 + k) as f32;
            let score = if ts {
                let s = self.sigma(nd, e, k, total, cfg);
                let g = match cfg.zmode {
                    ZMode::One => 1.0,
                    _ if s == 0.0 => 0.0,
                    mode => {
                        let z: f32 = StandardNormal.sample(rng);
                        match mode {
                            ZMode::Abs => z.abs(),
                            ZMode::Pos => z.max(0.0),
                            _ => z,
                        }
                    }
                };
                q + cfg.kappa * u + s * g
            } else {
                q + u
            };
            if score > best.0 {
                best = (score, lo + i);
            }
        }
        best.1
    }

    /// Gumbel MuZero's improved policy at node `n`: (global edge index,
    /// pi'(a)) for every edge.
    fn improved(&self, n: u32, cfg: &LabCfg) -> Vec<f32> {
        let nd = &self.nodes[n as usize];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        let edges = &self.edges[lo..hi];
        let (mut sp, mut spq, mut big_n, mut maxn) = (0.0f32, 0.0f32, 0u32, 0u32);
        let qs: Vec<Option<f32>> = edges.iter().map(|e| self.edge_q(nd.player, e, cfg)).collect();
        for (e, q) in edges.iter().zip(&qs) {
            if let Some(q) = q {
                sp += e.prior;
                spq += e.prior * q;
            }
            big_n += e.n;
            maxn = maxn.max(e.n);
        }
        let vmix = if sp > 0.0 {
            (nd.v + big_n as f32 * spq / sp) / (1 + big_n) as f32
        } else {
            nd.v
        };
        let scale = (cfg.c_visit + maxn as f32) * cfg.c_scale;
        let logits: Vec<f32> = edges
            .iter()
            .zip(&qs)
            .map(|(e, q)| {
                let q = q.or_else(|| self.peek_q(nd.player, e)).unwrap_or(vmix);
                e.prior.max(1e-12).ln() + scale * (q + 1.0) / 2.0
            })
            .collect();
        let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let ex: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
        let s: f32 = ex.iter().sum();
        ex.into_iter().map(|x| x / s).collect()
    }

    fn gumbel_edge(&self, n: u32, cfg: &LabCfg, _root: bool) -> usize {
        let nd = &self.nodes[n as usize];
        let lo = nd.first as usize;
        let pi = self.improved(n, cfg);
        let edges = self.edges_of(n);
        let total: u32 = edges.iter().map(|e| e.n + e.vloss as u32).sum();
        let mut best = (f32::NEG_INFINITY, lo);
        for (i, (e, p)) in edges.iter().zip(&pi).enumerate() {
            let s = p - (e.n + e.vloss as u32) as f32 / (1 + total) as f32;
            if s > best.0 {
                best = (s, lo + i);
            }
        }
        best.1
    }

    /// Index (into `edges_of(0)`) of the move to play.
    pub fn pick(&self, cfg: &LabCfg) -> usize {
        let edges = self.edges_of(0);
        let maxn = edges.iter().map(|e| e.n).max().unwrap_or(0);
        if maxn == 0 {
            return argmax(edges.iter().map(|e| e.prior));
        }
        let player = self.nodes[0].player;
        let q = |e: &Edge| self.edge_q(player, e, cfg).unwrap_or(-2.0);
        match cfg.pick {
            Pick::Visits => {
                let mut best = 0;
                for (i, e) in edges.iter().enumerate() {
                    let b = &edges[best];
                    if (e.n, q(e), e.prior) > (b.n, q(b), b.prior) {
                        best = i;
                    }
                }
                best
            }
            Pick::Q => {
                let floor = (cfg.pick_frac * maxn as f32).max(1.0);
                argmax(edges.iter().map(|e| if e.n as f32 >= floor { q(e) } else { -3.0 }))
            }
            Pick::Improved => argmax(self.improved(0, cfg).into_iter()),
        }
    }

    /// One row per root edge: action, prior, n, q (backup rule), w/n,
    /// w2/n, mean leaf variance, Bayes mean and variance (NaN unvisited),
    /// exact result (NaN unless terminal), child network value.
    pub fn report(&self, cfg: &LabCfg) -> (Vec<f32>, [f32; 4]) {
        let nd = &self.nodes[0];
        let mut rows = Vec::new();
        for e in self.edges_of(0) {
            let n = e.n.max(1) as f32;
            let (bm, bs) = self.edge_belief(nd.player, e).unwrap_or((f32::NAN, f32::NAN));
            rows.extend_from_slice(&[
                e.action as f32,
                e.prior,
                e.n as f32,
                self.edge_q(nd.player, e, cfg).unwrap_or(f32::NAN),
                if e.n > 0 { e.w / n } else { f32::NAN },
                if e.n > 0 { e.w2 / n } else { f32::NAN },
                if e.n > 0 { e.vs / n } else { f32::NAN },
                bm,
                bs,
                e.term,
                self.peek_q(nd.player, e).unwrap_or(f32::NAN),
            ]);
        }
        (rows, [nd.v, nd.var, nd.mu, self.edges_of(0).iter().map(|e| e.n).sum::<u32>() as f32])
    }
}

pub const REPORT_COLS: usize = 11;

fn argmax(xs: impl Iterator<Item = f32>) -> usize {
    let mut best = (f32::NEG_INFINITY, 0);
    for (i, x) in xs.enumerate() {
        if x > best.0 {
            best = (x, i);
        }
    }
    best.1
}

// ════════════════════════════════════════════════════════════════════════════
// Search service
// ════════════════════════════════════════════════════════════════════════════

enum Request<G> {
    Root(G),
    /// `visited`: reached by a simulation, not an eager evaluation.
    Child { edge: u32, state: G, depth: u16, visited: bool },
}

pub struct Report {
    pub slot: usize,
    pub rows: Vec<f32>,
    pub root: [f32; 4],
}

struct Slot<G> {
    active: bool,
    state: G,
    tree: Tree<G>,
    done: usize,
    moved: Option<u16>,
    report: Option<(Vec<f32>, [f32; 4])>,
    rng: SmallRng,
    requests: Vec<Request<G>>,
    pending: Vec<(usize, usize)>,
    paths: Vec<(u32, u32)>,
    path: Vec<(u32, u32)>,
    evals: u64,
}

/// A batch of independent searches: `search(slot, state)` starts one,
/// `collect`/`apply` advance every active slot by one wave of simulations,
/// `moves` returns the finished searches' moves.
pub struct LabEngine<G: Game> {
    pub cfg: LabCfg,
    pub sims: usize,
    pub wave: usize,
    slots: Vec<Slot<G>>,
    counts: Vec<usize>,
}

impl<G: Game> LabEngine<G> {
    pub fn new(cfg: LabCfg, slots: usize, sims: usize, wave: usize, seed: u64) -> Self {
        LabEngine {
            cfg,
            sims,
            wave,
            counts: vec![0; slots],
            slots: (0..slots)
                .map(|i| Slot {
                    active: false,
                    state: G::new(),
                    tree: Tree::default(),
                    done: 0,
                    moved: None,
                    report: None,
                    rng: SmallRng::seed_from_u64(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ i as u64),
                    requests: Vec::new(),
                    pending: Vec::new(),
                    paths: Vec::new(),
                    path: Vec::new(),
                    evals: 0,
                })
                .collect(),
        }
    }

    pub fn search(&mut self, slot: usize, state: G) -> Result<(), String> {
        let s = self.slots.get_mut(slot).ok_or("slot out of range")?;
        if s.active || s.moved.is_some() {
            return Err(format!("slot {slot} is busy"));
        }
        if state.is_terminal() {
            return Err("cannot search a finished game".into());
        }
        s.tree.clear();
        s.state = state;
        s.done = 0;
        s.active = true;
        Ok(())
    }

    pub fn collect(&mut self) -> Vec<f32> {
        let (cfg, sims, wave) = (&self.cfg, self.sims, self.wave);
        self.slots.par_iter_mut().for_each(|s| s.collect(cfg, sims, wave));
        for (s, c) in self.slots.iter().zip(&mut self.counts) {
            *c = s.requests.len();
        }
        let rows: usize = self.counts.iter().sum();
        let mut obs = vec![0f32; rows * G::OBS_SIZE];
        let mut chunks = Vec::with_capacity(self.slots.len());
        let mut rest = obs.as_mut_slice();
        for s in &self.slots {
            let (head, tail) = rest.split_at_mut(s.requests.len() * G::OBS_SIZE);
            chunks.push(head);
            rest = tail;
        }
        self.slots.par_iter().zip(chunks).for_each(|(s, out)| {
            for (q, row) in s.requests.iter().zip(out.chunks_exact_mut(G::OBS_SIZE)) {
                match q {
                    Request::Root(st) | Request::Child { state: st, .. } => st.encode(row),
                }
            }
        });
        obs
    }

    pub fn pending_rows(&self) -> usize {
        self.counts.iter().sum()
    }

    pub fn apply(&mut self, v: &[f32], d: Option<&[f32]>, logits: &[f32]) {
        let rows = self.pending_rows();
        assert_eq!(v.len(), rows);
        assert_eq!(logits.len(), rows * G::NUM_ACTIONS);
        let mut offsets = Vec::with_capacity(self.slots.len());
        let mut off = 0;
        for &c in &self.counts {
            offsets.push(off);
            off += c;
        }
        let (cfg, sims) = (&self.cfg, self.sims);
        self.slots.par_iter_mut().zip(offsets).for_each(|(s, off)| {
            let d = d.map(|d| &d[off..]);
            s.apply(cfg, sims, &v[off..], d, &logits[off * G::NUM_ACTIONS..])
        });
        self.counts.fill(0);
    }

    pub fn moves(&mut self) -> Vec<(usize, u16)> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.moved.take().map(|a| (i, a)))
            .collect()
    }

    pub fn reports(&mut self) -> Vec<Report> {
        self.slots
            .iter_mut()
            .enumerate()
            .filter_map(|(i, s)| s.report.take().map(|(rows, root)| Report { slot: i, rows, root }))
            .collect()
    }

    pub fn evals(&self) -> u64 {
        self.slots.iter().map(|s| s.evals).sum()
    }
}

impl<G: Game> Slot<G> {
    fn collect(&mut self, cfg: &LabCfg, sims: usize, wave: usize) {
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
        if self.tree.edges_of(0).len() == 1 {
            self.done = sims; // forced move
            self.finish(cfg);
            return;
        }
        let puct = (self.done as f32) < cfg.ts_from * sims as f32;
        for _ in 0..wave.min(sims.saturating_sub(self.done)) {
            match self.tree.select(cfg, puct, &mut self.path, &mut self.rng) {
                Selected::Done => self.done += 1,
                Selected::Leaf(state) => {
                    let edge = self.path.last().unwrap().1;
                    let depth = self.path.len() as u16;
                    self.request(edge, state, depth, true);
                    let start = self.paths.len();
                    self.paths.extend_from_slice(&self.path);
                    self.pending.push((start, self.paths.len()));
                }
            }
        }
        self.peek();
        if self.requests.is_empty() && self.done >= sims {
            self.finish(cfg);
        }
    }

    /// Queue an evaluation of edge `edge`'s child (once per wave).
    fn request(&mut self, edge: u32, state: G, depth: u16, visited: bool) {
        for q in &mut self.requests {
            if let Request::Child { edge: e, visited: v, .. } = q {
                if *e == edge {
                    *v |= visited;
                    return;
                }
            }
        }
        self.requests.push(Request::Child { edge, state, depth, visited });
    }

    /// Request eager evaluations of the queued nodes' children.
    fn peek(&mut self) {
        let queue = std::mem::take(&mut self.tree.peek_queue);
        for n in queue {
            let nd = &self.tree.nodes[n as usize];
            let (first, len, player, depth) = (nd.first, nd.len, nd.player as usize, nd.depth + 1);
            for e in first..first + len {
                let edge = &self.tree.edges[e as usize];
                if edge.child != NONE || !edge.term.is_nan() {
                    continue;
                }
                let mut st = self.tree.nodes[n as usize].state.clone();
                st.apply(edge.action);
                if let Some(r) = st.result_for(player) {
                    self.tree.set_term(e, r as f32);
                    continue;
                }
                self.request(e, st, depth, false);
            }
        }
    }

    fn apply(&mut self, cfg: &LabCfg, sims: usize, v: &[f32], d: Option<&[f32]>, logits: &[f32]) {
        if !self.active {
            return;
        }
        let na = G::NUM_ACTIONS;
        for (i, q) in self.requests.drain(..).enumerate() {
            let l = &logits[i * na..(i + 1) * na];
            let di = d.map_or(0.0, |d| d[i]);
            match q {
                Request::Root(st) => {
                    let n = self.tree.add_node(st, 0, v[i], di, l, cfg);
                    self.tree.on_visit(n, cfg);
                }
                Request::Child { edge, state, depth, visited } => {
                    self.tree.expand(edge, state, depth, v[i], di, l, cfg, visited)
                }
            }
            self.evals += 1;
        }
        for &(lo, hi) in &self.pending {
            self.tree.backup_leaf(&self.paths[lo..hi], cfg);
        }
        self.done += self.pending.len();
        self.pending.clear();
        self.paths.clear();
        if self.done >= sims && !self.tree.is_empty() {
            self.finish(cfg);
        }
    }

    fn finish(&mut self, cfg: &LabCfg) {
        let idx = self.tree.pick(cfg);
        self.moved = Some(self.tree.edges_of(0)[idx].action);
        if cfg.report {
            self.report = Some(self.tree.report(cfg));
        }
        self.tree.clear();
        self.active = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{Connect4, Othello};
    use crate::puct;

    /// Sequential search with a deterministic fake network.
    fn search_with<G: Game>(state: G, sims: usize, cfg: &LabCfg) -> Tree<G> {
        let mut t = Tree::default();
        let net = |s: &G| -> (f32, Vec<f32>) {
            let mut obs = vec![0f32; G::OBS_SIZE];
            s.encode(&mut obs);
            let h: f32 = obs.iter().enumerate().map(|(i, x)| x * ((i * 7919 % 13) as f32 - 6.0)).sum();
            let logits = (0..G::NUM_ACTIONS).map(|a| ((a * 31 + h as usize) % 7) as f32 * 0.3).collect();
            ((h * 0.37).sin() * 0.8, logits)
        };
        let (v, l) = net(&state);
        let root = t.add_node(state, 0, v, 0.0, &l, cfg);
        t.on_visit(root, cfg);
        let mut path = Vec::new();
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..sims {
            // eager evaluations first, as the engine's next wave would
            for n in std::mem::take(&mut t.peek_queue) {
                let nd = &t.nodes[n as usize];
                let (first, len, player, depth) = (nd.first, nd.len, nd.player as usize, nd.depth + 1);
                for e in first..first + len {
                    if t.edges[e as usize].child != NONE || !t.edges[e as usize].term.is_nan() {
                        continue;
                    }
                    let mut st = t.nodes[n as usize].state.clone();
                    st.apply(t.edges[e as usize].action);
                    match st.result_for(player) {
                        Some(r) => t.set_term(e, r as f32),
                        None => {
                            let (v, l) = net(&st);
                            t.expand(e, st, depth, v, 0.0, &l, cfg, false);
                        }
                    }
                }
            }
            if let Selected::Leaf(st) = t.select(cfg, false, &mut path, &mut rng) {
                let (v, l) = net(&st);
                let e = path.last().unwrap().1;
                t.expand(e, st, path.len() as u16, v, 0.0, &l, cfg, true);
                t.backup_leaf(&path, cfg);
            }
        }
        t
    }

    #[test]
    fn matches_the_production_puct_tree() {
        // the defaults, and production's mixmax backup + max-q pick
        let mm = LabCfg {
            backup: Backup::MixMax,
            bparam: 0.25,
            pick: Pick::Q,
            pick_frac: 0.2,
            ..LabCfg::default()
        };
        for (cfg, mixmax, pick_frac) in [(LabCfg::default(), 0.0, 0.0), (mm, 0.25, 0.2)] {
            check_against_production(&cfg, mixmax, pick_frac);
        }
    }

    fn check_against_production(cfg: &LabCfg, mixmax: f32, pick_frac: f32) {
        let lab = search_with(Othello::new(), 300, cfg);
        // the same search in crate::puct
        let pc = puct::PuctCfg {
            c_puct: 1.5,
            fpu_reduction: 0.2,
            virtual_loss: 1.0,
            select: puct::Select::Puct,
            ts_c: 1.5,
            mixmax,
            pick_frac,
        };
        let mut t: puct::Tree<Othello> = puct::Tree::default();
        let net = |s: &Othello| -> (f32, Vec<f32>) {
            let mut obs = vec![0f32; Othello::OBS_SIZE];
            s.encode(&mut obs);
            let h: f32 = obs.iter().enumerate().map(|(i, x)| x * ((i * 7919 % 13) as f32 - 6.0)).sum();
            let logits = (0..Othello::NUM_ACTIONS).map(|a| ((a * 31 + h as usize) % 7) as f32 * 0.3).collect();
            ((h * 0.37).sin() * 0.8, logits)
        };
        let (v, l) = net(&Othello::new());
        t.add_node(Othello::new(), v, 0.0, &l);
        let mut path = Vec::new();
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..300 {
            if let puct::Selected::Leaf(st) = t.select(&pc, &mut path, &mut rng) {
                let (v, l) = net(&st);
                let e = path.last().unwrap().1;
                t.expand(e, st, v, 0.0, &l);
                t.backup_leaf(&path, &pc);
            }
        }
        let a: Vec<u32> = lab.edges_of(0).iter().map(|e| e.n).collect();
        let b: Vec<u32> = t.edges_of(0).iter().map(|e| e.n).collect();
        assert_eq!(a, b);
        assert_eq!(lab.pick(cfg), t.pick(&mut rng, false, &pc));
        assert!((lab.nodes[0].mu - t.backed_up_value(0, &pc)).abs() < 1e-4);
    }

    #[test]
    fn power_one_backup_equals_the_mean() {
        let mean = search_with(Othello::new(), 300, &LabCfg::default());
        let cfg = LabCfg {
            backup: Backup::Power,
            bparam: 1.0,
            ..LabCfg::default()
        };
        let pow = search_with(Othello::new(), 300, &cfg);
        for (a, b) in mean.edges_of(0).iter().zip(pow.edges_of(0)) {
            assert_eq!(a.n, b.n);
        }
        assert!((mean.nodes[0].mu - pow.nodes[0].mu).abs() < 1e-4);
    }

    #[test]
    fn clark_max_matches_known_moments() {
        // max of two iid N(0, 1): mean 1/sqrt(pi), variance 1 - 1/pi
        let (m, v) = clark_max((0.0, 1.0), (0.0, 1.0));
        assert!((m - 0.5642).abs() < 1e-3 && (v - 0.6817).abs() < 1e-3);
        assert_eq!(clark_max((0.3, 0.0), (0.1, 0.0)), (0.3, 0.0));
    }

    #[test]
    fn every_rule_finds_the_immediate_win() {
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1] {
            s.apply(a);
        }
        let base = LabCfg::default();
        let ts = LabCfg { sel: Sel::Ts, ..base };
        let cfgs = [
            base,
            ts,
            LabCfg { sigma: Sigma::Count, kappa: 1.0, ..ts },
            LabCfg { sigma: Sigma::NetVar, zmode: ZMode::Abs, ..ts },
            LabCfg { sigma: Sigma::Emp, zmode: ZMode::One, ..ts },
            LabCfg { sigma: Sigma::Fit, kappa: 1.0, ..ts },
            LabCfg { sigma: Sigma::Bayes, backup: Backup::Bayes, ..ts },
            LabCfg { backup: Backup::MixMax, bparam: 0.5, ..base },
            LabCfg { backup: Backup::SoftMax, bparam: 0.2, ..base },
            LabCfg { sel: Sel::Gumbel, pick: Pick::Improved, ..base },
            LabCfg { pick: Pick::Q, ..base },
            LabCfg { eager: 0, ..base },
            LabCfg { eager: 99, ..ts },
            LabCfg { eager: 99, eager_min: 4, ..base },
        ];
        for cfg in cfgs {
            let t = search_with(s.clone(), 200, &cfg);
            assert!(t.nodes.len() > 1);
            let e = &t.edges_of(0)[t.pick(&cfg)];
            assert_eq!(e.action, 0, "{cfg:?}");
            assert!(t.edges.iter().all(|e| e.vloss == 0));
        }
    }
}
