//! The AlphaZero search tree: PUCT over network priors, mean-value backup.
//!
//! Expanding a node stores the network's value v(s) and a prior P(s,a) for
//! every legal move (a softmax of the policy logits over the legal moves
//! only).  Selection descends argmax Q + U with
//!
//! ```text
//!   U(s,a) = c_puct * P(s,a) * sqrt(1 + N(s)) / (1 + N(s,a))
//! ```
//!
//! where N(s) is the visit total of s's edges.  An unvisited edge's Q is the
//! node's running mean minus `fpu_reduction` (first-play urgency).  Each
//! in-flight simulation counts as a lost visit on its edges (virtual loss), so
//! a wave of simulations spreads out.  Values are scalars in [-1, 1] from the
//! perspective of the node's mover; a terminal child stores its exact result
//! on the edge and is never evaluated by the network.
//!
//! With Thompson selection (`Select::Dirichlet` or `Select::Gaussian`), each
//! edge instead draws one value and descent takes the argmax.  The draw has
//! the edge's backed-up value as its mean (the FPU value while unvisited) and
//! PUCT's exploration term U(s,a) as its standard deviation: the bonus
//! becomes noise of the same size.  `Dirichlet` draws p_win - p_loss from a
//! Dirichlet over (win, draw, loss) whose draw mean is the backed-up draw
//! probability (the node's mean while unvisited; it comes from a win/draw/loss
//! value head, 0 without one).  `Gaussian` draws from a normal distribution.
//!
//! Storage is two flat arenas (nodes, edges) indexed by u32.  The root is
//! always node 0; moving to a child rebuilds the arenas from that subtree.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::belief::gamma;
use crate::games::Game;

pub const NONE: u32 = u32::MAX;

#[derive(Clone, Copy, Debug)]
pub struct PuctCfg {
    pub c_puct: f32,
    /// Unvisited edges are valued at the node's mean value minus this.
    pub fpu_reduction: f32,
    /// Value an in-flight simulation is assumed to return (as a loss).
    pub virtual_loss: f32,
    /// Edge selection rule (see the module docs).
    pub select: Select,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Select {
    /// argmax Q + U
    Puct,
    /// argmax of a win/draw/loss Dirichlet draw per edge
    Dirichlet,
    /// argmax of a Gaussian draw per edge
    Gaussian,
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub action: u16,
    pub prior: f32,
    pub n: u32,
    /// Sum of backed-up values, from the node's mover's perspective.
    pub w: f32,
    /// Sum of backed-up draw probabilities.
    pub d: f32,
    vloss: u16,
    /// Exact result of a terminal child for the node's mover, else NaN.
    term: f32,
    pub child: u32,
}

#[derive(Clone, Debug)]
pub struct Node<G> {
    pub state: G,
    pub player: u8,
    first: u32,
    len: u32,
    /// Network value for the mover.
    pub v: f32,
    /// Network draw probability (0 without a win/draw/loss head).
    pub d: f32,
}

pub enum Selected<G> {
    /// Hit a terminal edge; already backed up.
    Terminal,
    /// Reached an unexpanded edge (the last on the path); `state` is its child.
    Leaf(G),
}

pub struct Tree<G> {
    pub nodes: Vec<Node<G>>,
    edges: Vec<Edge>,
    legal: Vec<u16>,
}

impl<G: Game> Default for Tree<G> {
    fn default() -> Self {
        Tree {
            nodes: Vec::new(),
            edges: Vec::new(),
            legal: Vec::new(),
        }
    }
}

impl<G: Game> Tree<G> {
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
    }

    pub fn edges_of(&self, n: u32) -> &[Edge] {
        let nd = &self.nodes[n as usize];
        &self.edges[nd.first as usize..(nd.first + nd.len) as usize]
    }

    /// Visit total of node `n`'s edges.
    pub fn visits(&self, n: u32) -> u32 {
        self.edges_of(n).iter().map(|e| e.n).sum()
    }

    /// Node `n`'s value for its mover: the network value and every
    /// simulation through it, equally weighted.
    pub fn mean_value(&self, n: u32) -> f32 {
        let (mut w, mut k) = (self.nodes[n as usize].v, 1u32);
        for e in self.edges_of(n) {
            w += e.w;
            k += e.n;
        }
        w / k as f32
    }

    /// Create a node for `state` from its evaluation: value `v` for the
    /// mover, draw probability `d` and policy `logits` indexed by action id.
    pub fn add_node(&mut self, state: G, v: f32, d: f32, logits: &[f32]) -> u32 {
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
                d: 0.0,
                vloss: 0,
                term: f32::NAN,
                child: NONE,
            });
        }
        for e in &mut self.edges[first as usize..] {
            e.prior /= total;
        }
        let player = state.current_player() as u8;
        self.nodes.push(Node {
            state,
            player,
            first,
            len: self.legal.len() as u32,
            v: v.clamp(-1.0, 1.0),
            d: d.clamp(0.0, 1.0),
        });
        (self.nodes.len() - 1) as u32
    }

    /// Expand the unexpanded edge `e` with its child's evaluation.
    pub fn expand(&mut self, e: u32, state: G, v: f32, d: f32, logits: &[f32]) -> u32 {
        let n = self.add_node(state, v, d, logits);
        self.edges[e as usize].child = n;
        n
    }

    /// Mix Dirichlet(alpha) noise into the root priors with weight `frac`,
    /// alpha = min(1, alpha_scale / legal moves) as in OLIVAW.
    pub fn add_root_noise<R: Rng>(&mut self, rng: &mut R, alpha_scale: f32, frac: f32) {
        let nd = &self.nodes[0];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        if frac <= 0.0 || hi - lo < 2 {
            return;
        }
        let alpha = (alpha_scale as f64 / (hi - lo) as f64).min(1.0);
        let g: Vec<f64> = (lo..hi).map(|_| gamma(rng, alpha)).collect();
        let s: f64 = g.iter().sum();
        if s <= 0.0 {
            return;
        }
        for (e, x) in self.edges[lo..hi].iter_mut().zip(g) {
            e.prior = (1.0 - frac) * e.prior + frac * (x / s) as f32;
        }
    }

    /// PUCT-descend (or Thompson-descend) from the root, applying virtual
    /// loss, until an unexpanded or terminal edge.  `path` receives (node,
    /// edge) pairs.
    pub fn select<R: Rng>(&mut self, cfg: &PuctCfg, path: &mut Vec<(u32, u32)>, rng: &mut R) -> Selected<G> {
        path.clear();
        let mut n = 0u32;
        loop {
            let e = match cfg.select {
                Select::Puct => self.select_edge(n, cfg),
                _ => self.thompson_edge(n, cfg, rng),
            };
            self.edges[e].vloss += 1;
            path.push((n, e as u32));
            let edge = &self.edges[e];
            if !edge.term.is_nan() {
                let (v, persp) = (edge.term, self.nodes[n as usize].player);
                self.backup(path, v, (v == 0.0) as u8 as f32, persp, cfg);
                return Selected::Terminal;
            }
            if edge.child != NONE {
                n = edge.child;
                continue;
            }
            let node = &self.nodes[n as usize];
            let mut st = node.state.clone();
            st.apply(edge.action);
            if let Some(r) = st.result_for(node.player as usize) {
                let persp = node.player;
                self.edges[e].term = r as f32;
                self.backup(path, r as f32, (r == 0) as u8 as f32, persp, cfg);
                return Selected::Terminal;
            }
            return Selected::Leaf(st);
        }
    }

    /// Global index of node `n`'s argmax Q + U edge.
    fn select_edge(&self, n: u32, cfg: &PuctCfg) -> usize {
        let nd = &self.nodes[n as usize];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        let (mut total, mut w, mut visited) = (0u32, nd.v, 1u32);
        for e in &self.edges[lo..hi] {
            total += e.n + e.vloss as u32;
            w += e.w;
            visited += e.n;
        }
        let fpu = w / visited as f32 - cfg.fpu_reduction;
        let scale = cfg.c_puct * ((1 + total) as f32).sqrt();
        let mut best = (f32::NEG_INFINITY, lo);
        for (i, e) in self.edges[lo..hi].iter().enumerate() {
            let k = e.n + e.vloss as u32;
            let q = if k == 0 {
                fpu
            } else {
                (e.w - cfg.virtual_loss * e.vloss as f32) / k as f32
            };
            let score = q + scale * e.prior / (1 + k) as f32;
            if score > best.0 {
                best = (score, lo + i);
            }
        }
        best.1
    }

    /// Global index of node `n`'s edge with the highest Thompson draw.
    fn thompson_edge<R: Rng>(&self, n: u32, cfg: &PuctCfg, rng: &mut R) -> usize {
        let nd = &self.nodes[n as usize];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        let (mut total, mut w, mut d, mut visited) = (0u32, nd.v, nd.d, 1u32);
        for e in &self.edges[lo..hi] {
            total += e.n + e.vloss as u32;
            w += e.w;
            d += e.d;
            visited += e.n;
        }
        let fpu = w / visited as f32 - cfg.fpu_reduction;
        let d_node = d / visited as f32;
        let scale = cfg.c_puct * ((1 + total) as f32).sqrt();
        let mut best = (f32::NEG_INFINITY, lo);
        for (i, e) in self.edges[lo..hi].iter().enumerate() {
            let k = e.n + e.vloss as u32;
            let x = if !e.term.is_nan() && e.vloss == 0 {
                e.term // exact
            } else {
                let (q, dm) = if k == 0 {
                    (fpu, d_node)
                } else {
                    // in-flight simulations count as losses (no draw mass)
                    ((e.w - cfg.virtual_loss * e.vloss as f32) / k as f32, e.d / k as f32)
                };
                let sigma = scale * e.prior / (1 + k) as f32;
                if cfg.select == Select::Gaussian {
                    let z: f32 = StandardNormal.sample(rng);
                    q + sigma * z
                } else {
                    thompson_draw(rng, q, dm, sigma)
                }
            };
            if x > best.0 {
                best = (x, lo + i);
            }
        }
        best.1
    }

    /// Fold value `v` and draw probability `d` (from `persp`'s point of view)
    /// into every edge on the path, `v` negated for nodes whose mover is the
    /// other player.
    pub fn backup(&mut self, path: &[(u32, u32)], v: f32, d: f32, persp: u8, _cfg: &PuctCfg) {
        for &(n, e) in path.iter().rev() {
            let x = if self.nodes[n as usize].player == persp { v } else { -v };
            let edge = &mut self.edges[e as usize];
            edge.vloss -= 1;
            edge.n += 1;
            edge.w += x;
            edge.d += d;
        }
    }

    /// Back up the freshly expanded child at the end of `path`.
    pub fn backup_leaf(&mut self, path: &[(u32, u32)], cfg: &PuctCfg) {
        let &(_, e) = path.last().unwrap();
        let child = &self.nodes[self.edges[e as usize].child as usize];
        let (v, d, persp) = (child.v, child.d, child.player);
        self.backup(path, v, d, persp, cfg);
    }

    /// Index (into `edges_of(0)`) of the move to play: sampled in proportion
    /// to visits if `sample`, else the most visited (ties: higher Q, then
    /// prior).  With no visits at all, the prior decides.
    pub fn pick<R: Rng>(&self, rng: &mut R, sample: bool) -> usize {
        let edges = self.edges_of(0);
        let total: u32 = edges.iter().map(|e| e.n).sum();
        if total == 0 {
            if sample {
                let mut u = rng.random::<f32>();
                for (i, e) in edges.iter().enumerate() {
                    u -= e.prior;
                    if u <= 0.0 {
                        return i;
                    }
                }
                return edges.len() - 1;
            }
            return argmax(edges.iter().map(|e| e.prior));
        }
        if sample {
            let mut u = rng.random_range(0..total);
            for (i, e) in edges.iter().enumerate() {
                if u < e.n {
                    return i;
                }
                u -= e.n;
            }
        }
        let q = |e: &Edge| if e.n > 0 { e.w / e.n as f32 } else { -2.0 };
        let mut best = 0;
        for (i, e) in edges.iter().enumerate() {
            let b = &edges[best];
            if (e.n, q(e), e.prior) > (b.n, q(b), b.prior) {
                best = i;
            }
        }
        best
    }

    /// Visit distribution over node `n`'s edges as (action, probability), or
    /// empty if nothing below `n` was searched.
    pub fn policy(&self, n: u32) -> Vec<(u16, f32)> {
        let total = self.visits(n);
        if total == 0 {
            return Vec::new();
        }
        self.edges_of(n)
            .iter()
            .filter(|e| e.n > 0)
            .map(|e| (e.action, e.n as f32 / total as f32))
            .collect()
    }

    /// Make the child under root edge `idx` the new root, keeping its subtree.
    pub fn advance(&mut self, idx: usize) {
        let child = self.edges_of(0)[idx].child;
        if child == NONE {
            self.clear();
            return;
        }
        let mut nodes = Vec::new();
        let mut edges = Vec::new();
        // breadth-first copy with index remapping: queue holds old node ids
        let mut queue = vec![child];
        let mut head = 0;
        while head < queue.len() {
            let old = queue[head] as usize;
            head += 1;
            let mut nd = self.nodes[old].clone();
            let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
            nd.first = edges.len() as u32;
            for e in &self.edges[lo..hi] {
                let mut e = e.clone();
                debug_assert_eq!(e.vloss, 0);
                if e.child != NONE {
                    queue.push(e.child);
                    e.child = (queue.len() - 1) as u32;
                }
                edges.push(e);
            }
            nodes.push(nd);
        }
        self.nodes = nodes;
        self.edges = edges;
    }
}

/// Concentrations above this are treated as certain (no draw).
const TS_MAX_CONC: f32 = 1e5;
/// ... and are floored here (wider than this adds nothing but noise).
const TS_MIN_CONC: f32 = 0.5;
const TS_EPS: f32 = 1e-3;

/// One draw of p_win - p_loss from the Dirichlet over (win, draw, loss) with
/// mean (from value `q` and draw probability `dm`) and the concentration
/// that gives p_win - p_loss standard deviation `sigma`.  For Dir(k m),
/// Var(p_w - p_l) = (m_w + m_l - (m_w - m_l)^2) / (k + 1).
pub fn thompson_draw<R: Rng>(rng: &mut R, q: f32, dm: f32, sigma: f32) -> f32 {
    let dm = dm.clamp(0.0, 1.0);
    let q = q.clamp(-(1.0 - dm), 1.0 - dm);
    let m = [
        ((1.0 - dm + q) / 2.0).max(TS_EPS),
        dm.max(TS_EPS),
        ((1.0 - dm - q) / 2.0).max(TS_EPS),
    ];
    let s = m[0] + m[1] + m[2];
    let m = [m[0] / s, m[1] / s, m[2] / s];
    let var0 = m[0] + m[2] - (m[0] - m[2]) * (m[0] - m[2]);
    let k = if sigma > 0.0 {
        var0 / (sigma * sigma) - 1.0
    } else {
        f32::INFINITY
    };
    if k >= TS_MAX_CONC {
        return m[0] - m[2];
    }
    let k = k.max(TS_MIN_CONC) as f64;
    let g = [
        gamma(rng, k * m[0] as f64),
        gamma(rng, k * m[1] as f64),
        gamma(rng, k * m[2] as f64),
    ];
    let t = g[0] + g[1] + g[2];
    if t <= 0.0 {
        return m[0] - m[2];
    }
    ((g[0] - g[2]) / t) as f32
}

fn argmax(xs: impl Iterator<Item = f32>) -> usize {
    let mut best = (f32::NEG_INFINITY, 0);
    for (i, x) in xs.enumerate() {
        if x > best.0 {
            best = (x, i);
        }
    }
    best.1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::{Connect4, Othello};
    use rand::{rngs::SmallRng, SeedableRng};

    const CFG: PuctCfg = PuctCfg {
        c_puct: 1.5,
        fpu_reduction: 0.2,
        virtual_loss: 1.0,
        select: Select::Puct,
    };
    const TS: PuctCfg = PuctCfg { select: Select::Dirichlet, ..CFG };

    /// Plain sequential search with a uniform network (v = 0, flat logits).
    fn search<G: Game>(tree: &mut Tree<G>, state: G, sims: usize) {
        search_with(tree, state, sims, &CFG, 0.0)
    }

    fn search_with<G: Game>(tree: &mut Tree<G>, state: G, sims: usize, cfg: &PuctCfg, d: f32) {
        let logits = vec![0.0; G::NUM_ACTIONS];
        if tree.is_empty() {
            tree.add_node(state, 0.0, d, &logits);
        }
        let mut path = Vec::new();
        let mut rng = SmallRng::seed_from_u64(7);
        for _ in 0..sims {
            if let Selected::Leaf(st) = tree.select(cfg, &mut path, &mut rng) {
                let e = path.last().unwrap().1;
                tree.expand(e, st, 0.0, d, &logits);
                tree.backup_leaf(&path, cfg);
            }
        }
    }

    #[test]
    fn priors_are_a_softmax_over_legal_moves() {
        let mut logits = vec![0.0f32; Othello::NUM_ACTIONS];
        logits[19] = 2.0; // d3
        logits[0] = 50.0; // illegal: must not take any mass
        let mut t: Tree<Othello> = Tree::default();
        t.add_node(Othello::new(), 0.3, 0.0, &logits);
        let p: Vec<f32> = t.edges_of(0).iter().map(|e| e.prior).collect();
        assert_eq!(p.len(), 4);
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        let e2 = 2f32.exp();
        assert!((p[0] - e2 / (e2 + 3.0)).abs() < 1e-6);
    }

    #[test]
    fn finds_immediate_win_and_backs_up_exact_results() {
        // x to move with three in column 0: playing 0 wins
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1] {
            s.apply(a);
        }
        let mut t = Tree::default();
        search(&mut t, s, 400);
        let mut rng = SmallRng::seed_from_u64(0);
        let i = t.pick(&mut rng, false);
        let e = &t.edges_of(0)[i];
        assert_eq!(e.action, 0);
        assert_eq!(e.w, e.n as f32, "every visit of a won edge returns +1");
        assert!(t.mean_value(0) > 0.5);
    }

    #[test]
    fn visits_balance_and_reuse_keeps_the_subtree() {
        let mut t: Tree<Othello> = Tree::default();
        search(&mut t, Othello::new(), 300);
        assert_eq!(t.visits(0), 300);
        assert!(t.edges.iter().all(|e| e.vloss == 0));
        // every expanded node's visits are one (its expansion) plus its edges'
        for (i, e) in t.edges.iter().enumerate() {
            if e.child != NONE {
                assert_eq!(e.n, 1 + t.visits(e.child), "edge {i}");
            }
        }
        let mut rng = SmallRng::seed_from_u64(1);
        let i = t.pick(&mut rng, false);
        let below = t.visits(t.edges_of(0)[i].child);
        t.advance(i);
        assert_eq!(t.visits(0), below);
        let pi = t.policy(0);
        assert!((pi.iter().map(|x| x.1).sum::<f32>() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn root_noise_keeps_a_distribution() {
        let mut t: Tree<Othello> = Tree::default();
        t.add_node(Othello::new(), 0.0, 0.0, &vec![0.0; Othello::NUM_ACTIONS]);
        let mut rng = SmallRng::seed_from_u64(2);
        t.add_root_noise(&mut rng, 10.0, 0.25);
        let p: Vec<f32> = t.edges_of(0).iter().map(|e| e.prior).collect();
        assert!((p.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        assert!(p.iter().all(|&x| x >= 0.75 * 0.25 - 1e-6));
        assert!(p.iter().any(|&x| (x - 0.25).abs() > 1e-4));
    }

    #[test]
    fn thompson_draws_have_the_requested_spread() {
        let mut rng = SmallRng::seed_from_u64(3);
        for &(q, dm, sigma) in &[(0.2f32, 0.3f32, 0.1f32), (-0.5, 0.0, 0.2), (0.0, 0.6, 0.05)] {
            let xs: Vec<f32> = (0..20000).map(|_| thompson_draw(&mut rng, q, dm, sigma)).collect();
            let mean = xs.iter().sum::<f32>() / xs.len() as f32;
            let sd = (xs.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / xs.len() as f32).sqrt();
            assert!((mean - q).abs() < 0.01, "mean {mean} for q {q}");
            assert!((sd - sigma).abs() < 0.1 * sigma, "sd {sd} for sigma {sigma}");
        }
        assert_eq!(thompson_draw(&mut rng, 0.3, 0.2, 0.0), 0.3);
    }

    #[test]
    fn thompson_search_finds_the_win_and_tracks_draws() {
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1] {
            s.apply(a);
        }
        let mut t = Tree::default();
        search_with(&mut t, s, 400, &TS, 0.25);
        let mut rng = SmallRng::seed_from_u64(0);
        let e = &t.edges_of(0)[t.pick(&mut rng, false)];
        assert_eq!(e.action, 0);
        assert_eq!(e.w, e.n as f32);
        assert_eq!(e.d, 0.0, "a won edge backs up no draw mass");
        assert!(t.edges.iter().all(|e| e.vloss == 0));
        // non-terminal visits back up the network's draw probability
        let other = t.edges_of(0).iter().find(|e| e.action != 0 && e.n > 0).unwrap();
        assert!(other.d > 0.0 && other.d <= other.n as f32 * 0.25 + 1e-4);
    }

    #[test]
    fn gaussian_thompson_search_finds_the_win() {
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1] {
            s.apply(a);
        }
        let mut t = Tree::default();
        let cfg = PuctCfg { select: Select::Gaussian, ..CFG };
        search_with(&mut t, s, 400, &cfg, 0.0);
        let mut rng = SmallRng::seed_from_u64(0);
        let e = &t.edges_of(0)[t.pick(&mut rng, false)];
        assert_eq!(e.action, 0);
        assert_eq!(e.w, e.n as f32);
        assert!(e.n > 200, "the won edge takes most visits: {}", e.n);
        assert!(t.edges.iter().all(|e| e.vloss == 0));
    }
}
