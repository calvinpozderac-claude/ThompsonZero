//! The Thompson-sampling search tree.
//!
//! Expanding a node stores the network's STATE belief V(s) and one ACTION
//! belief Q(s,a) per legal move.  Selection draws one value per edge from its
//! current belief and descends the argmax; the freshly expanded leaf's own
//! V(leaf) (or a proven-terminal result) is backed up along the path, flipped
//! into each node's mover perspective.  Edges proven won/lost/drawn override
//! their belief with a spike, and fully-solved nodes prove their parent edge
//! (MCTS-Solver).
//!
//! Storage is three flat arenas (nodes, edges, accumulators) indexed by u32.
//! Edge accumulators are allocated lazily: most edges are never searched.
//! The root is always node 0; moving to a child rebuilds the arenas from that
//! subtree, so memory stays proportional to the live tree.

use rand::Rng;

use crate::belief::*;
use crate::games::Game;

pub const NONE: u32 = u32::MAX;

/// Search-wide settings the tree needs.
#[derive(Clone, Copy, Debug)]
pub struct SearchCfg {
    /// Evidence rule for selection beliefs.
    pub search: Rule,
    /// Evidence rule for training targets.
    pub target: Rule,
    /// Leaves back up a Dirichlet draw instead of their mean.
    pub sample_leaf: bool,
    /// Penalty on the sampled value of an edge per in-flight selection.
    pub virtual_loss: f64,
    /// Keep every optional statistic (reports rebuild any rule offline).
    pub track_all: bool,
    /// State targets from the most-visited edge's target (a one-ply max
    /// backup) instead of the node's own accumulator.
    pub best_value: bool,
}

impl SearchCfg {
    /// Statistics an EDGE accumulator keeps: edges feed selection and targets.
    fn edge_track(&self) -> Track {
        if self.track_all {
            return Track { mle: true, corr: true };
        }
        let (a, b) = (self.search.track(), self.target.track());
        Track {
            mle: a.mle || b.mle,
            corr: a.corr || b.corr,
        }
    }

    /// Statistics a NODE accumulator keeps: nodes only feed state targets.
    /// (Skipping the MLE here is what makes search=additive_mle with
    /// target=avgconc cheap: the MLE refresh is the costliest thing in a
    /// backup, and a node's would never be read.)
    fn node_track(&self) -> Track {
        if self.track_all {
            return Track { mle: true, corr: true };
        }
        self.target.track()
    }

    /// Proven terminals go up as exact one-hots under `corr` (zero variance),
    /// as finite spikes otherwise.
    fn terminal_obs(&self, outcome: usize) -> Obs {
        if matches!(self.search, Rule::Corr { .. }) || matches!(self.target, Rule::Corr { .. }) {
            Obs::exact(outcome)
        } else {
            Obs::from_alpha(spike(outcome))
        }
    }
}

#[derive(Clone, Debug)]
pub struct Edge {
    pub action: u16,
    /// Proven outcome for the node's mover, or -1.
    pub term: i8,
    vloss: u16,
    pub child: u32,
    acc: u32,
    /// Network belief Q(s,a).
    prior: [f32; 3],
    /// What selection samples (prior + evidence, or a proof spike).
    sel: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct Node<G> {
    pub state: G,
    pub player: u8,
    first: u32,
    len: u32,
    /// Network belief V(s).
    pub v_alpha: Dir,
    acc: Acc,
}

/// One network evaluation, already activated: `v` = (p_win, p_draw, p_loss,
/// concentration); `a` = the same four numbers for every action id.
#[derive(Clone, Copy)]
pub struct Eval<'a> {
    pub v: &'a [f32],
    pub a: &'a [f32],
}

pub enum Selected<G> {
    /// Hit a terminal or proven edge; already backed up.
    Terminal,
    /// Reached an unexpanded edge (the last on the path); `state` is its child.
    Leaf(G),
}

/// The searched root as the search left it, with each accumulator's raw
/// statistics (see [`Acc::raw`]; zeros for an edge never searched).
pub struct Report {
    pub v_alpha: Dir,
    pub node: [f64; RAW],
    pub edges: Vec<EdgeReport>,
}

pub struct EdgeReport {
    pub action: u16,
    /// Proven outcome for the root's mover, or -1.
    pub term: i8,
    pub prior: Dir,
    pub sel: Dir,
    pub acc: [f64; RAW],
}

/// A training target read off a searched node.
pub struct Target {
    pub v: Dir,
    pub edges: Vec<(u16, Dir)>,
}

pub struct Tree<G> {
    pub nodes: Vec<Node<G>>,
    edges: Vec<Edge>,
    accs: Vec<Acc>,
    legal: Vec<u16>,
}

fn alpha(p: &[f32]) -> [f32; 3] {
    let c = p[3];
    let fl = ALPHA_FLOOR as f32;
    [(c * p[0]).max(fl), (c * p[1]).max(fl), (c * p[2]).max(fl)]
}

fn to64(a: [f32; 3]) -> Dir {
    [a[0] as f64, a[1] as f64, a[2] as f64]
}

fn to32(a: Dir) -> [f32; 3] {
    [a[0] as f32, a[1] as f32, a[2] as f32]
}

impl<G: Game> Default for Tree<G> {
    fn default() -> Self {
        Tree {
            nodes: Vec::new(),
            edges: Vec::new(),
            accs: Vec::new(),
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
        self.accs.clear();
    }

    pub fn root(&self) -> &Node<G> {
        &self.nodes[0]
    }

    pub fn edges_of(&self, n: u32) -> &[Edge] {
        let nd = &self.nodes[n as usize];
        &self.edges[nd.first as usize..(nd.first + nd.len) as usize]
    }

    /// Searched-evidence count per edge of node `n`.
    pub fn visits(&self, n: u32) -> impl Iterator<Item = f64> + '_ {
        self.edges_of(n).iter().map(|e| {
            if e.acc == NONE {
                0.0
            } else {
                self.accs[e.acc as usize].n
            }
        })
    }

    /// Create a node for `state` from its network evaluation.  The node's own
    /// belief is its first piece of evidence about itself.
    pub fn add_node<R: Rng>(&mut self, state: G, ev: Eval, cfg: &SearchCfg, rng: &mut R) -> u32 {
        state.legal_actions(&mut self.legal);
        let first = self.edges.len() as u32;
        for &a in &self.legal {
            let i = a as usize * 4;
            let prior = alpha(&ev.a[i..i + 4]);
            self.edges.push(Edge {
                action: a,
                term: -1,
                vloss: 0,
                child: NONE,
                acc: NONE,
                prior,
                sel: prior,
            });
        }
        let v_alpha = to64(alpha(ev.v));
        let mut acc = Acc::default();
        acc.add(&self.leaf_obs(v_alpha, cfg, rng), cfg.node_track());
        let player = state.current_player() as u8;
        self.nodes.push(Node {
            state,
            player,
            first,
            len: self.legal.len() as u32,
            v_alpha,
            acc,
        });
        (self.nodes.len() - 1) as u32
    }

    /// Expand the unexpanded edge `e` with its child's evaluation.
    pub fn expand<R: Rng>(&mut self, e: u32, state: G, ev: Eval, cfg: &SearchCfg, rng: &mut R) -> u32 {
        let n = self.add_node(state, ev, cfg, rng);
        self.edges[e as usize].child = n;
        n
    }

    fn leaf_obs<R: Rng>(&self, a: Dir, cfg: &SearchCfg, rng: &mut R) -> Obs {
        if cfg.sample_leaf {
            Obs::sampled(a, rng)
        } else {
            Obs::from_alpha(a)
        }
    }

    fn refresh(&mut self, e: usize, cfg: &SearchCfg) {
        let edge = &self.edges[e];
        let sel = if edge.term >= 0 {
            spike(edge.term as usize)
        } else if edge.acc == NONE {
            to64(edge.prior)
        } else {
            cfg.search
                .search_belief(to64(edge.prior), &self.accs[edge.acc as usize])
        };
        self.edges[e].sel = to32(sel);
    }

    fn set_term(&mut self, e: usize, outcome: usize, cfg: &SearchCfg) {
        self.edges[e].term = outcome as i8;
        self.refresh(e, cfg);
    }

    /// Thompson-descend from the root, applying virtual loss, until an
    /// unexpanded or terminal edge.  `path` receives (node, edge) pairs.
    pub fn select<R: Rng>(
        &mut self,
        cfg: &SearchCfg,
        rng: &mut R,
        temp: f64,
        path: &mut Vec<(u32, u32)>,
    ) -> Selected<G> {
        path.clear();
        let mut n = 0u32;
        loop {
            let e = self.sample_edge(n, rng, temp, cfg.virtual_loss);
            self.edges[e].vloss += 1;
            path.push((n, e as u32));
            let edge = &self.edges[e];
            if edge.term >= 0 {
                let out = edge.term as usize;
                self.backup_terminal(path, out, cfg);
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
                let out = outcome_index(r);
                self.set_term(e, out, cfg);
                self.backup_terminal(path, out, cfg);
                return Selected::Terminal;
            }
            return Selected::Leaf(st);
        }
    }

    /// One Thompson draw per edge of node `n` from Dir(temp * sel), minus the
    /// virtual-loss penalty; returns the global index of the argmax.
    fn sample_edge<R: Rng>(&self, n: u32, rng: &mut R, temp: f64, vl: f64) -> usize {
        let nd = &self.nodes[n as usize];
        let (lo, hi) = (nd.first as usize, (nd.first + nd.len) as usize);
        let mut best = (f64::NEG_INFINITY, lo);
        for i in lo..hi {
            let e = &self.edges[i];
            let a = to64(e.sel);
            let v = sample_value(rng, [a[0] * temp, a[1] * temp, a[2] * temp]) - vl * e.vloss as f64;
            if v > best.0 {
                best = (v, i);
            }
        }
        best.1
    }

    /// Fold one observation (in `persp`'s perspective) into every node and
    /// edge on the path, flipping it for nodes whose mover is the other player.
    pub fn backup(&mut self, path: &[(u32, u32)], obs: Obs, persp: u8, cfg: &SearchCfg) {
        let (edge_track, node_track) = (cfg.edge_track(), cfg.node_track());
        let flipped = obs.flipped();
        for &(n, e) in path.iter().rev() {
            let o = if self.nodes[n as usize].player == persp {
                &obs
            } else {
                &flipped
            };
            let edge = &mut self.edges[e as usize];
            edge.vloss -= 1;
            if edge.acc == NONE {
                edge.acc = self.accs.len() as u32;
                self.accs.push(Acc::default());
            }
            self.accs[edge.acc as usize].add(o, edge_track);
            self.nodes[n as usize].acc.add(o, node_track);
            self.refresh(e as usize, cfg);
        }
    }

    /// Back up the freshly expanded child at the end of `path`.
    pub fn backup_leaf<R: Rng>(&mut self, path: &[(u32, u32)], cfg: &SearchCfg, rng: &mut R) {
        let &(_, e) = path.last().unwrap();
        let child = &self.nodes[self.edges[e as usize].child as usize];
        let (obs, persp) = (self.leaf_obs(child.v_alpha, cfg, rng), child.player);
        self.backup(path, obs, persp, cfg);
    }

    fn backup_terminal(&mut self, path: &[(u32, u32)], outcome: usize, cfg: &SearchCfg) {
        let persp = self.nodes[path.last().unwrap().0 as usize].player;
        self.backup(path, cfg.terminal_obs(outcome), persp, cfg);
    }

    /// Proven outcome for node `n`'s mover: a win if any edge is a proven
    /// win; once every edge is proven, a draw if any is, else a loss.
    pub fn solved(&self, n: u32) -> Option<usize> {
        let mut all = true;
        let mut draw = false;
        for e in self.edges_of(n) {
            match e.term {
                t if t == WIN as i8 => return Some(WIN),
                t if t == DRAW as i8 => draw = true,
                t if t < 0 => all = false,
                _ => {}
            }
        }
        all.then_some(if draw { DRAW } else { LOSS })
    }

    /// After a terminal hit, walk up proving parent edges of newly solved
    /// nodes.  Pushes each newly solved node onto `out`.
    pub fn propagate_solved(&mut self, path: &[(u32, u32)], cfg: &SearchCfg, out: &mut Vec<u32>) {
        for k in (1..path.len()).rev() {
            let n = path[k].0;
            let Some(o) = self.solved(n) else { break };
            let (pn, pe) = path[k - 1];
            if self.edges[pe as usize].term >= 0 {
                break;
            }
            let same = self.nodes[pn as usize].player == self.nodes[n as usize].player;
            self.set_term(pe as usize, if same { o } else { 2 - o }, cfg);
            out.push(n);
        }
    }

    /// Index (into `edges_of(0)`) of the move to play.  `thompson`: one more
    /// Thompson sample of the root beliefs, drawn from Dir(temp * α).
    /// Otherwise the best posterior mean among edges search actually touched.
    pub fn pick<R: Rng>(&self, rng: &mut R, thompson: Option<f64>) -> usize {
        let lo = self.nodes[0].first as usize;
        if let Some(temp) = thompson {
            return self.sample_edge(0, rng, temp, 0.0) - lo;
        }
        let edges = self.edges_of(0);
        let touched = |e: &Edge| e.acc != NONE || e.term >= 0;
        let any = edges.iter().any(touched);
        let mut best = (f64::NEG_INFINITY, 0);
        for (i, e) in edges.iter().enumerate() {
            if any && !touched(e) {
                continue;
            }
            let v = value(&to64(e.sel));
            if v > best.0 {
                best = (v, i);
            }
        }
        best.1
    }

    /// Training target for node `n`: its state belief and one belief per
    /// edge search touched.  Proofs win over evidence.
    pub fn target(&self, n: u32, cfg: &SearchCfg) -> Target {
        let node = &self.nodes[n as usize];
        let mut edges = Vec::new();
        // (visits, target) of the most-visited edge with a target
        let mut best: Option<(f64, Dir)> = None;
        for e in self.edges_of(n) {
            let t = if e.term >= 0 {
                Some(spike(e.term as usize))
            } else if e.acc != NONE {
                cfg.target.edge_target(to64(e.prior), &self.accs[e.acc as usize])
            } else {
                None
            };
            if let Some(t) = t {
                let t = floor_conc(t, TARGET_CONC_FLOOR);
                let visits = if e.acc == NONE { 0.0 } else { self.accs[e.acc as usize].n };
                if best.is_none_or(|(b, _)| visits > b) {
                    best = Some((visits, t));
                }
                edges.push((e.action, t));
            }
        }
        let v = match (self.solved(n), best) {
            (Some(o), _) => spike(o),
            (None, Some((_, t))) if cfg.best_value => t,
            _ => cfg.target.state_target(node.v_alpha, &node.acc),
        };
        Target {
            v: floor_conc(v, TARGET_CONC_FLOOR),
            edges,
        }
    }

    pub fn report(&self) -> Report {
        let root = &self.nodes[0];
        let edges = self
            .edges_of(0)
            .iter()
            .map(|e| EdgeReport {
                action: e.action,
                term: e.term,
                prior: to64(e.prior),
                sel: to64(e.sel),
                acc: if e.acc == NONE { [0.0; RAW] } else { self.accs[e.acc as usize].raw() },
            })
            .collect();
        Report {
            v_alpha: root.v_alpha,
            node: root.acc.raw(),
            edges,
        }
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
        let mut accs = Vec::new();
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
                if e.acc != NONE {
                    accs.push(self.accs[e.acc as usize].clone());
                    e.acc = (accs.len() - 1) as u32;
                }
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
        self.accs = accs;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::games::Connect4;
    use rand::{rngs::SmallRng, SeedableRng};

    fn uniform_eval<G: Game>() -> (Vec<f32>, Vec<f32>) {
        let v = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 1.5];
        let a = (0..G::NUM_ACTIONS).flat_map(|_| v.clone()).collect();
        (v, a)
    }

    /// Run a plain sequential search with a uniform network.
    fn search(tree: &mut Tree<Connect4>, state: Connect4, sims: usize, cfg: &SearchCfg, rng: &mut SmallRng) {
        let (v, a) = uniform_eval::<Connect4>();
        let ev = Eval { v: &v, a: &a };
        if tree.is_empty() {
            tree.add_node(state, ev, cfg, rng);
        }
        let mut path = Vec::new();
        let mut solved = Vec::new();
        for _ in 0..sims {
            if tree.solved(0).is_some() {
                break;
            }
            match tree.select(cfg, rng, 1.0, &mut path) {
                Selected::Terminal => tree.propagate_solved(&path, cfg, &mut solved),
                Selected::Leaf(st) => {
                    let e = path.last().unwrap().1;
                    tree.expand(e, st, ev, cfg, rng);
                    tree.backup_leaf(&path, cfg, rng);
                }
            }
        }
    }

    fn cfg(rule: &str) -> SearchCfg {
        let r = Rule::parse(rule, 0.8).unwrap();
        SearchCfg {
            search: r,
            target: r,
            sample_leaf: false,
            virtual_loss: 1.0,
            track_all: false,
            best_value: false,
        }
    }

    #[test]
    fn best_value_targets_the_most_visited_edge() {
        let mut c = cfg("additive");
        c.best_value = true;
        let mut rng = SmallRng::seed_from_u64(4);
        let mut t: Tree<Connect4> = Tree::default();
        search(&mut t, Connect4::new(), 200, &c, &mut rng);
        let tg = t.target(0, &c);
        let (i, _) = t.visits(0).enumerate().fold((0, -1.0), |b, (i, n)| if n > b.1 { (i, n) } else { b });
        let a = t.edges_of(0)[i].action;
        let e = tg.edges.iter().find(|(x, _)| *x == a).unwrap().1;
        assert_eq!(tg.v, e);
    }

    #[test]
    fn finds_and_proves_immediate_win() {
        // x to move with three in column 0: playing 0 wins
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1] {
            s.apply(a);
        }
        for rule in [
            "mixture", "additive", "additive_mle", "avgconc", "corr", "additive_x16", "sumconc", "mean_mle", "point",
        ] {
            let c = cfg(rule);
            let mut rng = SmallRng::seed_from_u64(1);
            let mut t = Tree::default();
            search(&mut t, s.clone(), 400, &c, &mut rng);
            assert_eq!(t.solved(0), Some(WIN), "{rule}");
            let i = t.pick(&mut rng, None);
            assert_eq!(t.edges_of(0)[i].action, 0, "{rule}");
        }
    }

    #[test]
    fn proves_forced_loss_two_plies_deep() {
        // x: b1 c1 d1 (open at a1 and e1), o: b2 c2 d2 (unsupported).  After x
        // wastes a move, o can block only one end: o is lost.
        let mut s = Connect4::new();
        for a in [1, 1, 2, 2, 3, 3, 6] {
            s.apply(a);
        }
        let c = cfg("additive_mle");
        let mut rng = SmallRng::seed_from_u64(2);
        let mut t = Tree::default();
        search(&mut t, s, 5000, &c, &mut rng);
        assert_eq!(t.solved(0), Some(LOSS));
    }

    #[test]
    fn subtree_reuse_keeps_evidence_and_bookkeeping_balances() {
        let c = cfg("additive");
        let mut rng = SmallRng::seed_from_u64(3);
        let mut t: Tree<Connect4> = Tree::default();
        search(&mut t, Connect4::new(), 300, &c, &mut rng);
        let i = t.pick(&mut rng, None);
        let child = t.edges_of(0)[i].child;
        let n_before = t.nodes[child as usize].acc.n;
        t.advance(i);
        assert_eq!(t.nodes[0].acc.n, n_before);
        assert!(t.edges.iter().all(|e| e.vloss == 0));
        // additive: root acc count == 1 (own belief) + simulations through it
        let through: f64 = t.visits(0).sum();
        assert_eq!(t.nodes[0].acc.n, 1.0 + through);
    }
}
