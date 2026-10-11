//! Dirichlet value beliefs and the evidence rules that back them up.
//!
//! Every belief is a Dirichlet over (win, draw, loss) FROM THE PERSPECTIVE OF
//! THE PLAYER TO MOVE at that node.  Its scalar value is E[p_win] - E[p_loss].
//!
//! Each node and each searched edge keeps an evidence accumulator [`Acc`]: O(1)
//! running sums over the Dirichlets backed up through it.  An evidence RULE
//! ([`Rule`]) turns an accumulator (plus the network prior) back into one
//! Dirichlet, separately for SELECTION and for TRAINING TARGETS.
//!
//! ## Adding a backup rule
//! 1. add a variant to [`Rule`] and a name to [`Rule::parse`];
//! 2. say how it collapses evidence in [`Rule::observed`] (prior-free rules)
//!    or give it its own arm in [`Rule::search_belief`] / [`Rule::edge_target`]
//!    / [`Rule::state_target`] (rules that combine with the prior, like
//!    `corr`);
//! 3. if it needs a new running statistic, add it to [`Acc`] behind a
//!    [`Track`] flag so the other rules don't pay for it.

use rand::Rng;
use rand_distr::{Distribution, StandardNormal};

pub const WIN: usize = 0;
pub const DRAW: usize = 1;
pub const LOSS: usize = 2;

pub type Dir = [f64; 3];

/// Positivity epsilon for a Dirichlet component (never shapes anything).
pub const ALPHA_FLOOR: f64 = 1e-9;
/// Smoothing pseudo-count for training-target components.
pub const TARGET_EPS: f64 = 0.05;
/// A proven result is asserted as a finite spike: this much on the outcome...
pub const TERMINAL_CONC: f64 = 100.0;
/// ...and this much on the other two.
pub const TERMINAL_EPS: f64 = 0.05;
/// Minimum total concentration of a training target (direction-preserving).
pub const TARGET_CONC_FLOOR: f64 = 3.0 * TARGET_EPS;

/// Reflect a belief one perspective over: swap win and loss.
#[inline]
pub fn flip(a: Dir) -> Dir {
    [a[2], a[1], a[0]]
}

/// Outcome index from a +1 / 0 / -1 result.
#[inline]
pub fn outcome_index(r: i8) -> usize {
    match r {
        1 => WIN,
        0 => DRAW,
        _ => LOSS,
    }
}

pub fn spike(outcome: usize) -> Dir {
    let mut a = [TERMINAL_EPS; 3];
    a[outcome] = TERMINAL_CONC;
    a
}

/// Scalar value E[p_win] - E[p_loss].
#[inline]
pub fn value(a: &Dir) -> f64 {
    (a[WIN] - a[LOSS]) / (a[0] + a[1] + a[2])
}

/// Raise the total concentration to at least `min_total` by SCALING, which
/// keeps the mean (the direction is the information).  An all-zero row has no
/// direction and becomes uniform.
pub fn floor_conc(a: Dir, min_total: f64) -> Dir {
    let tot = a[0] + a[1] + a[2];
    if tot <= 0.0 {
        return [min_total / 3.0; 3];
    }
    let s = (min_total / tot).max(1.0);
    [a[0] * s, a[1] * s, a[2] * s]
}

// ════════════════════════════════════════════════════════════════════════════
//  One backed-up observation
// ════════════════════════════════════════════════════════════════════════════

/// A Dirichlet as the accumulators see it: mean, per-component variance and
/// total concentration.
#[derive(Clone, Copy, Debug)]
pub struct Obs {
    pub m: Dir,
    pub v: Dir,
    pub a0: f64,
}

impl Obs {
    pub fn from_alpha(a: Dir) -> Obs {
        let a0 = a[0] + a[1] + a[2];
        let m = [a[0] / a0, a[1] / a0, a[2] / a0];
        Obs::with_mean(m, a0)
    }

    fn with_mean(m: Dir, a0: f64) -> Obs {
        let inv = 1.0 / (a0 + 1.0);
        Obs {
            m,
            v: [
                m[0] * (1.0 - m[0]) * inv,
                m[1] * (1.0 - m[1]) * inv,
                m[2] * (1.0 - m[2]) * inv,
            ],
            a0,
        }
    }

    /// A proven result as an exact one-hot with zero variance (it is not an
    /// estimate).  Only the `corr` rule uses this; the others back up the
    /// finite [`spike`] like any other belief.
    pub fn exact(outcome: usize) -> Obs {
        let mut m = [0.0; 3];
        m[outcome] = 1.0;
        Obs {
            m,
            v: [0.0; 3],
            a0: TERMINAL_CONC + 2.0 * TERMINAL_EPS,
        }
    }

    /// The leaf contributes a DRAW from its Dirichlet instead of its mean (its
    /// concentration still counts as its evidence strength).
    pub fn sampled<R: Rng>(a: Dir, rng: &mut R) -> Obs {
        let g = [gamma(rng, a[0]), gamma(rng, a[1]), gamma(rng, a[2])];
        let s = g[0] + g[1] + g[2];
        if !s.is_finite() || s <= 0.0 {
            return Obs::from_alpha(a);
        }
        Obs::with_mean([g[0] / s, g[1] / s, g[2] / s], a[0] + a[1] + a[2])
    }

    #[inline]
    pub fn flipped(&self) -> Obs {
        Obs {
            m: flip(self.m),
            v: flip(self.v),
            a0: self.a0,
        }
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Evidence accumulator
// ════════════════════════════════════════════════════════════════════════════

/// Which optional statistics an accumulator maintains.
#[derive(Clone, Copy, Debug, Default)]
pub struct Track {
    pub mle: bool,
    pub corr: bool,
}

/// Running sums over the observations backed up through a node or edge.
#[derive(Clone, Debug)]
pub struct Acc {
    pub n: f64,
    /// Σ m
    pub sm: Dir,
    /// Σ m²
    pub sq: Dir,
    /// Σ per-component variance
    pub sv: Dir,
    /// Σ α₀
    pub sa: f64,
    /// Σ sqrt(total variance) — `corr` only
    pub st: f64,
    /// Σ log m and the warm-started MLE — `additive_mle` only
    pub lsum: Dir,
    pub mle: Dir,
    mle_next: f64,
}

impl Default for Acc {
    fn default() -> Self {
        Acc {
            n: 0.0,
            sm: [0.0; 3],
            sq: [0.0; 3],
            sv: [0.0; 3],
            sa: 0.0,
            st: 0.0,
            lsum: [0.0; 3],
            mle: [1.0; 3],
            mle_next: 1.0,
        }
    }
}

const MLE_FLOOR: f64 = 1e-9;
/// Refresh the MLE when n reaches n * GROWTH (~23 refreshes over 400 sims).
const MLE_GROWTH: f64 = 1.3;
const MLE_STEPS: usize = 12;
const MLE_NEWTON: usize = 2;

impl Acc {
    pub fn add(&mut self, o: &Obs, track: Track) {
        self.n += 1.0;
        for j in 0..3 {
            self.sm[j] += o.m[j];
            self.sq[j] += o.m[j] * o.m[j];
            self.sv[j] += o.v[j];
        }
        self.sa += o.a0;
        if track.corr {
            let u = o.v[0] + o.v[1] + o.v[2];
            if u > 0.0 {
                self.st += u.sqrt();
            }
        }
        if track.mle {
            for j in 0..3 {
                self.lsum[j] += o.m[j].max(MLE_FLOOR).ln();
            }
            if self.n >= self.mle_next {
                self.mle_next = (self.n + 1.0).max(self.n * MLE_GROWTH);
                self.refresh_mle();
            }
        }
    }

    /// Minka's fixed point for the Dirichlet MLE of the observed means:
    /// ψ(α_j) = ψ(α₀) + mean log x_j, warm-started from the last estimate.
    fn refresh_mle(&mut self) {
        let g = [self.lsum[0] / self.n, self.lsum[1] / self.n, self.lsum[2] / self.n];
        let mut a = self.mle;
        for _ in 0..MLE_STEPS {
            let c = digamma(a[0] + a[1] + a[2]);
            a = [inv_digamma(c + g[0]), inv_digamma(c + g[1]), inv_digamma(c + g[2])];
        }
        self.mle = a;
    }
}

// ════════════════════════════════════════════════════════════════════════════
//  Evidence rules
// ════════════════════════════════════════════════════════════════════════════

/// How a node/edge's accumulated evidence becomes one Dirichlet.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Rule {
    /// Moment-match the MIXTURE (1/n) Σ Dir(α_i): concentration measures how
    /// much the backed-up beliefs disagree; does not grow with visits.
    Mixture,
    /// α = Σ m_i: each simulation is one soft-labelled observation, so α₀ is
    /// exactly the visit count (textbook Thompson annealing).
    Additive,
    /// Dirichlet maximum-likelihood fit of the same leaf means: α₀ tracks
    /// min(n, intrinsic dispersion).  Falls back to `Additive` below 3 visits.
    AdditiveMle,
    /// Direction = mean of leaf means, concentration = mean leaf α₀.
    AvgConc,
    /// Correlated-error posterior: network prior and search mean are two
    /// estimators with error correlation ρ, combined by generalised least
    /// squares and moment-matched to a Dirichlet.
    Corr { rho: f64 },
}

impl Rule {
    pub fn parse(name: &str, rho: f64) -> Result<Rule, String> {
        Ok(match name {
            "mixture" => Rule::Mixture,
            "additive" => Rule::Additive,
            "additive_mle" => Rule::AdditiveMle,
            "avgconc" => Rule::AvgConc,
            "corr" => Rule::Corr { rho },
            _ => {
                return Err(format!(
                    "unknown evidence rule {name:?}; expected mixture | additive | additive_mle | avgconc | corr"
                ))
            }
        })
    }

    pub fn track(&self) -> Track {
        Track {
            mle: *self == Rule::AdditiveMle,
            corr: matches!(self, Rule::Corr { .. }),
        }
    }

    /// Prior-free collapse (every rule but `corr`).  None without evidence.
    pub fn observed(&self, acc: &Acc) -> Option<Dir> {
        let n = acc.n;
        if n <= 0.0 {
            return None;
        }
        let fl = |a: Dir| [a[0].max(ALPHA_FLOOR), a[1].max(ALPHA_FLOOR), a[2].max(ALPHA_FLOOR)];
        Some(match self {
            Rule::AdditiveMle if n >= 3.0 => fl(acc.mle),
            Rule::Additive | Rule::AdditiveMle => fl(acc.sm),
            Rule::AvgConc => {
                let k = acc.sa / n / n; // (SM/n) * (SA/n)
                fl([acc.sm[0] * k, acc.sm[1] * k, acc.sm[2] * k])
            }
            Rule::Mixture => {
                let m = [acc.sm[0] / n, acc.sm[1] / n, acc.sm[2] / n];
                let v = [
                    (acc.sv[0] + acc.sq[0]) / n - m[0] * m[0],
                    (acc.sv[1] + acc.sq[1]) / n - m[1] * m[1],
                    (acc.sv[2] + acc.sq[2]) / n - m[2] * m[2],
                ];
                let b0 = beta0_from_moments(m, v);
                fl([m[0] * b0, m[1] * b0, m[2] * b0])
            }
            Rule::Corr { .. } => unreachable!("corr combines with the prior; use search_belief"),
        })
    }

    /// What selection samples: prior + observed evidence (or, for `corr`, the
    /// GLS posterior, which already contains the prior).
    pub fn search_belief(&self, prior: Dir, acc: &Acc) -> Dir {
        match self {
            Rule::Corr { rho } => corr_posterior(prior, acc, *rho),
            _ => match self.observed(acc) {
                Some(o) => [prior[0] + o[0], prior[1] + o[1], prior[2] + o[2]],
                None => prior,
            },
        }
    }

    /// Training target for a searched edge.  None if it was never searched.
    pub fn edge_target(&self, prior: Dir, acc: &Acc) -> Option<Dir> {
        match self {
            Rule::Corr { rho } => (acc.n > 0.0).then(|| corr_target(corr_posterior(prior, acc, *rho))),
            _ => self.observed(acc),
        }
    }

    /// Training target for a node's own value (the tree handles solved nodes).
    pub fn state_target(&self, v_alpha: Dir, acc: &Acc) -> Dir {
        match self {
            Rule::Corr { rho } if acc.n > 0.0 => corr_target(corr_posterior(v_alpha, acc, *rho)),
            Rule::Corr { .. } => v_alpha,
            _ => self.observed(acc).unwrap_or(v_alpha),
        }
    }
}

/// Minka moment match: the Dirichlet with mean M and total variance ΣV has
/// β₀ = ΣM(1-M)/ΣV - 1.  A mixture can be more dispersed than any Dirichlet,
/// which makes β₀ negative; that floors (it is invalid, not large).
fn beta0_from_moments(m: Dir, v: Dir) -> f64 {
    let vs = v[0].max(1e-300) + v[1].max(1e-300) + v[2].max(1e-300);
    let b0 = (m[0] * (1.0 - m[0]) + m[1] * (1.0 - m[1]) + m[2] * (1.0 - m[2])) / vs - 1.0;
    b0.max(ALPHA_FLOOR)
}

// ── correlated-error posterior ("ThompsonZero v2", corr_posterior.py) ───────

const RHO_MIN: f64 = 1e-3;
const RHO_MAX: f64 = 0.999;
const CORR_EPS: f64 = 1e-12;
const CORR_T_MIN: f64 = 1e-10;
const CORR_A0_MIN: f64 = 0.05;
const CORR_A0_MAX: f64 = 1e6;
const CORR_FLOOR: f64 = 1e-4;

/// Posterior Dirichlet from the network prior and the search accumulator.
///
/// The prior (mean μ₀, total variance u₀) and the search mean μₛ (variance Tₛ)
/// share error with covariance r, so the GLS weight on the prior is
/// w = (Tₛ - r) / (u₀ + Tₛ - 2r); the combined mean and total variance are
/// inverted to a concentration α₀ = (1 - |μ|²)/T - 1.
pub fn corr_posterior(prior: Dir, acc: &Acc, rho: f64) -> Dir {
    let n = acc.n;
    if n <= 0.0 {
        return prior;
    }
    let rho = rho.clamp(RHO_MIN, RHO_MAX);
    let a0p = prior[0] + prior[1] + prior[2];
    let m0 = [prior[0] / a0p, prior[1] / a0p, prior[2] / a0p];
    let u0 = ((1.0 - (m0[0] * m0[0] + m0[1] * m0[1] + m0[2] * m0[2])) / (a0p + 1.0)).max(0.0);

    let inv_n = 1.0 / n;
    let ms = [acc.sm[0] * inv_n, acc.sm[1] * inv_n, acc.sm[2] * inv_n];
    let b_emp = if n >= 2.0 {
        let ss = acc.sm[0] * acc.sm[0] + acc.sm[1] * acc.sm[1] + acc.sm[2] * acc.sm[2];
        ((acc.sq[0] + acc.sq[1] + acc.sq[2] - ss * inv_n) / (n - 1.0)).max(0.0)
    } else {
        0.0
    };
    // what idiosyncratic leaf noise alone must produce; max() so the empirical
    // dispersion (which already contains that noise) is not counted twice
    let su = acc.sv[0] + acc.sv[1] + acc.sv[2];
    let b_eff = b_emp.max((1.0 - rho) * su * inv_n);
    let ts = b_eff * inv_n + rho * acc.st * acc.st * inv_n * inv_n;
    let r = rho * u0.sqrt() * acc.st * inv_n;

    let den = u0 + ts - 2.0 * r;
    let w = if den > CORR_EPS {
        ((ts - r) / den).clamp(0.0, 1.0)
    } else if ts <= u0 {
        0.0
    } else {
        1.0
    };
    let ww = 1.0 - w;
    let mu = [w * m0[0] + ww * ms[0], w * m0[1] + ww * ms[1], w * m0[2] + ww * ms[2]];
    let t = w * w * u0 + ww * ww * ts + 2.0 * w * ww * r;
    let num = 1.0 - (mu[0] * mu[0] + mu[1] * mu[1] + mu[2] * mu[2]);
    let a0 = if t <= CORR_T_MIN || num <= CORR_EPS {
        CORR_A0_MAX // resolved
    } else {
        (num / t - 1.0).clamp(CORR_A0_MIN, CORR_A0_MAX)
    };
    [
        (a0 * mu[0]).max(CORR_FLOOR),
        (a0 * mu[1]).max(CORR_FLOOR),
        (a0 * mu[2]).max(CORR_FLOOR),
    ]
}

/// A `corr` belief made safe as a KL target: total capped at TERMINAL_CONC
/// (the 1e6 resolved spike would explode the loss), then components floored at
/// TARGET_EPS (after the cap, so the floor survives).
fn corr_target(a: Dir) -> Dir {
    let tot = a[0] + a[1] + a[2];
    let s = if tot > TERMINAL_CONC { TERMINAL_CONC / tot } else { 1.0 };
    [
        (a[0] * s).max(TARGET_EPS),
        (a[1] * s).max(TARGET_EPS),
        (a[2] * s).max(TARGET_EPS),
    ]
}

// ════════════════════════════════════════════════════════════════════════════
//  Special functions and sampling
// ════════════════════════════════════════════════════════════════════════════

const DIGAMMA_1: f64 = -0.577_215_664_901_532_9;

pub fn digamma(mut x: f64) -> f64 {
    let mut r = 0.0;
    while x < 6.0 {
        r -= 1.0 / x;
        x += 1.0;
    }
    let f = 1.0 / (x * x);
    r + x.ln() - 0.5 / x + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0))))
}

/// (digamma(x), trigamma(x)) sharing one recurrence: the Newton step of
/// `inv_digamma` needs both, and it is the inner loop of the MLE refresh.
fn digamma_trigamma(mut x: f64) -> (f64, f64) {
    let (mut d, mut t) = (0.0, 0.0);
    while x < 6.0 {
        let inv = 1.0 / x;
        d -= inv;
        t += 1.0 / (x * x);
        x += 1.0;
    }
    let inv = 1.0 / x;
    let f = 1.0 / (x * x);
    (
        d + x.ln() - 0.5 / x + f * (-1.0 / 12.0 + f * (1.0 / 120.0 + f * (-1.0 / 252.0 + f * (1.0 / 240.0)))),
        t + inv * (1.0 + 0.5 / x + f * (1.0 / 6.0 + f * (-1.0 / 30.0 + f * (1.0 / 42.0 + f * (-1.0 / 30.0))))),
    )
}

fn inv_digamma(y: f64) -> f64 {
    let mut x = if y >= -2.22 {
        y.exp() + 0.5
    } else {
        -1.0 / (y - DIGAMMA_1)
    };
    for _ in 0..MLE_NEWTON {
        let (d, t) = digamma_trigamma(x);
        x = (x - (d - y) / t).max(1e-9);
    }
    x
}

/// Gamma(shape, 1) by Marsaglia-Tsang (with the U^(1/a) boost below 1).
#[inline]
pub fn gamma<R: Rng>(rng: &mut R, shape: f64) -> f64 {
    let (a, boost) = if shape < 1.0 {
        (shape + 1.0, true)
    } else {
        (shape, false)
    };
    let d = a - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    let g = loop {
        let x: f64 = StandardNormal.sample(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 {
            continue;
        }
        let v = v * v * v;
        let u: f64 = rng.random();
        if u < 1.0 - 0.0331 * x * x * x * x || u.ln() < 0.5 * x * x + d * (1.0 - v + v.ln()) {
            break d * v;
        }
    };
    if boost {
        g * rng.random::<f64>().powf(1.0 / shape)
    } else {
        g
    }
}

/// One Thompson draw of v = p_win - p_loss from Dir(α).
///
/// As α₀ -> 0 every gamma underflows to 0; the exact limit of Dir(α) there is
/// a point mass on corner i with probability α_i/α₀, so that is drawn instead.
#[inline]
pub fn sample_value<R: Rng>(rng: &mut R, a: Dir) -> f64 {
    let g = [gamma(rng, a[0]), gamma(rng, a[1]), gamma(rng, a[2])];
    let s = g[0] + g[1] + g[2];
    if s > 0.0 {
        return (g[WIN] - g[LOSS]) / s;
    }
    let u = rng.random::<f64>() * (a[0] + a[1] + a[2]);
    if u < a[WIN] {
        1.0
    } else if u < a[WIN] + a[DRAW] {
        0.0
    } else {
        -1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::SmallRng, SeedableRng};

    fn acc_of(alphas: &[Dir], track: Track) -> Acc {
        let mut acc = Acc::default();
        for a in alphas {
            acc.add(&Obs::from_alpha(*a), track);
        }
        acc
    }

    #[test]
    fn mixture_of_one_round_trips() {
        let a = [3.0, 1.5, 0.5];
        let got = Rule::Mixture.observed(&acc_of(&[a], Track::default())).unwrap();
        for j in 0..3 {
            assert!((got[j] - a[j]).abs() < 1e-9, "{got:?}");
        }
    }

    #[test]
    fn additive_concentration_is_visit_count() {
        let acc = acc_of(&[[1.0, 1.0, 1.0], [5.0, 0.1, 0.1], [0.2, 0.2, 9.0]], Track::default());
        let got = Rule::Additive.observed(&acc).unwrap();
        assert!((got.iter().sum::<f64>() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn avgconc_mean_and_concentration() {
        let acc = acc_of(&[[2.0, 1.0, 1.0], [1.0, 1.0, 6.0]], Track::default());
        let got = Rule::AvgConc.observed(&acc).unwrap();
        let tot: f64 = got.iter().sum();
        assert!((tot - 6.0).abs() < 1e-12); // mean of 4 and 8
        assert!((got[0] / tot - (0.5 + 0.125) / 2.0).abs() < 1e-12);
    }

    #[test]
    fn mle_recovers_dispersion() {
        // leaf means drawn from Dir(4, 2, 2): the fit should land near α₀ = 8
        let mut rng = SmallRng::seed_from_u64(0);
        let mut acc = Acc::default();
        let track = Rule::AdditiveMle.track();
        for _ in 0..4000 {
            let g = [gamma(&mut rng, 4.0), gamma(&mut rng, 2.0), gamma(&mut rng, 2.0)];
            let s: f64 = g.iter().sum();
            acc.add(
                &Obs::from_alpha([g[0] / s * 1e6, g[1] / s * 1e6, g[2] / s * 1e6]),
                track,
            );
        }
        acc.refresh_mle();
        let a = Rule::AdditiveMle.observed(&acc).unwrap();
        let a0: f64 = a.iter().sum();
        assert!((a0 - 8.0).abs() < 0.5, "{a:?}");
        assert!((a[0] / a0 - 0.5).abs() < 0.02, "{a:?}");
    }

    #[test]
    fn digamma_matches_reference() {
        assert!((digamma(1.0) - DIGAMMA_1).abs() < 1e-9);
        assert!((digamma(0.5) - (-1.963_510_026_021_423_5)).abs() < 1e-9);
        assert!((inv_digamma(digamma(3.7)) - 3.7).abs() < 1e-2);
        for x in [1e-3, 0.37, 1.0, 5.99, 6.0, 42.0] {
            assert_eq!(digamma_trigamma(x).0, digamma(x)); // bit-identical
        }
        assert!((digamma_trigamma(1.0).1 - std::f64::consts::PI.powi(2) / 6.0).abs() < 1e-9);
    }

    #[test]
    fn corr_limits() {
        let track = Track { mle: false, corr: true };
        let prior = [2.0, 1.0, 1.0];
        // no evidence -> prior
        assert_eq!(corr_posterior(prior, &Acc::default(), 0.5), prior);
        // exact proofs -> resolved spike on the proven corner
        let mut acc = Acc::default();
        for _ in 0..5 {
            acc.add(&Obs::exact(WIN), track);
        }
        let p = corr_posterior(prior, &acc, 0.5);
        assert!(p[WIN] > 1e5 && p[LOSS] <= 1e-3, "{p:?}");
        // many agreeing confident leaves at low rho: concentration grows with n
        let leaf = Obs::from_alpha([8.0, 1.0, 1.0]);
        let (mut a10, mut a100) = (Acc::default(), Acc::default());
        for i in 0..100 {
            if i < 10 {
                a10.add(&leaf, track);
            }
            a100.add(&leaf, track);
        }
        let s10: f64 = corr_posterior(prior, &a10, 0.05).iter().sum();
        let s100: f64 = corr_posterior(prior, &a100, 0.05).iter().sum();
        assert!(s100 > s10 * 2.0, "{s10} {s100}");
        // ...and saturates near the leaf's own concentration as rho -> 1
        let hi: f64 = corr_posterior(prior, &a100, 0.999).iter().sum();
        assert!(hi < 2.0 * 10.0, "{hi}");
    }

    #[test]
    fn gamma_mean_and_degenerate_values() {
        let mut rng = SmallRng::seed_from_u64(3);
        for shape in [0.3, 1.0, 7.5] {
            let m: f64 = (0..20000).map(|_| gamma(&mut rng, shape)).sum::<f64>() / 20000.0;
            assert!((m - shape).abs() < 0.05 * shape.max(1.0), "{shape} {m}");
        }
        let v = sample_value(&mut rng, [1e-12, 1e-12, 1e-12]);
        assert!(v == 1.0 || v == 0.0 || v == -1.0);
    }
}
