//! ThompsonZero core: games, Dirichlet value beliefs, the Thompson-sampling
//! search tree, and the batched self-play / arena engine.  The network and
//! training live in Python (`python/thompsonzero`); everything here is
//! CPU-side and talks to it through `python.rs`.

pub mod belief;
pub mod engine;
pub mod games;
pub mod tree;

#[cfg(feature = "python")]
mod python;
