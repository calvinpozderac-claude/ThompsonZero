//! Game interface plus the three games.
//!
//! The search only ever needs what is in [`Game`]: who moves, which actions
//! are legal, how an action changes the state, whether it is over, and a
//! mover-relative encoding for the network.  Adding a game is implementing
//! this trait and listing it in `engine::AnyEngine` / `python.rs`.

pub mod boop;
pub mod connect4;
pub mod othello;

pub use boop::Boop;
pub use connect4::Connect4;
pub use othello::Othello;

/// A two-player, zero-sum, deterministic, perfect-information game.
///
/// States are small `Copy`-cheap values: the tree stores one per expanded
/// node, so descending never replays moves from the root.
pub trait Game: Clone + Send + Sync + 'static {
    const NAME: &'static str;
    /// Size of the action space (network action-head width).
    const NUM_ACTIONS: usize;
    /// Network input as (planes, rows, cols).
    const OBS_SHAPE: [usize; 3];
    const OBS_SIZE: usize = Self::OBS_SHAPE[0] * Self::OBS_SHAPE[1] * Self::OBS_SHAPE[2];
    /// Ply cap for self-play; a game cut off here never observes its outcome.
    const MAX_PLIES: usize;

    fn new() -> Self;
    /// Player to move (0 or 1).  Meaningless once terminal.
    fn current_player(&self) -> usize;
    /// Clears `out` and fills it with the legal actions in ascending order.
    fn legal_actions(&self, out: &mut Vec<u16>);
    fn apply(&mut self, action: u16);
    /// `Some(+1 / 0 / -1)` from player 0's perspective once the game is over.
    fn outcome(&self) -> Option<i8>;
    /// Writes `OBS_SIZE` floats, from the perspective of the player to move.
    fn encode(&self, out: &mut [f32]);
    fn render(&self) -> String;

    fn is_terminal(&self) -> bool {
        self.outcome().is_some()
    }

    /// Terminal result for `player`: +1 win, 0 draw, -1 loss.
    fn result_for(&self, player: usize) -> Option<i8> {
        self.outcome().map(|r| if player == 0 { r } else { -r })
    }

    /// Per-square ownership for `player` (+1 theirs, -1 the opponent's, 0
    /// empty) into `out` (OBS_SHAPE[1] * OBS_SHAPE[2] floats); false if the
    /// game has no such notion.
    fn ownership(&self, _player: usize, _out: &mut [f32]) -> bool {
        false
    }
}
