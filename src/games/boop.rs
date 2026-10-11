//! Boop (6x6), the "faithful" rules from `boop_thompson_worker.py`.
//!
//! * Each player owns 8 pieces, starting as 8 kittens in hand.
//! * A turn places a kitten or a cat on an empty cell; the placed piece
//!   "boops" every adjacent piece one step away.  Kittens push only kittens,
//!   cats push both.  A piece pushed off the board returns to its owner's hand;
//!   a push into an occupied cell does nothing.
//! * Win: three own cats in a line (any direction), or all eight pieces cats
//!   on the board.
//! * Graduation: three own pieces in a line (kittens and/or cats) leave the
//!   board and return to hand as three cats.
//! * A player with an empty hand must graduate one of its kittens on the board
//!   instead of placing (it returns to hand as a cat).
//! * 500 plies without a winner is a draw.
//!
//! Actions: `piece * 36 + cell` for placements (piece 0 = kitten, 1 = cat),
//! `72 + cell` for a forced graduation.
//!
//! One deliberate difference from the Python original: when several lines
//! qualify for graduation at once, Python picked one uniformly at random,
//! which made the game stochastic (the same action could lead to different
//! states), quietly invalidating tree reuse and the MCTS-solver proofs.  Here
//! the line graduating the most kittens is chosen (ties: first in scan
//! order), which keeps transitions deterministic and is the choice a player
//! would almost always make anyway.

use super::Game;
use std::sync::OnceLock;

const N: usize = 6;
const CELLS: usize = N * N;
const BOARD: u64 = (1u64 << CELLS) - 1;
const PIECES: u8 = 8;
const GRADUATE: u16 = 2 * CELLS as u16;
const MAX_MOVES: u16 = 500;

/// Every 3-in-a-row on the board as a bitmask, in (row, col, direction) order.
fn lines() -> &'static [u64] {
    static LINES: OnceLock<Vec<u64>> = OnceLock::new();
    LINES.get_or_init(|| {
        let mut v = Vec::new();
        for r in 0..N as i32 {
            for c in 0..N as i32 {
                for (dr, dc) in [(0, 1), (1, 0), (1, 1), (1, -1)] {
                    let (er, ec) = (r + 2 * dr, c + 2 * dc);
                    if (0..N as i32).contains(&er) && (0..N as i32).contains(&ec) {
                        v.push((0..3).fold(0u64, |m, k| m | 1u64 << ((r + k * dr) * N as i32 + c + k * dc)));
                    }
                }
            }
        }
        v
    })
}

#[derive(Clone, Debug)]
pub struct Boop {
    kittens: [u64; 2],
    cats: [u64; 2],
    hand: [[u8; 2]; 2], // [player][kitten, cat]
    to_move: u8,
    moves: u16,
    outcome: Option<i8>,
}

impl Boop {
    fn occupied(&self) -> u64 {
        self.kittens[0] | self.kittens[1] | self.cats[0] | self.cats[1]
    }

    fn boop(&mut self, r: i32, c: i32, is_cat: bool) {
        let on_board = |r: i32, c: i32| (0..N as i32).contains(&r) && (0..N as i32).contains(&c);
        for dr in -1..=1 {
            for dc in -1..=1 {
                if (dr, dc) == (0, 0) || !on_board(r + dr, c + dc) {
                    continue;
                }
                let from = 1u64 << ((r + dr) * N as i32 + c + dc);
                let Some((owner, cat)) = (0..2).find_map(|q| {
                    if self.kittens[q] & from != 0 {
                        Some((q, false))
                    } else if self.cats[q] & from != 0 {
                        Some((q, true))
                    } else {
                        None
                    }
                }) else {
                    continue;
                };
                if cat && !is_cat {
                    continue; // kittens cannot push cats
                }
                let (tr, tc) = (r + 2 * dr, c + 2 * dc);
                let dest = if on_board(tr, tc) {
                    1u64 << (tr * N as i32 + tc)
                } else {
                    0
                };
                if dest != 0 && self.occupied() & dest != 0 {
                    continue; // blocked: no chain reaction
                }
                let set = if cat {
                    &mut self.cats[owner]
                } else {
                    &mut self.kittens[owner]
                };
                *set = (*set & !from) | dest;
                if dest == 0 {
                    self.hand[owner][cat as usize] += 1; // pushed off the edge
                }
            }
        }
    }

    fn has_won(&self, p: usize) -> bool {
        let cats = self.cats[p];
        cats.count_ones() >= PIECES as u32 || lines().iter().any(|&l| l & !cats == 0)
    }

    /// Graduate qualifying lines one at a time: clearing one line can break
    /// an overlapping one, while an independent line still graduates.
    fn graduate_lines(&mut self, p: usize) {
        loop {
            let mine = self.kittens[p] | self.cats[p];
            let mut best: Option<(u32, u64)> = None;
            for &l in lines() {
                let k = (self.kittens[p] & l).count_ones();
                if mine & l == l && best.is_none_or(|(bk, _)| k > bk) {
                    best = Some((k, l));
                }
            }
            let Some((_, line)) = best else { return };
            self.kittens[p] &= !line;
            self.cats[p] &= !line;
            self.hand[p][1] += 3;
        }
    }

    fn check_wins(&mut self, p: usize) -> bool {
        for q in [p, 1 - p] {
            if self.has_won(q) {
                self.outcome = Some(if q == 0 { 1 } else { -1 });
                return true;
            }
        }
        false
    }

    /// Everything after a placement or graduation: the move cap, wins (mover
    /// first), line graduation for both players, wins again, hand-off.
    fn resolve(&mut self, p: usize) {
        if self.moves >= MAX_MOVES {
            self.outcome = Some(0);
            return;
        }
        if self.check_wins(p) {
            return;
        }
        self.graduate_lines(p);
        self.graduate_lines(1 - p);
        if self.check_wins(p) {
            return;
        }
        self.to_move ^= 1;
        let q = self.to_move as usize;
        if self.hand[q] == [0, 0] && self.kittens[q] == 0 {
            self.outcome = Some(0); // defensive: cannot happen under these rules
        }
    }
}

impl Game for Boop {
    const NAME: &'static str = "boop";
    const NUM_ACTIONS: usize = 3 * CELLS;
    /// Planes: my kittens, my cats, opponent kittens, opponent cats, empty,
    /// then four constant planes for the hands (my kittens, my cats, opponent
    /// kittens, opponent cats), each divided by 8.
    const OBS_SHAPE: [usize; 3] = [9, N, N];
    const MAX_PLIES: usize = MAX_MOVES as usize;

    fn new() -> Self {
        Boop {
            kittens: [0; 2],
            cats: [0; 2],
            hand: [[PIECES, 0]; 2],
            to_move: 0,
            moves: 0,
            outcome: None,
        }
    }

    fn current_player(&self) -> usize {
        self.to_move as usize
    }

    fn legal_actions(&self, out: &mut Vec<u16>) {
        out.clear();
        if self.outcome.is_some() {
            return;
        }
        let p = self.to_move as usize;
        let [hk, hc] = self.hand[p];
        let push_bits = |out: &mut Vec<u16>, mut b: u64, base: u16| {
            while b != 0 {
                out.push(base + b.trailing_zeros() as u16);
                b &= b - 1;
            }
        };
        if hk == 0 && hc == 0 {
            push_bits(out, self.kittens[p], GRADUATE);
            return;
        }
        let empty = !self.occupied() & BOARD;
        if hk > 0 {
            push_bits(out, empty, 0);
        }
        if hc > 0 {
            push_bits(out, empty, CELLS as u16);
        }
    }

    fn apply(&mut self, action: u16) {
        debug_assert!(self.outcome.is_none());
        let p = self.to_move as usize;
        if action >= GRADUATE {
            let bit = 1u64 << (action - GRADUATE);
            debug_assert!(self.kittens[p] & bit != 0);
            self.kittens[p] &= !bit;
            self.hand[p][1] += 1;
        } else {
            let is_cat = action as usize >= CELLS;
            let cell = action as usize % CELLS;
            let bit = 1u64 << cell;
            debug_assert!(self.occupied() & bit == 0);
            self.hand[p][is_cat as usize] -= 1;
            if is_cat {
                self.cats[p] |= bit;
            } else {
                self.kittens[p] |= bit;
            }
            self.boop((cell / N) as i32, (cell % N) as i32, is_cat);
        }
        self.moves += 1;
        self.resolve(p);
    }

    fn outcome(&self) -> Option<i8> {
        self.outcome
    }

    fn encode(&self, out: &mut [f32]) {
        let p = self.to_move as usize;
        let o = 1 - p;
        let planes = [
            self.kittens[p],
            self.cats[p],
            self.kittens[o],
            self.cats[o],
            !self.occupied() & BOARD,
        ];
        for (k, b) in planes.iter().enumerate() {
            for i in 0..CELLS {
                out[k * CELLS + i] = (b >> i & 1) as f32;
            }
        }
        let hands = [self.hand[p][0], self.hand[p][1], self.hand[o][0], self.hand[o][1]];
        for (k, &h) in hands.iter().enumerate() {
            out[(5 + k) * CELLS..(6 + k) * CELLS].fill(h as f32 / PIECES as f32);
        }
    }

    fn render(&self) -> String {
        let mut s = String::new();
        for r in 0..N {
            for c in 0..N {
                let bit = 1u64 << (r * N + c);
                s.push(if self.kittens[0] & bit != 0 {
                    'k'
                } else if self.cats[0] & bit != 0 {
                    'K'
                } else if self.kittens[1] & bit != 0 {
                    'o'
                } else if self.cats[1] & bit != 0 {
                    'O'
                } else {
                    '.'
                });
            }
            s.push('\n');
        }
        s.push_str(&format!(
            "P0: {}k {}K  P1: {}k {}K  move={}\n",
            self.hand[0][0], self.hand[0][1], self.hand[1][0], self.hand[1][1], self.moves
        ));
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(r: usize, c: usize) -> u16 {
        (r * N + c) as u16
    }

    #[test]
    fn placement_boops_neighbours_and_edges_return_to_hand() {
        let mut s = Boop::new();
        s.apply(cell(0, 0)); // p0 kitten in the corner
        s.apply(cell(1, 1)); // p1 kitten diagonally next to it: pushes it off
        assert_eq!(s.kittens[0], 0);
        assert_eq!(s.hand[0], [PIECES, 0]);
        s.apply(cell(1, 2)); // p0 kitten pushes p1's (1,1) kitten to (1,0)
        assert_eq!(s.kittens[1], 1u64 << cell(1, 0));
    }

    #[test]
    fn kittens_cannot_push_cats_and_blocked_pieces_stay() {
        let mut s = Boop::new();
        s.cats[1] = 1u64 << cell(2, 2);
        s.kittens[1] = 1u64 << cell(2, 4);
        s.apply(cell(2, 3)); // p0 kitten between them
        assert_eq!(s.cats[1], 1u64 << cell(2, 2)); // cat not pushed
        assert_eq!(s.kittens[1], 1u64 << cell(2, 5)); // kitten pushed east
        s.kittens[1] |= 1u64 << cell(4, 4);
        s.kittens[0] |= 1u64 << cell(5, 5);
        s.to_move = 0;
        s.apply(cell(3, 3)); // pushes (4,4) toward (5,5): occupied, stays
        assert!(s.kittens[1] & (1u64 << cell(4, 4)) != 0);
    }

    #[test]
    fn three_in_a_row_graduates() {
        let mut s = Boop::new();
        s.kittens[0] = (1u64 << cell(0, 0)) | (1u64 << cell(0, 1)) | (1u64 << cell(3, 3));
        s.cats[0] = 1u64 << cell(4, 0);
        s.hand[0] = [4, 0];
        s.apply(cell(0, 2)); // completes (0,0)(0,1)(0,2): all three graduate
        assert_eq!(s.kittens[0], 1u64 << cell(3, 3));
        assert_eq!(s.hand[0], [3, 3]);
        assert_eq!(s.outcome, None);
        assert_eq!(s.to_move, 1);
    }

    #[test]
    fn three_cats_in_a_row_win() {
        let mut s = Boop::new();
        s.cats[1] = (1u64 << cell(3, 0)) | (1u64 << cell(3, 1));
        s.hand[1] = [6, 1];
        s.to_move = 1;
        // the cat at (3,2) pushes (3,1) toward (3,0): occupied, so it stays
        s.apply(CELLS as u16 + cell(3, 2));
        assert_eq!(s.outcome, Some(-1));
    }

    #[test]
    fn empty_hand_forces_graduation() {
        let mut s = Boop::new();
        s.hand[0] = [0, 0];
        s.kittens[0] = (1u64 << cell(0, 0)) | (1u64 << cell(2, 4));
        let mut a = Vec::new();
        s.legal_actions(&mut a);
        assert_eq!(a, vec![GRADUATE + cell(0, 0), GRADUATE + cell(2, 4)]);
        s.apply(GRADUATE + cell(2, 4));
        assert_eq!(s.hand[0], [0, 1]);
        assert_eq!(s.kittens[0], 1);
        assert_eq!(s.to_move, 1);
    }

    #[test]
    fn random_games_terminate_and_conserve_pieces() {
        use rand::{rngs::SmallRng, Rng, SeedableRng};
        let mut rng = SmallRng::seed_from_u64(1);
        let mut acts = Vec::new();
        for _ in 0..200 {
            let mut s = Boop::new();
            while !s.is_terminal() {
                for p in 0..2 {
                    let on_board = (s.kittens[p] | s.cats[p]).count_ones() as u8;
                    assert_eq!(on_board + s.hand[p][0] + s.hand[p][1], PIECES);
                }
                s.legal_actions(&mut acts);
                s.apply(acts[rng.random_range(0..acts.len())]);
            }
        }
    }
}
