//! Connect Four (6 rows x 7 columns), same rules and action ids as
//! open_spiel's `connect_four`: action = column, first to four in a row wins,
//! a full board is a draw.
//!
//! Bitboard layout: bit `col * 7 + row` (row 0 = bottom).  Each column has one
//! spare bit on top so vertical/diagonal shifts never wrap into the next column.

use super::Game;

const ROWS: usize = 6;
const COLS: usize = 7;
const STRIDE: usize = ROWS + 1;

#[derive(Clone, Debug)]
pub struct Connect4 {
    stones: [u64; 2],
    heights: [u8; COLS],
    to_move: u8,
    plies: u8,
    outcome: Option<i8>,
}

fn has_four(b: u64) -> bool {
    for d in [1, STRIDE, STRIDE - 1, STRIDE + 1] {
        let m = b & (b >> d);
        if m & (m >> (2 * d)) != 0 {
            return true;
        }
    }
    false
}

impl Game for Connect4 {
    const NAME: &'static str = "connect_four";
    const NUM_ACTIONS: usize = COLS;
    const OBS_SHAPE: [usize; 3] = [3, ROWS, COLS];
    const MAX_PLIES: usize = ROWS * COLS;

    fn new() -> Self {
        Connect4 {
            stones: [0; 2],
            heights: [0; COLS],
            to_move: 0,
            plies: 0,
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
        out.extend((0..COLS as u16).filter(|&c| (self.heights[c as usize] as usize) < ROWS));
    }

    fn apply(&mut self, action: u16) {
        let c = action as usize;
        let p = self.to_move as usize;
        debug_assert!(self.outcome.is_none() && (self.heights[c] as usize) < ROWS);
        self.stones[p] |= 1u64 << (c * STRIDE + self.heights[c] as usize);
        self.heights[c] += 1;
        self.plies += 1;
        if has_four(self.stones[p]) {
            self.outcome = Some(if p == 0 { 1 } else { -1 });
        } else if self.plies as usize == ROWS * COLS {
            self.outcome = Some(0);
        }
        self.to_move ^= 1;
    }

    fn outcome(&self) -> Option<i8> {
        self.outcome
    }

    /// Planes: mover's discs, opponent's discs, empty.  Row 0 of the tensor is
    /// the top of the board.
    fn encode(&self, out: &mut [f32]) {
        let me = self.stones[self.to_move as usize];
        let opp = self.stones[1 - self.to_move as usize];
        let plane = ROWS * COLS;
        for r in 0..ROWS {
            for c in 0..COLS {
                let bit = 1u64 << (c * STRIDE + r);
                let i = (ROWS - 1 - r) * COLS + c;
                let (m, o) = (me & bit != 0, opp & bit != 0);
                out[i] = m as u8 as f32;
                out[plane + i] = o as u8 as f32;
                out[2 * plane + i] = (!m && !o) as u8 as f32;
            }
        }
    }

    fn render(&self) -> String {
        let mut s = String::new();
        for r in (0..ROWS).rev() {
            for c in 0..COLS {
                let bit = 1u64 << (c * STRIDE + r);
                s.push(if self.stones[0] & bit != 0 {
                    'x'
                } else if self.stones[1] & bit != 0 {
                    'o'
                } else {
                    '.'
                });
            }
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(s: &Connect4, depth: usize) -> u64 {
        if depth == 0 || s.is_terminal() {
            return 1;
        }
        let mut acts = Vec::new();
        s.legal_actions(&mut acts);
        acts.iter()
            .map(|&a| {
                let mut c = s.clone();
                c.apply(a);
                perft(&c, depth - 1)
            })
            .sum()
    }

    #[test]
    fn perft_before_any_win_is_seven_to_the_n() {
        let s = Connect4::new();
        for d in 0..=6 {
            assert_eq!(perft(&s, d), 7u64.pow(d as u32));
        }
    }

    #[test]
    fn detects_lines() {
        let mut s = Connect4::new();
        for a in [0, 1, 0, 1, 0, 1, 0] {
            s.apply(a); // vertical four for player 0
        }
        assert_eq!(s.outcome(), Some(1));
        let mut s = Connect4::new();
        for a in [0, 0, 1, 1, 2, 2, 6, 3, 6, 3, 6, 3] {
            s.apply(a); // horizontal four for player 1 on row 1
        }
        assert_eq!(s.outcome(), Some(-1));
        let mut s = Connect4::new();
        // diagonal for player 0: (0,0) (1,1) (2,2) (3,3)
        for a in [0, 1, 1, 2, 2, 3, 2, 3, 3, 6, 3] {
            s.apply(a);
        }
        assert_eq!(s.outcome(), Some(1));
    }

    #[test]
    fn encode_is_mover_relative() {
        let mut s = Connect4::new();
        s.apply(3);
        let mut o = vec![0.0; Connect4::OBS_SIZE];
        s.encode(&mut o);
        // player 1 to move: the disc belongs to the opponent plane
        assert_eq!(o[42 + 5 * 7 + 3], 1.0);
        assert_eq!(o[5 * 7 + 3], 0.0);
    }
}
