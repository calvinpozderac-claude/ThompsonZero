//! Othello on 8x8, same rules and action ids as open_spiel's `othello`:
//! action = row * 8 + col, 64 = pass (legal only when nothing else is),
//! player 0 is black and moves first, the game ends when neither side can
//! place a disc, and the disc majority wins.

use super::Game;

const PASS: u16 = 64;
const FILE_A: u64 = 0x0101_0101_0101_0101; // col 0
const FILE_H: u64 = 0x8080_8080_8080_8080; // col 7

#[derive(Clone, Debug)]
pub struct Othello {
    discs: [u64; 2], // [black, white]
    to_move: u8,
    outcome: Option<i8>,
}

/// Shift every disc one step in direction `d` (0..8), dropping discs that
/// would wrap around a board edge.
#[inline(always)]
fn shift(b: u64, d: usize) -> u64 {
    match d {
        0 => (b << 1) & !FILE_A, // east
        1 => (b >> 1) & !FILE_H, // west
        2 => b << 8,             // south
        3 => b >> 8,             // north
        4 => (b << 9) & !FILE_A, // south-east
        5 => (b << 7) & !FILE_H, // south-west
        6 => (b >> 7) & !FILE_A, // north-east
        _ => (b >> 9) & !FILE_H, // north-west
    }
}

fn moves(me: u64, opp: u64) -> u64 {
    let empty = !(me | opp);
    let mut out = 0;
    for d in 0..8 {
        let mut x = shift(me, d) & opp;
        for _ in 0..5 {
            x |= shift(x, d) & opp;
        }
        out |= shift(x, d) & empty;
    }
    out
}

fn flips(me: u64, opp: u64, sq: u64) -> u64 {
    let mut out = 0;
    for d in 0..8 {
        let mut line = 0;
        let mut x = shift(sq, d);
        while x & opp != 0 {
            line |= x;
            x = shift(x, d);
        }
        if x & me != 0 {
            out |= line;
        }
    }
    out
}

impl Othello {
    /// The position in Edax's `setboard` format: 64 squares from a1 in
    /// row-major order ('X' black, 'O' white, '-' empty), a space, and the
    /// side to move.
    pub fn edax_board(&self) -> String {
        let mut s: String = (0..64)
            .map(|i| {
                let bit = 1u64 << i;
                if self.discs[0] & bit != 0 {
                    'X'
                } else if self.discs[1] & bit != 0 {
                    'O'
                } else {
                    '-'
                }
            })
            .collect();
        s.push_str(if self.to_move == 0 { " X" } else { " O" });
        s
    }
}

impl Game for Othello {
    const NAME: &'static str = "othello";
    const NUM_ACTIONS: usize = 65;
    const OBS_SHAPE: [usize; 3] = [3, 8, 8];
    const MAX_PLIES: usize = 128;

    fn new() -> Self {
        let black = (1u64 << 28) | (1u64 << 35); // (3,4) and (4,3)
        let white = (1u64 << 27) | (1u64 << 36); // (3,3) and (4,4)
        Othello {
            discs: [black, white],
            to_move: 0,
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
        let mut m = moves(self.discs[p], self.discs[1 - p]);
        if m == 0 {
            out.push(PASS);
        }
        while m != 0 {
            out.push(m.trailing_zeros() as u16);
            m &= m - 1;
        }
    }

    fn apply(&mut self, action: u16) {
        debug_assert!(self.outcome.is_none());
        let p = self.to_move as usize;
        self.to_move ^= 1;
        if action == PASS {
            return; // open_spiel checks for the end only after a placement
        }
        let (me, opp) = (self.discs[p], self.discs[1 - p]);
        let sq = 1u64 << action;
        let f = flips(me, opp, sq);
        debug_assert!(f != 0 && (me | opp) & sq == 0, "illegal othello move");
        self.discs[p] = me | sq | f;
        self.discs[1 - p] = opp & !f;
        let (b, w) = (self.discs[0], self.discs[1]);
        if moves(b, w) == 0 && moves(w, b) == 0 {
            let (nb, nw) = (b.count_ones(), w.count_ones());
            self.outcome = Some((nb > nw) as i8 - (nb < nw) as i8);
        }
    }

    fn outcome(&self) -> Option<i8> {
        self.outcome
    }

    /// Planes: mover's discs, opponent's discs, empty.
    fn encode(&self, out: &mut [f32]) {
        let me = self.discs[self.to_move as usize];
        let opp = self.discs[1 - self.to_move as usize];
        for i in 0..64 {
            let bit = 1u64 << i;
            out[i] = (me & bit != 0) as u8 as f32;
            out[64 + i] = (opp & bit != 0) as u8 as f32;
            out[128 + i] = ((me | opp) & bit == 0) as u8 as f32;
        }
    }

    fn ownership(&self, player: usize, out: &mut [f32]) -> bool {
        let (me, opp) = (self.discs[player], self.discs[1 - player]);
        for (i, x) in out[..64].iter_mut().enumerate() {
            let bit = 1u64 << i;
            *x = (me & bit != 0) as u8 as f32 - (opp & bit != 0) as u8 as f32;
        }
        true
    }

    fn render(&self) -> String {
        let mut s = String::from("  a b c d e f g h\n");
        for r in 0..8 {
            s.push_str(&format!("{} ", r + 1));
            for c in 0..8 {
                let bit = 1u64 << (r * 8 + c);
                s.push(if self.discs[0] & bit != 0 {
                    'x'
                } else if self.discs[1] & bit != 0 {
                    'o'
                } else {
                    '-'
                });
                s.push(' ');
            }
            s.push('\n');
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn perft(s: &Othello, depth: usize) -> u64 {
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
    fn perft_matches_known_counts() {
        // Standard Othello perft (passes counted as moves).
        let s = Othello::new();
        let want = [1u64, 4, 12, 56, 244, 1396, 8200, 55092];
        for (d, &n) in want.iter().enumerate() {
            assert_eq!(perft(&s, d), n, "depth {d}");
        }
    }

    #[test]
    fn opening_moves_are_the_standard_four() {
        let mut a = Vec::new();
        Othello::new().legal_actions(&mut a);
        assert_eq!(a, vec![19, 26, 37, 44]);
    }
}
