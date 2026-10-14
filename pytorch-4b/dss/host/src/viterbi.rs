// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Exact-rational port of the probability-domain Viterbi decoder in
//! sovereign-ruby-kernel `lib/golog/dyna_golog_viterbi.pl` (sections 1-3 and
//! the three domain models of section 7). Probabilities are read from their
//! decimal text exactly (0.6 = 3/5) and all products/comparisons are exact.
//!
//! Faithful to the Prolog control flow:
//! - trellis init: w(0,q) = start(q) * emit(q, o0), missing facts = 0;
//! - step: candidates scanned in the given state order, starting from
//!   (0, none), replaced only on strict `>` (first maximum wins);
//! - final: same strict-`>` scan starting from (none, 0);
//! - all weights 0 with T > 1: the Prolog predicate fails ("no path").
//! Divergences (documented in RUBY_KERNEL_PORTS.md):
//! - Prolog's `viterbi_traceback/4` returns an unbound first element (the
//!   time-0 state is never bound); this port returns the time-0 state from
//!   the time-1 backpointer, like `reconstruct_path` in `_v1.py`;
//! - T = 1 with all weights 0: Prolog returns `[_]`, weight 0.0; this port
//!   returns "no path";
//! - exact rationals vs IEEE doubles: on exact ties the first state wins
//!   here, while float rounding may break the tie differently in Prolog.

use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::{One, Zero};

pub type Q = BigRational;

/// Exact rational from a plain decimal "0.6", "1.0", "0.05", "1", "0".
pub fn dec(s: &str) -> Result<Q, String> {
    let (ip, fp) = match s.split_once('.') { Some((a, b)) => (a, b), None => (s, "") };
    if ip.is_empty() || !ip.bytes().all(|b| b.is_ascii_digit()) || !fp.bytes().all(|b| b.is_ascii_digit())
        || (s.contains('.') && fp.is_empty()) || fp.len() > 30 || ip.len() > 30 {
        return Err(format!("not a plain decimal: {s:?}"));
    }
    let num: BigInt = format!("{ip}{fp}").parse().map_err(|_| "bad decimal".to_string())?;
    let den = BigInt::from(10u32).pow(fp.len() as u32);
    Ok(Q::new(num, den))
}

#[derive(Clone, Debug)]
pub struct Hmm {
    pub start: Vec<(String, Q)>,
    pub trans: Vec<(String, String, Q)>,
    pub emit: Vec<(String, String, Q)>,
}

impl Hmm {
    // First matching fact, like `( hmm_start(Q, W) -> true ; W = 0.0 )`.
    fn start(&self, q: &str) -> Q { self.start.iter().find(|x| x.0 == q).map(|x| x.1.clone()).unwrap_or_else(Q::zero) }
    fn trans(&self, a: &str, b: &str) -> Q {
        self.trans.iter().find(|x| x.0 == a && x.1 == b).map(|x| x.2.clone()).unwrap_or_else(Q::zero)
    }
    fn emit(&self, q: &str, o: &str) -> Q {
        self.emit.iter().find(|x| x.0 == q && x.1 == o).map(|x| x.2.clone()).unwrap_or_else(Q::zero)
    }
    /// `assert_*` range check: every probability in [0, 1].
    pub fn validate(&self) -> Result<(), String> {
        let one = Q::one();
        let all = self.start.iter().map(|x| &x.1).chain(self.trans.iter().map(|x| &x.2)).chain(self.emit.iter().map(|x| &x.2));
        for p in all { if *p < Q::zero() || *p > one { return Err(format!("invalid_prob {p}")); } }
        Ok(())
    }
    pub fn symbols(&self) -> Vec<&str> {
        let mut v: Vec<&str> = vec![];
        for e in &self.emit { if !v.contains(&e.1.as_str()) { v.push(&e.1) } }
        v
    }
}

#[derive(Debug, PartialEq)]
pub enum Decode { Path { path: Vec<String>, weight: Q }, NoPath }

pub fn decode(h: &Hmm, obs: &[&str], states: &[&str]) -> Decode {
    if obs.is_empty() || states.is_empty() { return Decode::NoPath; }
    let n = states.len();
    let mut w: Vec<Q> = states.iter().map(|q| h.start(q) * h.emit(q, obs[0])).collect();
    let mut bp: Vec<Vec<Option<usize>>> = vec![vec![None; n]];
    for o in &obs[1..] {
        let mut nw = Vec::with_capacity(n);
        let mut nb = Vec::with_capacity(n);
        for q in states {
            let e = h.emit(q, o);
            let (mut best, mut arg) = (Q::zero(), None);
            for (j, qp) in states.iter().enumerate() {
                let c = &w[j] * h.trans(qp, q) * &e;
                if c > best { best = c; arg = Some(j); }
            }
            nw.push(best);
            nb.push(arg);
        }
        w = nw;
        bp.push(nb);
    }
    let (mut best, mut arg) = (Q::zero(), None);
    for j in 0..n { if w[j] > best { best = w[j].clone(); arg = Some(j); } }
    let Some(mut cur) = arg else { return Decode::NoPath };
    let mut rev = vec![cur];
    for t in (1..obs.len()).rev() {
        cur = bp[t][cur].expect("positive weight implies a backpointer");
        rev.push(cur);
    }
    rev.reverse();
    Decode::Path { path: rev.into_iter().map(|i| states[i].to_string()).collect(), weight: best }
}

pub fn fmt(d: &Decode) -> String {
    match d {
        Decode::Path { path, weight } => format!("path={} weight={}", path.join(","), weight),
        Decode::NoPath => "no path (all weights 0)".into(),
    }
}

/// Correctly rounded f64 of a non-negative rational (for parity output).
pub fn to_f64(q: &Q) -> f64 {
    use num_traits::ToPrimitive;
    if q.is_zero() { return 0.0; }
    // scale to 2^k so the integer quotient has ~80 significant bits
    let (n, d) = (q.numer().clone(), q.denom().clone());
    let shift = 80i64 - (n.bits() as i64 - d.bits() as i64);
    let (a, b) = if shift >= 0 { (n << (shift as usize), d) } else { (n, d << ((-shift) as usize)) };
    let quo = &a / &b;
    let rem_nonzero = !(&a % &b).is_zero();
    let mut m = quo.to_u128().unwrap();
    if rem_nonzero { m |= 1; } // sticky bit for correct rounding
    // m has 80-81 bits; u128 -> f64 rounds to nearest-even
    let f = m as f64;
    f * 2f64.powi(-(shift as i32))
}

fn model(start: &[(&str, &str)], trans: &[(&str, &str, &str)], emit: &[(&str, &str, &str)]) -> Hmm {
    Hmm {
        start: start.iter().map(|(s, p)| (s.to_string(), dec(p).unwrap())).collect(),
        trans: trans.iter().map(|(a, b, p)| (a.to_string(), b.to_string(), dec(p).unwrap())).collect(),
        emit: emit.iter().map(|(a, b, p)| (a.to_string(), b.to_string(), dec(p).unwrap())).collect(),
    }
}

/// The three domain models of `dyna_golog_viterbi.pl` section 7, with the
/// state order used by its demo. POS start weights are `1.0/5.0` in the
/// source, i.e. exactly 1/5 (the float 0.2 differs from 1/5 by < 2^-55).
pub fn builtin(name: &str) -> Option<(Hmm, Vec<&'static str>)> {
    match name {
        "toy_weather" => Some((model(
            &[("sunny", "0.6"), ("rainy", "0.4")],
            &[("sunny", "sunny", "0.7"), ("sunny", "rainy", "0.3"), ("rainy", "sunny", "0.4"), ("rainy", "rainy", "0.6")],
            &[("sunny", "walk", "0.1"), ("sunny", "shop", "0.4"), ("sunny", "clean", "0.5"),
              ("rainy", "walk", "0.6"), ("rainy", "shop", "0.3"), ("rainy", "clean", "0.1")]),
            vec!["sunny", "rainy"])),
        "pos_tagger" => {
            let st = ["det", "nn", "vb", "in", "punct"];
            let mut h = model(&[],
                &[("det","nn","0.8"),("det","vb","0.05"),("det","in","0.05"),("det","det","0.05"),("det","punct","0.05"),
                  ("nn","vb","0.4"),("nn","in","0.3"),("nn","punct","0.2"),("nn","nn","0.05"),("nn","det","0.05"),
                  ("vb","det","0.3"),("vb","nn","0.2"),("vb","in","0.3"),("vb","punct","0.15"),("vb","vb","0.05"),
                  ("in","det","0.6"),("in","nn","0.3"),("in","punct","0.05"),("in","vb","0.03"),("in","in","0.02"),
                  ("punct","det","0.4"),("punct","nn","0.3"),("punct","vb","0.2"),("punct","in","0.05"),("punct","punct","0.05")],
                &[("det","the","0.6"),("det","a","0.3"),("det","an","0.1"),
                  ("nn","cat","0.2"),("nn","dog","0.2"),("nn","man","0.15"),("nn","woman","0.15"),
                  ("nn","park","0.1"),("nn","house","0.1"),("nn","car","0.1"),
                  ("vb","saw","0.25"),("vb","walked","0.25"),("vb","ate","0.2"),("vb","ran","0.15"),("vb","is","0.15"),
                  ("in","in","0.4"),("in","on","0.3"),("in","with","0.2"),("in","by","0.1"),
                  ("punct",".","0.7"),("punct","!","0.2"),("punct","?","0.1")]);
            h.start = st.iter().map(|s| (s.to_string(), Q::new(1.into(), 5.into()))).collect();
            Some((h, st.to_vec()))
        }
        "cpg_island" => Some((model(
            &[("island", "0.2"), ("ocean", "0.8")],
            &[("island","island","0.95"),("island","ocean","0.05"),("ocean","ocean","0.95"),("ocean","island","0.05")],
            &[("island","A","0.15"),("island","C","0.35"),("island","G","0.35"),("island","T","0.15"),
              ("ocean","A","0.30"),("ocean","C","0.20"),("ocean","G","0.20"),("ocean","T","0.30")]),
            vec!["island", "ocean"])),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn source_tests() {
        // test_toy_weather: [walk, shop, walk], length 3, weight > 0.
        let (h, st) = builtin("toy_weather").unwrap();
        let d = decode(&h, &["walk", "shop", "walk"], &st);
        assert_eq!(fmt(&d), "path=rainy,rainy,rainy weight=243/15625");
        // test_golog_wired: [walk, shop, clean]; Prolog prints [_,sunny,sunny] 0.013440
        let d = decode(&h, &["walk", "shop", "clean"], &st);
        assert_eq!(fmt(&d), "path=rainy,sunny,sunny weight=42/3125");
        assert_eq!(to_f64(&dec("0.1").unwrap()), 0.1);
        assert_eq!(to_f64(&Q::new(243.into(), 15625.into())), 243.0 / 15625.0);
    }
}

/// True if any argmax in the trellis (step or final) had two candidates
/// exactly equal to a positive maximum: there the first-in-order rule picks
/// the winner, and a float implementation may pick differently.
pub fn has_exact_tie(h: &Hmm, obs: &[&str], states: &[&str]) -> bool {
    if obs.is_empty() { return false; }
    let tie = |c: &[Q]| { let m = c.iter().max().unwrap(); *m > Q::zero() && c.iter().filter(|x| *x == m).count() > 1 };
    let mut w: Vec<Q> = states.iter().map(|q| h.start(q) * h.emit(q, obs[0])).collect();
    let mut any = false;
    for o in &obs[1..] {
        let mut nw = vec![];
        for q in states {
            let e = h.emit(q, o);
            let c: Vec<Q> = states.iter().enumerate().map(|(j, qp)| &w[j] * h.trans(qp, q) * &e).collect();
            any |= tie(&c);
            nw.push(c.into_iter().max().unwrap());
        }
        w = nw;
    }
    any || tie(&w)
}
