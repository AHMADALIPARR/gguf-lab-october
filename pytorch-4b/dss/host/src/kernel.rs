// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! DSS kernel (CPU). A set S of positive integers is DSS when any two subsets
//! with equal sum have equal size. Port of dss/oracle.py: an incremental
//! sum -> subset-size table (card[s], -1 = unreachable), updated like a 0/1
//! knapsack, instead of enumerating 2^n subsets per candidate.

use std::collections::HashMap;

pub const NONE: i16 = -1;

pub struct DssTable {
    pub elems: Vec<u64>,
    pub card: Vec<i16>,
}

impl DssTable {
    pub fn new() -> Self {
        DssTable { elems: vec![], card: vec![0] }
    }

    /// None if S + {x} stays DSS, else Some((sum, size_a, size_b)).
    pub fn conflict(&self, x: u64) -> Option<(u64, i16, i16)> {
        let n = self.card.len() as u64;
        for (s, &c) in self.card.iter().enumerate() {
            if c == NONE { continue; }
            let t = s as u64 + x;
            if t >= n { break; }
            let ct = self.card[t as usize];
            if ct != NONE && ct != c + 1 {
                return Some((t, ct, c + 1));
            }
        }
        None
    }

    pub fn add(&mut self, x: u64) -> bool {
        if x == 0 || self.elems.contains(&x) || self.conflict(x).is_some() {
            return false;
        }
        let old = self.card.len();
        let mut new = vec![NONE; old + x as usize];
        new[..old].copy_from_slice(&self.card);
        for s in 0..old {
            if self.card[s] != NONE {
                new[s + x as usize] = self.card[s] + 1;
            }
        }
        self.card = new;
        self.elems.push(x);
        true
    }

    pub fn cardinality(&self, s: u64) -> i64 {
        if (s as usize) < self.card.len() { self.card[s as usize] as i64 } else { -1 }
    }
}

pub fn build(s: &[u64]) -> Option<DssTable> {
    let mut t = DssTable::new();
    for &x in s {
        if !t.add(x) { return None; }
    }
    Some(t)
}

pub fn is_dss(s: &[u64]) -> bool {
    build(s).is_some()
}

/// Same search order as oracle.collision_witness, so witnesses are identical.
pub fn collision_witness(s: &[u64]) -> Option<(Vec<u64>, Vec<u64>)> {
    let mut order: Vec<(u64, Vec<u64>)> = vec![(0, vec![])];
    let mut idx: HashMap<u64, usize> = HashMap::from([(0, 0)]);
    for &x in s {
        let snapshot = order.clone();
        for (sum, sub) in snapshot {
            let ns = sum + x;
            let mut nsub = sub.clone();
            nsub.push(x);
            if let Some(&i) = idx.get(&ns) {
                if order[i].1.len() != nsub.len() {
                    return Some((order[i].1.clone(), nsub));
                }
            } else {
                idx.insert(ns, order.len());
                order.push((ns, nsub));
            }
        }
    }
    None
}

pub fn greedy_next(s: &[u64]) -> Option<u64> {
    let t = build(s)?;
    let mut x = *s.last()? + 1;
    while t.conflict(x).is_some() { x += 1; }
    Some(x)
}

pub fn quadratic_bound(k: u64) -> u64 { (k * k + 1) / 2 }

fn join(v: &[u64], sep: &str) -> String {
    v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(sep)
}

/// Tool results, formatted exactly like dss/agent.py run_tool.
pub fn tool_isdss(s: &[u64]) -> String {
    if is_dss(s) { return "yes".into(); }
    let (mut a, mut b) = collision_witness(s).expect("non-DSS set has a witness");
    if a.len() > b.len() { std::mem::swap(&mut a, &mut b); }
    format!("no {}={}", join(&a, "+"), join(&b, "+"))
}

pub fn tool_card(w: &[u64], sum: u64) -> Option<String> {
    Some(build(w)?.cardinality(sum).to_string())
}

pub fn tool_next(s: &[u64]) -> Option<String> {
    Some(greedy_next(s)?.to_string())
}

pub fn tool_bound(k: u64, a: u64) -> String {
    let b = quadratic_bound(k);
    if a >= b { format!("b={b}; {a}>={b} yes") } else { format!("b={b}; {a}<{b} no") }
}
