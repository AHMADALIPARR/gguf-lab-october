// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Strict per-task prompt parser. The host, not the model, extracts the
//! integers. Anything that does not match exactly one task template is
//! rejected: extra or missing spaces, signs, leading zeros, non-ASCII digits,
//! duplicates, unsorted sets, more than 8 elements, values above 10^6.
//! Phi prompts: integers are signed, |x| <= 10^12, exponents 0..=80, at most
//! 8 terms; elements are written "(a + b phi)" or "(a - b phi)".

use crate::phi::ZPhi;

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    IsDss(Vec<u64>),
    Card(Vec<u64>, u64),
    Next(Vec<u64>),
    Bound(u64, u64),
    PhiNormalize(Vec<(i128, u32)>),
    PhiMul(ZPhi, ZPhi),
    PhiConj(ZPhi),
    PhiNorm(ZPhi),
}

impl Task {
    pub fn intent(&self) -> &'static str {
        match self {
            Task::IsDss(_) => "CHECK_IS_DSS",
            Task::Card(..) => "CARD_FROM_SUM",
            Task::Next(_) => "NEXT_GREEDY",
            Task::Bound(..) => "CHECK_BOUND",
            Task::PhiNormalize(_) => "PHI_NORMALIZE",
            Task::PhiMul(..) => "PHI_MUL",
            Task::PhiConj(_) => "PHI_CONJ",
            Task::PhiNorm(_) => "PHI_NORM",
        }
    }
    pub fn tool(&self) -> &'static str {
        match self {
            Task::IsDss(_) => "isdss",
            Task::Card(..) => "card",
            Task::Next(_) => "next",
            Task::Bound(..) => "bound",
            Task::PhiNormalize(_) => "phi_normalize",
            Task::PhiMul(..) => "phi_mul",
            Task::PhiConj(_) => "phi_conj",
            Task::PhiNorm(_) => "phi_norm",
        }
    }
}

const MAX_VAL: u64 = 1_000_000;
const MAX_LEN: usize = 8;

fn int(tok: &str) -> Result<u64, String> {
    if tok.is_empty() || tok.len() > 7 || !tok.bytes().all(|b| b.is_ascii_digit()) {
        return Err(format!("not a plain integer: {tok:?}"));
    }
    if tok.len() > 1 && tok.starts_with('0') {
        return Err(format!("leading zero: {tok:?}"));
    }
    let v: u64 = tok.parse().map_err(|e| format!("{e}"))?;
    if v == 0 || v > MAX_VAL { return Err(format!("out of range: {v}")); }
    Ok(v)
}

fn set(toks: &[&str]) -> Result<Vec<u64>, String> {
    if toks.is_empty() || toks.len() > MAX_LEN { return Err("bad set size".into()); }
    let v: Vec<u64> = toks.iter().map(|t| int(t)).collect::<Result<_, _>>()?;
    if v.windows(2).any(|w| w[0] >= w[1]) {
        return Err("set must be strictly increasing (sorted, no duplicates)".into());
    }
    Ok(v)
}

const PHI_MAX: i128 = 1_000_000_000_000;

/// Signed integer: optional '-', no '+', no leading zeros, |x| <= 10^12.
fn sint(tok: &str) -> Result<i128, String> {
    let (neg, body) = match tok.strip_prefix('-') { Some(b) => (true, b), None => (false, tok) };
    if body.is_empty() || body.len() > 13 || !body.bytes().all(|b| b.is_ascii_digit())
        || (body.len() > 1 && body.starts_with('0')) || (neg && body == "0") {
        return Err(format!("not a plain signed integer: {tok:?}"));
    }
    let v: i128 = body.parse().map_err(|e| format!("{e}"))?;
    if v > PHI_MAX { return Err(format!("out of range: {tok}")); }
    Ok(if neg { -v } else { v })
}

/// Unsigned magnitude for "a + b phi" / "a - b phi" (b written without sign).
fn uint(tok: &str) -> Result<i128, String> {
    if tok.starts_with('-') { return Err(format!("unexpected sign: {tok:?}")); }
    sint(tok)
}

/// "(a + b phi)" or "(a - b phi)" as 5 whitespace tokens: "(a", "+", "b", "phi)".
fn elem(t: &[&str]) -> Result<ZPhi, String> {
    if t.len() != 4 || !t[0].starts_with('(') || t[3] != "phi)" {
        return Err("element must look like (a + b phi)".into());
    }
    let a = sint(&t[0][1..])?;
    let b = uint(t[2])?;
    match t[1] { "+" => Ok(ZPhi { a, b }), "-" => Ok(ZPhi { a, b: -b }), _ => Err("bad sign".into()) }
}

/// Polynomial "c1 phi^k1 + c2 phi^k2 - c3 ..." : first coefficient signed,
/// later ones unsigned after an explicit + / -; a term is "c" or "c phi^k".
fn poly(t: &[&str]) -> Result<Vec<(i128, u32)>, String> {
    let mut terms = vec![];
    let mut i = 0;
    let mut sign = 1i128;
    while i < t.len() {
        let c = if terms.is_empty() { sint(t[i])? } else { uint(t[i])? } * sign;
        i += 1;
        let mut k = 0u32;
        if i < t.len() && t[i].starts_with("phi^") {
            let e = &t[i][4..];
            if e.is_empty() || !e.bytes().all(|b| b.is_ascii_digit()) || (e.len() > 1 && e.starts_with('0')) {
                return Err(format!("bad exponent {:?}", t[i]));
            }
            k = e.parse().map_err(|x| format!("{x}"))?;
            if k > 80 { return Err("exponent above 80".into()); }
            i += 1;
        }
        terms.push((c, k));
        if i < t.len() {
            sign = match t[i] { "+" => 1, "-" => -1, x => return Err(format!("expected + or -, got {x:?}")) };
            i += 1;
            if i == t.len() { return Err("dangling operator".into()); }
        }
    }
    if terms.is_empty() || terms.len() > 8 { return Err("polynomial must have 1..8 terms".into()); }
    Ok(terms)
}

pub fn parse_prompt(p: &str) -> Result<Task, String> {
    if p.contains("  ") || p.starts_with(' ') || p.ends_with(' ') || !p.is_ascii() {
        return Err("irregular whitespace or non-ASCII".into());
    }
    let t: Vec<&str> = p.split(' ').collect();
    let n = t.len();
    let mut found = vec![];
    // is DSS <set> ->
    if n >= 4 && t[0] == "is" && t[1] == "DSS" && t[n - 1] == "->" {
        found.push(set(&t[2..n - 1]).map(Task::IsDss));
    }
    // DSS W <set> sum <s> -> size
    if n >= 7 && t[0] == "DSS" && t[1] == "W" && t[n - 4] == "sum" && t[n - 2] == "->" && t[n - 1] == "size" {
        found.push(set(&t[2..n - 4]).and_then(|w| Ok(Task::Card(w, int(t[n - 3])?))));
    }
    // DSS greedy next after <set> ->
    if n >= 6 && t[0] == "DSS" && t[1] == "greedy" && t[2] == "next" && t[3] == "after" && t[n - 1] == "->" {
        found.push(set(&t[4..n - 1]).map(Task::Next));
    }
    // bound k <k> a <a> ->
    if n == 6 && t[0] == "bound" && t[1] == "k" && t[3] == "a" && t[5] == "->" {
        found.push(int(t[2]).and_then(|k| Ok(Task::Bound(k, int(t[4])?))));
    }
    // phi normalize <poly> ->
    if n >= 4 && t[0] == "phi" && t[1] == "normalize" && t[n - 1] == "->" {
        found.push(poly(&t[2..n - 1]).map(Task::PhiNormalize));
    }
    // phi mul (a + b phi) (c + d phi) ->
    if n == 11 && t[0] == "phi" && t[1] == "mul" && t[n - 1] == "->" {
        found.push(elem(&t[2..6]).and_then(|x| Ok(Task::PhiMul(x, elem(&t[6..10])?))));
    }
    // phi conj (a + b phi) -> ; phi norm (a + b phi) ->
    if n == 7 && t[0] == "phi" && t[1] == "conj" && t[n - 1] == "->" {
        found.push(elem(&t[2..6]).map(Task::PhiConj));
    }
    if n == 7 && t[0] == "phi" && t[1] == "norm" && t[n - 1] == "->" {
        found.push(elem(&t[2..6]).map(Task::PhiNorm));
    }
    match found.len() {
        0 => Err("prompt matches no task template".into()),
        1 => found.pop().unwrap(),
        _ => Err("ambiguous prompt".into()),
    }
}

/// Parse an arg-copy call body "TOOL ARGS" emitted by the model (args mode).
pub fn parse_call(body: &str) -> Result<Task, String> {
    let t: Vec<&str> = body.split(' ').collect();
    match t.first().copied() {
        Some("isdss") => set(&t[1..]).map(Task::IsDss),
        Some("next") => set(&t[1..]).map(Task::Next),
        Some("card") if t.len() >= 4 && t[t.len() - 2] == "sum" => {
            let w = set(&t[1..t.len() - 2])?;
            Ok(Task::Card(w, int(t[t.len() - 1])?))
        }
        Some("bound") if t.len() == 3 => Ok(Task::Bound(int(t[1])?, int(t[2])?)),
        _ => Err(format!("malformed call: {body:?}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strict() {
        assert_eq!(parse_prompt("is DSS 101 202 303 ->"), Ok(Task::IsDss(vec![101, 202, 303])));
        assert_eq!(parse_prompt("bound k 25 a 313 ->"), Ok(Task::Bound(25, 313)));
        assert_eq!(parse_prompt("DSS W 1 2 4 9 sum 11 -> size"), Ok(Task::Card(vec![1, 2, 4, 9], 11)));
        assert!(parse_prompt("is DSS 3  5 ->").is_err());
        assert!(parse_prompt("is DSS 5 3 ->").is_err());
        assert!(parse_prompt("is DSS 05 7 ->").is_err());
        assert!(parse_prompt("is DSS -5 7 ->").is_err());
        assert!(parse_prompt("is DSS 3 5 -> yes").is_err());
        assert!(parse_prompt("bound k 25 a 313 a 4 ->").is_err());
        assert_eq!(parse_prompt("phi norm (3 - 2 phi) ->"), Ok(Task::PhiNorm(ZPhi { a: 3, b: -2 })));
        assert_eq!(parse_prompt("phi normalize 2 phi^5 - 1 phi^2 + 3 ->"),
                   Ok(Task::PhiNormalize(vec![(2, 5), (-1, 2), (3, 0)])));
        assert!(parse_prompt("phi norm (3 + -2 phi) ->").is_err());
        assert!(parse_prompt("phi norm (03 + 2 phi) ->").is_err());
        assert!(parse_prompt("phi normalize 2 phi^5 + ->").is_err());
        assert!(parse_prompt("phi normalize 2 phi^81 ->").is_err());
        assert!(parse_prompt("phi mul (1 + 1 phi) ->").is_err());
    }
}
