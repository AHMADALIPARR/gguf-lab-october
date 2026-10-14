// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Strict per-task prompt parser. The host, not the model, extracts the
//! integers. Anything that does not match exactly one task template is
//! rejected: extra or missing spaces, signs, leading zeros, non-ASCII digits,
//! duplicates, unsorted sets, more than 8 elements, values above 10^6.

#[derive(Debug, Clone, PartialEq)]
pub enum Task {
    IsDss(Vec<u64>),
    Card(Vec<u64>, u64),
    Next(Vec<u64>),
    Bound(u64, u64),
}

impl Task {
    pub fn intent(&self) -> &'static str {
        match self {
            Task::IsDss(_) => "CHECK_IS_DSS",
            Task::Card(..) => "CARD_FROM_SUM",
            Task::Next(_) => "NEXT_GREEDY",
            Task::Bound(..) => "CHECK_BOUND",
        }
    }
    pub fn tool(&self) -> &'static str {
        match self {
            Task::IsDss(_) => "isdss",
            Task::Card(..) => "card",
            Task::Next(_) => "next",
            Task::Bound(..) => "bound",
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
    }
}
