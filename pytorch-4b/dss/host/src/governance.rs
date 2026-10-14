// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Port of sovereign-ruby-kernel `lib/kernel/governance.rb` (Gate state
//! machine) and `lib/kernel/worm.rb` (append-only SHA-256 chain).
//!
//! WORM.seal(label, payload): prev = last seal or "0"*64,
//! ts = Time.now.utc.iso8601, seal = SHA256(JSON.generate({label:, payload:,
//! ts:, prev:})). The clock is an input here (the host never reads time), and
//! JSON.generate is reproduced for the value subset strings / integers /
//! true / false / null / arrays / objects (json 2.7.2 escaping: `"`, `\`,
//! \b \f \n \r \t, other C0 controls as lowercase \u00xx, everything else
//! raw UTF-8). Floats are rejected rather than imitating Ruby Float#to_s.

use sha2::{Digest, Sha256};

pub const STATES: [&str; 4] = ["init", "verified", "sealed", "rejected"];

pub fn allowed(from: &str, to: &str) -> bool {
    matches!((from, to), ("init", "verified") | ("init", "rejected") | ("verified", "sealed")
        | ("verified", "rejected") | ("rejected", "init"))
}

#[derive(Clone, Debug, PartialEq)]
pub enum J { Null, Bool(bool), Int(String), Str(String), Arr(Vec<J>), Obj(Vec<(String, J)>) }

pub fn esc(s: &str, out: &mut String) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""), '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"), '\u{c}' => out.push_str("\\f"),
            '\n' => out.push_str("\\n"), '\r' => out.push_str("\\r"), '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out.push('"');
}

pub fn gen(j: &J, out: &mut String) {
    match j {
        J::Null => out.push_str("null"),
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Int(i) => out.push_str(i),
        J::Str(s) => esc(s, out),
        J::Arr(v) => { out.push('['); for (i, x) in v.iter().enumerate() { if i > 0 { out.push(',') } gen(x, out) } out.push(']') }
        J::Obj(v) => {
            out.push('{');
            for (i, (k, x)) in v.iter().enumerate() { if i > 0 { out.push(',') } esc(k, out); out.push(':'); gen(x, out) }
            out.push('}')
        }
    }
}

pub fn to_json(j: &J) -> String { let mut s = String::new(); gen(j, &mut s); s }

/// Strict parser: accepts only text that is already in canonical
/// JSON.generate form (re-generation must reproduce it byte for byte),
/// no floats, no duplicate keys, depth <= 16.
pub fn parse_canonical(s: &str) -> Result<J, String> {
    struct P<'a> { b: &'a [u8], i: usize }
    impl<'a> P<'a> {
        fn val(&mut self, d: usize) -> Result<J, String> {
            if d > 16 { return Err("too deep".into()); }
            match self.b.get(self.i) {
                Some(b'n') => self.lit("null", J::Null), Some(b't') => self.lit("true", J::Bool(true)),
                Some(b'f') => self.lit("false", J::Bool(false)),
                Some(b'"') => Ok(J::Str(self.string()?)),
                Some(b'[') => {
                    self.i += 1; let mut v = vec![];
                    if self.b.get(self.i) == Some(&b']') { self.i += 1; return Ok(J::Arr(v)); }
                    loop {
                        v.push(self.val(d + 1)?);
                        match self.b.get(self.i) { Some(b',') => self.i += 1, Some(b']') => { self.i += 1; return Ok(J::Arr(v)); } _ => return Err("bad array".into()) }
                    }
                }
                Some(b'{') => {
                    self.i += 1; let mut v: Vec<(String, J)> = vec![];
                    if self.b.get(self.i) == Some(&b'}') { self.i += 1; return Ok(J::Obj(v)); }
                    loop {
                        if self.b.get(self.i) != Some(&b'"') { return Err("bad key".into()); }
                        let k = self.string()?;
                        if v.iter().any(|x| x.0 == k) { return Err("duplicate key".into()); }
                        if self.b.get(self.i) != Some(&b':') { return Err("missing :".into()); }
                        self.i += 1;
                        let x = self.val(d + 1)?; v.push((k, x));
                        match self.b.get(self.i) { Some(b',') => self.i += 1, Some(b'}') => { self.i += 1; return Ok(J::Obj(v)); } _ => return Err("bad object".into()) }
                    }
                }
                Some(c) if *c == b'-' || c.is_ascii_digit() => {
                    let st = self.i; if self.b[self.i] == b'-' { self.i += 1 }
                    let ds = self.i; while self.b.get(self.i).is_some_and(|c| c.is_ascii_digit()) { self.i += 1 }
                    if self.i == ds || matches!(self.b.get(self.i), Some(b'.') | Some(b'e') | Some(b'E')) { return Err("floats not supported".into()); }
                    let t = std::str::from_utf8(&self.b[st..self.i]).unwrap();
                    if (t.len() > ds - st + 1 && self.b[ds] == b'0') || t == "-0" || self.i - ds > 40 { return Err("non-canonical integer".into()); }
                    Ok(J::Int(t.to_string()))
                }
                _ => Err("unexpected token".into()),
            }
        }
        fn lit(&mut self, w: &str, j: J) -> Result<J, String> {
            if self.b[self.i..].starts_with(w.as_bytes()) { self.i += w.len(); Ok(j) } else { Err("bad literal".into()) }
        }
        fn string(&mut self) -> Result<String, String> {
            self.i += 1; let mut out = String::new();
            let rest = std::str::from_utf8(&self.b[self.i..]).map_err(|_| "utf8")?;
            let mut it = rest.char_indices();
            while let Some((k, c)) = it.next() {
                match c {
                    '"' => { self.i += k + 1; return Ok(out); }
                    '\\' => match it.next().map(|x| x.1) {
                        Some('"') => out.push('"'), Some('\\') => out.push('\\'), Some('b') => out.push('\u{8}'),
                        Some('f') => out.push('\u{c}'), Some('n') => out.push('\n'), Some('r') => out.push('\r'),
                        Some('t') => out.push('\t'),
                        Some('u') => {
                            let h: String = (0..4).filter_map(|_| it.next().map(|x| x.1)).collect();
                            let v = u32::from_str_radix(&h, 16).map_err(|_| "bad \\u")?;
                            out.push(char::from_u32(v).ok_or("bad \\u")?);
                        }
                        _ => return Err("bad escape".into()),
                    },
                    c if (c as u32) < 0x20 => return Err("raw control char".into()),
                    c => out.push(c),
                }
            }
            Err("unterminated string".into())
        }
    }
    let mut p = P { b: s.as_bytes(), i: 0 };
    let j = p.val(0)?;
    if p.i != s.len() { return Err("trailing text".into()); }
    if to_json(&j) != s { return Err("not in canonical JSON.generate form".into()); }
    Ok(j)
}

pub const GENESIS: &str = "0000000000000000000000000000000000000000000000000000000000000000";

/// The raw JSON string hashed by WORM.seal.
pub fn seal_raw(label: &str, payload: &J, ts: &str, prev: &str) -> String {
    to_json(&J::Obj(vec![("label".into(), J::Str(label.into())), ("payload".into(), payload.clone()),
                         ("ts".into(), J::Str(ts.into())), ("prev".into(), J::Str(prev.into()))]))
}

pub fn seal(label: &str, payload: &J, ts: &str, prev: &str) -> String {
    let h = Sha256::digest(seal_raw(label, payload, ts, prev).as_bytes());
    h.iter().map(|b| format!("{b:02x}")).collect()
}

/// Governance payload sealed on every transition: {from:, to:, evidence: {}}.
pub fn transition_payload(from: &str, to: &str) -> J {
    J::Obj(vec![("from".into(), J::Str(from.into())), ("to".into(), J::Str(to.into())), ("evidence".into(), J::Obj(vec![]))])
}

/// Apply transitions from a fresh Gate (state :init). Ok(final state) or the
/// StateError message exactly as governance.rb raises it.
pub fn run_steps(steps: &[&str]) -> Result<String, String> {
    let mut st = "init";
    for s in steps {
        if !allowed(st, s) { return Err(format!("{st} → {s} forbidden")); }
        st = s;
    }
    Ok(st.to_string())
}

pub fn fmt_steps(r: &Result<String, String>) -> String {
    match r { Ok(s) => s.clone(), Err(e) => format!("StateError: {e}") }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The WORM / Governance tests of spec/kernel_spec.rb, restated.
    #[test]
    fn source_spec() {
        let ts = "2026-01-01T00:00:00Z";
        let obj = |k: &str, v: J| J::Obj(vec![(k.into(), v)]);
        let a = seal("test1", &obj("a", J::Int("1".into())), ts, GENESIS);
        let b = seal("test2", &obj("b", J::Int("2".into())), ts, &a);
        let c = seal("test3", &obj("c", J::Int("3".into())), ts, &b);
        assert_eq!([a.len(), b.len(), c.len()], [64, 64, 64]);
        assert_eq!(seal("v1", &obj("x", J::Int("42".into())), ts, GENESIS), seal("v1", &obj("x", J::Int("42".into())), ts, GENESIS));
        assert_eq!(run_steps(&["verified", "sealed"]), Ok("sealed".into()));
        assert!(run_steps(&["sealed"]).is_err());
    }
    #[test]
    fn basics() {
        assert_eq!(fmt_steps(&run_steps(&["verified", "sealed"])), "sealed");
        assert_eq!(fmt_steps(&run_steps(&["verified", "sealed", "init"])), "StateError: sealed → init forbidden");
        assert_eq!(fmt_steps(&run_steps(&["rejected", "init", "verified"])), "verified");
        let j = parse_canonical(r#"{"a":"\u0001\b/ é","b":[1,-2,null,true,{}]}"#).unwrap();
        assert_eq!(to_json(&j), r#"{"a":"\u0001\b/ é","b":[1,-2,null,true,{}]}"#);
        assert!(parse_canonical(r#"{"a": 1}"#).is_err());
        assert!(parse_canonical(r#"{"a":1.5}"#).is_err());
        assert!(parse_canonical(r#"{"a":1,"a":2}"#).is_err());
        assert!(parse_canonical(r#"{"a":"\u0041"}"#).is_err());
    }
}
