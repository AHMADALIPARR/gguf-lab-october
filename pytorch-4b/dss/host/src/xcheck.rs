// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Parity drivers for the ports of sovereign-ruby-kernel files. Each `gen-*`
//! writes a case file that the original-runtime driver (swipl / ruby) and the
//! Rust side both read; each kernel subcommand prints one canonical line per
//! case so that the two outputs can be compared line by line
//! (see kernels/ruby_ports/run_parity.sh).
//!
//! Viterbi case line (TAB-separated):
//!   id  states(csv)  start(s=p;..)  trans(a>b=p;..)  emit(s>o=p;..)  obs(csv)

use crate::{edaulc, governance as gv, viterbi as vt};
use std::io::{BufRead, Write};

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 { self.0 ^= self.0 << 13; self.0 ^= self.0 >> 7; self.0 ^= self.0 << 17; self.0 }
    fn below(&mut self, n: u64) -> u64 { self.next() % n }
}

fn lines(path: &str) -> Vec<String> {
    let f = std::fs::File::open(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    std::io::BufReader::new(f).lines().map(|l| l.unwrap()).collect()
}

fn hmm_line(id: &str, states: &[String], h: &vt::Hmm, start_txt: &[(String, String)],
            trans_txt: &[(String, String, String)], emit_txt: &[(String, String, String)], obs: &[String]) -> String {
    let _ = h;
    format!("{id}\t{}\t{}\t{}\t{}\t{}", states.join(","),
        start_txt.iter().map(|(s, p)| format!("{s}={p}")).collect::<Vec<_>>().join(";"),
        trans_txt.iter().map(|(a, b, p)| format!("{a}>{b}={p}")).collect::<Vec<_>>().join(";"),
        emit_txt.iter().map(|(a, b, p)| format!("{a}>{b}={p}")).collect::<Vec<_>>().join(";"),
        obs.join(","))
}

fn parse_case(l: &str) -> (String, Vec<String>, vt::Hmm, Vec<String>) {
    let f: Vec<&str> = l.split('\t').collect();
    let items = |s: &str| -> Vec<String> { if s.is_empty() { vec![] } else { s.split(';').map(|x| x.to_string()).collect() } };
    let states = f[1].split(',').map(|s| s.to_string()).collect();
    let start = items(f[2]).iter().map(|x| { let (s, p) = x.split_once('=').unwrap(); (s.to_string(), vt::dec(p).unwrap()) }).collect();
    let pair = |x: &String| { let (ab, p) = x.split_once('=').unwrap(); let (a, b) = ab.split_once('>').unwrap();
                              (a.to_string(), b.to_string(), vt::dec(p).unwrap()) };
    let trans = items(f[3]).iter().map(pair).collect();
    let emit = items(f[4]).iter().map(pair).collect();
    let obs = f[5].split(',').map(|s| s.to_string()).collect();
    (f[0].to_string(), states, vt::Hmm { start, trans, emit }, obs)
}

fn gen_viterbi(n: usize, seed: u64, mode: &str) {
    let mut r = Rng(seed.max(1));
    let mut out = std::io::stdout().lock();
    for i in 0..n {
        let id = format!("{mode}{i}");
        if mode == "builtin" || mode == "source" {
            // "source": the observation sequences used by the tests and demo of
            // dyna_golog_viterbi.pl and the smoke tests of _v1.py.
            const SRC: [(&str, &str); 8] = [
                ("toy_weather", "walk shop walk"), ("toy_weather", "walk shop clean walk"),
                ("toy_weather", "walk shop clean"), ("toy_weather", "walk shop walk clean walk"),
                ("pos_tagger", "the man saw a cat in the park ."), ("pos_tagger", "the cat saw the dog ."),
                ("cpg_island", "T C G C G C G A T C G C G C G A T A A T C G C G C G"),
                ("cpg_island", "C G C G C G A T A T C G C G C G")];
            if mode == "source" && i >= SRC.len() { break; }
            let name = if mode == "source" { SRC[i].0 } else { ["toy_weather", "pos_tagger", "cpg_island"][i % 3] };
            let (h, st) = vt::builtin(name).unwrap();
            let voc = h.symbols();
            let len = 1 + r.below(if name == "cpg_island" { 26 } else { 12 }) as usize;
            let obs: Vec<String> = if mode == "source" { SRC[i].1.split(' ').map(|x| x.to_string()).collect() }
                else { (0..len).map(|_| voc[r.below(voc.len() as u64) as usize].to_string()).collect() };
            // probabilities as decimal text; POS start is 1/5 = "0.2" exactly as a rational
            let q = |x: &vt::Q| { let f = vt::to_f64(x); format!("{f}") };
            let s: Vec<(String, String)> = h.start.iter().map(|(a, p)| (a.clone(), q(p))).collect();
            let t: Vec<(String, String, String)> = h.trans.iter().map(|(a, b, p)| (a.clone(), b.clone(), q(p))).collect();
            let e: Vec<(String, String, String)> = h.emit.iter().map(|(a, b, p)| (a.clone(), b.clone(), q(p))).collect();
            let stv: Vec<String> = st.iter().map(|x| x.to_string()).collect();
            writeln!(out, "{}", hmm_line(&id, &stv, &h, &s, &t, &e, &obs)).unwrap();
            continue;
        }
        let ns = 2 + r.below(3) as usize;
        let nv = 2 + r.below(3) as usize;
        let states: Vec<String> = (0..ns).map(|k| format!("s{k}")).collect();
        let voc: Vec<String> = (0..nv).map(|k| format!("o{k}")).collect();
        let p = |r: &mut Rng| -> String {
            if mode == "ties" { ["0.0", "0.25", "0.5", "0.75", "1.0"][r.below(5) as usize].to_string() }
            else if r.below(7) == 0 { "0.0".into() } else { format!("0.{:03}", 1 + r.below(999)) }
        };
        let s: Vec<(String, String)> = states.iter().map(|a| (a.clone(), p(&mut r))).collect();
        let mut t = vec![]; for a in &states { for b in &states { t.push((a.clone(), b.clone(), p(&mut r))); } }
        let mut e = vec![]; for a in &states { for o in &voc { e.push((a.clone(), o.clone(), p(&mut r))); } }
        let len = 1 + r.below(10) as usize;
        let obs: Vec<String> = (0..len).map(|_| voc[r.below(nv as u64) as usize].clone()).collect();
        let h = vt::Hmm { start: vec![], trans: vec![], emit: vec![] };
        writeln!(out, "{}", hmm_line(&id, &states, &h, &s, &t, &e, &obs)).unwrap();
    }
}

fn run_viterbi(path: &str) {
    for l in lines(path) {
        let (id, st, h, obs) = parse_case(&l);
        h.validate().unwrap();
        let s: Vec<&str> = st.iter().map(|x| x.as_str()).collect();
        let o: Vec<&str> = obs.iter().map(|x| x.as_str()).collect();
        let tie = vt::has_exact_tie(&h, &o, &s) as u8;
        match vt::decode(&h, &o, &s) {
            vt::Decode::Path { path, weight } => println!("{id}\t{}\t{:?}\t{}\ttie={tie}", path.join(","), vt::to_f64(&weight), weight),
            vt::Decode::NoPath => println!("{id}\tNOPATH\ttie={tie}"),
        }
    }
}

fn gen_edaulc() {
    let mut tasks: Vec<String> = edaulc::TASK_CLASS.iter().map(|x| x.0.to_string()).collect();
    let mut roles: Vec<String> = edaulc::ROLES.iter().map(|x| x.to_string()).collect();
    let extra = ["colonize_mars", "wizard", "sa_representative", "technical", "operational", "advisory_legal",
                 "devops", "system_architecture", "safety", "edaulc", "permit", "a", "x9", "diagnose_illness_2"];
    let all: Vec<String> = tasks.iter().chain(roles.iter()).cloned().chain(extra.iter().map(|s| s.to_string())).collect();
    tasks = all.clone(); roles = all;
    for t in &tasks { for r in &roles { println!("{t} {r}"); } }
}

fn run_edaulc(path: &str) {
    for l in lines(path) {
        let (t, r) = l.split_once(' ').unwrap();
        println!("{l}\t{}", edaulc::execution_decision(t, r).unwrap_or_else(|| "FAIL".into()));
    }
}

fn gen_gov(maxlen: usize) {
    let mut cur: Vec<Vec<&str>> = vec![vec![]];
    for _ in 0..=maxlen {
        let mut next = vec![];
        for s in &cur {
            println!("{}", if s.is_empty() { "-".to_string() } else { s.join(" ") });
            for x in gv::STATES { let mut v = s.clone(); v.push(x); next.push(v); }
        }
        cur = next;
    }
}

fn run_gov(path: &str, ts: &str) {
    for l in lines(path) {
        let steps: Vec<&str> = if l == "-" { vec![] } else { l.split(' ').collect() };
        let res = gv::run_steps(&steps);
        // chain: one seal per successful transition, prev-linked from genesis
        let (mut st, mut prev, mut seals) = ("init".to_string(), gv::GENESIS.to_string(), vec![]);
        for s in &steps {
            if !gv::allowed(&st, s) { break; }
            let sl = gv::seal(&format!("governance:{s}"), &gv::transition_payload(&st, s), ts, &prev);
            seals.push(sl.clone()); prev = sl; st = s.to_string();
        }
        println!("{l}\t{}\t{}", gv::fmt_steps(&res), if seals.is_empty() { "-".into() } else { seals.join(",") });
    }
}

fn rand_str(r: &mut Rng) -> String {
    let pool: Vec<char> = "abcXYZ019 _-:/.\"\\\u{0}\u{1}\u{8}\u{9}\u{a}\u{c}\u{d}\u{1f}\u{7f}é中\u{2028}😀".chars().collect();
    (0..r.below(12)).map(|_| pool[r.below(pool.len() as u64) as usize]).collect()
}

fn rand_json(r: &mut Rng, d: usize) -> gv::J {
    match if d >= 3 { r.below(4) } else { r.below(6) } {
        0 => gv::J::Null,
        1 => gv::J::Bool(r.below(2) == 1),
        2 => { let v = r.next() as i64 >> r.below(60); gv::J::Int(if v != 0 && r.below(5) == 0 { format!("{v}{}", r.next()) } else { v.to_string() }) }
        3 => gv::J::Str(rand_str(r)),
        4 => gv::J::Arr((0..r.below(4)).map(|_| rand_json(r, d + 1)).collect()),
        _ => {
            let mut v: Vec<(String, gv::J)> = vec![];
            for _ in 0..r.below(4) { let k = rand_str(r); if !v.iter().any(|x| x.0 == k) { v.push((k, rand_json(r, d + 1))); } }
            gv::J::Obj(v)
        }
    }
}

fn gen_worm(n: usize, seed: u64) {
    let mut r = Rng(seed.max(1));
    for i in 0..n {
        for _ in 0..1 + r.below(5) {
            let label: String = format!("lbl{}:{}", r.below(100), ["kernel:init", "stage:apl", "governance:sealed", "x"][r.below(4) as usize]);
            let ts = format!("20{:02}-{:02}-{:02}T{:02}:{:02}:{:02}Z", 20 + r.below(10), 1 + r.below(12), 1 + r.below(28),
                             r.below(24), r.below(60), r.below(60));
            let payload = if r.below(4) == 0 { rand_json(&mut r, 0) } else {
                gv::J::Obj((0..1 + r.below(4)).map(|k| (format!("k{k}"), rand_json(&mut r, 1))).collect()) };
            println!("w{i}\t{label}\t{ts}\t{}", gv::to_json(&payload));
        }
    }
}

fn run_worm(path: &str) {
    let (mut cur, mut prev) = (String::new(), gv::GENESIS.to_string());
    for l in lines(path) {
        let f: Vec<&str> = l.splitn(4, '\t').collect();
        if f[0] != cur { cur = f[0].to_string(); prev = gv::GENESIS.to_string(); }
        let j = gv::parse_canonical(f[3]).expect("case payload is canonical");
        let s = gv::seal(f[1], &j, f[2], &prev);
        println!("{}\t{s}", f[0]);
        prev = s;
    }
}

/// Compare Rust viterbi output with a driver's output (prolog | python).
fn cmp_viterbi(rust: &str, other: &str, kind: &str) {
    let (r, o) = (lines(rust), lines(other));
    assert_eq!(r.len(), o.len(), "line count differs");
    let (mut n, mut nopath, mut path_ok, mut w_bits, mut w_close, mut tie_cases) = (0, 0, 0, 0, 0, 0);
    let mut bad = vec![];
    let (mut t1_zero, mut tie_div, mut other) = (0, 0, 0);
    for (a, b) in r.iter().zip(&o) {
        let fa: Vec<&str> = a.split('\t').collect();
        let fb: Vec<&str> = b.split('\t').collect();
        assert_eq!(fa[0], fb[0], "case ids differ");
        n += 1;
        let tie = a.ends_with("tie=1");
        if tie { tie_cases += 1; }
        if fa[1] == "NOPATH" || fb[1] == "NOPATH" {
            if fa[1] == fb[1] { nopath += 1; path_ok += 1; w_bits += 1; w_close += 1; }
            else if kind == "prolog" && fa[1] == "NOPATH" && fb[1] == "_" && fb[2].parse::<f64>() == Ok(0.0) { t1_zero += 1; }
            else { other += 1; bad.push(format!("{}: rust={} other={} tie={tie}", fa[0], fa[1], fb[1])); }
            continue;
        }
        let pa: Vec<&str> = fa[1].split(',').collect();
        let pb: Vec<&str> = fb[1].split(',').collect();
        let same = pa.len() == pb.len() && if kind == "prolog" {
            pb[0] == "_" && pa[1..] == pb[1..]   // Prolog leaves the time-0 state unbound
        } else { pa == pb };
        let (x, y): (f64, f64) = (fa[2].parse().unwrap(), fb[2].parse().unwrap());
        let bits = x.to_bits() == y.to_bits();
        let close = (x - y).abs() <= 1e-12 * x.abs().max(y.abs());
        if same { path_ok += 1; }
        if bits { w_bits += 1; }
        if close { w_close += 1; }
        if !same || !close { if tie { tie_div += 1 } else { other += 1 } bad.push(format!("{}: rust={} {} other={} {} tie={tie}", fa[0], fa[1], fa[2], fb[1], fb[2])); }
    }
    println!("{kind}: cases={n} nopath_agree={nopath} path_agree={path_ok} weight_bit_equal={w_bits} weight_rel1e-12={w_close} exact_tie_cases={tie_cases} | divergences: t1_all_zero(Prolog [_] w=0.0)={t1_zero} exact_tie_broken_by_float={tie_div} unexplained={other}");
    for b in bad.iter().take(20) { println!("  {b}"); }
}

pub fn main(a: &[String]) {
    let g = |i: usize| a.get(i).map(|s| s.as_str()).unwrap_or("");
    match g(0) {
        "gen-viterbi" => gen_viterbi(g(1).parse().unwrap(), g(2).parse().unwrap(), g(3)),
        "viterbi" => run_viterbi(g(1)),
        "cmp-viterbi" => cmp_viterbi(g(1), g(2), g(3)),
        "gen-edaulc" => gen_edaulc(),
        "edaulc" => run_edaulc(g(1)),
        "gen-gov" => gen_gov(g(1).parse().unwrap()),
        "gov" => run_gov(g(1), g(2)),
        "gen-worm" => gen_worm(g(1).parse().unwrap(), g(2).parse().unwrap()),
        "worm" => run_worm(g(1)),
        _ => { eprintln!("xcheck gen-viterbi N SEED generic|ties|builtin | viterbi F | gen-edaulc | edaulc F | gen-gov L | gov F TS | gen-worm N SEED | worm F"); std::process::exit(2) }
    }
}
