#!/usr/bin/env bash
# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# Parity of the dss-host Rust ports against the ORIGINAL sovereign-ruby-kernel
# files run in their own runtimes (swipl 9.x, ruby 3.3 with -rtime).
# Usage: run_parity.sh [SRC_ROOT] [WORKDIR]
set -euo pipefail
S=${1:-/workspace/sovereign-ruby-kernel}
W=${2:-$(mktemp -d)}; mkdir -p "$W"
HERE=$(cd "$(dirname "$0")" && pwd)
B=$HERE/../../dss/host/target/release/dss-host
[ -x "$B" ] || (cd "$HERE/../../dss/host" && cargo build --release -q)
echo "source: $S ($(git -C "$S" rev-parse --short HEAD 2>/dev/null || echo '?'))  workdir: $W"

echo "== original test suites =="
(cd "$S" && swipl -g run_tests -t halt lib/golog/edaulc_agent.pl 2>&1 | tail -1)
(cd "$S" && swipl lib/golog/dyna_golog_viterbi.pl 2>/dev/null | grep -E '^\[PASS\]|ALL TESTS' || true)
(cd "$S" && ruby -rtime spec/kernel_spec.rb | tail -1)

echo "== viterbi (dyna_golog_viterbi.pl) =="
for m in source builtin generic ties; do
  n=3000; [ $m = source ] && n=8
  $B xcheck gen-viterbi $n 7 $m > $W/v_$m.txt
  $B xcheck viterbi $W/v_$m.txt > $W/v_$m.rust
  swipl "$HERE/viterbi_xcheck.pl" -- "$S/lib/golog/dyna_golog_viterbi.pl" $W/v_$m.txt > $W/v_$m.pl 2>/dev/null
  printf '%-8s ' $m; $B xcheck cmp-viterbi $W/v_$m.rust $W/v_$m.pl prolog | head -1
done

echo "== edaulc (edaulc_agent.pl), exhaustive over KB atoms x extra atoms =="
$B xcheck gen-edaulc > $W/e.txt
$B xcheck edaulc $W/e.txt > $W/e.rust
swipl "$HERE/edaulc_xcheck.pl" -- "$S/lib/golog/edaulc_agent.pl" $W/e.txt > $W/e.pl 2>/dev/null
echo "cases=$(wc -l < $W/e.txt) identical=$(paste $W/e.rust $W/e.pl | awk -F'\t' '$1==$3 && $2==$4' | wc -l)"

echo "== governance (governance.rb + worm.rb), all step sequences of length 0..6 =="
$B xcheck gen-gov 6 > $W/g.txt
for TS in 2026-10-14T14:18:00Z 1999-12-31T23:59:59Z; do
  $B xcheck gov $W/g.txt $TS > $W/g.rust
  ruby -rtime "$HERE/governance_xcheck.rb" "$S" gov $W/g.txt $TS > $W/g.rb
  echo "ts=$TS cases=$(wc -l < $W/g.txt) identical(result+all seals)=$(paste $W/g.rust $W/g.rb | awk -F'\t' '$1==$4 && $2==$5 && $3==$6' | wc -l) seals=$(awk -F'\t' '$3!="-"{n+=split($3,a,",")} END{print n}' $W/g.rust)"
done

echo "== worm (worm.rb), random chains with random JSON payloads =="
for seed in 11 12; do
  $B xcheck gen-worm 3000 $seed > $W/w.txt
  $B xcheck worm $W/w.txt > $W/w.rust
  ruby -rtime "$HERE/governance_xcheck.rb" "$S" worm $W/w.txt > $W/w.rb
  echo "seed=$seed chains=3000 seals=$(wc -l < $W/w.txt) identical=$(paste $W/w.rust $W/w.rb | awk -F'\t' '$1==$3 && $2==$4' | wc -l)"
done

echo "== prompt level: dss-host answer vs original-runtime output =="
awk -F'\t' '{split($1,a," "); print "permission " a[1] " " a[2] " ->"}' $W/e.txt | $B answer > $W/p_e.ans
echo "permission prompts=$(wc -l < $W/p_e.ans) identical=$(paste $W/p_e.ans <(cut -f2 $W/e.pl) | awk -F'\t' '$1==$2' | wc -l)"
grep -v '^-$' $W/g.txt | awk '{print "governance steps " $0 " ->"}' | $B answer > $W/p_g.ans
echo "governance prompts=$(wc -l < $W/p_g.ans) identical=$(paste $W/p_g.ans <(grep -v $'^-\t' $W/g.rb | cut -f2) | awk -F'\t' '$1==$2' | wc -l)"
# viterbi prompts on the source sequences: path (minus Prolog's unbound t=0) and weight vs Prolog
models=(toy_weather toy_weather toy_weather toy_weather pos_tagger pos_tagger cpg_island cpg_island)
i=0; : > $W/p_v.txt
while IFS=$'\t' read -r id st s t e obs; do echo "viterbi ${models[$i]} ${obs//,/ } ->" >> $W/p_v.txt; i=$((i+1)); done < $W/v_source.txt
$B answer < $W/p_v.txt > $W/p_v.ans
ok=0; i=0
while IFS= read -r a; do
  pl=$(sed -n "$((i+1))p" $W/v_source.pl | cut -f2); ru=$(sed -n "$((i+1))p" $W/v_source.rust)
  p=${a#path=}; p=${p%% weight=*}; w=${a##* weight=}
  [ "_,${p#*,}" = "$pl" ] && [ "$(echo "$ru" | cut -f4)" = "$w" ] && ok=$((ok+1)); i=$((i+1))
done < $W/p_v.ans
echo "viterbi prompts=$i path(t>=1)==prolog && weight==exact: $ok"
