// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
// Emits greedy.mjs reference outputs as JSON for crosscheck.py.
const { isDSS, computeGreedySequence, verifyQuadraticBound, buildCardinalityTable } = await import(process.argv[2]);
const sets = JSON.parse(process.argv[3]);
const K = parseInt(process.argv[4]);
const seq = computeGreedySequence(K);
console.log(JSON.stringify({
  greedy: seq,
  bound: verifyQuadraticBound(seq).map(r => [r.k, r.bound, r.holds]),
  dss: sets.map(s => isDSS(s)),
  tables: sets.slice(0, 50).map(s => [...buildCardinalityTable(s).entries()]),
}));
