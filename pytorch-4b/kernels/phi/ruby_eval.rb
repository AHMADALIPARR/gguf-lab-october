# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# Evaluates queries with sovereign-ruby-kernel's own PhiAlgebra (f64), for the
# cross-check in phi_crosscheck.py. stdin: JSON array; stdout: JSON array of
# "%.17g" strings.
require 'json'
$LOAD_PATH.unshift File.join(ARGV[0], 'lib')
require 'ast/phi_algebra'
P = Sovereign::AST::PhiAlgebra
def elem(a, b) = P.add(P.literal(a), P.mul(P.literal(b), P.phi(1)))
out = JSON.parse($stdin.read).map do |q|
  v = case q['op']
      when 'normalize'
        q['terms'].map { |c, k| P.mul(P.literal(c), P.phi(k)) }.reduce { |x, y| P.add(x, y) }.then { |n| P.evaluate(n) }
      when 'mul'  then P.evaluate(P.mul(elem(*q['x']), elem(*q['y'])))
      when 'conj' then P.evaluate(P.galois_conjugate(elem(*q['x'])))
      when 'norm' then P.evaluate(P.norm(elem(*q['x'])))
      end
  format('%.17g', v)
end
puts JSON.generate(out)
