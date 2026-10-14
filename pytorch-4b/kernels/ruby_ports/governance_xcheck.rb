# Copyright (C) 2026 Ahmad Ali Parr
# SPDX-License-Identifier: AGPL-3.0-only
# Parity driver: runs the ORIGINAL Governance::Gate and WORM from
# sovereign-ruby-kernel lib/kernel/governance.rb and lib/kernel/worm.rb.
# The clock is stubbed (Time.now returns the case timestamp); the kernel
# code itself is loaded unmodified.
# Usage: ruby -rtime governance_xcheck.rb SRC_ROOT gov CASES TS
#        ruby -rtime governance_xcheck.rb SRC_ROOT worm CASES
require 'json'
require 'time'
root, mode, cases, fixed = ARGV
require File.join(root, 'lib/kernel/governance')

$now = Time.iso8601(fixed || '2026-01-01T00:00:00Z')
class << Time
  def now = $now
end
W = Sovereign::Kernel::WORM

case mode
when 'gov'
  File.readlines(cases, chomp: true).each do |l|
    W.reset!
    gate = Sovereign::Kernel::Governance.new_gate
    steps = l == '-' ? [] : l.split(' ')
    res = begin
      steps.each { |s| gate.transition!(s.to_sym) }
      gate.state.to_s
    rescue Sovereign::Kernel::Governance::StateError => e
      "StateError: #{e.message}"
    end
    raise 'chain invalid' unless W.valid? && (0...W.length).all? { |i| W.verify_entry(i) }
    seals = W.chain.map { |e| e[:seal] }
    puts "#{l}\t#{res}\t#{seals.empty? ? '-' : seals.join(',')}"
  end
when 'worm'
  cur = nil
  File.readlines(cases, chomp: true).each do |l|
    id, label, ts, payload = l.split("\t", 4)
    if id != cur
      W.reset!
      cur = id
    end
    $now = Time.iso8601(ts)
    e = W.seal(label, JSON.parse(payload))
    raise 'chain invalid' unless W.valid? && W.verify_entry(W.length - 1)
    puts "#{id}\t#{e[:seal]}"
  end
end
