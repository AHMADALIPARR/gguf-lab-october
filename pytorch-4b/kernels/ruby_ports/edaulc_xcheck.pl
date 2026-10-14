%% Copyright (C) 2026 Ahmad Ali Parr
%% SPDX-License-Identifier: AGPL-3.0-only
%% Parity driver: queries the ORIGINAL execution_decision/3 from
%% sovereign-ruby-kernel lib/golog/edaulc_agent.pl.
%% Usage: swipl edaulc_xcheck.pl -- SRC_PL CASES   (case line: "task role")
:- initialization(main, main).

main :-
    current_prolog_flag(argv, Argv), append(_, [Src, Cases], Argv),
    use_module(Src),
    read_file_to_string(Cases, S, []),
    split_string(S, "\n", "", Lines0), exclude(==(""), Lines0, Lines),
    forall(member(L, Lines), (
        split_string(L, " ", "", [T, R]), atom_string(TA, T), atom_string(RA, R),
        (   findall(D, edaulc_agent:execution_decision(TA, RA, D), Ds), Ds = [D1|_]
        ->  length(Ds, N), ( N > 1 -> Extra = "\tMULTI" ; Extra = "" ),
            format("~s\t~w~s~n", [L, D1, Extra])
        ;   format("~s\tFAIL~n", [L])
        ))).
