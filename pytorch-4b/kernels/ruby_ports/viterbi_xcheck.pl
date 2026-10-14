%% Copyright (C) 2026 Ahmad Ali Parr
%% SPDX-License-Identifier: AGPL-3.0-only
%% Parity driver: runs the ORIGINAL viterbi_decode/4 from
%% sovereign-ruby-kernel lib/golog/dyna_golog_viterbi.pl on a case file.
%% Usage: swipl viterbi_xcheck.pl -- SRC_PL CASES
%% Output per case: id TAB path(csv, unbound element printed as _) TAB weight
%% or id TAB NOPATH when the predicate fails.
:- initialization(main, main).

%% The source ends with `:- initialization(demo, main).`, which would run its
%% demo and halt. Only that directive is dropped at load time; every other
%% clause of the source is loaded unmodified.
:- multifile user:term_expansion/2.
user:term_expansion((:- initialization(demo, main)), []).

main :-
    current_prolog_flag(argv, Argv), append(_, [Src, Cases], Argv),
    load_files(Src, [if(true), silent(true)]),
    read_file_to_string(Cases, S, []),
    split_string(S, "\n", "", Lines0), exclude(==(""), Lines0, Lines),
    forall(member(L, Lines), run_case(L)).

num(T, N) :- number_string(N0, T), N is float(N0).

run_case(L) :-
    split_string(L, "\t", "", [Id, StS, StartS, TrS, EmS, ObsS]),
    dyna_golog_viterbi:reset_model,
    items(StartS, Starts), forall(member(I, Starts), (split_string(I, "=", "", [Q, P]), atom_string(QA, Q), num(P, PN), dyna_golog_viterbi:assert_start(QA, PN))),
    items(TrS, Trs), forall(member(I, Trs), (split_string(I, ">=", "", [A, B, P]), atom_string(AA, A), atom_string(BA, B), num(P, PN), dyna_golog_viterbi:assert_transition(AA, BA, PN))),
    items(EmS, Ems), forall(member(I, Ems), (split_string(I, ">=", "", [A, B, P]), atom_string(AA, A), atom_string(BA, B), num(P, PN), dyna_golog_viterbi:assert_emission(AA, BA, PN))),
    split_string(StS, ",", "", StL), maplist([X, Y]>>atom_string(Y, X), StL, States),
    split_string(ObsS, ",", "", OL), maplist([X, Y]>>atom_string(Y, X), OL, Obs),
    (   catch(dyna_golog_viterbi:viterbi_decode(Obs, States, Path, W), E, (print_message(error, E), fail))
    ->  maplist([X, Y]>>(var(X) -> Y = '_' ; Y = X), Path, P2),
        atomic_list_concat(P2, ',', PA),
        format("~s\t~w\t~w~n", [Id, PA, W])
    ;   format("~s\tNOPATH~n", [Id])
    ).

items("", []) :- !.
items(S, L) :- split_string(S, ";", "", L).
