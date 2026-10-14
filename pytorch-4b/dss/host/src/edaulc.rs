// Copyright (C) 2026 Ahmad Ali Parr
// SPDX-License-Identifier: AGPL-3.0-only
//! Port of `execution_decision/3` from sovereign-ruby-kernel
//! `lib/golog/edaulc_agent.pl` (EDAULC agent v2.0). The knowledge base is
//! copied fact for fact; the clauses are tried in source order and the first
//! one whose body succeeds wins (each is cut-terminated in the source).

pub const STATUS_ACTIVE: bool = true; // status(active).
pub const GOVERNING_PRINCIPLES: [&str; 4] = ["safety", "accuracy", "neutrality", "transparency"];
pub const COMPETENCIES: [&str; 4] = ["devops", "system_architecture", "enterprise_communications", "organizational_design"];
pub const ROLES: [&str; 4] = ["devops_specialist", "system_architect", "communications_lead", "org_designer"];
pub const MAPS_TO: [(&str, &str); 4] = [
    ("devops_specialist", "devops"), ("system_architect", "system_architecture"),
    ("communications_lead", "enterprise_communications"), ("org_designer", "organizational_design")];
pub const TASK_CLASS: [(&str, &str); 12] = [
    ("configure_aws_infrastructure", "technical"), ("design_ci_cd_pipeline", "technical"),
    ("draft_architecture_decision_record", "technical"), ("design_team_topology", "operational"),
    ("draft_stakeholder_memo", "communication"), ("diagnose_illness", "advisory_medical"),
    ("prescribe_treatment", "advisory_medical"), ("draft_contract", "advisory_legal"),
    ("interpret_statute", "advisory_legal"), ("recommend_stock_allocation", "advisory_financial"),
    ("prepare_tax_strategy", "advisory_financial"), ("summarize_public_topic", "general")];
pub const PROHIBITED_CLASS: [&str; 3] = ["advisory_medical", "advisory_legal", "advisory_financial"];
pub const ROLE_DOMAIN: [(&str, &str); 5] = [
    ("devops_specialist", "technical"), ("devops_specialist", "operational"), ("system_architect", "technical"),
    ("communications_lead", "communication"), ("org_designer", "operational")];

fn task_class(t: &str) -> Option<&'static str> { TASK_CLASS.iter().find(|x| x.0 == t).map(|x| x.1) }
fn known_task(t: &str) -> bool { task_class(t).is_some() }
fn is_role(r: &str) -> bool { ROLES.contains(&r) }
fn prohibited_task_class(t: &str) -> bool { task_class(t).is_some_and(|c| PROHIBITED_CLASS.contains(&c)) }
fn qualified(r: &str) -> bool {
    is_role(r) && MAPS_TO.iter().any(|(rr, c)| *rr == r && COMPETENCIES.contains(c))
}
fn role_can_take(r: &str, t: &str) -> bool {
    task_class(t).is_some_and(|c| ROLE_DOMAIN.iter().any(|(rr, cc)| *rr == r && *cc == c))
}
fn principle_gate(p: &str, t: &str) -> bool {
    match p { "safety" => !prohibited_task_class(t), "accuracy" | "neutrality" | "transparency" => true, _ => false }
}

/// Decision term exactly as SWI-Prolog prints it, or None where the source
/// predicate fails (no clause applies; unreachable for the shipped KB).
pub fn execution_decision(task: &str, role: &str) -> Option<String> {
    if !STATUS_ACTIVE { return Some("deny(agent_inactive)".into()); }
    if !known_task(task) { return Some("deny(unknown_task)".into()); }
    if !is_role(role) { return Some("deny(unknown_role)".into()); }
    if let Some(c) = task_class(task) {
        if PROHIBITED_CLASS.contains(&c) { return Some(format!("deny(prohibited_class({c}))")); }
    }
    if !qualified(role) { return Some("deny(role_not_qualified)".into()); }
    if !role_can_take(role, task) { return Some("deny(out_of_domain)".into()); }
    for p in GOVERNING_PRINCIPLES {
        if !principle_gate(p, task) { return Some(format!("deny(principle_violation({p}))")); }
    }
    if GOVERNING_PRINCIPLES.iter().all(|p| principle_gate(p, task)) { Some("permit".into()) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;
    // The decision-level plunit tests of edaulc_agent.pl, restated.
    #[test]
    fn source_plunit_decisions() {
        let d = |t, r| execution_decision(t, r).unwrap();
        assert_eq!(d("diagnose_illness", "devops_specialist"), "deny(prohibited_class(advisory_medical))");
        assert_eq!(d("draft_contract", "system_architect"), "deny(prohibited_class(advisory_legal))");
        for r in ROLES { assert_eq!(d("prepare_tax_strategy", r), "deny(prohibited_class(advisory_financial))"); }
        assert_eq!(d("colonize_mars", "devops_specialist"), "deny(unknown_task)");
        assert_eq!(d("design_ci_cd_pipeline", "wizard"), "deny(unknown_role)");
        assert_eq!(d("draft_stakeholder_memo", "devops_specialist"), "deny(out_of_domain)");
        assert_eq!(d("draft_stakeholder_memo", "communications_lead"), "permit");
        assert_eq!(d("configure_aws_infrastructure", "devops_specialist"), "permit");
        assert_eq!(d("draft_contract", "devops_specialist"), "deny(prohibited_class(advisory_legal))");
        assert!(qualified("devops_specialist"));
        assert!(!qualified("sa_representative"));
    }
}
