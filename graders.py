"""
ClinicalOps — Deterministic graders for all three tasks.

All scores are strictly in (0.0, 1.0) — never exactly 0.0 or 1.0.
  • Minimum returned score: 0.05  (attempted but all wrong)
  • Maximum returned score: 0.95  (perfect)

Scoring systems used:
  NEWS2  — National Early Warning Score 2 (Royal College of Physicians, 2017)
  SOFA   — Sequential Organ Failure Assessment (Vincent et al., 1996)
"""

from __future__ import annotations
from typing import Dict, List, Any


# ─────────────────────────────────────────────────────────────────────────────
# NEWS2 scorer  (Task 1)
# ─────────────────────────────────────────────────────────────────────────────

def compute_news2(vitals: Dict[str, Any]) -> int:
    """Return NEWS2 integer score (0–20+). Higher = more urgent."""
    score = 0

    # Respiratory rate
    rr = vitals.get("respiratory_rate", 16)
    if rr <= 8:           score += 3
    elif rr <= 11:        score += 1
    elif rr <= 20:        score += 0
    elif rr <= 24:        score += 2
    else:                 score += 3

    # SpO2 (Scale 1 — no hypercapnic drive)
    spo2 = vitals.get("spo2", 98.0)
    if spo2 <= 91:        score += 3
    elif spo2 <= 93:      score += 2
    elif spo2 <= 95:      score += 1
    else:                 score += 0

    # Systolic BP
    sbp = vitals.get("systolic_bp", 120)
    if sbp <= 90:         score += 3
    elif sbp <= 100:      score += 2
    elif sbp <= 110:      score += 1
    elif sbp <= 219:      score += 0
    else:                 score += 3

    # Heart rate
    hr = vitals.get("heart_rate", 75)
    if hr <= 40:          score += 3
    elif hr <= 50:        score += 1
    elif hr <= 90:        score += 0
    elif hr <= 110:       score += 1
    elif hr <= 130:       score += 2
    else:                 score += 3

    # Temperature
    temp = vitals.get("temperature", 37.0)
    if temp <= 35.0:      score += 3
    elif temp <= 36.0:    score += 1
    elif temp <= 38.0:    score += 0
    elif temp <= 39.0:    score += 1
    else:                 score += 2

    # Consciousness (AVPU)
    avpu = vitals.get("consciousness", "Alert")
    if avpu == "Alert":   score += 0
    else:                 score += 3

    return score


def grade_triage(ranked_ids: List[str], patients: List[Dict[str, Any]]) -> float:
    """
    Task 1 grader.
    Score patient list by NEWS2, then compare agent ranking vs correct ranking.
    Returns score strictly in (0.05, 0.95).
    """
    if not ranked_ids or not patients:
        return 0.05

    # Build correct ranking (descending NEWS2)
    scored = [(p["patient_id"], compute_news2(p.get("vitals", {}))) for p in patients]
    correct_order = [pid for pid, _ in sorted(scored, key=lambda x: -x[1])]

    n = len(correct_order)
    if n == 0:
        return 0.05

    # Only score the patients the agent actually ranked
    agent_ranked = [pid for pid in ranked_ids if pid in correct_order]

    # Position-weighted scoring: position 0 (most urgent) worth most
    # Critical patients (NEWS2 >= 7) that are misranked get a heavy penalty
    critical_ids = {pid for pid, s in scored if s >= 7}

    raw_score = 0.0
    for agent_pos, pid in enumerate(agent_ranked):
        correct_pos = correct_order.index(pid)
        # Distance penalty — 0 penalty for exact match, up to 1.0 for worst swap
        position_error = abs(agent_pos - correct_pos) / max(n - 1, 1)
        position_score = 1.0 - position_error

        # Weight: critical patients count 2×
        weight = 2.0 if pid in critical_ids else 1.0
        raw_score += position_score * weight

    max_possible = sum(2.0 if pid in critical_ids else 1.0 for pid in correct_order)
    normalised = raw_score / max_possible if max_possible > 0 else 0.0

    # Critical patient penalty: any critical patient ranked outside top 3 → deduct
    for pid in critical_ids:
        if pid in agent_ranked and agent_ranked.index(pid) >= 3:
            normalised -= 0.15

    return max(0.06, min(0.94, round(normalised, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Medication conflict grader  (Task 2)
# ─────────────────────────────────────────────────────────────────────────────

def grade_medication(
    flagged_conflicts: List[Dict[str, Any]],
    resolved_conflicts: List[Dict[str, Any]],
    ground_truth_conflicts: List[Dict[str, Any]],
) -> float:
    """
    Task 2 grader.
    Compares flagged+resolved conflicts against ground truth.
    Returns score strictly in (0.05, 0.95).
    """
    if not ground_truth_conflicts:
        return 0.85  # no conflicts → safe discharge is correct

    total_gt = len(ground_truth_conflicts)
    severe_gt = [c for c in ground_truth_conflicts if c.get("severity") == "severe"]

    # Build lookup sets for matching (drug pair order-insensitive)
    def pair_key(c: Dict) -> frozenset:
        return frozenset([c.get("drug_a", "").lower(), c.get("drug_b", "").lower()])

    gt_keys = {pair_key(c): c for c in ground_truth_conflicts}
    severe_keys = {pair_key(c) for c in severe_gt}

    flagged_keys = {pair_key(c) for c in flagged_conflicts}
    resolved_keys = {pair_key(c) for c in resolved_conflicts}

    # True positives
    tp_flagged = flagged_keys & set(gt_keys.keys())
    tp_resolved = resolved_keys & set(gt_keys.keys())

    # Recall: what fraction of gt conflicts were found
    recall = len(tp_flagged) / total_gt

    # Resolution bonus: extra credit for correct resolution
    resolution_bonus = (len(tp_resolved) / total_gt) * 0.2

    # Severe miss penalty
    severe_missed = severe_keys - flagged_keys
    severe_penalty = len(severe_missed) * 0.20

    # False positive penalty (flagging non-existent conflicts)
    fp = flagged_keys - set(gt_keys.keys())
    fp_penalty = len(fp) * 0.05

    raw = recall + resolution_bonus - severe_penalty - fp_penalty
    return max(0.05, min(0.94, round(raw, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Sepsis Watch grader  (Task 3)
# ─────────────────────────────────────────────────────────────────────────────

# Required investigations for sepsis workup (Sepsis-6 bundle)
SEPSIS_INVESTIGATIONS = {
    "blood_cultures", "lactate", "cbc", "renal_panel", "abg"
}

def compute_sofa_partial(labs: Dict[str, Any]) -> int:
    """
    Simplified SOFA score from available labs (0–10 range here).
    Full SOFA is 0–24; we use the components we can simulate.
    """
    score = 0

    # Renal (creatinine umol/L)
    cr = labs.get("creatinine", 80)
    if cr >= 440:    score += 4
    elif cr >= 300:  score += 3
    elif cr >= 171:  score += 2
    elif cr >= 110:  score += 1

    # Coagulation (platelets x10^9/L)
    plt = labs.get("platelets", 200)
    if plt < 20:     score += 4
    elif plt < 50:   score += 3
    elif plt < 100:  score += 2
    elif plt < 150:  score += 1

    # Hepatic (bilirubin umol/L)
    bil = labs.get("bilirubin", 15)
    if bil >= 204:   score += 4
    elif bil >= 102: score += 3
    elif bil >= 33:  score += 2
    elif bil >= 20:  score += 1

    # Respiratory (PaO2/FiO2)
    pf = labs.get("pao2_fio2", 400)
    if pf < 100:     score += 4
    elif pf < 200:   score += 3
    elif pf < 300:   score += 2
    elif pf < 400:   score += 1

    return score


def grade_sepsis(
    investigations_ordered: List[str],
    escalation_level: str,
    escalation_step: int,
    sofa_at_escalation: int,
    final_sofa: int,
    total_steps: int,
) -> float:
    """
    Task 3 grader.
    Rewards: early escalation, correct investigations, low SOFA at escalation.
    Penalises: late/no escalation, SOFA≥6 before escalation.
    Returns score strictly in (0.05, 0.95).
    """
    score = 0.0

    # 1. Investigation coverage (up to 0.35)
    ordered_set = set(i.lower() for i in investigations_ordered)
    coverage = len(ordered_set & SEPSIS_INVESTIGATIONS) / len(SEPSIS_INVESTIGATIONS)
    score += coverage * 0.35

    # 2. Escalation quality (up to 0.40)
    if escalation_level in ("rapid_response", "icu_transfer"):
        # Bonus for escalating before SOFA crosses danger threshold
        if sofa_at_escalation < 4:
            score += 0.40   # early, correct
        elif sofa_at_escalation < 6:
            score += 0.28   # slightly late but correct level
        else:
            score += 0.10   # escalated after danger threshold
    elif escalation_level == "senior_review":
        score += 0.15       # partial credit — right idea, wrong urgency
    else:
        score += 0.0        # no escalation

    # 3. Timing bonus (escalated early in episode = better)
    if escalation_step > 0 and total_steps > 0:
        earliness = 1.0 - (escalation_step / total_steps)
        score += earliness * 0.15

    # 4. Penalty: SOFA ≥ 6 reached before escalation
    if final_sofa >= 6 and escalation_step == 0:
        score -= 0.30

    # 5. Penalty: never escalated
    if escalation_level == "":
        score -= 0.20

    return max(0.06, min(0.94, round(score, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Ventilator Weaning grader  (Task 4)
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_VENT_CHECKS = {
    "assess_oxygenation", "assess_consciousness", "assess_secretions",
    "check_rsbi", "reduce_fio2", "reduce_peep", "perform_sbt"
}

def grade_vent_weaning(
    checks_completed: List[str],
    sbt_performed: bool,
    extubation_decision: str,
    premature_extubation: bool,
) -> float:
    """
    Task 4 grader — SBT checklist coverage + correct extubation decision.
    Returns score strictly in (0.06, 0.94).
    """
    completed_set = set(c.lower() for c in checks_completed)

    # Coverage of required checks
    coverage = len(completed_set & REQUIRED_VENT_CHECKS) / len(REQUIRED_VENT_CHECKS)
    score = coverage * 0.55

    # SBT performed bonus
    if sbt_performed:
        score += 0.20

    # Extubation decision
    if extubation_decision == "extubate" and sbt_performed and not premature_extubation:
        score += 0.19
    elif extubation_decision == "continue_weaning":
        score += 0.08
    elif premature_extubation:
        score -= 0.20

    return max(0.06, min(0.94, round(score, 4)))


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic Reasoning grader  (Task 5)
# ─────────────────────────────────────────────────────────────────────────────

HIGH_YIELD_INVESTIGATIONS = {"chest_xray", "sputum_afb_smear", "ct_chest", "mantoux_test", "igra_test"}
DEFINITIVE_INVESTIGATIONS = {"sputum_afb_smear", "sputum_culture", "bronchoscopy_bal"}

def grade_diagnostic(
    investigations_ordered: List[str],
    final_diagnosis: str,
    correct_diagnosis: str,
    steps_taken: int,
    max_steps: int,
) -> float:
    """
    Task 5 grader — investigation efficiency + diagnostic accuracy.
    Returns score strictly in (0.06, 0.94).
    """
    ordered_set = set(i.lower() for i in investigations_ordered)
    score = 0.0

    # 1. Correct diagnosis (up to 0.40)
    if final_diagnosis.lower() == correct_diagnosis.lower():
        # Bonus for efficiency — fewer investigations = higher reward
        efficiency = 1.0 - (len(ordered_set) / max(max_steps, 1))
        score += 0.25 + efficiency * 0.15
    else:
        score += 0.0

    # 2. High-yield investigation coverage (up to 0.30)
    hyi_coverage = len(ordered_set & HIGH_YIELD_INVESTIGATIONS) / len(HIGH_YIELD_INVESTIGATIONS)
    score += hyi_coverage * 0.30

    # 3. Definitive test ordered bonus (up to 0.20)
    if ordered_set & DEFINITIVE_INVESTIGATIONS:
        score += 0.20

    # 4. Efficiency penalty — too many low-yield tests
    low_yield = ordered_set - HIGH_YIELD_INVESTIGATIONS - DEFINITIVE_INVESTIGATIONS
    if len(low_yield) > 3:
        score -= (len(low_yield) - 3) * 0.04

    # 5. Speed bonus — diagnosed early
    if final_diagnosis.lower() == correct_diagnosis.lower() and steps_taken <= 5:
        score += 0.08

    return max(0.06, min(0.94, round(score, 4)))
