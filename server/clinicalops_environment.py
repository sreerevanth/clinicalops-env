"""
ClinicalOps — Core Environment implementation.
5 tasks: ed_triage, medication_review, sepsis_watch, vent_weaning, diagnostic_reasoning
"""

from __future__ import annotations
import copy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import ClinicalOpsAction, ClinicalOpsObservation
    from ..graders import (
        compute_news2, grade_triage,
        grade_medication, grade_sepsis, SEPSIS_INVESTIGATIONS,
        grade_vent_weaning, REQUIRED_VENT_CHECKS,
        grade_diagnostic, HIGH_YIELD_INVESTIGATIONS, DEFINITIVE_INVESTIGATIONS,
    )
    from ..scenarios import (
        TRIAGE_PATIENTS, MED_PATIENT, MED_GROUND_TRUTH,
        SEPSIS_PATIENT_BASE, SEPSIS_TRAJECTORY,
        VENT_PATIENT_BASE, VENT_WEANING_CHECKLIST,
        DIAGNOSTIC_PATIENT, DIAGNOSTIC_INVESTIGATIONS,
        DIFFERENTIAL_DIAGNOSES, CORRECT_DIAGNOSIS,
    )
except ImportError:
    from models import ClinicalOpsAction, ClinicalOpsObservation
    from graders import (
        compute_news2, grade_triage,
        grade_medication, grade_sepsis, SEPSIS_INVESTIGATIONS,
        grade_vent_weaning, REQUIRED_VENT_CHECKS,
        grade_diagnostic, HIGH_YIELD_INVESTIGATIONS, DEFINITIVE_INVESTIGATIONS,
    )
    from scenarios import (
        TRIAGE_PATIENTS, MED_PATIENT, MED_GROUND_TRUTH,
        SEPSIS_PATIENT_BASE, SEPSIS_TRAJECTORY,
        VENT_PATIENT_BASE, VENT_WEANING_CHECKLIST,
        DIAGNOSTIC_PATIENT, DIAGNOSTIC_INVESTIGATIONS,
        DIFFERENTIAL_DIAGNOSES, CORRECT_DIAGNOSIS,
    )


TASKS = ["ed_triage", "medication_review", "sepsis_watch", "vent_weaning", "diagnostic_reasoning"]
MAX_STEPS = {
    "ed_triage": 1,
    "medication_review": 8,
    "sepsis_watch": 10,
    "vent_weaning": 8,
    "diagnostic_reasoning": 12,
}


class ClinicalOpsEnvironment(Environment):
    """
    ClinicalOps — 5-task hospital clinical workflow environment.

    Tasks:
      ed_triage              Easy       Rank 8 ED patients by NEWS2 urgency
      medication_review      Medium     Find 5 drug conflicts before discharge
      sepsis_watch           Hard       Detect & escalate sepsis before SOFA >= 6
      vent_weaning           Med-Hard   Assess ICU patient for ventilator weaning
      diagnostic_reasoning   Hard       Order investigations to reach correct diagnosis
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task: str = "ed_triage"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done: bool = False
        self._cumulative_reward: float = 0.0

        # Task 2
        self._flagged_conflicts: List[Dict[str, Any]] = []
        self._resolved_conflicts: List[Dict[str, Any]] = []

        # Task 3
        self._investigations_ordered: List[str] = []
        self._escalation_level: str = ""
        self._escalation_step: int = 0
        self._sofa_at_escalation: int = 0

        # Task 4
        self._vent_checks: List[str] = []
        self._sbt_performed: bool = False
        self._extubation_decision: str = ""
        self._premature_extubation: bool = False

        # Task 5
        self._diag_investigations: List[str] = []
        self._final_diagnosis: str = ""
        self._remaining_differentials: List[str] = list(DIFFERENTIAL_DIAGNOSES)

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self, task: Optional[str] = None, **kwargs) -> ClinicalOpsObservation:
        self._task = task if task in TASKS else "ed_triage"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._cumulative_reward = 0.0

        self._flagged_conflicts = []
        self._resolved_conflicts = []
        self._investigations_ordered = []
        self._escalation_level = ""
        self._escalation_step = 0
        self._sofa_at_escalation = 0
        self._vent_checks = []
        self._sbt_performed = False
        self._extubation_decision = ""
        self._premature_extubation = False
        self._diag_investigations = []
        self._final_diagnosis = ""
        self._remaining_differentials = list(DIFFERENTIAL_DIAGNOSES)

        return self._build_obs(reward=0.0, feedback="Episode started. Good luck.")

    def step(self, action: ClinicalOpsAction) -> ClinicalOpsObservation:
        if self._done:
            return self._build_obs(reward=0.0, feedback="Episode already finished.")

        self._state.step_count += 1
        reward, feedback = self._dispatch(action)
        self._cumulative_reward += reward

        if self._state.step_count >= MAX_STEPS[self._task]:
            self._done = True

        return self._build_obs(reward=reward, feedback=feedback)

    @property
    def state(self) -> State:
        return self._state

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def _dispatch(self, action: ClinicalOpsAction):
        at = action.action_type
        if self._task == "ed_triage":
            return self._handle_triage(action)
        elif self._task == "medication_review":
            if at == "flag_conflict":    return self._handle_flag(action)
            elif at == "resolve_conflict": return self._handle_resolve(action)
            elif at == "no_action":
                self._done = True
                return 0.0, "Discharge approved. Episode complete."
            return -0.05, "Unknown action for medication_review."
        elif self._task == "sepsis_watch":
            if at == "order_investigation": return self._handle_investigation(action)
            elif at in ("escalate", "escalate_care"): return self._handle_escalation(action)
            elif at == "no_action": return self._handle_no_action_sepsis()
            return -0.05, "Unknown action for sepsis_watch."
        elif self._task == "vent_weaning":
            if at == "vent_check":     return self._handle_vent_check(action)
            elif at == "perform_sbt":  return self._handle_sbt(action)
            elif at == "extubate":     return self._handle_extubate(action)
            elif at == "no_action":    return -0.03, "No assessment made this step."
            return -0.03, "Unknown action for vent_weaning."
        elif self._task == "diagnostic_reasoning":
            if at == "order_investigation": return self._handle_diag_investigation(action)
            elif at == "submit_diagnosis":  return self._handle_diagnosis(action)
            elif at == "no_action":         return -0.02, "No investigation ordered."
            return -0.03, "Unknown action for diagnostic_reasoning."
        return 0.0, "No-op."

    # ── Task 1 ────────────────────────────────────────────────────────────────

    def _handle_triage(self, action: ClinicalOpsAction):
        ranked = action.ranked_patient_ids
        score = grade_triage(ranked, TRIAGE_PATIENTS)
        self._done = True
        ordered = sorted(TRIAGE_PATIENTS, key=lambda p: -compute_news2(p["vitals"]))
        correct_ids = [p["patient_id"] for p in ordered]
        return score, (
            "Triage complete. Score: " + str(round(score, 3)) +
            ". Correct order: " + str(correct_ids) +
            ". Your order: " + str(ranked) + "."
        )

    # ── Task 2 ────────────────────────────────────────────────────────────────

    def _handle_flag(self, action: ClinicalOpsAction):
        conflict = {"drug_a": action.drug_a, "drug_b": action.drug_b, "conflict_type": action.conflict_type}
        pair = frozenset([action.drug_a.lower(), action.drug_b.lower()])
        gt_pairs = [frozenset([c["drug_a"].lower(), c["drug_b"].lower()]) for c in MED_GROUND_TRUTH]

        if pair in gt_pairs:
            gt = MED_GROUND_TRUTH[gt_pairs.index(pair)]
            already = any(frozenset([f["drug_a"].lower(), f["drug_b"].lower()]) == pair for f in self._flagged_conflicts)
            if already:
                return -0.02, "Already flagged " + action.drug_a + "/" + action.drug_b + "."
            self._flagged_conflicts.append(conflict)
            sev = gt["severity"]
            reward = 0.20 if sev == "severe" else (0.12 if sev == "moderate" else 0.06)
            return reward, "Correct! " + gt["description"]
        return -0.05, "No significant conflict between " + action.drug_a + " and " + action.drug_b + "."

    def _handle_resolve(self, action: ClinicalOpsAction):
        pair = frozenset([action.drug_a.lower(), action.drug_b.lower()])
        flagged_pairs = [frozenset([f["drug_a"].lower(), f["drug_b"].lower()]) for f in self._flagged_conflicts]
        if pair not in flagged_pairs:
            return -0.03, "Flag the conflict first before resolving."
        gt_pairs = [frozenset([c["drug_a"].lower(), c["drug_b"].lower()]) for c in MED_GROUND_TRUTH]
        already_resolved = any(frozenset([r["drug_a"].lower(), r["drug_b"].lower()]) == pair for r in self._resolved_conflicts)
        if already_resolved:
            return -0.01, "Already resolved this conflict."
        resolution = {**self._flagged_conflicts[flagged_pairs.index(pair)], "resolution": action.resolution}
        self._resolved_conflicts.append(resolution)
        if pair in gt_pairs:
            gt = MED_GROUND_TRUTH[gt_pairs.index(pair)]
            keywords = [w for w in gt["resolution"].lower().split() if len(w) > 4]
            matches = sum(1 for k in keywords if k in action.resolution.lower())
            quality = min(1.0, matches / max(len(keywords), 1))
            reward = 0.05 + quality * 0.10
            return reward, "Resolution recorded. Suggested: " + gt["resolution"]
        return 0.03, "Resolution noted."

    # ── Task 3 ────────────────────────────────────────────────────────────────

    def _handle_investigation(self, action: ClinicalOpsAction):
        inv = action.investigation.lower().replace(" ", "_")
        if inv in self._investigations_ordered:
            return -0.02, inv + " already ordered."
        self._investigations_ordered.append(inv)
        relevant = inv in SEPSIS_INVESTIGATIONS
        reward = 0.08 if relevant else -0.03
        return reward, "Ordered " + inv + ". " + ("Relevant for sepsis workup." if relevant else "Low yield.")

    def _handle_escalation(self, action: ClinicalOpsAction):
        if self._escalation_level:
            return -0.02, "Already escalated to " + self._escalation_level + "."
        level = action.escalation_level or "senior_review"
        self._escalation_level = level
        self._escalation_step = self._state.step_count
        step_idx = min(self._state.step_count - 1, len(SEPSIS_TRAJECTORY) - 1)
        self._sofa_at_escalation = SEPSIS_TRAJECTORY[step_idx]["sofa"]
        if level in ("rapid_response", "icu_transfer"):
            if self._sofa_at_escalation < 4:
                reward, fb = 0.40, "Excellent early escalation! SOFA=" + str(self._sofa_at_escalation)
            elif self._sofa_at_escalation < 6:
                reward, fb = 0.28, "Correct escalation, slightly late. SOFA=" + str(self._sofa_at_escalation)
            else:
                reward, fb = 0.10, "Escalation too late. SOFA=" + str(self._sofa_at_escalation)
        else:
            reward, fb = 0.12, "Senior review requested. SOFA=" + str(self._sofa_at_escalation)
        self._done = True
        return reward, fb

    def _handle_no_action_sepsis(self):
        step_idx = min(self._state.step_count - 1, len(SEPSIS_TRAJECTORY) - 1)
        sofa = SEPSIS_TRAJECTORY[step_idx]["sofa"]
        if sofa >= 6:   return -0.10, "DANGER: SOFA=" + str(sofa) + ". Escalate NOW."
        elif sofa >= 4: return -0.04, "Warning: SOFA=" + str(sofa) + " and rising."
        return 0.0, "Monitoring. SOFA=" + str(sofa) + "."

    # ── Task 4 ────────────────────────────────────────────────────────────────

    def _handle_vent_check(self, action: ClinicalOpsAction):
        check = (action.investigation or action.reasoning or "").lower().replace(" ", "_")
        if not check:
            return -0.02, "Specify which weaning criterion to assess."
        if check in self._vent_checks:
            return -0.01, "Already assessed " + check + "."
        self._vent_checks.append(check)
        required = check in REQUIRED_VENT_CHECKS
        reward = 0.10 if required else 0.02
        descriptions = {c["check"]: c["description"] for c in VENT_WEANING_CHECKLIST}
        desc = descriptions.get(check, "Assessment recorded.")
        return reward, "Checked: " + desc

    def _handle_sbt(self, action: ClinicalOpsAction):
        if self._sbt_performed:
            return -0.01, "SBT already performed."
        prereqs = {"reduce_fio2", "reduce_peep", "check_rsbi"}
        missing = prereqs - set(self._vent_checks)
        if missing:
            self._premature_extubation = True
            return -0.10, "Prerequisites missing before SBT: " + str(missing) + ". SBT performed prematurely."
        self._sbt_performed = True
        self._vent_checks.append("perform_sbt")
        return 0.20, "SBT performed successfully. Patient tolerated 30 minutes of spontaneous breathing. RSBI=68. Consider extubation."

    def _handle_extubate(self, action: ClinicalOpsAction):
        if not self._sbt_performed:
            self._premature_extubation = True
            self._extubation_decision = "extubate"
            self._done = True
            return -0.20, "Premature extubation without SBT — high reintubation risk."
        self._extubation_decision = "extubate"
        self._done = True
        return 0.19, "Patient successfully extubated. Monitoring on high-flow nasal oxygen."

    # ── Task 5 ────────────────────────────────────────────────────────────────

    def _handle_diag_investigation(self, action: ClinicalOpsAction):
        inv = action.investigation.lower().replace(" ", "_")
        if inv in self._diag_investigations:
            return -0.02, inv + " already ordered."
        self._diag_investigations.append(inv)

        if inv in DIAGNOSTIC_INVESTIGATIONS:
            info = DIAGNOSTIC_INVESTIGATIONS[inv]
            # Narrow differentials
            if info["narrows_to"]:
                self._remaining_differentials = [
                    d for d in self._remaining_differentials
                    if d in info["narrows_to"] or d not in (info.get("rules_out", []))
                ]
            for ruled_out in info.get("rules_out", []):
                if ruled_out in self._remaining_differentials:
                    self._remaining_differentials.remove(ruled_out)

            yield_val = info["yield"]
            reward = 0.15 if yield_val == "definitive" else (0.10 if yield_val == "high" else 0.04)
            return reward, "Result: " + info["result"] + " Remaining differentials: " + str(len(self._remaining_differentials))
        return 0.02, "Investigation ordered. Result pending — not in standard panel."

    def _handle_diagnosis(self, action: ClinicalOpsAction):
        diagnosis = (action.reasoning or "").lower().replace(" ", "_")
        self._final_diagnosis = diagnosis
        self._done = True
        score = grade_diagnostic(
            self._diag_investigations, diagnosis, CORRECT_DIAGNOSIS,
            self._state.step_count, MAX_STEPS["diagnostic_reasoning"]
        )
        if diagnosis == CORRECT_DIAGNOSIS:
            return score, "CORRECT diagnosis: " + CORRECT_DIAGNOSIS + ". Score: " + str(round(score, 3))
        return 0.06, "Incorrect. Correct diagnosis was: " + CORRECT_DIAGNOSIS + ". Investigations suggested: " + str(self._diag_investigations)

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, reward: float, feedback: str) -> ClinicalOpsObservation:
        score = 0.0
        if self._done:
            if self._task == "ed_triage":
                score = max(0.06, reward)
            elif self._task == "medication_review":
                score = grade_medication(self._flagged_conflicts, self._resolved_conflicts, MED_GROUND_TRUTH)
            elif self._task == "sepsis_watch":
                step_idx = min(self._state.step_count, len(SEPSIS_TRAJECTORY) - 1)
                score = grade_sepsis(
                    self._investigations_ordered, self._escalation_level,
                    self._escalation_step, self._sofa_at_escalation,
                    SEPSIS_TRAJECTORY[step_idx]["sofa"], self._state.step_count,
                )
            elif self._task == "vent_weaning":
                score = grade_vent_weaning(
                    self._vent_checks, self._sbt_performed,
                    self._extubation_decision, self._premature_extubation,
                )
            elif self._task == "diagnostic_reasoning":
                score = grade_diagnostic(
                    self._diag_investigations, self._final_diagnosis, CORRECT_DIAGNOSIS,
                    self._state.step_count, MAX_STEPS["diagnostic_reasoning"],
                )

        # Build patient list
        if self._task == "ed_triage":
            patients = TRIAGE_PATIENTS
        elif self._task == "medication_review":
            patients = [MED_PATIENT]
        elif self._task == "sepsis_watch":
            step_idx = min(self._state.step_count, len(SEPSIS_TRAJECTORY) - 1)
            traj = SEPSIS_TRAJECTORY[step_idx]
            patient = copy.deepcopy(SEPSIS_PATIENT_BASE)
            patient["vitals"] = traj["vitals"]
            patient["labs"] = traj["labs"]
            patient["sofa_score"] = traj["sofa"]
            patients = [patient]
        elif self._task == "vent_weaning":
            patients = [VENT_PATIENT_BASE]
        else:
            patients = [DIAGNOSTIC_PATIENT]

        return ClinicalOpsObservation(
            task=self._task,
            step=self._state.step_count,
            done=self._done,
            reward=round(reward, 4),
            score=round(score, 4),
            patients=patients,
            context=self._build_context(),
            feedback=feedback,
            grader_info={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "investigations_ordered": self._investigations_ordered,
                "escalation_level": self._escalation_level,
                "flagged_conflicts": len(self._flagged_conflicts),
                "resolved_conflicts": len(self._resolved_conflicts),
                "vent_checks": self._vent_checks,
                "sbt_performed": self._sbt_performed,
                "remaining_differentials": self._remaining_differentials,
            },
        )

    def _build_context(self) -> Dict[str, Any]:
        if self._task == "ed_triage":
            return {
                "objective": "Rank all 8 patients from highest to lowest urgency using NEWS2.",
                "available_actions": ["triage_rank"],
                "patient_ids": [p["patient_id"] for p in TRIAGE_PATIENTS],
            }
        elif self._task == "medication_review":
            return {
                "objective": "Find all drug conflicts before the patient is discharged.",
                "available_actions": ["flag_conflict", "resolve_conflict", "no_action"],
                "flagged_so_far": self._flagged_conflicts,
                "resolved_so_far": self._resolved_conflicts,
            }
        elif self._task == "sepsis_watch":
            step_idx = min(self._state.step_count, len(SEPSIS_TRAJECTORY) - 1)
            return {
                "objective": "Detect sepsis and escalate before SOFA reaches 6.",
                "available_actions": ["order_investigation", "escalate", "no_action"],
                "valid_investigations": list(SEPSIS_INVESTIGATIONS) + ["urine_culture", "chest_xray", "ecg"],
                "valid_escalation_levels": ["senior_review", "rapid_response", "icu_transfer"],
                "current_sofa": SEPSIS_TRAJECTORY[step_idx]["sofa"],
                "investigations_ordered": self._investigations_ordered,
                "escalated": bool(self._escalation_level),
            }
        elif self._task == "vent_weaning":
            return {
                "objective": "Systematically assess patient for ventilator weaning using SBT criteria.",
                "available_actions": ["vent_check", "perform_sbt", "extubate", "no_action"],
                "valid_checks": list(REQUIRED_VENT_CHECKS),
                "checks_completed": self._vent_checks,
                "sbt_performed": self._sbt_performed,
                "hint": "Complete all SBT prerequisites before performing SBT. SBT before extubation.",
            }
        else:
            return {
                "objective": "Order investigations to identify the correct diagnosis efficiently.",
                "available_actions": ["order_investigation", "submit_diagnosis", "no_action"],
                "valid_investigations": list(DIAGNOSTIC_INVESTIGATIONS.keys()),
                "remaining_differentials": self._remaining_differentials,
                "investigations_ordered": self._diag_investigations,
                "hint": "Start with high-yield tests. Submit diagnosis using action_type=submit_diagnosis with reasoning=<diagnosis_name>.",
                "possible_diagnoses": DIFFERENTIAL_DIAGNOSES,
            }
