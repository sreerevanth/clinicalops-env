"""
ClinicalOps — Core Environment implementation.

Implements the OpenEnv Environment interface for all three tasks.
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
    )
    from ..scenarios import (
        TRIAGE_PATIENTS, MED_PATIENT, MED_GROUND_TRUTH,
        SEPSIS_PATIENT_BASE, SEPSIS_TRAJECTORY,
    )
except ImportError:
    from models import ClinicalOpsAction, ClinicalOpsObservation
    from graders import (
        compute_news2, grade_triage,
        grade_medication, grade_sepsis, SEPSIS_INVESTIGATIONS,
    )
    from scenarios import (
        TRIAGE_PATIENTS, MED_PATIENT, MED_GROUND_TRUTH,
        SEPSIS_PATIENT_BASE, SEPSIS_TRAJECTORY,
    )


TASKS = ["ed_triage", "medication_review", "sepsis_watch"]
MAX_STEPS = {"ed_triage": 1, "medication_review": 8, "sepsis_watch": 10}


class ClinicalOpsEnvironment(Environment):
    """
    ClinicalOps multi-task clinical workflow environment.

    Tasks:
      ed_triage          — rank 8 ED patients by NEWS2 urgency (1-step)
      medication_review  — find & resolve drug conflicts before discharge (8-step)
      sepsis_watch       — detect and escalate sepsis before SOFA ≥ 6 (10-step)
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._task: str = "ed_triage"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done: bool = False
        self._cumulative_reward: float = 0.0

        # Task 2 bookkeeping
        self._flagged_conflicts: List[Dict[str, Any]] = []
        self._resolved_conflicts: List[Dict[str, Any]] = []

        # Task 3 bookkeeping
        self._investigations_ordered: List[str] = []
        self._escalation_level: str = ""
        self._escalation_step: int = 0
        self._sofa_at_escalation: int = 0

    # ── OpenEnv interface ────────────────────────────────────────────────────

    def reset(self, task: Optional[str] = None, **kwargs) -> ClinicalOpsObservation:
        """Reset episode. Pass task='ed_triage'|'medication_review'|'sepsis_watch'."""
        self._task = task if task in TASKS else "ed_triage"
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._done = False
        self._cumulative_reward = 0.0

        # Reset per-task state
        self._flagged_conflicts = []
        self._resolved_conflicts = []
        self._investigations_ordered = []
        self._escalation_level = ""
        self._escalation_step = 0
        self._sofa_at_escalation = 0

        return self._build_obs(reward=0.0, feedback="Episode started. Good luck.")

    def step(self, action: ClinicalOpsAction) -> ClinicalOpsObservation:  # type: ignore[override]
        """Execute one action. Returns updated observation with reward and feedback."""
        if self._done:
            return self._build_obs(reward=0.0, feedback="Episode already finished.")

        self._state.step_count += 1
        reward, feedback = self._dispatch(action)
        self._cumulative_reward += reward

        max_steps = MAX_STEPS[self._task]
        if self._state.step_count >= max_steps:
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
            if at == "flag_conflict":
                return self._handle_flag(action)
            elif at == "resolve_conflict":
                return self._handle_resolve(action)
            elif at == "no_action":
                self._done = True
                return 0.0, "Discharge approved. Episode complete."
            else:
                return -0.05, f"Unknown action '{at}' for medication_review task."
        elif self._task == "sepsis_watch":
            if at == "order_investigation":
                return self._handle_investigation(action)
            elif at in ("escalate", "escalate_care"):
                return self._handle_escalation(action)
            elif at == "no_action":
                return self._handle_no_action()
            else:
                return -0.05, f"Unknown action '{at}' for sepsis_watch task."
        return 0.0, "No-op."

    # ── Task 1 ────────────────────────────────────────────────────────────────

    def _handle_triage(self, action: ClinicalOpsAction):
        ranked = action.ranked_patient_ids
        score = grade_triage(ranked, TRIAGE_PATIENTS)
        self._done = True
        ordered = sorted(TRIAGE_PATIENTS, key=lambda p: -compute_news2(p["vitals"]))
        correct_ids = [p["patient_id"] for p in ordered]
        feedback = (
            f"Triage complete. Score: {score:.3f}. "
            f"Correct order: {correct_ids}. "
            f"Your order: {ranked}."
        )
        return score, feedback

    # ── Task 2 ────────────────────────────────────────────────────────────────

    def _handle_flag(self, action: ClinicalOpsAction):
        conflict = {
            "drug_a": action.drug_a,
            "drug_b": action.drug_b,
            "conflict_type": action.conflict_type,
        }
        pair = frozenset([action.drug_a.lower(), action.drug_b.lower()])
        gt_pairs = [frozenset([c["drug_a"].lower(), c["drug_b"].lower()])
                    for c in MED_GROUND_TRUTH]

        if pair in gt_pairs:
            gt = MED_GROUND_TRUTH[gt_pairs.index(pair)]
            already = any(frozenset([f["drug_a"].lower(), f["drug_b"].lower()]) == pair
                          for f in self._flagged_conflicts)
            if already:
                return -0.02, f"Already flagged {action.drug_a}/{action.drug_b}."
            self._flagged_conflicts.append(conflict)
            sev = gt["severity"]
            reward = 0.20 if sev == "severe" else (0.12 if sev == "moderate" else 0.06)
            return reward, f"Correct! {gt['description']}"
        else:
            return -0.05, f"No significant conflict between {action.drug_a} and {action.drug_b}."

    def _handle_resolve(self, action: ClinicalOpsAction):
        pair = frozenset([action.drug_a.lower(), action.drug_b.lower()])
        flagged_pairs = [frozenset([f["drug_a"].lower(), f["drug_b"].lower()])
                         for f in self._flagged_conflicts]
        if pair not in flagged_pairs:
            return -0.03, "Flag the conflict first before resolving."

        gt_pairs = [frozenset([c["drug_a"].lower(), c["drug_b"].lower()])
                    for c in MED_GROUND_TRUTH]
        already_resolved = any(frozenset([r["drug_a"].lower(), r["drug_b"].lower()]) == pair
                                for r in self._resolved_conflicts)
        if already_resolved:
            return -0.01, "Already resolved this conflict."

        resolution = {**self._flagged_conflicts[flagged_pairs.index(pair)],
                      "resolution": action.resolution}
        self._resolved_conflicts.append(resolution)

        if pair in gt_pairs:
            gt = MED_GROUND_TRUTH[gt_pairs.index(pair)]
            correct_hint = gt["resolution"].lower()
            provided = action.resolution.lower()
            keywords = [w for w in correct_hint.split() if len(w) > 4]
            matches = sum(1 for k in keywords if k in provided)
            quality = min(1.0, matches / max(len(keywords), 1))
            reward = 0.05 + quality * 0.10
            return reward, f"Resolution recorded. Suggested: {gt['resolution']}"
        return 0.03, "Resolution noted."

    # ── Task 3 ────────────────────────────────────────────────────────────────

    def _handle_investigation(self, action: ClinicalOpsAction):
        inv = action.investigation.lower().replace(" ", "_")
        if inv in self._investigations_ordered:
            return -0.02, f"{inv} already ordered."
        self._investigations_ordered.append(inv)

        relevant = inv in SEPSIS_INVESTIGATIONS
        reward = 0.08 if relevant else -0.03
        fb = (f"Ordered {inv}. {'Relevant for sepsis workup.' if relevant else 'Low yield for this presentation.'}")
        return reward, fb

    def _handle_escalation(self, action: ClinicalOpsAction):
        if self._escalation_level:
            return -0.02, f"Already escalated to {self._escalation_level}."

        level = action.escalation_level or "senior_review"
        self._escalation_level = level
        self._escalation_step = self._state.step_count

        step_idx = min(self._state.step_count - 1, len(SEPSIS_TRAJECTORY) - 1)
        self._sofa_at_escalation = SEPSIS_TRAJECTORY[step_idx]["sofa"]

        if level in ("rapid_response", "icu_transfer"):
            if self._sofa_at_escalation < 4:
                reward = 0.40
                fb = f"Excellent early escalation! SOFA={self._sofa_at_escalation} at time of call. ICU notified."
            elif self._sofa_at_escalation < 6:
                reward = 0.28
                fb = f"Correct escalation, slightly late. SOFA={self._sofa_at_escalation}."
            else:
                reward = 0.10
                fb = f"Escalation too late — SOFA already {self._sofa_at_escalation}."
        else:
            reward = 0.12
            fb = f"Senior review requested. SOFA={self._sofa_at_escalation}. Consider rapid_response."

        self._done = True
        return reward, fb

    def _handle_no_action(self):
        step_idx = min(self._state.step_count - 1, len(SEPSIS_TRAJECTORY) - 1)
        sofa = SEPSIS_TRAJECTORY[step_idx]["sofa"]
        if sofa >= 6:
            return -0.10, f"DANGER: SOFA now {sofa}. Patient is critically ill. Escalate NOW."
        elif sofa >= 4:
            return -0.04, f"Warning: SOFA={sofa} and rising. Consider escalation."
        return 0.0, f"Monitoring. SOFA={sofa}. Watch closely."

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_obs(self, reward: float, feedback: str) -> ClinicalOpsObservation:
        step_idx = min(self._state.step_count, len(SEPSIS_TRAJECTORY) - 1)
        final_sofa = SEPSIS_TRAJECTORY[step_idx]["sofa"] if self._task == "sepsis_watch" else 0

        # Final score calculation
        score = 0.0
        if self._done:
            if self._task == "ed_triage":
                score = reward  # single-step task, reward IS the score
            elif self._task == "medication_review":
                score = grade_medication(
                    self._flagged_conflicts, self._resolved_conflicts, MED_GROUND_TRUTH
                )
            elif self._task == "sepsis_watch":
                score = grade_sepsis(
                    self._investigations_ordered,
                    self._escalation_level,
                    self._escalation_step,
                    self._sofa_at_escalation,
                    final_sofa,
                    self._state.step_count,
                )

        # Build patient list depending on task
        if self._task == "ed_triage":
            patients = TRIAGE_PATIENTS
        elif self._task == "medication_review":
            patients = [MED_PATIENT]
        else:
            traj = SEPSIS_TRAJECTORY[step_idx]
            patient = copy.deepcopy(SEPSIS_PATIENT_BASE)
            patient["vitals"] = traj["vitals"]
            patient["labs"] = traj["labs"]
            patient["sofa_score"] = traj["sofa"]
            patients = [patient]

        context = self._build_context()

        return ClinicalOpsObservation(
            task=self._task,
            step=self._state.step_count,
            done=self._done,
            reward=round(reward, 4),
            score=round(score, 4),
            patients=patients,
            context=context,
            feedback=feedback,
            grader_info={
                "cumulative_reward": round(self._cumulative_reward, 4),
                "investigations_ordered": self._investigations_ordered,
                "escalation_level": self._escalation_level,
                "flagged_conflicts": len(self._flagged_conflicts),
                "resolved_conflicts": len(self._resolved_conflicts),
            },
        )

    def _build_context(self) -> Dict[str, Any]:
        if self._task == "ed_triage":
            return {
                "objective": "Rank all 8 patients from highest to lowest urgency using NEWS2 clinical reasoning.",
                "available_actions": ["triage_rank"],
                "hint": "Consider: RR, SpO2, BP, HR, Temp, Consciousness. Higher NEWS2 = more urgent.",
                "num_patients": len(TRIAGE_PATIENTS),
                "patient_ids": [p["patient_id"] for p in TRIAGE_PATIENTS],
            }
        elif self._task == "medication_review":
            return {
                "objective": "Identify all drug conflicts, contraindications, and duplicates before the patient is discharged.",
                "available_actions": ["flag_conflict", "resolve_conflict", "no_action"],
                "hint": "Check: drug interactions, allergy contraindications, duplicate medications, renal dosing.",
                "flagged_so_far": self._flagged_conflicts,
                "resolved_so_far": self._resolved_conflicts,
            }
        else:
            step_idx = min(self._state.step_count, len(SEPSIS_TRAJECTORY) - 1)
            sofa = SEPSIS_TRAJECTORY[step_idx]["sofa"]
            return {
                "objective": "Detect sepsis onset and escalate before SOFA reaches 6. Order relevant investigations.",
                "available_actions": ["order_investigation", "escalate", "no_action"],
                "valid_investigations": list(SEPSIS_INVESTIGATIONS) + ["urine_culture", "chest_xray", "ecg"],
                "valid_escalation_levels": ["senior_review", "rapid_response", "icu_transfer"],
                "current_sofa": sofa,
                "hint": "Sepsis-6 bundle: blood_cultures, lactate, cbc, renal_panel, abg + early escalation.",
                "investigations_ordered": self._investigations_ordered,
                "escalated": bool(self._escalation_level),
            }
