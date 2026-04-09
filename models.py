"""
ClinicalOps — Action and Observation models.

Three tasks:
  Task 1 · ED Triage         (easy)   — rank 8 patients by NEWS2 urgency
  Task 2 · Medication Review (medium) — find drug conflicts in a discharge list
  Task 3 · Sepsis Watch      (hard)   — detect & escalate sepsis over 10 live steps
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


# ── shared sub-models ────────────────────────────────────────────────────────

class Vitals(Action):
    """Re-used as a plain data holder (not an action itself)."""
    heart_rate: int = 75
    respiratory_rate: int = 16
    spo2: float = 98.0
    systolic_bp: int = 120
    temperature: float = 37.0
    consciousness: str = "Alert"   # Alert / Voice / Pain / Unresponsive


class PatientSummary(Action):
    patient_id: str = ""
    name: str = ""
    age: int = 50
    chief_complaint: str = ""
    vitals: Dict[str, Any] = Field(default_factory=dict)
    labs: Dict[str, Any] = Field(default_factory=dict)
    medications: List[Dict[str, Any]] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    history: List[str] = Field(default_factory=list)


# ── ACTIONS ──────────────────────────────────────────────────────────────────

class ClinicalOpsAction(Action):
    """
    Unified action for all three ClinicalOps tasks.

    Set `action_type` to one of:
      "triage_rank"        — Task 1: submit ordered list of patient IDs
      "flag_conflict"      — Task 2: flag a drug conflict
      "resolve_conflict"   — Task 2: mark a conflict resolved with a fix
      "order_investigation"— Task 3: order a diagnostic test
      "escalate"           — Task 3: escalate care level
      "no_action"          — Task 3: do nothing this step
    """
    action_type: str = Field(..., description=(
        "One of: triage_rank | flag_conflict | resolve_conflict | "
        "order_investigation | escalate | no_action"
    ))

    # Task 1 — triage
    ranked_patient_ids: List[str] = Field(
        default_factory=list,
        description="Task 1: patient IDs ordered highest→lowest priority"
    )

    # Task 2 — medication
    conflict_patient_id: str = Field(
        default="",
        description="Task 2: ID of patient whose meds have a conflict"
    )
    drug_a: str = Field(default="", description="Task 2: first drug in conflict")
    drug_b: str = Field(default="", description="Task 2: second drug / allergen")
    conflict_type: str = Field(
        default="",
        description="Task 2: interaction | dosing_error | duplicate | contraindication"
    )
    resolution: str = Field(default="", description="Task 2: proposed fix text")

    # Task 3 — sepsis watch
    investigation: str = Field(
        default="",
        description=(
            "Task 3: one of blood_cultures | lactate | cbc | "
            "renal_panel | chest_xray | ecg | abg | urine_culture"
        )
    )
    escalation_level: str = Field(
        default="",
        description="Task 3: senior_review | rapid_response | icu_transfer"
    )
    reasoning: str = Field(default="", description="Optional free-text reasoning")


# ── OBSERVATIONS ─────────────────────────────────────────────────────────────

class ClinicalOpsObservation(Observation):
    """Returned by reset() and step() for all three tasks."""

    task: str = Field(default="", description="Active task name")
    step: int = Field(default=0)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    score: float = Field(default=0.0, description="Cumulative normalised score [0,1]")

    # The full patient list (populated by reset, updated each step for Task 3)
    patients: List[Dict[str, Any]] = Field(default_factory=list)

    # Task-specific context
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Task-specific guidance and available actions"
    )

    # Human-readable feedback on last action
    feedback: str = Field(default="")

    # Grader bookkeeping (hidden from agent in prompts, exposed in obs for transparency)
    grader_info: Dict[str, Any] = Field(default_factory=dict)
