"""
ClinicalOps — Action and Observation models.

Five tasks:
  Task 1 · ED Triage              (easy)      rank 8 patients by NEWS2
  Task 2 · Medication Review      (medium)    find 5 drug conflicts
  Task 3 · Sepsis Watch           (hard)      detect & escalate sepsis
  Task 4 · Ventilator Weaning     (med-hard)  SBT criteria assessment
  Task 5 · Diagnostic Reasoning   (hard)      narrow diagnosis efficiently
"""

from typing import Any, Dict, List, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation


class ClinicalOpsAction(Action):
    """
    Unified action for all five ClinicalOps tasks.

    action_type values:
      triage_rank          Task 1: submit ranked patient ID list
      flag_conflict        Task 2: flag a drug conflict
      resolve_conflict     Task 2: propose a resolution
      order_investigation  Task 3 & 5: order a diagnostic test
      escalate             Task 3: escalate care
      vent_check           Task 4: assess a weaning criterion
      perform_sbt          Task 4: conduct spontaneous breathing trial
      extubate             Task 4: decision to extubate
      submit_diagnosis     Task 5: submit final diagnosis
      no_action            Any: do nothing this step
    """
    action_type: str = Field(..., description=(
        "One of: triage_rank | flag_conflict | resolve_conflict | "
        "order_investigation | escalate | vent_check | perform_sbt | "
        "extubate | submit_diagnosis | no_action"
    ))

    # Task 1
    ranked_patient_ids: List[str] = Field(default_factory=list)

    # Task 2
    conflict_patient_id: str = Field(default="")
    drug_a: str = Field(default="")
    drug_b: str = Field(default="")
    conflict_type: str = Field(default="")
    resolution: str = Field(default="")

    # Task 3 & 4 & 5
    investigation: str = Field(default="")

    # Task 3
    escalation_level: str = Field(default="")

    # Task 4 & 5 — free text reasoning / diagnosis name
    reasoning: str = Field(default="")


class ClinicalOpsObservation(Observation):
    """Returned by reset() and step() for all five tasks."""

    task: str = Field(default="")
    step: int = Field(default=0)
    done: bool = Field(default=False)
    reward: float = Field(default=0.0)
    score: float = Field(default=0.0)
    patients: List[Dict[str, Any]] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    feedback: str = Field(default="")
    grader_info: Dict[str, Any] = Field(default_factory=dict)
