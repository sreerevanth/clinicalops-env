"""ClinicalOps Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import ClinicalOpsAction, ClinicalOpsObservation


class ClinicalOpsEnv(EnvClient[ClinicalOpsAction, ClinicalOpsObservation, State]):
    """
    Client for the ClinicalOps Environment.

    Maintains a persistent WebSocket connection to the environment server.

    Example (async):
        async with ClinicalOpsEnv(base_url="http://localhost:7860") as env:
            result = await env.reset(task="ed_triage")
            result = await env.step(ClinicalOpsAction(
                action_type="triage_rank",
                ranked_patient_ids=["PT005", "PT001", "PT003", ...]
            ))

    Example (Docker):
        client = ClinicalOpsEnv.from_docker_image("clinicalops-env:latest")
    """

    def _step_payload(self, action: ClinicalOpsAction) -> Dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[ClinicalOpsObservation]:
        obs_data = payload.get("observation", {})
        observation = ClinicalOpsObservation(**obs_data) if obs_data else ClinicalOpsObservation()
        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        return State(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
        )
