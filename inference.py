"""
ClinicalOps — Baseline Inference Script
========================================

Runs an LLM agent through all three ClinicalOps tasks using the OpenAI client.

Environment variables:
    API_BASE_URL  — LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — HuggingFace / API key  (REQUIRED)

Stdout format (strictly enforced):
    [START] task=<name> env=clinicalops model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List, Optional

from openai import OpenAI

# ── Environment variables ─────────────────────────────────────────────────────
API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

ENV_BASE_URL: str = os.getenv("CLINICALOPS_URL", "http://localhost:7860")
BENCHMARK: str    = "clinicalops"
MAX_STEPS: int    = 12
TEMPERATURE: float = 0.2

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

# ── Stdout loggers ────────────────────────────────────────────────────────────

def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool,
             error: Optional[str]) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        f"[STEP] step={step} action={action_safe} "
        f"reward={reward:.2f} done={done_val} error={err_val}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )

# ── LLM call ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert clinical decision support AI.
You will be given clinical data about patients and must respond with a JSON action.

Always respond with ONLY a valid JSON object — no preamble, no markdown fences.
The JSON must have an "action_type" field and any relevant fields for that action type.

Action types and their fields:
  triage_rank:         {"action_type": "triage_rank", "ranked_patient_ids": ["PT001", ...]}
  flag_conflict:       {"action_type": "flag_conflict", "drug_a": "...", "drug_b": "...", "conflict_type": "interaction|dosing_error|duplicate|contraindication"}
  resolve_conflict:    {"action_type": "resolve_conflict", "drug_a": "...", "drug_b": "...", "resolution": "..."}
  order_investigation: {"action_type": "order_investigation", "investigation": "blood_cultures|lactate|cbc|renal_panel|abg|urine_culture|chest_xray|ecg"}
  escalate:            {"action_type": "escalate", "escalation_level": "senior_review|rapid_response|icu_transfer", "reasoning": "..."}
  no_action:           {"action_type": "no_action"}
"""

def call_llm(user_prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON action. Returns fallback on failure."""
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=512,
        )
        text = (resp.choices[0].message.content or "").strip()
        # Strip markdown fences if model adds them
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text.strip())
    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr)
        return {"action_type": "no_action"}

# ── Prompt builders ───────────────────────────────────────────────────────────

def build_triage_prompt(obs: Dict[str, Any]) -> str:
    patients = obs.get("patients", [])
    lines = []
    for p in patients:
        v = p.get("vitals", {})
        lines.append(
            f"  {p['patient_id']} | {p['name']} age {p['age']} | "
            f"CC: {p['chief_complaint']} | "
            f"HR={v.get('heart_rate')} RR={v.get('respiratory_rate')} "
            f"SpO2={v.get('spo2')}% BP={v.get('systolic_bp')} "
            f"Temp={v.get('temperature')} AVPU={v.get('consciousness')}"
        )
    patient_list = "\n".join(lines)
    ids = [p["patient_id"] for p in patients]
    return (
        f"TASK: ED Triage\n"
        f"Rank the following {len(patients)} patients from HIGHEST to LOWEST urgency "
        f"using NEWS2 scoring.\n\nPatients:\n{patient_list}\n\n"
        f"Available IDs: {ids}\n"
        f"Return: {{\"action_type\": \"triage_rank\", \"ranked_patient_ids\": [<ordered list>]}}"
    )

def build_med_prompt(obs: Dict[str, Any], step: int) -> str:
    patient = obs.get("patients", [{}])[0]
    ctx = obs.get("context", {})
    meds = patient.get("medications", [])
    allergies = patient.get("allergies", [])
    flagged = ctx.get("flagged_so_far", [])
    resolved = ctx.get("resolved_so_far", [])
    med_lines = "\n".join(
        f"  {m['name']} {m['dose']} {m['frequency']} [{m['source']}]"
        for m in meds
    )
    return (
        f"TASK: Medication Reconciliation (step {step})\n"
        f"Patient: {patient.get('name')}, age {patient.get('age')}\n"
        f"Allergies: {allergies}\n"
        f"History: {patient.get('history', [])}\n"
        f"Medications:\n{med_lines}\n\n"
        f"Already flagged: {[f'{c[\"drug_a\"]}/{c[\"drug_b\"]}' for c in flagged]}\n"
        f"Already resolved: {[f'{c[\"drug_a\"]}/{c[\"drug_b\"]}' for c in resolved]}\n"
        f"Feedback: {obs.get('feedback', '')}\n\n"
        f"Find the next drug conflict (interaction, contraindication, duplicate, or dosing error). "
        f"If all conflicts are found and resolved, respond with no_action."
    )

def build_sepsis_prompt(obs: Dict[str, Any], step: int) -> str:
    patient = obs.get("patients", [{}])[0]
    ctx = obs.get("context", {})
    v = patient.get("vitals", {})
    labs = patient.get("labs", {})
    return (
        f"TASK: Sepsis Watch (step {step}/10)\n"
        f"Patient: {patient.get('name')}, age {patient.get('age')}\n"
        f"History: {patient.get('history', [])}\n"
        f"Vitals: HR={v.get('heart_rate')} RR={v.get('respiratory_rate')} "
        f"SpO2={v.get('spo2')}% BP={v.get('systolic_bp')} "
        f"Temp={v.get('temperature')} AVPU={v.get('consciousness')}\n"
        f"Labs: WBC={labs.get('wbc')} Creatinine={labs.get('creatinine')} "
        f"Lactate={labs.get('lactate')} Bilirubin={labs.get('bilirubin')} "
        f"Platelets={labs.get('platelets')} PaO2/FiO2={labs.get('pao2_fio2')}\n"
        f"SOFA score: {ctx.get('current_sofa', '?')}\n"
        f"Investigations ordered: {ctx.get('investigations_ordered', [])}\n"
        f"Escalated: {ctx.get('escalated', False)}\n"
        f"Feedback: {obs.get('feedback', '')}\n\n"
        f"Decide: order an investigation, escalate care, or no_action. "
        f"Sepsis threshold is SOFA >= 6. Act early."
    )

# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(task: str) -> None:
    """Run one complete task episode."""
    from openenv.core import EnvClient
    try:
        from clinicalops.client import ClinicalOpsEnv
        from clinicalops.models import ClinicalOpsAction
    except ImportError:
        from client import ClinicalOpsEnv
        from models import ClinicalOpsAction

    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    last_error: Optional[str] = None

    log_start(task=task, model=MODEL_NAME)

    try:
        async with ClinicalOpsEnv(base_url=ENV_BASE_URL) as env:
            result = await env.reset(task=task)
            obs = result.observation.model_dump() if hasattr(result, "observation") else {}

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                # Build prompt based on task
                if task == "ed_triage":
                    prompt = build_triage_prompt(obs)
                elif task == "medication_review":
                    prompt = build_med_prompt(obs, step)
                else:
                    prompt = build_sepsis_prompt(obs, step)

                action_dict = call_llm(prompt)
                action_str = json.dumps(action_dict)

                try:
                    action = ClinicalOpsAction(**action_dict)
                    result = await env.step(action)
                    reward = result.reward or 0.0
                    done = result.done
                    obs = result.observation.model_dump() if hasattr(result, "observation") else {}
                    last_error = None
                    score = obs.get("score", 0.0)
                except Exception as exc:
                    reward = 0.0
                    done = False
                    last_error = str(exc)[:120]

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward,
                         done=done, error=last_error)

                if done:
                    break

            # Final score from last observation
            score = float(obs.get("score", 0.0)) if obs else 0.0
            success = score >= 0.40

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task {task} error: {exc}", file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    tasks = ["ed_triage", "medication_review", "sepsis_watch"]
    for task in tasks:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
