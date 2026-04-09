"""
ClinicalOps — Baseline Inference Script
========================================
MANDATORY environment variables:
    API_BASE_URL  — LLM API endpoint  (default: https://router.huggingface.co/v1)
    MODEL_NAME    — Model identifier   (default: Qwen/Qwen2.5-72B-Instruct)
    HF_TOKEN      — HuggingFace / API key  (REQUIRED)

STDOUT FORMAT:
    [START] task=<task_name> env=clinicalops model=<model_name>
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
SUCCESS_SCORE_THRESHOLD: float = 0.40

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

SYSTEM_PROMPT = (
    "You are an expert clinical decision support AI. "
    "Respond with ONLY a valid JSON object — no preamble, no markdown fences.\n\n"
    "Action types:\n"
    "  triage_rank:         {\"action_type\": \"triage_rank\", \"ranked_patient_ids\": [...]}\n"
    "  flag_conflict:       {\"action_type\": \"flag_conflict\", \"drug_a\": \"...\", \"drug_b\": \"...\", \"conflict_type\": \"interaction|dosing_error|duplicate|contraindication\"}\n"
    "  resolve_conflict:    {\"action_type\": \"resolve_conflict\", \"drug_a\": \"...\", \"drug_b\": \"...\", \"resolution\": \"...\"}\n"
    "  order_investigation: {\"action_type\": \"order_investigation\", \"investigation\": \"blood_cultures|lactate|cbc|renal_panel|abg|urine_culture|chest_xray|ecg\"}\n"
    "  escalate:            {\"action_type\": \"escalate\", \"escalation_level\": \"senior_review|rapid_response|icu_transfer\", \"reasoning\": \"...\"}\n"
    "  no_action:           {\"action_type\": \"no_action\"}"
)


def call_llm(user_prompt: str) -> Dict[str, Any]:
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
        if text.startswith("```"):
            parts = text.split("```")
            text = parts[1] if len(parts) > 1 else text
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
        line = (
            "  " + p["patient_id"] + " | " + p["name"] +
            " age " + str(p["age"]) +
            " | CC: " + p["chief_complaint"] +
            " | HR=" + str(v.get("heart_rate")) +
            " RR=" + str(v.get("respiratory_rate")) +
            " SpO2=" + str(v.get("spo2")) + "%" +
            " BP=" + str(v.get("systolic_bp")) +
            " Temp=" + str(v.get("temperature")) +
            " AVPU=" + str(v.get("consciousness"))
        )
        lines.append(line)
    patient_block = "\n".join(lines)
    ids = [p["patient_id"] for p in patients]
    return (
        "TASK: ED Triage\n"
        "Rank the following " + str(len(patients)) + " patients from HIGHEST to LOWEST urgency "
        "using NEWS2 scoring.\n\nPatients:\n" + patient_block + "\n\n"
        "Available IDs: " + str(ids) + "\n"
        "Return: {\"action_type\": \"triage_rank\", \"ranked_patient_ids\": [<ordered list>]}"
    )


def build_med_prompt(obs: Dict[str, Any], step: int) -> str:
    patient = obs.get("patients", [{}])[0]
    ctx = obs.get("context", {})
    meds = patient.get("medications", [])
    allergies = patient.get("allergies", [])
    flagged = ctx.get("flagged_so_far", [])
    resolved = ctx.get("resolved_so_far", [])

    med_lines = "\n".join(
        "  " + m["name"] + " " + m["dose"] + " " + m["frequency"] + " [" + m["source"] + "]"
        for m in meds
    )

    flagged_pairs = [c["drug_a"] + "/" + c["drug_b"] for c in flagged]
    resolved_pairs = [c["drug_a"] + "/" + c["drug_b"] for c in resolved]

    return (
        "TASK: Medication Reconciliation (step " + str(step) + ")\n"
        "Patient: " + str(patient.get("name")) + ", age " + str(patient.get("age")) + "\n"
        "Allergies: " + str(allergies) + "\n"
        "History: " + str(patient.get("history", [])) + "\n"
        "Medications:\n" + med_lines + "\n\n"
        "Already flagged: " + str(flagged_pairs) + "\n"
        "Already resolved: " + str(resolved_pairs) + "\n"
        "Feedback: " + str(obs.get("feedback", "")) + "\n\n"
        "Find the next drug conflict (interaction, contraindication, duplicate, dosing error). "
        "If all conflicts found and resolved, respond with no_action."
    )


def build_sepsis_prompt(obs: Dict[str, Any], step: int) -> str:
    patient = obs.get("patients", [{}])[0]
    ctx = obs.get("context", {})
    v = patient.get("vitals", {})
    labs = patient.get("labs", {})
    return (
        "TASK: Sepsis Watch (step " + str(step) + "/10)\n"
        "Patient: " + str(patient.get("name")) + ", age " + str(patient.get("age")) + "\n"
        "History: " + str(patient.get("history", [])) + "\n"
        "Vitals: HR=" + str(v.get("heart_rate")) +
        " RR=" + str(v.get("respiratory_rate")) +
        " SpO2=" + str(v.get("spo2")) + "%" +
        " BP=" + str(v.get("systolic_bp")) +
        " Temp=" + str(v.get("temperature")) +
        " AVPU=" + str(v.get("consciousness")) + "\n"
        "Labs: WBC=" + str(labs.get("wbc")) +
        " Creatinine=" + str(labs.get("creatinine")) +
        " Lactate=" + str(labs.get("lactate")) +
        " Bilirubin=" + str(labs.get("bilirubin")) +
        " Platelets=" + str(labs.get("platelets")) +
        " PaO2/FiO2=" + str(labs.get("pao2_fio2")) + "\n"
        "SOFA score: " + str(ctx.get("current_sofa", "?")) + "\n"
        "Investigations ordered: " + str(ctx.get("investigations_ordered", [])) + "\n"
        "Escalated: " + str(ctx.get("escalated", False)) + "\n"
        "Feedback: " + str(obs.get("feedback", "")) + "\n\n"
        "Decide: order an investigation, escalate care, or no_action. "
        "Sepsis threshold is SOFA >= 6. Act early."
    )

# ── Task runner ───────────────────────────────────────────────────────────────

async def run_task(task: str) -> None:
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
    obs: Dict[str, Any] = {}

    log_start(task=task, model=MODEL_NAME)

    try:
        async with ClinicalOpsEnv(base_url=ENV_BASE_URL) as env:
            try:
                result = await env.reset(task=task)
                obs = result.observation.model_dump() if hasattr(result, "observation") else {}
            except Exception as exc:
                last_error = str(exc)
                print(f"[DEBUG] reset failed: {exc}", file=sys.stderr)

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                try:
                    if task == "ed_triage":
                        prompt = build_triage_prompt(obs)
                    elif task == "medication_review":
                        prompt = build_med_prompt(obs, step)
                    else:
                        prompt = build_sepsis_prompt(obs, step)

                    action_dict = call_llm(prompt)
                    action_str = json.dumps(action_dict)

                    action = ClinicalOpsAction(**action_dict)
                    result = await env.step(action)
                    reward = result.reward or 0.0
                    done = result.done
                    obs = result.observation.model_dump() if hasattr(result, "observation") else {}
                    score = float(obs.get("score", 0.0))
                    last_error = None

                except Exception as exc:
                    reward = 0.0
                    done = False
                    last_error = str(exc)[:120]
                    print(f"[DEBUG] step {step} error: {exc}", file=sys.stderr)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str if "action_str" in dir() else "no_action",
                         reward=reward, done=done if "done" in dir() else False, error=last_error)

                if obs.get("done", False):
                    break

        score = float(obs.get("score", 0.0)) if obs else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        last_error = str(exc)
        print(f"[DEBUG] Task {task} fatal error: {exc}", file=sys.stderr)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


# ── Main ──────────────────────────────────────────────────────────────────────

async def main() -> None:
    tasks = ["ed_triage", "medication_review", "sepsis_watch"]
    for task in tasks:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())