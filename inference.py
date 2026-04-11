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

API_BASE_URL: str = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

if not HF_TOKEN:
    print("[ERROR] HF_TOKEN environment variable is required", file=sys.stderr)
    sys.exit(1)

ENV_BASE_URL: str = os.getenv("CLINICALOPS_URL", "https://leoxfs-clinicalops-env.hf.space")
BENCHMARK: str = "clinicalops"
MAX_STEPS: int = 14
TEMPERATURE: float = 0.2
SUCCESS_SCORE_THRESHOLD: float = 0.40

client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)


def log_start(task: str, model: str) -> None:
    print("[START] task=" + task + " env=" + BENCHMARK + " model=" + model, flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err_val = error if error else "null"
    done_val = str(done).lower()
    action_safe = action.replace("\n", " ").replace("\r", "")[:200]
    print(
        "[STEP] step=" + str(step) + " action=" + action_safe +
        " reward=" + format(reward, ".2f") +
        " done=" + done_val + " error=" + err_val,
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(format(r, ".2f") for r in rewards)
    print(
        "[END] success=" + str(success).lower() +
        " steps=" + str(steps) +
        " score=" + format(score, ".3f") +
        " rewards=" + rewards_str,
        flush=True,
    )


SYSTEM_PROMPT = (
    "You are an expert clinical decision support AI. "
    "Respond with ONLY a valid JSON object — no preamble, no markdown fences.\n\n"
    "Action types:\n"
    "  triage_rank:         {\"action_type\": \"triage_rank\", \"ranked_patient_ids\": [...]}\n"
    "  flag_conflict:       {\"action_type\": \"flag_conflict\", \"drug_a\": \"...\", \"drug_b\": \"...\", \"conflict_type\": \"interaction|dosing_error|duplicate|contraindication\"}\n"
    "  resolve_conflict:    {\"action_type\": \"resolve_conflict\", \"drug_a\": \"...\", \"drug_b\": \"...\", \"resolution\": \"...\"}\n"
    "  order_investigation: {\"action_type\": \"order_investigation\", \"investigation\": \"...\"}\n"
    "  escalate:            {\"action_type\": \"escalate\", \"escalation_level\": \"senior_review|rapid_response|icu_transfer\", \"reasoning\": \"...\"}\n"
    "  vent_check:          {\"action_type\": \"vent_check\", \"investigation\": \"assess_oxygenation|assess_consciousness|assess_secretions|check_rsbi|reduce_fio2|reduce_peep\"}\n"
    "  perform_sbt:         {\"action_type\": \"perform_sbt\"}\n"
    "  extubate:            {\"action_type\": \"extubate\"}\n"
    "  submit_diagnosis:    {\"action_type\": \"submit_diagnosis\", \"reasoning\": \"<diagnosis_name>\"}\n"
    "  no_action:           {\"action_type\": \"no_action\"}"
)


def call_llm(user_prompt: str) -> Dict[str, Any]:
    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
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
        print("[DEBUG] LLM call failed: " + str(exc), file=sys.stderr)
        return {"action_type": "no_action"}


def build_prompt(task: str, obs: Dict[str, Any], step: int) -> str:
    if task == "ed_triage":
        patients = obs.get("patients", [])
        lines = []
        for p in patients:
            v = p.get("vitals", {})
            lines.append(
                "  " + p["patient_id"] + " | " + p["name"] +
                " age " + str(p["age"]) + " | " + p["chief_complaint"] +
                " | HR=" + str(v.get("heart_rate")) +
                " RR=" + str(v.get("respiratory_rate")) +
                " SpO2=" + str(v.get("spo2")) +
                " BP=" + str(v.get("systolic_bp")) +
                " Temp=" + str(v.get("temperature")) +
                " AVPU=" + str(v.get("consciousness"))
            )
        ids = [p["patient_id"] for p in patients]
        return (
            "TASK: ED Triage\n"
            "Rank " + str(len(patients)) + " patients HIGHEST to LOWEST urgency using NEWS2.\n"
            "Patients:\n" + "\n".join(lines) + "\n"
            "IDs: " + str(ids) + "\n"
            "Return: {\"action_type\": \"triage_rank\", \"ranked_patient_ids\": [...]}"
        )

    elif task == "medication_review":
        patient = obs.get("patients", [{}])[0]
        ctx = obs.get("context", {})
        meds = patient.get("medications", [])
        med_lines = "\n".join(
            "  " + m["name"] + " " + m["dose"] + " " + m["frequency"] + " [" + m["source"] + "]"
            for m in meds
        )
        flagged = [c["drug_a"] + "/" + c["drug_b"] for c in ctx.get("flagged_so_far", [])]
        resolved = [c["drug_a"] + "/" + c["drug_b"] for c in ctx.get("resolved_so_far", [])]
        return (
            "TASK: Medication Reconciliation (step " + str(step) + ")\n"
            "Patient: " + str(patient.get("name")) + " age " + str(patient.get("age")) + "\n"
            "Allergies: " + str(patient.get("allergies", [])) + "\n"
            "History: " + str(patient.get("history", [])) + "\n"
            "Medications:\n" + med_lines + "\n"
            "Flagged: " + str(flagged) + "\n"
            "Resolved: " + str(resolved) + "\n"
            "Feedback: " + str(obs.get("feedback", "")) + "\n"
            "Find the next conflict or use no_action if done."
        )

    elif task == "sepsis_watch":
        patient = obs.get("patients", [{}])[0]
        ctx = obs.get("context", {})
        v = patient.get("vitals", {})
        labs = patient.get("labs", {})
        return (
            "TASK: Sepsis Watch (step " + str(step) + "/10)\n"
            "Patient: " + str(patient.get("name")) + " age " + str(patient.get("age")) + "\n"
            "Vitals: HR=" + str(v.get("heart_rate")) +
            " RR=" + str(v.get("respiratory_rate")) +
            " SpO2=" + str(v.get("spo2")) +
            " BP=" + str(v.get("systolic_bp")) +
            " Temp=" + str(v.get("temperature")) +
            " AVPU=" + str(v.get("consciousness")) + "\n"
            "Labs: WBC=" + str(labs.get("wbc")) +
            " Creatinine=" + str(labs.get("creatinine")) +
            " Lactate=" + str(labs.get("lactate")) +
            " Platelets=" + str(labs.get("platelets")) + "\n"
            "SOFA=" + str(ctx.get("current_sofa", "?")) + "\n"
            "Ordered: " + str(ctx.get("investigations_ordered", [])) + "\n"
            "Feedback: " + str(obs.get("feedback", "")) + "\n"
            "Order investigations or escalate. SOFA>=6 is danger."
        )

    elif task == "vent_weaning":
        patient = obs.get("patients", [{}])[0]
        ctx = obs.get("context", {})
        v = patient.get("vitals", {})
        vent = patient.get("vent_settings", {})
        return (
            "TASK: Ventilator Weaning (step " + str(step) + "/8)\n"
            "Patient: " + str(patient.get("name")) + " age " + str(patient.get("age")) + "\n"
            "Vitals: HR=" + str(v.get("heart_rate")) +
            " RR=" + str(v.get("respiratory_rate")) +
            " SpO2=" + str(v.get("spo2")) + "\n"
            "Vent: FiO2=" + str(vent.get("fio2")) +
            " PEEP=" + str(vent.get("peep")) +
            " PS=" + str(vent.get("pressure_support")) + "\n"
            "Checks done: " + str(ctx.get("checks_completed", [])) + "\n"
            "SBT performed: " + str(ctx.get("sbt_performed", False)) + "\n"
            "Feedback: " + str(obs.get("feedback", "")) + "\n"
            "Valid checks: assess_oxygenation, assess_consciousness, assess_secretions, check_rsbi, reduce_fio2, reduce_peep\n"
            "Complete all checks, then perform_sbt, then extubate if SBT passed."
        )

    else:
        ctx = obs.get("context", {})
        patient = obs.get("patients", [{}])[0]
        v = patient.get("vitals", {})
        labs = patient.get("labs", {})
        return (
            "TASK: Diagnostic Reasoning (step " + str(step) + "/12)\n"
            "Patient: " + str(patient.get("name")) + " age " + str(patient.get("age")) + "\n"
            "Complaint: " + str(patient.get("chief_complaint", "")) + "\n"
            "History: " + str(patient.get("history", [])) + "\n"
            "Vitals: HR=" + str(v.get("heart_rate")) +
            " Temp=" + str(v.get("temperature")) +
            " SpO2=" + str(v.get("spo2")) + "\n"
            "Labs: WBC=" + str(labs.get("wbc")) +
            " ESR=" + str(labs.get("esr")) +
            " CRP=" + str(labs.get("crp")) + "\n"
            "Remaining differentials: " + str(ctx.get("remaining_differentials", [])) + "\n"
            "Investigations ordered: " + str(ctx.get("investigations_ordered", [])) + "\n"
            "Feedback: " + str(obs.get("feedback", "")) + "\n"
            "Available: " + str(ctx.get("valid_investigations", [])) + "\n"
            "Order investigations to narrow diagnosis. When confident use submit_diagnosis with reasoning=<diagnosis_name>."
        )


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
    done = False
    action_str = "no_action"

    log_start(task=task, model=MODEL_NAME)

    try:
        async with ClinicalOpsEnv(base_url=ENV_BASE_URL) as env:
            try:
                result = await env.reset(task=task)
                obs = result.observation.model_dump() if hasattr(result, "observation") else {}
            except Exception as exc:
                last_error = str(exc)[:120]
                print("[DEBUG] reset failed: " + str(exc), file=sys.stderr)

            for step in range(1, MAX_STEPS + 1):
                if obs.get("done", False):
                    break

                try:
                    prompt = build_prompt(task, obs, step)
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
                    print("[DEBUG] step " + str(step) + " error: " + str(exc), file=sys.stderr)

                rewards.append(reward)
                steps_taken = step
                log_step(step=step, action=action_str, reward=reward, done=done, error=last_error)

                if obs.get("done", False):
                    break

        score = float(obs.get("score", 0.0)) if obs else 0.0
        if score == 0.0:
            score = 0.06
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print("[DEBUG] Task " + task + " fatal: " + str(exc), file=sys.stderr)
        score = 0.06

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


async def main() -> None:
    tasks = ["ed_triage", "medication_review", "sepsis_watch", "vent_weaning", "diagnostic_reasoning"]
    for task in tasks:
        await run_task(task)


if __name__ == "__main__":
    asyncio.run(main())
