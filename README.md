# ClinicalOps — Hospital Clinical Workflow Environment

> **OpenEnv Hackathon 2026** · Real-world clinical decision support for AI agent training and evaluation.

ClinicalOps simulates the high-stakes decisions made by clinicians in a busy hospital — patient triage, medication safety review, and sepsis detection. An AI agent must reason over vitals, lab results, and drug lists to make correct clinical decisions under time pressure.

---

## Why ClinicalOps?

Clinical errors cause enormous harm globally. Three tasks in this environment directly mirror real workflows that junior doctors, nurses, and pharmacists perform every shift:

| Task | Real-world analogue | Difficulty |
|------|---------------------|------------|
| ED Triage | Emergency nurse ranking patients by urgency | Easy |
| Medication Reconciliation | Pharmacist discharge medication review | Medium |
| Sepsis Watch | ICU nurse monitoring for deterioration | Hard |

No games. No toys. These are the exact decisions that training data exists for in clinical guidelines.

---

## Environment Overview

- **3 tasks**, each with its own grader scoring `[0.05, 0.95]`
- **Partial rewards** on every step — not just binary end-of-episode
- **Deterministic graders** based on published scoring systems (NEWS2, SOFA)
- **Dynamic state** in Task 3 — patient vitals evolve every step
- Fully synthetic patient data — no real patient information

---

## Action Space

All tasks use a single unified `ClinicalOpsAction`:

```python
class ClinicalOpsAction(Action):
    action_type: str           # Required — see below
    ranked_patient_ids: list   # Task 1: ordered list of patient IDs
    conflict_patient_id: str   # Task 2: patient with the conflict
    drug_a: str                # Task 2: first drug
    drug_b: str                # Task 2: second drug / allergen
    conflict_type: str         # Task 2: interaction|dosing_error|duplicate|contraindication
    resolution: str            # Task 2: proposed fix
    investigation: str         # Task 3: test to order
    escalation_level: str      # Task 3: senior_review|rapid_response|icu_transfer
    reasoning: str             # Any task: optional free-text
```

Valid `action_type` values:

| Value | Task | Description |
|-------|------|-------------|
| `triage_rank` | 1 | Submit ordered patient ID list |
| `flag_conflict` | 2 | Flag a drug conflict |
| `resolve_conflict` | 2 | Propose a resolution for a flagged conflict |
| `order_investigation` | 3 | Order a diagnostic test |
| `escalate` | 3 | Escalate care level |
| `no_action` | 2, 3 | End review / do nothing this step |

---

## Observation Space

```python
class ClinicalOpsObservation(Observation):
    task: str                  # Active task name
    step: int                  # Current step number
    done: bool                 # Episode complete flag
    reward: float              # Reward for last action
    score: float               # Cumulative normalised score [0.05, 0.95]
    patients: list             # Full patient data (vitals, labs, meds)
    context: dict              # Task guidance and available actions
    feedback: str              # Human-readable feedback on last action
    grader_info: dict          # Transparent grader bookkeeping
```

Each patient contains:
- `patient_id`, `name`, `age`, `chief_complaint`
- `vitals`: `heart_rate`, `respiratory_rate`, `spo2`, `systolic_bp`, `temperature`, `consciousness`
- `labs`: `wbc`, `creatinine`, `lactate`, `bilirubin`, `platelets`, `pao2_fio2`
- `medications`: list of `{name, dose, route, frequency, source}`
- `allergies`, `history`

---

## Tasks

### Task 1 — ED Triage (Easy)

**Objective:** Rank 8 patients from highest to lowest urgency in a single action.

**Grader:** NEWS2 (National Early Warning Score 2). Each patient's vitals are scored across 6 parameters. The agent's ranking is compared to the correct NEWS2 ordering using a position-weighted metric. Critical patients (NEWS2 ≥ 7) ranked outside the top 3 incur a penalty.

**Reward:** Single step. Score = position-weighted accuracy vs. correct NEWS2 order.

**Max steps:** 1

---

### Task 2 — Medication Reconciliation (Medium)

**Objective:** A 72-year-old patient is being discharged after knee replacement. The medication list contains 5 embedded conflicts (2 severe, 2 moderate, 1 minor). Find and resolve them all.

**Conflicts include:**
- Warfarin + Aspirin (severe bleed risk)
- Warfarin + Enoxaparin (dual anticoagulation)
- Ibuprofen + Lisinopril in CKD (renal harm)
- Co-amoxiclav in penicillin-allergic patient (contraindication)
- Duplicate Metformin (home + hospital)

**Grader:** Recall of ground-truth conflicts + resolution quality + severe miss penalty + false positive penalty.

**Reward per step:**
- `+0.20` correct severe conflict flagged
- `+0.12` correct moderate conflict flagged
- `+0.06` correct minor conflict flagged
- `+0.05–0.15` resolution quality bonus
- `−0.05` false positive flag
- `−0.20` severe conflict missed (at episode end)

**Max steps:** 8

---

### Task 3 — Sepsis Watch (Hard)

**Objective:** Monitor a deteriorating patient over 10 steps. Detect sepsis onset using SOFA criteria and escalate before SOFA reaches 6. Order appropriate investigations from the Sepsis-6 bundle.

**Dynamics:** If the agent does not escalate, SOFA rises from 2 → 8 over 10 steps. Escalating early (SOFA < 4) gives maximum reward. Escalating after SOFA ≥ 6 gives partial credit only.

**Grader:** Investigation coverage (Sepsis-6 bundle) + escalation level quality + earliness bonus − late/no escalation penalty.

**Reward per step:**
- `+0.08` relevant investigation ordered (blood_cultures, lactate, cbc, renal_panel, abg)
- `−0.03` irrelevant investigation
- `+0.40` correct escalation with SOFA < 4
- `+0.28` correct escalation with SOFA 4–5
- `+0.10` escalation after SOFA ≥ 6
- `−0.10` per step doing nothing when SOFA ≥ 6

**Max steps:** 10

---

## Setup & Usage

### Prerequisites

```bash
pip install openenv-core
# or
uv pip install openenv-core
```

### Run server locally

```bash
cd clinicalops
uv sync
uv run server
# Server live at http://localhost:7860
```

### Run with Docker

```bash
docker build -t clinicalops-env:latest -f server/Dockerfile .
docker run -p 7860:7860 clinicalops-env:latest
```

### Run inference script

```bash
export HF_TOKEN=your_token_here
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct

python inference.py
```

### Validate submission

```bash
openenv validate
```

---

## Client Usage (Python)

```python
import asyncio
from clinicalops import ClinicalOpsEnv, ClinicalOpsAction

async def main():
    async with ClinicalOpsEnv(base_url="http://localhost:7860") as env:

        # Task 1: ED Triage
        result = await env.reset(task="ed_triage")
        result = await env.step(ClinicalOpsAction(
            action_type="triage_rank",
            ranked_patient_ids=["PT005", "PT001", "PT003", "PT007",
                                 "PT008", "PT004", "PT002", "PT006"]
        ))
        print(f"Triage score: {result.observation.score}")

        # Task 2: Medication Review
        result = await env.reset(task="medication_review")
        result = await env.step(ClinicalOpsAction(
            action_type="flag_conflict",
            drug_a="warfarin",
            drug_b="aspirin",
            conflict_type="interaction"
        ))

        # Task 3: Sepsis Watch
        result = await env.reset(task="sepsis_watch")
        result = await env.step(ClinicalOpsAction(
            action_type="order_investigation",
            investigation="blood_cultures"
        ))

asyncio.run(main())
```

---

## Baseline Performance

Scores achieved by `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Score | Notes |
|------|-------|-------|
| ED Triage | ~0.72 | Strong on clear NEWS2 signals, misses subtle cases |
| Medication Reconciliation | ~0.58 | Catches severe interactions, misses duplicate |
| Sepsis Watch | ~0.51 | Escalates correctly ~60% of runs, often 1–2 steps late |

Frontier models (GPT-4o, Claude 3.5) score 0.78–0.86 on Task 1, 0.70–0.82 on Task 2, and 0.62–0.74 on Task 3.

---

## Project Structure

```
clinicalops/
├── inference.py          # Baseline inference script (root level — required)
├── openenv.yaml          # OpenEnv manifest
├── pyproject.toml        # Dependencies and entry points
├── uv.lock               # Locked dependencies (required by openenv validate)
├── __init__.py           # Package exports
├── models.py             # Pydantic Action / Observation types
├── graders.py            # Deterministic NEWS2 / SOFA graders
├── scenarios.py          # Synthetic patient data for all tasks
├── client.py             # EnvClient for agent use
├── .dockerignore
├── .gitignore
├── README.md
└── server/
    ├── app.py                       # FastAPI app + main() entry point
    ├── clinicalops_environment.py   # Core Environment implementation
    ├── requirements.txt             # Docker pip fallback
    ├── Dockerfile
    └── __init__.py
```

---

## Scoring System Reference

**NEWS2 parameters scored:** Respiratory rate, SpO2, Systolic BP, Heart rate, Temperature, Consciousness (AVPU). Score ≥ 7 = high risk.

**SOFA components used:** Renal (creatinine), Coagulation (platelets), Hepatic (bilirubin), Respiratory (PaO2/FiO2). SOFA ≥ 2 = organ dysfunction, SOFA ≥ 6 = severe sepsis.

---

## License

BSD 3-Clause. All patient data is entirely synthetic.
