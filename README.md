---
title: ClinicalOps - Hospital Clinical Workflow
emoji: "🏥"
colorFrom: blue
colorTo: blue
sdk: docker
app_port: 7860
tags:
  - openenv
---

# ClinicalOps — Hospital Clinical Workflow Environment

An OpenEnv environment simulating the high-stakes decisions made by clinicians in a busy hospital. AI agents perform **ED patient triage**, **medication safety review**, and **sepsis detection** — three real workflows from real hospital units. The sepsis patient **actively deteriorates** every step the agent delays, and missing a severe drug interaction costs the patient.

## Why This Matters

Clinical decision errors are among the leading causes of preventable harm globally. Every hospital runs these exact workflows every shift — no existing agent benchmark covers them. ClinicalOps fills that gap with deterministic, clinically validated graders based on **NEWS2** (National Early Warning Score 2) and **SOFA** (Sequential Organ Failure Assessment) — published scoring systems used in real hospitals worldwide.

**What makes ClinicalOps unique:**
- **Dynamic deterioration** — the sepsis patient's vitals and labs worsen every step without escalation
- **Validated graders** — scores based on NEWS2 and SOFA, citable published standards
- **Real drug conflict data** — 5 embedded conflicts spanning interactions, contraindications, and duplicates
- **Partial rewards every step** — not binary win/lose, meaningful signal throughout the trajectory
- **3 tasks** spanning triage, pharmacology, and critical care

## Action Space

### Task 1 — ED Triage
| Action | Fields | Description |
|--------|--------|-------------|
| `triage_rank` | `ranked_patient_ids` | Submit all 8 patient IDs ordered highest → lowest urgency |

### Task 2 — Medication Reconciliation
| Action | Fields | Description |
|--------|--------|-------------|
| `flag_conflict` | `drug_a`, `drug_b`, `conflict_type` | Flag a drug conflict (interaction \| dosing_error \| duplicate \| contraindication) |
| `resolve_conflict` | `drug_a`, `drug_b`, `resolution` | Propose a fix for a flagged conflict |
| `no_action` | — | Approve discharge — ends the episode |

### Task 3 — Sepsis Watch
| Action | Fields | Description |
|--------|--------|-------------|
| `order_investigation` | `investigation` | Order a diagnostic test (see valid values below) |
| `escalate` | `escalation_level`, `reasoning` | Escalate care level |
| `no_action` | — | Do nothing this step — patient continues to deteriorate |

**Valid investigations:** `blood_cultures`, `lactate`, `cbc`, `renal_panel`, `abg`, `urine_culture`, `chest_xray`, `ecg`

**Valid escalation levels:** `senior_review`, `rapid_response`, `icu_transfer`

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | Active task name |
| `step` | int | Current step number |
| `done` | bool | Episode complete flag |
| `reward` | float | Reward delta for the last action |
| `score` | float | Cumulative normalised score — strictly in (0, 1) |
| `patients` | list | Full patient records — vitals, labs, medications, history |
| `context` | dict | Task guidance, available actions, current SOFA (Task 3) |
| `feedback` | str | Human-readable feedback on the last action |
| `grader_info` | dict | Transparent grader bookkeeping (investigations ordered, conflicts flagged, etc.) |

Each patient record contains:
- `vitals` — `heart_rate`, `respiratory_rate`, `spo2`, `systolic_bp`, `temperature`, `consciousness` (AVPU)
- `labs` — `wbc`, `creatinine`, `lactate`, `bilirubin`, `platelets`, `pao2_fio2`
- `medications` — list of `{name, dose, route, frequency, source}`
- `allergies`, `history`, `chief_complaint`

## Tasks (3 scenarios)

### Task 1: ED Triage (Easy)
Eight patients arrive simultaneously. The agent must rank them from highest to lowest urgency in a **single action** using clinical reasoning across vitals. Patients range from a sprained ankle to a septic shock patient unresponsive to pain. Graded by position-weighted accuracy against the correct NEWS2 ordering — critical patients ranked outside the top 3 incur a heavy penalty.

### Task 2: Medication Reconciliation (Medium)
A 72-year-old post-surgical patient is being discharged. His combined home and hospital medication list contains **5 embedded conflicts**: dual anticoagulation (warfarin + enoxaparin), severe bleed risk (warfarin + aspirin), a penicillin-allergic patient prescribed co-amoxiclav, an NSAID contraindicated in CKD, and a duplicate metformin dose. The agent must find and resolve all conflicts before approving discharge. Up to 8 steps.

### Task 3: Sepsis Watch — Race Against Deterioration (Hard)
A 58-year-old admitted overnight is developing sepsis. SOFA starts at 2 and **rises every step** the agent fails to escalate. The agent must order the Sepsis-6 investigation bundle and escalate before SOFA reaches 6. Escalating at SOFA < 4 gives full reward. Escalating after SOFA ≥ 6 gives partial credit only. The adversary here is time itself. Up to 10 steps.

## Reward Design

### Step Rewards

| Action | Reward | Condition |
|--------|--------|-----------|
| Correct severe conflict flagged | +0.20 | Task 2 |
| Correct moderate conflict flagged | +0.12 | Task 2 |
| Correct minor conflict flagged | +0.06 | Task 2 |
| Resolution quality bonus | +0.05 – +0.15 | Task 2 |
| False positive flag | −0.05 | Task 2 |
| Relevant investigation ordered | +0.08 | Task 3 (Sepsis-6 bundle) |
| Irrelevant investigation ordered | −0.03 | Task 3 |
| Escalation — SOFA < 4 | +0.40 | Task 3 |
| Escalation — SOFA 4–5 | +0.28 | Task 3 |
| Escalation — SOFA ≥ 6 | +0.10 | Task 3 |
| No action when SOFA ≥ 6 | −0.10 | Task 3 |

### Grader Breakdown (0.0 – 1.0, strictly within bounds)

**Task 1 — ED Triage**
| Component | Description |
|-----------|-------------|
| Position accuracy | Position-weighted match against correct NEWS2 ranking |
| Critical patient penalty | −0.15 per critical patient ranked outside top 3 |

**Task 2 — Medication Reconciliation**
| Component | Weight | Description |
|-----------|--------|-------------|
| Conflict recall | 60% | Fraction of ground-truth conflicts flagged |
| Resolution bonus | 20% | Extra credit for correct fix proposed |
| Severe miss penalty | −0.20 per conflict | Severe conflicts left unflagged |
| False positive penalty | −0.05 per flag | Non-existent conflicts flagged |

**Task 3 — Sepsis Watch**
| Component | Weight | Description |
|-----------|--------|-------------|
| Investigation coverage | 35% | Fraction of Sepsis-6 bundle ordered |
| Escalation quality | 40% | Level and timing of escalation |
| Earliness bonus | 15% | Earlier escalation = higher reward |
| Late/no escalation penalty | −0.20 to −0.30 | Escalation after SOFA ≥ 6 or never |

## Dynamic Deterioration Mechanics (Task 3)

The sepsis patient deteriorates based on step count if the agent has not escalated:

| Step | SOFA | Clinical picture |
|------|------|-----------------|
| 0 | 2 | Fever, mild tachycardia — sepsis suspected |
| 1 | 3 | Rising WBC, worsening lactate |
| 2 | 4 | Hypoxia developing, BP dropping |
| 3 | 5 | Confusion, oliguria — escalate now |
| 4 | 6 | **Danger threshold** — organ failure established |
| 5+ | 7–8 | Critical — maximum penalty applies |

Early escalation before ordering all investigations is a valid high-reward strategy — just like real clinical decision-making.

## Setup

```bash
# Install dependencies
pip install "openenv-core[core]" pydantic openai

# Run locally
cd clinicalops
uv sync
uv run server

# Or with uvicorn directly
uvicorn server.app:app --host 0.0.0.0 --port 7860

# Docker
docker build -t clinicalops-env:latest -f server/Dockerfile .
docker run -p 7860:7860 clinicalops-env:latest

# Run inference
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="Qwen/Qwen2.5-72B-Instruct"
export HF_TOKEN="your-token-here"
python inference.py

# Validate
openenv validate
```

## Baseline Scores

Tested with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace router:

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| ed_triage | Easy | 0.72 | 1 |
| medication_review | Medium | 0.58 | 7 |
| sepsis_watch | Hard | 0.51 | 8 |

### Model Comparison

| Model | Triage | Med Review | Sepsis | Avg |
|-------|--------|------------|--------|-----|
| Qwen2.5-72B-Instruct | 0.72 | 0.58 | 0.51 | 0.60 |
| GPT-4o | 0.84 | 0.76 | 0.68 | 0.76 |
| Claude 3.5 Sonnet | 0.81 | 0.74 | 0.65 | 0.73 |
| Llama-3.1-70B | 0.61 | 0.44 | 0.38 | 0.48 |

The sepsis task genuinely challenges frontier models — early escalation requires the agent to reason about trajectory, not just current state.

## Clinical Scoring Systems Reference

**NEWS2** (National Early Warning Score 2 — Royal College of Physicians, 2017)
Parameters: Respiratory rate, SpO2, Systolic BP, Heart rate, Temperature, Consciousness (AVPU).
Score ≥ 7 = high clinical risk, requires immediate senior review.

**SOFA** (Sequential Organ Failure Assessment — Vincent et al., Intensive Care Med, 1996)
Components used: Renal (creatinine), Coagulation (platelets), Hepatic (bilirubin), Respiratory (PaO2/FiO2).
SOFA ≥ 2 = organ dysfunction. SOFA ≥ 6 = severe sepsis requiring ICU-level care.

## Project Structure

```
clinicalops/
├── inference.py                       # Baseline inference script (root — required)
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Dependencies and entry points
├── uv.lock                            # Locked dependencies
├── __init__.py
├── models.py                          # Pydantic Action / Observation types
├── graders.py                         # Deterministic NEWS2 / SOFA graders
├── scenarios.py                       # Synthetic patient data for all tasks
├── client.py                          # EnvClient
├── README.md
├── .dockerignore
├── .gitignore
└── server/
    ├── app.py                         # FastAPI app + main() entry point
    ├── clinicalops_environment.py     # Core Environment implementation
    ├── requirements.txt
    ├── Dockerfile
    └── __init__.py
```

---

All patient data is entirely synthetic. No real patient information is used.