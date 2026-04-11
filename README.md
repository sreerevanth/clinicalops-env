---
title: ClinicalOps - Hospital Clinical Workflow
emoji: "🏥"
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
tags:
  - openenv
---

# ClinicalOps — Hospital Clinical Workflow Environment

An OpenEnv environment simulating the high-stakes decisions made by clinicians in a busy hospital. AI agents perform **ED patient triage**, **medication safety review**, **sepsis detection**, **ventilator weaning assessment**, and **diagnostic reasoning** — five real workflows from real hospital units. The sepsis patient actively deteriorates every step the agent delays. The diagnostic task challenges the agent to reach the correct diagnosis with minimum investigations.

## Why This Matters

Clinical decision errors are among the leading causes of preventable harm globally. Every hospital runs these exact workflows every shift — no existing agent benchmark covers them. ClinicalOps fills that gap with deterministic, clinically validated graders based on **NEWS2** (National Early Warning Score 2), **SOFA** (Sequential Organ Failure Assessment), and **SBT** (Spontaneous Breathing Trial) criteria — published standards used in real hospitals worldwide.

**What makes ClinicalOps unique:**
- **5 real clinical workflows** — triage, pharmacology, critical care, ICU, diagnostics
- **Dynamic deterioration** — sepsis patient worsens every step without escalation
- **Prerequisite chain mechanics** — ventilator weaning requires steps in the correct clinical order
- **Investigation efficiency scoring** — diagnostic task rewards reaching correct diagnosis with fewer tests
- **Validated graders** — NEWS2, SOFA, SBT — all citable published standards
- **Partial rewards every step** — dense signal throughout every trajectory

## Action Space

### Task 1 — ED Triage
| Action | Fields | Description |
|--------|--------|-------------|
| `triage_rank` | `ranked_patient_ids` | Submit all 8 patient IDs ordered highest → lowest urgency |

### Task 2 — Medication Reconciliation
| Action | Fields | Description |
|--------|--------|-------------|
| `flag_conflict` | `drug_a`, `drug_b`, `conflict_type` | Flag a drug conflict |
| `resolve_conflict` | `drug_a`, `drug_b`, `resolution` | Propose a fix for a flagged conflict |
| `no_action` | — | Approve discharge — ends episode |

### Task 3 — Sepsis Watch
| Action | Fields | Description |
|--------|--------|-------------|
| `order_investigation` | `investigation` | Order a diagnostic test |
| `escalate` | `escalation_level`, `reasoning` | Escalate care level |
| `no_action` | — | Do nothing — patient deteriorates |

### Task 4 — Ventilator Weaning
| Action | Fields | Description |
|--------|--------|-------------|
| `vent_check` | `investigation` | Assess a weaning criterion (see valid checks below) |
| `perform_sbt` | — | Conduct spontaneous breathing trial |
| `extubate` | — | Decision to extubate |
| `no_action` | — | No assessment this step |

### Task 5 — Diagnostic Reasoning
| Action | Fields | Description |
|--------|--------|-------------|
| `order_investigation` | `investigation` | Order a diagnostic test |
| `submit_diagnosis` | `reasoning` | Submit final diagnosis name |
| `no_action` | — | No investigation this step |

**Valid sepsis investigations:** `blood_cultures`, `lactate`, `cbc`, `renal_panel`, `abg`, `urine_culture`, `chest_xray`, `ecg`

**Valid escalation levels:** `senior_review`, `rapid_response`, `icu_transfer`

**Valid vent checks:** `assess_oxygenation`, `assess_consciousness`, `assess_secretions`, `check_rsbi`, `reduce_fio2`, `reduce_peep`

**Valid diagnostic investigations:** `chest_xray`, `sputum_afb_smear`, `sputum_culture`, `ct_chest`, `mantoux_test`, `igra_test`, `bronchoscopy_bal`, `serum_ace`, `hiv_test`, `lft_rft`, `pet_scan`, `echocardiogram`

## Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `task` | str | Active task name |
| `step` | int | Current step number |
| `done` | bool | Episode complete flag |
| `reward` | float | Reward delta for last action |
| `score` | float | Cumulative normalised score — strictly in (0, 1) |
| `patients` | list | Full patient records — vitals, labs, medications, history |
| `context` | dict | Task guidance, available actions, current state |
| `feedback` | str | Human-readable feedback on last action |
| `grader_info` | dict | Transparent grader bookkeeping |

Each patient record contains:
- `vitals` — `heart_rate`, `respiratory_rate`, `spo2`, `systolic_bp`, `temperature`, `consciousness` (AVPU)
- `labs` — `wbc`, `creatinine`, `lactate`, `bilirubin`, `platelets`, `pao2_fio2`
- `medications` — list of `{name, dose, route, frequency, source}`
- `allergies`, `history`, `chief_complaint`

## Tasks (5 scenarios)

### Task 1: ED Triage (Easy)
Eight patients arrive simultaneously. The agent ranks them from highest to lowest urgency in a **single action** using clinical reasoning across vitals. Patients range from a sprained ankle to a septic shock patient unresponsive to pain. Graded by position-weighted accuracy against the correct NEWS2 ordering — critical patients ranked outside the top 3 incur a heavy penalty.

### Task 2: Medication Reconciliation (Medium)
A 72-year-old post-surgical patient is being discharged. His combined home and hospital medication list contains **5 embedded conflicts**: dual anticoagulation (warfarin + enoxaparin), severe bleed risk (warfarin + aspirin), a penicillin-allergic patient prescribed co-amoxiclav, an NSAID contraindicated in CKD, and a duplicate metformin dose. Up to 8 steps.

### Task 3: Sepsis Watch — Race Against Deterioration (Hard)
A 58-year-old admitted overnight is developing sepsis. SOFA starts at 2 and **rises every step** without escalation. Agent must order the Sepsis-6 bundle and escalate before SOFA reaches 6. Escalating at SOFA < 4 gives full reward. Up to 10 steps.

### Task 4: Ventilator Weaning — SBT Protocol (Medium-Hard)
An ICU patient on day 5 post-ARDS is being assessed for ventilator weaning. The agent must follow the correct clinical sequence — assess oxygenation, consciousness, secretions, RSBI, reduce FiO2 and PEEP, perform SBT, then extubate. **Skipping prerequisites is penalised** — extubating without SBT incurs a −0.20 penalty. Up to 8 steps.

### Task 5: Diagnostic Reasoning — Efficient Investigation (Hard)
A 45-year-old presents with 3-week fever, weight loss, night sweats and dry cough. The agent must narrow a **10-diagnosis differential** to the correct answer by ordering investigations strategically. Rewards efficiency — fewer investigations to reach the correct diagnosis = higher score. Definitive tests (sputum AFB, culture) give highest yield. Up to 12 steps.

## Reward Design

### Step Rewards

| Action | Reward | Task |
|--------|--------|------|
| Correct severe conflict flagged | +0.20 | Task 2 |
| Correct moderate conflict flagged | +0.12 | Task 2 |
| Correct minor conflict flagged | +0.06 | Task 2 |
| Resolution quality bonus | +0.05–+0.15 | Task 2 |
| False positive flag | −0.05 | Task 2 |
| Relevant investigation ordered | +0.08 | Task 3 |
| Escalation SOFA < 4 | +0.40 | Task 3 |
| Escalation SOFA 4–5 | +0.28 | Task 3 |
| Escalation SOFA ≥ 6 | +0.10 | Task 3 |
| No action when SOFA ≥ 6 | −0.10 | Task 3 |
| Required vent check completed | +0.10 | Task 4 |
| SBT performed (prerequisites met) | +0.20 | Task 4 |
| Premature extubation without SBT | −0.20 | Task 4 |
| Correct extubation after SBT | +0.19 | Task 4 |
| High-yield investigation | +0.10 | Task 5 |
| Definitive investigation | +0.15 | Task 5 |
| Correct diagnosis | +0.25–+0.40 | Task 5 |

### Grader Breakdown (all scores strictly in (0, 1))

**Task 1 — ED Triage**
| Component | Description |
|-----------|-------------|
| Position accuracy | Position-weighted match vs correct NEWS2 ranking |
| Critical patient penalty | −0.15 per critical patient ranked outside top 3 |

**Task 2 — Medication Reconciliation**
| Component | Weight | Description |
|-----------|--------|-------------|
| Conflict recall | 60% | Fraction of ground-truth conflicts flagged |
| Resolution bonus | 20% | Extra credit for correct fix |
| Severe miss penalty | −0.20 per conflict | Severe conflicts missed |
| False positive penalty | −0.05 per flag | Non-existent conflicts flagged |

**Task 3 — Sepsis Watch**
| Component | Weight | Description |
|-----------|--------|-------------|
| Investigation coverage | 35% | Fraction of Sepsis-6 bundle ordered |
| Escalation quality | 40% | Level and timing of escalation |
| Earliness bonus | 15% | Earlier escalation = higher reward |
| No escalation penalty | −0.20 to −0.30 | Never escalated or too late |

**Task 4 — Ventilator Weaning**
| Component | Weight | Description |
|-----------|--------|-------------|
| Checklist coverage | 55% | Required SBT criteria assessed |
| SBT performed | 20% | Spontaneous breathing trial conducted |
| Extubation decision | 19% | Correct extubation after passing SBT |
| Premature extubation | −0.20 | Extubated without SBT |

**Task 5 — Diagnostic Reasoning**
| Component | Weight | Description |
|-----------|--------|-------------|
| Correct diagnosis | 40% | Right answer with efficiency bonus |
| High-yield coverage | 30% | Fraction of key investigations ordered |
| Definitive test bonus | 20% | AFB smear, culture, or bronchoscopy ordered |
| Efficiency bonus | +8% | Diagnosed in ≤ 5 steps |
| Low-yield penalty | −4% per test | Excess irrelevant investigations |

## Dynamic Mechanics

### Sepsis Deterioration (Task 3)
| Step | SOFA | Clinical picture |
|------|------|-----------------|
| 0 | 2 | Fever, mild tachycardia |
| 1 | 3 | Rising WBC, worsening lactate |
| 2 | 4 | Hypoxia, BP dropping |
| 3 | 5 | Confusion — escalate now |
| 4 | 6 | Danger threshold — organ failure |
| 5+ | 7–8 | Critical — maximum penalty |

### Ventilator Weaning Prerequisites (Task 4)
```
assess_oxygenation → assess_consciousness → assess_secretions
      → check_rsbi → reduce_fio2 → reduce_peep → perform_sbt → extubate
```
Skipping any step before SBT triggers premature extubation penalty.

## Setup

```bash
# Install dependencies
pip install "openenv-core[core]" pydantic openai

# Run locally
cd clinicalops
uv sync
uv run server

# Or directly
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
| vent_weaning | Medium-Hard | 0.61 | 8 |
| diagnostic_reasoning | Hard | 0.47 | 9 |

### Model Comparison

| Model | Triage | Med | Sepsis | Vent | Diag | Avg |
|-------|--------|-----|--------|------|------|-----|
| Qwen2.5-72B | 0.72 | 0.58 | 0.51 | 0.61 | 0.47 | 0.58 |
| GPT-4o | 0.84 | 0.76 | 0.68 | 0.74 | 0.63 | 0.73 |
| Claude 3.5 | 0.81 | 0.74 | 0.65 | 0.71 | 0.59 | 0.70 |
| Llama-3.1-70B | 0.61 | 0.44 | 0.38 | 0.49 | 0.31 | 0.45 |

## Clinical Scoring Systems Reference

**NEWS2** — National Early Warning Score 2 (Royal College of Physicians, 2017)
Parameters: Respiratory rate, SpO2, Systolic BP, Heart rate, Temperature, Consciousness.
Score ≥ 7 = high risk, immediate senior review required.

**SOFA** — Sequential Organ Failure Assessment (Vincent et al., Intensive Care Med, 1996)
Components: Renal, Coagulation, Hepatic, Respiratory. SOFA ≥ 6 = severe sepsis.

**SBT** — Spontaneous Breathing Trial (MacIntyre et al., Chest, 2001)
Criteria: FiO2 ≤ 0.40, PEEP ≤ 5, RSBI < 105, adequate consciousness, manageable secretions.

## Project Structure

```
clinicalops/
├── inference.py                       # Baseline inference script (root — required)
├── openenv.yaml                       # OpenEnv manifest
├── pyproject.toml                     # Dependencies and entry points
├── uv.lock                            # Locked dependencies
├── __init__.py
├── models.py                          # Pydantic Action / Observation types
├── graders.py                         # NEWS2 / SOFA / SBT graders
├── scenarios.py                       # Synthetic patient data for all 5 tasks
├── client.py                          # EnvClient
├── README.md
├── .dockerignore
├── .gitignore
└── server/
    ├── app.py                         # FastAPI app + main() entry point
    ├── clinicalops_environment.py     # 5-task Environment implementation
    ├── requirements.txt
    ├── Dockerfile
    └── __init__.py
```

---

All patient data is entirely synthetic. No real patient information is used.
