"""
ClinicalOps — Synthetic patient scenarios.

All data is fictional. No real patient data is used.
Designed to produce clear, deterministic NEWS2 / SOFA scores.
"""

from __future__ import annotations
from typing import Any, Dict, List


# ─────────────────────────────────────────────────────────────────────────────
# Task 1 — ED Triage  (8 patients, correct NEWS2 ordering encoded)
# ─────────────────────────────────────────────────────────────────────────────

TRIAGE_PATIENTS: List[Dict[str, Any]] = [
    {
        "patient_id": "PT001",
        "name": "Arjun Mehta",
        "age": 67,
        "chief_complaint": "shortness of breath, confusion",
        "vitals": {"heart_rate": 118, "respiratory_rate": 28, "spo2": 88.0,
                   "systolic_bp": 88, "temperature": 38.9, "consciousness": "Voice"},
        "history": ["COPD", "hypertension"],
    },
    {
        "patient_id": "PT002",
        "name": "Priya Nair",
        "age": 34,
        "chief_complaint": "mild headache",
        "vitals": {"heart_rate": 74, "respiratory_rate": 15, "spo2": 99.0,
                   "systolic_bp": 122, "temperature": 37.1, "consciousness": "Alert"},
        "history": [],
    },
    {
        "patient_id": "PT003",
        "name": "Samuel Okonkwo",
        "age": 55,
        "chief_complaint": "chest pain, diaphoresis",
        "vitals": {"heart_rate": 105, "respiratory_rate": 22, "spo2": 94.0,
                   "systolic_bp": 96, "temperature": 37.4, "consciousness": "Alert"},
        "history": ["diabetes", "smoker"],
    },
    {
        "patient_id": "PT004",
        "name": "Lakshmi Suresh",
        "age": 78,
        "chief_complaint": "fall, right hip pain",
        "vitals": {"heart_rate": 88, "respiratory_rate": 18, "spo2": 96.0,
                   "systolic_bp": 134, "temperature": 36.8, "consciousness": "Alert"},
        "history": ["osteoporosis"],
    },
    {
        "patient_id": "PT005",
        "name": "Carlos Mendez",
        "age": 42,
        "chief_complaint": "unresponsive, found at home",
        "vitals": {"heart_rate": 132, "respiratory_rate": 32, "spo2": 84.0,
                   "systolic_bp": 78, "temperature": 39.6, "consciousness": "Pain"},
        "history": ["alcohol use disorder"],
    },
    {
        "patient_id": "PT006",
        "name": "Fatima Al-Hassan",
        "age": 29,
        "chief_complaint": "ankle sprain",
        "vitals": {"heart_rate": 78, "respiratory_rate": 14, "spo2": 99.0,
                   "systolic_bp": 118, "temperature": 36.9, "consciousness": "Alert"},
        "history": [],
    },
    {
        "patient_id": "PT007",
        "name": "David Chen",
        "age": 61,
        "chief_complaint": "fever, productive cough 3 days",
        "vitals": {"heart_rate": 98, "respiratory_rate": 23, "spo2": 93.0,
                   "systolic_bp": 108, "temperature": 38.6, "consciousness": "Alert"},
        "history": ["type 2 diabetes"],
    },
    {
        "patient_id": "PT008",
        "name": "Meera Krishnan",
        "age": 50,
        "chief_complaint": "nausea, vomiting 1 day",
        "vitals": {"heart_rate": 92, "respiratory_rate": 17, "spo2": 97.0,
                   "systolic_bp": 114, "temperature": 37.5, "consciousness": "Alert"},
        "history": ["migraine"],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Task 2 — Medication Reconciliation
# 5 conflicts embedded: 2 severe, 2 moderate, 1 minor
# ─────────────────────────────────────────────────────────────────────────────

MED_PATIENT: Dict[str, Any] = {
    "patient_id": "PT101",
    "name": "Rajan Pillai",
    "age": 72,
    "chief_complaint": "elective knee replacement — discharge medication review",
    "allergies": ["penicillin", "sulfonamides"],
    "history": ["atrial fibrillation", "type 2 diabetes", "chronic kidney disease stage 3"],
    "vitals": {"heart_rate": 76, "respiratory_rate": 16, "spo2": 98.0,
               "systolic_bp": 138, "temperature": 36.9, "consciousness": "Alert"},
    "medications": [
        # ── Home medications ──
        {"name": "warfarin",     "dose": "5 mg",   "route": "oral", "frequency": "once daily",  "source": "home"},
        {"name": "metformin",    "dose": "1000 mg", "route": "oral", "frequency": "twice daily", "source": "home"},
        {"name": "lisinopril",   "dose": "10 mg",  "route": "oral", "frequency": "once daily",  "source": "home"},
        {"name": "atorvastatin", "dose": "40 mg",  "route": "oral", "frequency": "at night",    "source": "home"},
        # ── Hospital medications (newly prescribed) ──
        {"name": "aspirin",      "dose": "100 mg",  "route": "oral", "frequency": "once daily",  "source": "hospital"},
        {"name": "co-amoxiclav", "dose": "625 mg",  "route": "oral", "frequency": "three times", "source": "hospital"},
        {"name": "ibuprofen",    "dose": "400 mg",  "route": "oral", "frequency": "three times", "source": "hospital"},
        {"name": "metformin",    "dose": "500 mg",  "route": "oral", "frequency": "once daily",  "source": "hospital"},
        {"name": "enoxaparin",   "dose": "40 mg",   "route": "SC",   "frequency": "once daily",  "source": "hospital"},
    ],
}

# Ground truth conflicts — what the grader expects the agent to find
MED_GROUND_TRUTH: List[Dict[str, Any]] = [
    {
        "drug_a": "warfarin",
        "drug_b": "aspirin",
        "conflict_type": "interaction",
        "severity": "severe",
        "description": "Concurrent warfarin + aspirin greatly increases bleeding risk",
        "resolution": "Discuss with cardiology — consider stopping aspirin or adjusting warfarin dose with close INR monitoring",
    },
    {
        "drug_a": "warfarin",
        "drug_b": "enoxaparin",
        "conflict_type": "interaction",
        "severity": "severe",
        "description": "Dual anticoagulation without bridging indication — high bleed risk",
        "resolution": "Stop enoxaparin; warfarin alone sufficient for AF if INR therapeutic",
    },
    {
        "drug_a": "ibuprofen",
        "drug_b": "lisinopril",
        "conflict_type": "interaction",
        "severity": "moderate",
        "description": "NSAIDs reduce ACE-inhibitor efficacy and worsen renal function in CKD",
        "resolution": "Switch ibuprofen to paracetamol for analgesia given CKD stage 3",
    },
    {
        "drug_a": "co-amoxiclav",
        "drug_b": "penicillin",
        "conflict_type": "contraindication",
        "severity": "moderate",
        "description": "Co-amoxiclav contains amoxicillin — a penicillin. Patient is allergic.",
        "resolution": "Switch to clindamycin or consult allergy/microbiology",
    },
    {
        "drug_a": "metformin",
        "drug_b": "metformin",
        "conflict_type": "duplicate",
        "severity": "minor",
        "description": "Metformin prescribed from both home and hospital lists — duplicate dose",
        "resolution": "Rationalise to single dose; review renal function before continuing",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# Task 3 — Sepsis Watch  (10-step dynamic episode)
# Vitals/labs evolve each step unless agent escalates correctly
# ─────────────────────────────────────────────────────────────────────────────

SEPSIS_PATIENT_BASE: Dict[str, Any] = {
    "patient_id": "PT201",
    "name": "Vikram Anand",
    "age": 58,
    "chief_complaint": "fever and confusion — admitted overnight from ED",
    "allergies": [],
    "history": ["type 2 diabetes", "chronic liver disease"],
    "vitals": {"heart_rate": 95, "respiratory_rate": 20, "spo2": 96.0,
               "systolic_bp": 108, "temperature": 38.4, "consciousness": "Alert"},
    "labs": {
        "wbc": 14.2,
        "creatinine": 125,
        "lactate": 1.8,
        "bilirubin": 28,
        "platelets": 148,
        "pao2_fio2": 340,
    },
}

# Deterioration trajectory — indexed by step number (0-based)
# Each entry shows how the state changes if agent has NOT escalated yet
SEPSIS_TRAJECTORY: List[Dict[str, Any]] = [
    # step 0 — initial state (same as base)
    {"vitals": {"heart_rate": 95,  "respiratory_rate": 20, "spo2": 96.0, "systolic_bp": 108, "temperature": 38.4, "consciousness": "Alert"},
     "labs":   {"wbc": 14.2, "creatinine": 125, "lactate": 1.8, "bilirubin": 28, "platelets": 148, "pao2_fio2": 340}, "sofa": 2},
    # step 1
    {"vitals": {"heart_rate": 102, "respiratory_rate": 22, "spo2": 95.0, "systolic_bp": 104, "temperature": 38.7, "consciousness": "Alert"},
     "labs":   {"wbc": 16.1, "creatinine": 140, "lactate": 2.2, "bilirubin": 32, "platelets": 138, "pao2_fio2": 310}, "sofa": 3},
    # step 2
    {"vitals": {"heart_rate": 110, "respiratory_rate": 24, "spo2": 94.0, "systolic_bp": 98,  "temperature": 39.1, "consciousness": "Alert"},
     "labs":   {"wbc": 18.0, "creatinine": 162, "lactate": 2.8, "bilirubin": 40, "platelets": 122, "pao2_fio2": 280}, "sofa": 4},
    # step 3 — SOFA=5, agent should have escalated by now
    {"vitals": {"heart_rate": 118, "respiratory_rate": 26, "spo2": 93.0, "systolic_bp": 92,  "temperature": 39.3, "consciousness": "Voice"},
     "labs":   {"wbc": 20.4, "creatinine": 188, "lactate": 3.4, "bilirubin": 52, "platelets": 104, "pao2_fio2": 245}, "sofa": 5},
    # step 4 — SOFA=6, danger threshold crossed
    {"vitals": {"heart_rate": 126, "respiratory_rate": 30, "spo2": 91.0, "systolic_bp": 84,  "temperature": 39.6, "consciousness": "Voice"},
     "labs":   {"wbc": 22.1, "creatinine": 220, "lactate": 4.1, "bilirubin": 68, "platelets": 88,  "pao2_fio2": 210}, "sofa": 6},
    # step 5 — critical
    {"vitals": {"heart_rate": 134, "respiratory_rate": 32, "spo2": 89.0, "systolic_bp": 78,  "temperature": 39.8, "consciousness": "Pain"},
     "labs":   {"wbc": 24.0, "creatinine": 262, "lactate": 5.0, "bilirubin": 84, "platelets": 72,  "pao2_fio2": 185}, "sofa": 7},
    # steps 6-9 — plateau (max deterioration)
    {"vitals": {"heart_rate": 136, "respiratory_rate": 33, "spo2": 88.0, "systolic_bp": 76,  "temperature": 39.9, "consciousness": "Pain"},
     "labs":   {"wbc": 24.5, "creatinine": 280, "lactate": 5.4, "bilirubin": 92, "platelets": 68,  "pao2_fio2": 178}, "sofa": 8},
    {"vitals": {"heart_rate": 138, "respiratory_rate": 34, "spo2": 87.0, "systolic_bp": 74,  "temperature": 40.0, "consciousness": "Pain"},
     "labs":   {"wbc": 25.0, "creatinine": 295, "lactate": 5.8, "bilirubin": 96, "platelets": 64,  "pao2_fio2": 172}, "sofa": 8},
    {"vitals": {"heart_rate": 140, "respiratory_rate": 34, "spo2": 86.0, "systolic_bp": 72,  "temperature": 40.1, "consciousness": "Pain"},
     "labs":   {"wbc": 25.2, "creatinine": 302, "lactate": 6.0, "bilirubin": 98, "platelets": 62,  "pao2_fio2": 170}, "sofa": 8},
    {"vitals": {"heart_rate": 140, "respiratory_rate": 34, "spo2": 86.0, "systolic_bp": 72,  "temperature": 40.1, "consciousness": "Pain"},
     "labs":   {"wbc": 25.2, "creatinine": 302, "lactate": 6.0, "bilirubin": 98, "platelets": 62,  "pao2_fio2": 170}, "sofa": 8},
]
# ─────────────────────────────────────────────────────────────────────────────
# Task 4 — Ventilator Weaning  (Medium-Hard, 8 steps)
# Agent assesses ICU patient for readiness to wean off mechanical ventilation
# using SBT (Spontaneous Breathing Trial) criteria
# ─────────────────────────────────────────────────────────────────────────────

VENT_PATIENT_BASE = {
    "patient_id": "PT301",
    "name": "Aditya Rao",
    "age": 64,
    "chief_complaint": "Day 5 post-ARDS — assess for ventilator weaning",
    "allergies": [],
    "history": ["ARDS secondary to pneumonia", "hypertension", "type 2 diabetes"],
    "vitals": {
        "heart_rate": 88, "respiratory_rate": 18, "spo2": 97.0,
        "systolic_bp": 118, "temperature": 37.2, "consciousness": "Alert"
    },
    "vent_settings": {
        "mode": "SIMV",
        "fio2": 0.40,
        "peep": 8,
        "pressure_support": 12,
        "tidal_volume": 480,
        "rr_set": 14,
        "pip": 22,
    },
    "labs": {
        "pao2": 88.0,
        "paco2": 42.0,
        "ph": 7.38,
        "hco3": 24.0,
        "pao2_fio2": 220.0,
        "wbc": 9.2,
        "hemoglobin": 10.8,
    },
    "sbt_attempted": False,
    "rsbi": 68,  # Rapid Shallow Breathing Index — <105 is good
}

# Ground truth weaning readiness checklist (SBT criteria)
VENT_WEANING_CHECKLIST = [
    {"check": "assess_oxygenation",   "required": True,  "description": "Check PaO2/FiO2 ratio — must be > 150"},
    {"check": "assess_consciousness", "required": True,  "description": "Confirm patient is awake and following commands"},
    {"check": "assess_secretions",    "required": True,  "description": "Check secretion burden — must be manageable"},
    {"check": "check_rsbi",           "required": True,  "description": "RSBI < 105 predicts successful extubation"},
    {"check": "reduce_fio2",          "required": True,  "description": "Reduce FiO2 to <= 0.40 before SBT"},
    {"check": "reduce_peep",          "required": True,  "description": "Reduce PEEP to <= 5 cmH2O"},
    {"check": "perform_sbt",          "required": True,  "description": "Conduct 30-minute spontaneous breathing trial"},
    {"check": "extubate",             "required": False, "description": "Decision to extubate based on SBT result"},
]

# ─────────────────────────────────────────────────────────────────────────────
# Task 5 — Diagnostic Reasoning  (Hard, 12 steps)
# Patient with ambiguous presentation — agent must order investigations
# strategically to narrow to correct diagnosis
# ─────────────────────────────────────────────────────────────────────────────

DIAGNOSTIC_PATIENT = {
    "patient_id": "PT401",
    "name": "Neha Sharma",
    "age": 45,
    "chief_complaint": "3-week history of fatigue, weight loss 6kg, night sweats, dry cough",
    "allergies": ["sulfonamides"],
    "history": ["visited rural Maharashtra 2 months ago", "works as schoolteacher"],
    "vitals": {
        "heart_rate": 94, "respiratory_rate": 20, "spo2": 95.0,
        "systolic_bp": 108, "temperature": 37.8, "consciousness": "Alert"
    },
    "labs": {
        "wbc": 11.2,
        "hemoglobin": 10.4,
        "esr": 88,
        "crp": 42,
        "ldh": 380,
        "calcium": 2.62,
        "albumin": 32,
    },
}

# Differential diagnoses — agent must narrow to correct one
DIFFERENTIAL_DIAGNOSES = [
    "pulmonary_tuberculosis",
    "sarcoidosis",
    "lymphoma",
    "lung_adenocarcinoma",
    "community_acquired_pneumonia",
    "hypersensitivity_pneumonitis",
    "pulmonary_histoplasmosis",
    "miliary_tuberculosis",
    "bronchiectasis",
    "pulmonary_embolism",
]

# Correct diagnosis
CORRECT_DIAGNOSIS = "pulmonary_tuberculosis"

# Investigation results — what each test reveals
DIAGNOSTIC_INVESTIGATIONS = {
    "chest_xray": {
        "result": "Bilateral upper lobe infiltrates with cavitation. Hilar lymphadenopathy.",
        "narrows_to": ["pulmonary_tuberculosis", "miliary_tuberculosis", "sarcoidosis"],
        "rules_out": ["pulmonary_embolism", "bronchiectasis"],
        "yield": "high",
    },
    "sputum_afb_smear": {
        "result": "Acid-fast bacilli 2+ on ZN staining. Consistent with active tuberculosis.",
        "narrows_to": ["pulmonary_tuberculosis", "miliary_tuberculosis"],
        "rules_out": ["sarcoidosis", "lymphoma", "lung_adenocarcinoma", "hypersensitivity_pneumonitis"],
        "yield": "definitive",
    },
    "sputum_culture": {
        "result": "Mycobacterium tuberculosis complex — pending sensitivity (result in 6 weeks).",
        "narrows_to": ["pulmonary_tuberculosis"],
        "rules_out": [],
        "yield": "definitive",
    },
    "ct_chest": {
        "result": "Tree-in-bud pattern, upper lobe consolidation with central cavitation. Mediastinal lymphadenopathy. No pleural effusion.",
        "narrows_to": ["pulmonary_tuberculosis", "miliary_tuberculosis"],
        "rules_out": ["lung_adenocarcinoma", "pulmonary_embolism", "bronchiectasis"],
        "yield": "high",
    },
    "mantoux_test": {
        "result": "Induration 18mm at 48h — strongly positive.",
        "narrows_to": ["pulmonary_tuberculosis", "miliary_tuberculosis"],
        "rules_out": [],
        "yield": "moderate",
    },
    "igra_test": {
        "result": "Interferon-Gamma Release Assay POSITIVE — Mycobacterium tuberculosis infection confirmed.",
        "narrows_to": ["pulmonary_tuberculosis", "miliary_tuberculosis"],
        "rules_out": ["sarcoidosis", "hypersensitivity_pneumonitis"],
        "yield": "high",
    },
    "bronchoscopy_bal": {
        "result": "BAL: lymphocytosis, AFB positive in lavage. No malignant cells.",
        "narrows_to": ["pulmonary_tuberculosis"],
        "rules_out": ["lung_adenocarcinoma", "lymphoma"],
        "yield": "high",
    },
    "serum_ace": {
        "result": "ACE level 28 U/L — normal. Makes sarcoidosis less likely.",
        "narrows_to": ["pulmonary_tuberculosis", "lymphoma", "miliary_tuberculosis"],
        "rules_out": ["sarcoidosis"],
        "yield": "moderate",
    },
    "hiv_test": {
        "result": "HIV antibody NEGATIVE. CD4 count not indicated.",
        "narrows_to": [],
        "rules_out": [],
        "yield": "low",
    },
    "lft_rft": {
        "result": "LFTs mildly elevated (ALT 52, AST 48). Renal function normal.",
        "narrows_to": [],
        "rules_out": [],
        "yield": "low",
    },
    "pet_scan": {
        "result": "FDG-avid mediastinal and hilar nodes. Upper lobe hypermetabolic lesions. No extrathoracic disease.",
        "narrows_to": ["pulmonary_tuberculosis", "sarcoidosis", "lymphoma"],
        "rules_out": ["pulmonary_embolism"],
        "yield": "moderate",
    },
    "echocardiogram": {
        "result": "Normal LV function. No pericardial effusion. No features of cardiac sarcoidosis.",
        "narrows_to": [],
        "rules_out": ["sarcoidosis"],
        "yield": "low",
    },
}

# Minimum efficient path — what a good clinician would order
OPTIMAL_INVESTIGATION_PATH = [
    "chest_xray", "sputum_afb_smear", "ct_chest", "mantoux_test"
]
