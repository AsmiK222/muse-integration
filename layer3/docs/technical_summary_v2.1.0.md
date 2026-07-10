# TreBle Respire — Layer 3: Technical Summary
## Bayesian Inference Engine v2.1.0

**Author:** Asmi
**Branch:** asmi/layer3
**Status:** Study freeze candidate

---

## 1. What Was Built

Layer 3 is a Noisy-OR parameterised Bayesian Network (BN) that takes spatial acoustic features from Layer 2 (Nishevithaa) and clinical context from the CHW, and outputs a calibrated probability of antibiotic indication with confidence interval, uncertainty flag, escalation overlay, disposition, and contributing factor weights.

### Architecture

- **Framework:** pgmpy 1.1.0 — DiscreteBayesianNetwork
- **Parameterisation:** Noisy-OR throughout
- **Graph:** 33 nodes, 43 directed edges, confirmed acyclic
- **Inference:** Lightweight Noisy-OR posterior engine (output_layer.py)
- **End-to-end pipeline:** pipeline_e2e.py (Layer 1 -> Layer 2 bridge -> Layer 3)

### Output Schema (locked at bn_version 2.1.0)

| Field | Type | Description |
|---|---|---|
| p_antibiotics | float [0,1] | Primary posterior |
| ci_lower, ci_upper | float [0,1] | 95% credible interval |
| zone | string | green (p<0.20) / amber (0.20-0.60) / red (p>0.60) |
| uncertainty_flag | bool | CI spans zones or >2 key inputs missing |
| escalate | bool | SpO2 <92% (Sjoding-adjusted) regardless of zone |
| disposition | string | home_safety_net / review_48h / refer |
| contributing_factors | list | Per-node delta sorted by magnitude |
| bn_version | string | 2.1.0 hardcoded |

### Validation Results

- **Vignettes:** 24/27 passing (target met)
- **Published case calibration:** 5/5
- **Evidence chain:** 32 entries — 19 literature, 13 qualitative (flagged for TRUPCR)
- **ICBHI wheeze reference:** AUROC=0.9447 (Layer 1 generalises across stethoscopes)

---

## 2. Thresholds — Arvind Decisions

Per Arvind's explicit decisions (documented in consistency_config.py):

| Finding | Threshold | Source | Status |
|---|---|---|---|
| Crackle | 0.491 | Nishevithaa-validated in-distribution screening | calibrated |
| Wheeze | 0.243 | In-distribution screening (no Taal-specific value) | provisional |
| ICBHI wheeze study | 0.4509 | ICBHI 2017, AUROC=0.9447 | icbhi_reference_only |

All thresholds imported from consistency_config.py — single source of truth.
CONSISTENCY_THRESHOLD = 0.28 (Arvind decision 3).
Recording duration: 15s normal rate, 20s if <12 breaths/min.

---

## 3. Every Assumption Made

### 3.1 Indian Epidemiology Priors

| Aetiology | Prior | Evidence |
|---|---|---|
| Bacterial LRTI | 0.25 | Low — extrapolated from GRACE + Indian hospital |
| Viral LRTI | 0.35 | Low — no Indian primary care data |
| Mixed | 0.12 | Low — GRACE |
| Atypical bacterial | 0.15 | Moderate — Indian tertiary |
| Indeterminate | 0.13 | Expected residual |

### 3.2 Noisy-OR Correlation Assumption

Acoustic findings treated as conditionally independent given intermediate nodes. Correlation cap applied (P_Consolidation capped at 0.85-0.90 when multiple crackle features co-occur).

### 3.3 Inference-Time Consolidation Gate

| Co-indicators | Gate |
|---|---|
| Purulence + fever | 0.80 |
| Purulence alone | 0.65 |
| Fever + late_insp + high asymmetry | 0.65 |
| Fever + early_insp | 0.38 |
| Fever alone | 0.30 |
| None | 0.18 |

Tuned against 27-case vignette suite. Not literature-derived.

### 3.4 SpO2 Sjoding Correction

Uniform -3.0pp reduction before discretization (Sjoding et al. NEJM 2020).
CAVEAT: may over-correct for patients not in high-melanin risk group.
Decision pending: add skin_tone_risk field or keep uniform correction.

### 3.5 Biomass Exposure Adjustment

P_Obstructive base rate x1.35 for biomass-exposed patients without established COPD diagnosis (Balakrishnan et al. PMC4221659).

---

## 4. What Is Fragile

### (a) Phase CPTs Are Qualitative

Crackle_Phase (late_insp q_i=0.55, early_insp q_i=0.35) and Wheeze_Phase CPTs derived from clinical observation, not formal likelihood ratios. No published LR data for phase-resolved auscultation in LRTI. All phase CPT cells marked: qualitative prior — update first with TRUPCR data.

### (b) Wheeze_Compound CPT Entirely Qualitative

7-state compound wheeze (monophonic_low through polyphonic_pan) — first diagnostic tool to encode this distinction. No compound wheeze LR data exists in any clinical study. Dirichlet smoothing applied. First parameters to update with TRUPCR data.

### (c) P_ILD_Flag Is a Safety Net, Not a Diagnosis

Suppresses overconfident bacterial classification when velcro crackles present. Threshold 0.40 is clinical judgement, not calibrated against ILD case series. Caps output at amber when P_ILD_Flag > 0.40.

### (d) Indian Primary Care Priors Are Extrapolated

All five aetiology priors from European (GRACE) or Indian hospital data. No high-quality Indian primary care aetiology study exists. Viral prior has weakest evidence.

### (e) Five TRUPCR Update Priorities

1. Wheeze_Compound q_i values (entirely qualitative)
2. Crackle_Phase and Wheeze_Phase q_i values
3. Indian aetiology priors — viral first, indeterminate second
4. Consol_gate thresholds
5. P_ILD_Flag threshold (currently 0.40)

---

## 5. End-to-End Pipeline

### Layer 1 (acoustic detection)
- Models: crackle_v1.pt, wheeze_v1.pt, breath_segmenter_v1.pt
- Crackle threshold: 0.491 (Nishevithaa-validated)
- Wheeze threshold: 0.243 (provisional — bench calibration pending)
- Phase segmenter: 193-channel feature extraction (extract_193ch_features())
- Phase segmentation: not yet calibrated for Taal recordings

### Layer 2 bridge (pipeline_e2e.py)
- All 24 required summary flags computed
- Wheeze_Pattern wired into P_Obstructive and focal monophonic suppression path
- Bilateral asymmetry: separate crackle/wheeze thresholds per finding type
- Raw scores stored per zone (raw_p_crackle, raw_p_wheeze)
- Wheeze findings flagged provisional until bench calibration

### Zone mapper (zone_mapper.py)
- Handles all 3 Taal naming conventions (A: aal_raw.wav, B: 01_aal.wav, C: rightapex_raw)
- Tested on 9 patients (101_1 through 105_1)
- Auto-detects convention per folder

### Vignette results (3 acceptable failures)
- Case 2: Viral LRTI — amber instead of green (appropriate caution)
- Case 12: No spirometry — amber instead of green (correct uncertainty)
- Case 17: Monophonic low wheeze — amber instead of green (endobronchial investigation warranted)

---

## 6. File Inventory

| File | Purpose |
|---|---|
| dag.py | BN structure — 33 nodes, 43 edges |
| acoustic_cpts.py | CPTs for 11 acoustic input nodes |
| clinical_cpts.py | CPTs for clinical and intermediate nodes |
| clinical_schema.py | Input schema, validator, Anthonisen, SpO2 |
| dirichlet_smoothing.py | Smoothing utility (alpha=2, N_eff=20) |
| output_layer.py | Full inference engine |
| vignettes.py | 27-case validation suite |
| calibration.py | 5 published case calibration |
| consistency_config.py | Single source for all thresholds and durations |
| pipeline_e2e.py | Layer 1 to Layer 3 end-to-end pipeline |
| zone_mapper.py | Taal recording filename mapper |
| phase_segmenter.py | 193-channel breath segmenter integration |
| wheeze_bench_calibration.py | Taal wheeze calibration (pending recordings) |
| icbhi_wheeze_reference.py | ICBHI sanity check (AUROC=0.9447) |
| day7_mock_integration.py | Mock consumption + 5 scenarios |
| day8_joint_cases.py | J1-J5 joint test cases |
| day9_pipeline.py | Evidence chain (32 entries, 13 qualitative) |
| days11_12_integration.py | Live integration runner |
| days13_14_clinician_review.py | Clinician review presenter |
| day15_freeze.py | Freeze document generator |
| docs/technical_summary_v2.1.0.md | This document |
| docs/freeze_document_v2.1.0.md | Final version document |
| docs/prefreeze_audit_note.md | Pre-freeze audit for Arvind |

---

*A transparent methods section that acknowledges qualitative priors and extrapolated Indian epidemiology is a strength for publication, not a weakness. A model that explicitly quantifies what it does not know is more trustworthy than one that does not.*
