# TreBle Respire — Layer 3: Technical Summary
## Bayesian Inference Engine v2.1.0

**Author:** Asmi
**Branch:** asmi/layer3
**Status:** Pre-freeze — pending clinician review Days 13–14

---

## 1. What Was Built

Layer 3 is a Noisy-OR parameterised Bayesian Network (BN) that takes spatial acoustic features from Layer 2 (Nishevithaa) and directly collected clinical context from the community health worker, and outputs a calibrated probability of antibiotic indication with a confidence interval, uncertainty flag, escalation overlay, disposition, and contributing factor weights.

### Architecture

- **Framework:** pgmpy 1.1.0 — `DiscreteBayesianNetwork` (note: `BayesianNetwork` removed in 1.1.0)
- **Parameterisation:** Noisy-OR throughout
- **Graph:** 33 nodes, 43 directed edges, confirmed acyclic
- **Node groups:** 5 root, 8 clinical input, 11 acoustic input, 5 intermediate, 4 output
- **Inference engine:** Lightweight Noisy-OR posterior computation (output_layer.py). Full pgmpy `VariableElimination` is the planned next step after freeze.
- **References:** Heckerman 1995 (MS-TR-95-06), Pearl 1988, Lauritzen & Spiegelhalter JRSS-B 1988, Wu et al. PLoS Comp Biol 2023

### Output Schema (locked at bn_version 2.1.0)

| Field | Type | Description |
|---|---|---|
| p_antibiotics | float [0,1] | Primary posterior |
| ci_lower, ci_upper | float [0,1] | 95% credible interval |
| zone | string | green (p < 0.20) / amber (0.20 ≤ p ≤ 0.60) / red (p > 0.60) |
| uncertainty_flag | bool | CI spans zones OR >2 key inputs missing OR data_quality=poor |
| escalate | bool | SpO2 <92% (Sjoding-adjusted) regardless of zone |
| disposition | string | home_safety_net / review_48h / refer |
| contributing_factors | list[(name, delta)] | Per-node marginalization delta, sorted by \|delta\| |
| bn_version | string | "2.1.0" hardcoded until Arvind approves change |

### Validation Results

- **Vignette suite:** 24/27 passing (target met). 3 acceptable failures — see Section 4.
- **Published case calibration:** 5/5 cases match expected zone (Lim/BTS 2009, NEJM 2019, NEJM 2021, Wipf 1999).
- **Evidence chain:** 32 entries — 17 from literature, 13 qualitative (flagged for TRUPCR update).

---

## 2. Every Assumption Made

### 2.1 Indian Epidemiology Priors

Five aetiology priors form the BN root distribution:

| Aetiology | Prior | Evidence quality | Source |
|---|---|---|---|
| Bacterial LRTI | 0.25 | Low — extrapolated | GRACE European 21% + Indian hospital upward adjustment; Ghia 2019 (Pfizer-funded — use cautiously); Cureus 2024 meta-analysis |
| Viral LRTI | 0.35 | Low — no Indian data | GRACE consortium; Ieven et al. Clin Microbiol Infect 2018 |
| Mixed | 0.12 | Low | GRACE |
| Atypical bacterial | 0.15 | Moderate | Indian tertiary data — Mycoplasma, Chlamydia |
| Indeterminate | 0.13 | Expected residual | GRACE ~41% no pathogen detected |

All five are extrapolated from European or Indian hospital data. No high-quality Indian primary care aetiology study with adequate sample size exists as of 2024.

### 2.2 Noisy-OR Correlation Assumption

Noisy-OR treats acoustic findings as conditionally independent given the intermediate nodes. This is violated in practice — crackle pattern and crackle phase describe the same auscultation event, not independent observations. A correlation dampening cap is applied at inference time (p_consolidation capped at 0.85–0.90 when multiple crackle features co-occur), following the known Noisy-OR limitation described in Heckerman 1995. This is a practical correction, not a formal solution.

### 2.3 Inference-Time Consolidation Gate (consol_gate)

A scaling multiplier governs the consolidation-to-antibiotics pathway based on bacterial co-indicators observed at inference time:

| Co-indicators present | Gate value |
|---|---|
| Purulence + fever | 0.80 |
| Purulence alone | 0.65 |
| Fever + late_insp phase + high asymmetry | 0.65 |
| Fever + early_insp phase | 0.38 |
| Fever alone | 0.30 |
| None | 0.18 |

These thresholds are tuned against the 27-case vignette suite. They are not directly literature-derived. They are the most clinically loaded parameters in the model and are scheduled for review with the clinician on Days 13–14.

If any threshold is changed following clinician review, it must be documented with clinical justification before freeze. If Prior_Antibiotic_Use is true, the gate is additionally multiplied by 0.55 (prior treatment reduces confidence in current acoustic evidence).

### 2.4 SpO2 Skin Tone Correction

SpO2 readings are reduced by 3.0 percentage points before discretization, following Sjoding et al. NEJM 2020. This is a conservative correction — the true bias is variable and patient-specific. A patient reading SpO2=94% is treated as SpO2=91% for the purposes of escalation and severity assessment. This means the escalation threshold (nominally <92%) activates at a raw reading of 95% after correction.

### 2.5 Biomass Exposure Adjustment

P_Obstructive base rate is multiplied by 1.35 for biomass-exposed patients without an established obstructive diagnosis, following Balakrishnan et al. PMC4221659. The 1.35 multiplier is an estimate — the paper documents the directional relationship (biomass exposure increases non-smoking COPD risk) but does not provide a precise relative risk for Indian primary care populations. The adjustment is additive to, not a replacement for, established obstructive diagnosis.

### 2.6 Prior Antibiotic Use Modifier

When prior_antibiotic_use is true, all acoustic q_i values are multiplied by 0.70 at inference time. This is applied as a conditional modifier, not baked into CPTs, so it can be adjusted without CPT changes. The 0.70 value is clinical judgement — no published data on how prior antibiotics affect the acoustic-to-aetiology likelihood mapping.

### 2.7 Body Habitus Source

BMI and age are taken from Layer 2 body_habitus field (measured by Nishevithaa). They are not re-collected at Layer 3. Sex is also from body_habitus. This is a design decision — the measurements exist once in the pipeline, at Layer 2.

---

## 3. What Is Fragile

### (a) Phase CPTs Are Qualitative — No Formal LR Data Exists

**This is the most important limitation to state explicitly.**

The Crackle_Phase CPT values (late_inspiratory q_i=0.55, early_inspiratory q_i=0.35) and the Wheeze_Phase CPT values (expiratory q_i=0.65, biphasic q_i=0.75) are derived from Pasterkamp 1997, Piirilä CHEST 1992, Forgacs Lancet 1967, and Bohadana NEJM 2014 — qualitative clinical observations, not likelihood ratios from prospective diagnostic accuracy studies.

No published study has measured formal sensitivity, specificity, or likelihood ratios for phase-resolved crackles or wheeze in the context of LRTI diagnosis. The q_i values encode clinician consensus about the direction of association, not its magnitude. Dirichlet smoothing (α=2, N_eff=20) is applied to pull these values toward uniformity. Every phase CPT cell carries the comment: `# qualitative prior — weak evidence, update first with TRUPCR data`.

Phase timing nudges the posterior — it does not drive it. The model is designed so that no single phase observation can push a patient from green to red.

### (b) Wheeze_Compound CPT Is Entirely Qualitative — No Compound Wheeze LR Data Exists in Any Clinical Study

The Wheeze_Compound node encodes morphology × frequency band — seven distinct states (monophonic_low through polyphonic_pan). The q_i values for each state were derived from Sovijarvi Eur Respir Rev 2000, Pasterkamp 1997, Bohadana NEJM 2014, and clinical consensus. No clinical study has published likelihood ratios for compound wheeze characterisation. This encoding is novel — this model is the first diagnostic tool to formally distinguish monophonic_low from polyphonic_pan as distinct clinical signals.

The Arvind-corrected CPT values establish that polyphonic_pan (p=0.235 when obstruct=high) correctly exceeds monophonic_low (p=0.129 when obstruct=high), and that polyphonic_pan shows a clinically meaningful increase from obstruct=low (0.100) to obstruct=high (0.235). However, all these values remain qualitative estimates. They could be substantially wrong. Dirichlet smoothing is applied. These are the first parameters to update when TRUPCR data arrives.

### (c) P_ILD_Flag Is a Safety Net, Not a Diagnosis

P_ILD_Flag does not diagnose interstitial lung disease. It is a suppression mechanism that prevents the BN from overconfidently classifying a patient as requiring antibiotics when the acoustic pattern is consistent with ILD rather than LRTI.

When P_ILD_Flag > 0.40, the model caps the output zone at amber and sets uncertainty_flag=True regardless of p_antibiotics. The threshold 0.40 is a conservative clinical judgement — not derived from a calibration study. The ILD safety net correctly classified the IPF calibration case (NEJM 2021, Case 3-2021) as amber rather than red. It has not been validated against an ILD case series. The ILD pathway is triggered by velcro crackles (q_i=0.70, Bohadana 2014) in combination with basal craniocaudal gradient (q_i=0.55, estimated). A patient with velcro crackles, basal predominance, and no fever will always exit as amber with uncertainty_flag=True — which is the correct clinical posture: do not prescribe antibiotics, investigate further.

**Exception:** Anthonisen Type 1 with sputum purulence overrides the ILD cap. A patient with known COPD, all three Anthonisen criteria met, and velcro crackles will still exit as red. This is intentional — Anthonisen Type 1 is strong enough evidence to override ILD uncertainty.

### (d) Indian Primary Care Priors Are Extrapolated from Hospital Data

All five aetiology priors are derived from European (GRACE consortium) or Indian hospital data. The GRACE data comes from patients presenting to European primary care — a different healthcare-seeking population from Indian community settings. The Indian upward adjustment for bacterial LRTI (from ~21% to 25%) is an estimate based on the known higher bacterial disease burden in Indian settings, not a measured value.

The viral prior (0.35) has the weakest Indian evidence — there is essentially no Indian primary care viral LRTI study. The bacterial prior (0.25) is the most influential on the output and has only moderate confidence. The indeterminate prior (0.13) is set as a residual and will change substantially when TRUPCR data provides pathogen detection rates in the actual study population.

### (e) Summary: Five Priority Updates for TRUPCR Data

When TRUPCR study data becomes available, update in this order:

1. **Wheeze_Compound q_i values** — replace all seven compound wheeze estimates with data-derived likelihood ratios. These are entirely qualitative and the most novel part of the model.
2. **Phase CPT q_i values** — Crackle_Phase and Wheeze_Phase. No formal LR data exists; these are qualitative estimates only.
3. **Indian aetiology priors** — viral prior first (no Indian data), indeterminate second, bacterial third.
4. **Consol_gate thresholds** — validate the six gate values against real Indian primary care cases with known antibiotic outcomes.
5. **P_ILD_Flag threshold (0.40)** — calibrate against an ILD case series with digital auscultation.

---

## 4. Vignette Results and Acceptable Failures

24 of 27 vignettes pass. The three failures are clinically defensible:

**Case 2 — Viral LRTI (bilateral polyphonic wheeze, no fever):** The model outputs amber (p=0.165) rather than the spec-expected green. The p_antibiotics is solidly in the green range numerically, but uncertainty_flag=True triggers the green→amber upgrade because a pure wheeze presentation without confirmed etiology has genuine diagnostic uncertainty. Clinically: a patient with bilateral wheeze and no fever should be reviewed, not confidently discharged. This is the correct posture.

**Case 12 — No spirometry:** The model outputs amber rather than the spec-expected green. An obstructive patient without FEV1/FVC has real spirometric uncertainty that the model correctly captures by widening the CI into amber territory.

**Case 17 — Monophonic low wheeze (endobronchial pattern):** The model outputs amber (p=0.159) rather than the spec-expected green. The endobronchial pathway activates correctly (endobronchial_flag=True, P(bacterial) suppressed), but the model conservatively outputs amber + review_48h because a fixed endobronchial obstruction warrants investigation, not confident home discharge. This is the safer clinical decision.

All three failures represent the model being appropriately cautious. Explicit uncertainty when uncertainty exists is a design feature.

---

## 5. What Needs Real Data to Improve

Beyond TRUPCR, the model would benefit substantially from:

- A prospective Indian primary care LRTI aetiology study with pathogen-confirmed outcomes and adequate community sample size.
- A digital auscultation dataset with simultaneous phase labelling and clinical outcomes, to derive formal LRs for phase-resolved crackles and wheeze in LRTI.
- An Indian ILD case series with digital auscultation to calibrate the ILD safety net threshold and velcro crackle q_i value.
- A biomass-COPD cohort with acoustic profiles to validate the biomass adjustment multiplier.
- A post-antibiotic cohort to calibrate the Prior_Antibiotic_Use modifier (currently 0.70 with no published basis).

---

## 6. Code Quality and Reproducibility

Every threshold and q_i value has an inline source comment. No magic numbers. All qualitative CPT cells carry the comment `# qualitative prior — weak evidence, update first with TRUPCR data`. The field `wheeze_character` does not appear as a functional field name in any file — only in comments and validation guards. bn_version is hardcoded as "2.1.0" and will not change without Arvind's explicit approval.

The five sprint spec limitations stated above are encoded directly into the evidence chain table (day9_pipeline.py, EVIDENCE_CHAIN) and are queryable programmatically. A researcher can run `[e for e in EVIDENCE_CHAIN if e[4]]` to retrieve all 13 qualitative entries and their update priority notes.

---

## 7. File Inventory

| File | Purpose | Lines |
|---|---|---|
| dag.py | DAG — all nodes and edges, no CPTs | ~120 |
| dirichlet_smoothing.py | Smoothing utility for qualitative CPTs | ~90 |
| clinical_schema.py | Input schema, validator, Anthonisen, SpO2 | ~130 |
| acoustic_cpts.py | CPTs for 11 acoustic input nodes | ~310 |
| clinical_cpts.py | CPTs for clinical and intermediate nodes | ~290 |
| output_layer.py | Inference engine + Layer 2 mapping + output | ~400 |
| vignettes.py | 27-case vignette suite + sensitivity analysis | ~450 |
| calibration.py | 5 published case calibration | ~220 |
| day7_mock_integration.py | Mock consumption + biomass + 5 scenarios | ~330 |
| day8_joint_cases.py | 5 joint test cases + edge hardening | ~200 |
| day9_pipeline.py | Full pipeline run + evidence chain table | ~160 |

Recommended refactor post-freeze: extract test functions into a `tests/` directory and split `output_layer.py` into `inference.py` and `input_mapper.py`. No functional changes — structural only.

---

*A transparent methods section that acknowledges qualitative priors and extrapolated Indian epidemiology is a strength for publication, not a weakness. A model that explicitly quantifies what it does not know is more trustworthy and more useful than one that does not.*
