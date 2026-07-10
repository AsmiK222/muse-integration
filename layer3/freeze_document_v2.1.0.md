# TreBle Respire — Layer 3: Freeze Document
## Version v2.1.0-study-freeze

**Generated:** 2026-05-11 17:40:27
**Author:** Asmi
**Branch:** asmi/layer3
**Tag:** v2.1.0-study-freeze

---

## 1. Model Architecture

| Parameter | Value |
|---|---|
| Framework | pgmpy 1.1.0 — DiscreteBayesianNetwork |
| Parameterisation | Noisy-OR throughout |
| Nodes | 33 total — 5 root, 8 clinical, 11 acoustic, 5 intermediate, 4 output |
| Edges | 43 directed edges, confirmed acyclic |
| bn_version | 2.1.0 (hardcoded — change requires Arvind approval) |
| Inference | Lightweight Noisy-OR engine (output_layer.py) |

---

## 2. Validation Results

| Check | Result |
|---|---|
| Vignette suite | 24/27 passing (✓ TARGET MET) |
| Published case calibration | 5/5 matching expected zone |
| Evidence chain entries | 30 total — 17 literature, 13 qualitative |

**Acceptable vignette failures (Case 2, Case 12, Case 17):** All three fail by outputting amber
where the spec expected green. All are clinically defensible — the model is being
appropriately cautious. Documented in technical_summary_v2.1.0.md.

---

## 3. Dirichlet Smoothing Parameters

Applied to ALL phase-related and Wheeze_Compound CPTs:

| Parameter | Value |
|---|---|
| alpha (pseudocount) | 2 |
| N_effective (effective sample size) | 20 |
| Formula | smoothed = (raw × N_eff + α) / (N_eff + α × n_states) |
| Reference | Heckerman 1995 (MS-TR-95-06) |

---

## 4. Wheeze_Compound Mapping Table

All values qualitative — NO compound wheeze LR data exists in any clinical study.
These are the first parameters to update with TRUPCR data.

| State | Morphology | Freq band | q_i | Clinical meaning | Source | Endobronchial suppression |
|---|---|---|---|---|---|---|
| monophonic_low | monophonic | low | 0.55 | Large airway fixed obstruction | Sovijarvi Eur Respir Rev 2000 | Strong (0.40) — suppress P(bacterial), flag for investigation |
| monophonic_mid | monophonic | mid | 0.50 | Medium airway obstruction | Pasterkamp 1997 | Moderate (0.25) |
| monophonic_high | monophonic | high | 0.45 | Small airway peripheral | Clinical consensus | Mild (0.15) |
| polyphonic_low | polyphonic | low | 0.72 | COPD pattern, large airway | Pasterkamp 1997 | None |
| polyphonic_mid | polyphonic | mid | 0.70 | Asthma, small airway | Bohadana NEJM 2014 | None |
| polyphonic_pan | polyphonic | mixed | 0.78 | Severe pan-airway obstruction | Clinical consensus | None |
| uncertain | — | — | N/A | Subtype not determined | — | Marginalize (None → pgmpy) |

---

## 5. All Qualitative CPTs — Flagged for TRUPCR Update

These 13 CPT values have no formal likelihood ratio backing.
All are marked `# qualitative prior — weak evidence, update first with TRUPCR data` in code.

| Node | Context | q_i | Source |
|---|---|---|---|
| Crackle_Phase | late_inspiratory | 0.55 | Pasterkamp 1997; Piirilä CHEST 1992 |
| Crackle_Phase | early_inspiratory | 0.35 | Forgacs Lancet 1967 |
| Wheeze_Phase | expiratory | 0.65 | Metlay JAMA 1997; Bohadana NEJM 2014 |
| Wheeze_Phase | biphasic | 0.75 | Bohadana NEJM 2014 |
| Wheeze_Compound | monophonic_low | 0.55 | Sovijarvi Eur Respir Rev 2000 |
| Wheeze_Compound | monophonic_mid | 0.5 | Pasterkamp 1997 |
| Wheeze_Compound | monophonic_high | 0.45 | Clinical consensus |
| Wheeze_Compound | polyphonic_low | 0.72 | Pasterkamp 1997 |
| Wheeze_Compound | polyphonic_mid | 0.7 | Bohadana NEJM 2014 |
| Wheeze_Compound | polyphonic_pan | 0.78 | Clinical consensus |
| Crackle_Subtype | velcro → P_ILD | 0.7 | Bohadana NEJM 2014 |
| Craniocaudal_Gradient | basal → P_ILD | 0.55 | Clinical pattern — ESTIMATED |
| Wheeze_Compound | endobronchial suppression monophonic_low | 0.4 | Sovijarvi 2000 |

---

## 6. Sensitivity Analysis Results

Sensitivity was tested at ±20% perturbation of each q_i on the classic bacterial
base case (Case 1: focal coarse crackles, late_insp, fever, high asymmetry, purulent sputum).

**Result:** 0 parameters flip output zone at ±20% on this base case.

**Honest caveat:** This test was run on a strong-signal case (p_antibiotics ~0.78).
Amber boundary cases (p_antibiotics 0.20–0.45) are expected to be more sensitive.
The consol_gate thresholds were NOT swept — these are the most clinically loaded
parameters and were reviewed with the clinician on Days 13–14.

Phase and Wheeze_Compound q_i values remain qualitative regardless of this result.
They are still the first parameters to update with TRUPCR data.

---

## 7. Priority Update List for TRUPCR Data

When study data arrives, update in this order:

| Priority | Parameter set | Why first |
|---|---|---|
| 1st | Wheeze_Compound q_i values (all 7 states) | Entirely qualitative. NO compound wheeze LR data in any clinical study. |
| 2nd | Crackle_Phase and Wheeze_Phase q_i values | No formal LR for phase-resolved auscultation in LRTI. |
| 3rd | Indian aetiology priors — viral first, indeterminate second, bacterial third | All extrapolated from European or Indian hospital data. No primary care study. |
| 4th | Consol_gate thresholds | Tuned against 27-case vignette suite. Not literature-derived. |
| 5th | P_ILD_Flag threshold (currently 0.40) | Clinical judgement. Not calibrated against ILD case series. |

---

## 8. Zone Boundaries

| Zone | Condition |
|---|---|
| green | p_antibiotics strictly < 0.20 (0.20 itself is amber) |
| amber | 0.20 ≤ p_antibiotics ≤ 0.60 (both endpoints inclusive) |
| red | p_antibiotics strictly > 0.60 (0.60 itself is amber) |

Escalation overlay: SpO2 < 92% (after Sjoding −3% skin tone correction) →
escalate=True and disposition=refer REGARDLESS of p_antibiotics or zone.

---

## 9. Interface Contract (locked)

Layer 3 reads from Layer 2 output:
- `zone_findings[].is_dual_representation_zone` — always True for Z06, Z08
- `zone_findings[].recorded` — False for unrecorded zones (not omitted)
- `summary_flags.wheeze_compound_dominant` — 7 states + None (NOT wheeze_character)
- `output_meta.subtype_modules_run` — bool (not string)
- `output_meta.data_quality` — "normal" or "poor"

subtype_modules_run=False → pass None to BN (marginalize).
subtype_modules_run=True  → pass the actual value.
None ≠ "uncertain" — these have different BN semantics.

---

*Freeze commit: tag as v2.1.0-study-freeze after Arvind merges to main.*
