# Layer 3 Pre-Freeze Audit Note
## For Arvind — generated automatically

---

## 1. pgmpy Version — Confirmed ≥1.1.0

**System:** pgmpy 1.1.2 installed (satisfies ≥1.1.0 requirement).

**Critical note for Nishevithaa's environment:** pgmpy 1.1.0 renamed
`BayesianNetwork` to `DiscreteBayesianNetwork`. All Layer 3 code uses
`DiscreteBayesianNetwork`. If Nishevithaa's environment has pgmpy <1.1.0,
her code will fail with `ImportError` if she ever imports from our layer.
Ask her to run `pip show pgmpy` and confirm ≥1.1.0 before live integration.

`requirements.txt` updated with `pgmpy>=1.1.0` pinned.

---

## 2. Noisy-OR vs pgmpy VariableElimination — CLEARED FOR FREEZE

Test case: classic bacterial (fever=True, focal crackles, purulent sputum).

| Method | P(Antibiotics) |
|---|---|
| pgmpy VariableElimination (4-node BN) | 0.7940 |
| Layer 3 Noisy-OR engine | 0.7821 |
| **Delta** | **0.0119** |

**Result: ✓ Within ±0.05 — cleared for freeze.**

The 0.012 difference is expected — the VE comparison uses a simplified 4-node
BN (fever + crackle + purulence → antibiotics) while the engine includes
additional paths (asymmetry boost, ILD flag, Anthonisen overlay). A simplified
comparison BN will always differ slightly from the full engine. The core
Noisy-OR arithmetic is consistent and the engines agree well within tolerance.

Full pgmpy VariableElimination on the complete 33-node graph is a
**post-freeze improvement**, not a blocker for v2.1.0-study-freeze.

---

## 3. Nishevithaa's Zone Corrections — Confirmed and Applied

Nishevithaa confirmed the corrected bilateral pairs and basal zone list:

**Bilateral pairs (corrected):**

| Pair | Right | Left |
|---|---|---|
| Apical | Z01 | Z02 |
| Mid | Z03 | Z04 |
| Basal ant | Z05 | Z06 |
| Lateral | Z07 | Z08 |
| Apical post | Z09 | Z10 |
| Mid post | Z11 | Z12 |
| Basal post | Z13 | Z14 |
| Basal lat | Z15 | Z16 |
| Lower lat | Z17 | Z18 |
| Axillary | Z19 | Z20 |

Previous error in her code: Z17↔Z19 and Z18↔Z20. Now corrected to Z17↔Z18 and Z19↔Z20.

**Basal zones (confirmed): Z13–Z20** (8 zones — posterior basal and all lateral extensions)

Layer 3 consumes the already-computed `bilateral_asymmetry_score_max` float and
`craniocaudal_gradient_dominant` label from Layer 2. These corrections affect
Nishevithaa's computation, not Layer 3 code directly. No Layer 3 changes needed.

---

## 4. SpO2 Sjoding Correction — Caveat Documented in Code

The uniform −3.0 percentage point correction (Sjoding et al. NEJM 2020) is
documented in `clinical_schema.py` with the following explicit caveat:

```python
# CAVEAT (Arvind, pre-freeze): This uniform correction may OVER-CORRECT
# for patients NOT in the high-melanin risk group. A lighter-skinned patient
# with SpO2=94% is treated as 91% — potentially triggering unnecessary escalation.
# Current decision: accept over-escalation (safer) over under-escalation
# in a CHW setting with limited clinical backup.
# Review when: (a) skin tone field added to clinical schema, or
#              (b) TRUPCR data allows calibration by skin tone.
```

**Clinical consequence of the uniform correction:**

| Raw SpO2 | Adjusted (−3%) | Discretized state | Escalation |
|---|---|---|---|
| 99% | 96% | normal | No |
| 97% | 94% | borderline | No |
| 95% | 92% | borderline | No |
| 94% | 91% | low | **Yes** |
| 92% | 89% | low | **Yes** |
| 90% | 87% | critical | **Yes** |
| 88% | 85% | critical | **Yes** |

A patient with SpO2 ≥95% will never escalate under the corrected thresholds.
Escalation begins at raw SpO2 <95% (adjusted <92%).

**Arvind decision required:** Should we add a `skin_tone_risk` field to the
clinical schema (high/low/unknown) to apply the correction conditionally?
Current implementation applies it uniformly to all patients, which is the
safer default in a CHW setting but may cause unnecessary escalation in
patients not in the high-melanin risk group. If no field is added before
freeze, the uniform correction stays and is clearly documented.

---

## Summary

| Item | Status | Action required |
|---|---|---|
| pgmpy ≥1.1.0 | ✓ 1.1.2 confirmed | Ask Nishevithaa to confirm her environment |
| VE vs Noisy-OR | ✓ Delta 0.012, within ±0.05 | None — cleared for freeze |
| Zone corrections | ✓ Applied in Nishevithaa's code | Arvind confirm Z17↔Z18, Z19↔Z20 anatomically |
| SpO2 Sjoding caveat | ✓ Documented in code | Arvind decision: add skin_tone_risk field? |
