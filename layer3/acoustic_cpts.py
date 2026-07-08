"""
TreBle Respire — Layer 3: Acoustic CPTs
Author: Asmi | Day 2

ALL CPTs for acoustic input nodes, using Noisy-OR parameterisation.
Every q_i value is documented with:
  (1) source LR+
  (2) Bayes algebra conversion
  (3) resulting q_i
  (4) whether it is from literature or estimated

Noisy-OR conversion from LR+:
    sensitivity ≈ LR+ / (LR+ + specificity/sensitivity)  [simplified]
    q_i = P(node=1 | parent=1, all others=0)
        = 1 - (1 - base_rate) / (1 - base_rate * (1 - LR+ conversion))

Practical formula used (Pearl 1988):
    If LR+ known: q_i ≈ LR+ / (1 + LR+)   [maps LR+ to [0,1] monotonically]
    This is an approximation — see comments per node for exact reasoning.

Dirichlet smoothing (alpha=2, N_eff=20) applied to ALL phase and
Wheeze_Compound CPTs immediately after setting q_i values.
These are marked: # qualitative prior — weak evidence, update first with TRUPCR data

References:
  - Wipf et al. Arch Intern Med 1999       (crackle LRs)
  - Htun 2019 meta-analysis                (crackles pooled LR+ 2.42)
  - Heckerling Ann Intern Med 1990         (bronchial breath, diminished sounds)
  - McGee / Diehr                          (bilateral asymmetry LR)
  - Metlay JAMA 1997                       (wheeze non-discriminatory)
  - Pasterkamp 1997                        (coarse crackle, phase, compound wheeze)
  - Piirilä CHEST 1992                     (crackle phase)
  - Forgacs Lancet 1967                    (early insp crackles)
  - Sovijarvi Eur Respir Rev 2000          (monophonic wheeze)
  - Bohadana NEJM 2014                     (velcro, biphasic wheeze)
  - Pearl 1988                             (Noisy-OR)
  - Heckerman 1995 MS-TR-95-06            (BN parameterisation)
"""

import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))
from dirichlet_smoothing import smooth_and_verify, verify_cpt_rows_sum_to_one


# ---------------------------------------------------------------------------
# NOISY-OR HELPER
# ---------------------------------------------------------------------------

def lr_to_qi(lr_plus: float) -> float:
    """
    Converts a published LR+ to a Noisy-OR q_i value.
    Formula: q_i = LR+ / (1 + LR+)   — Pearl 1988 monotonic mapping.
    Clips to [0.05, 0.95] to avoid deterministic CPT cells.
    """
    qi = lr_plus / (1.0 + lr_plus)
    return float(np.clip(qi, 0.05, 0.95))


def noisy_or_cpt(qi_values: list[float], n_parent_states: list[int]) -> np.ndarray:
    """
    Builds a Noisy-OR CPT for a binary child node.
    qi_values[i] = probability child=1 when only parent i is active.
    Returns array shape (2, product(n_parent_states)).
    Columns = all parent state combinations (pgmpy convention).
    """
    import itertools
    parent_combos = list(itertools.product(*[range(n) for n in n_parent_states]))
    n_cols = len(parent_combos)

    cpt = np.zeros((2, n_cols))
    for col_idx, combo in enumerate(parent_combos):
        # Noisy-OR: P(child=0) = product of (1 - q_i) for each active parent
        p_child_0 = 1.0
        for parent_idx, parent_state in enumerate(combo):
            if parent_state == 1:  # parent is active
                p_child_0 *= (1.0 - qi_values[parent_idx])
        p_child_1 = 1.0 - p_child_0
        cpt[0, col_idx] = p_child_0
        cpt[1, col_idx] = p_child_1

    return cpt


# ---------------------------------------------------------------------------
# ACOUSTIC NODE CPTs
# Each node: states defined, q_i documented, CPT built, verified.
# ---------------------------------------------------------------------------

def build_acoustic_cpts() -> dict:
    """
    Builds all acoustic input node CPTs.
    Returns dict: {node_name: {"states": [...], "cpt": np.ndarray, "parents": [...]}}
    CPTs are NOT yet attached to pgmpy model — see attach_cpts.py (Day 3).
    """
    cpts = {}

    # -----------------------------------------------------------------------
    # CRACKLE_PATTERN
    # States: absent / focal / bilateral_basal / diffuse
    # Parent: Season (monsoon/winter/other) — affects base rates
    # LR source: Wipf et al. 1999, Htun 2019 meta-analysis
    #
    # q_i conversion:
    #   focal crackles LR+ = 2.5–4.0 (Wipf 1999) → use midpoint 3.0 → q_i = 3/(1+3) = 0.75
    #   pooled crackles LR+ = 2.42 (Htun 2019)    → q_i = 2.42/3.42 = 0.71
    #   bilateral_basal: estimated from CHF/pneumonia literature → q_i = 0.55 (estimated)
    #   diffuse: estimated → q_i = 0.45 (estimated, less specific)
    # -----------------------------------------------------------------------
    # NOTE: Crackle_Pattern is an OBSERVATION node from Layer 2.
    # Its CPT here represents P(pattern observed | season context).
    # Season only modestly affects base prevalence — main evidence comes from
    # the pattern value itself propagating to P_Consolidation.
    #
    # Shape: (4 states) × (3 season states) = (4, 3)
    # Rows: absent, focal, bilateral_basal, diffuse
    # Cols: monsoon, winter, other
    # qualitative prior — weak evidence, update first with TRUPCR data
    crackle_pattern_raw = np.array([
        [0.55, 0.45, 0.60],   # absent: less absent in winter (more LRTI)
        [0.20, 0.28, 0.18],   # focal: higher in winter (bacterial pneumonia peak)
        [0.15, 0.17, 0.13],   # bilateral_basal: moderate seasonal variation
        [0.10, 0.10, 0.09],   # diffuse: relatively season-stable
    ])
    crackle_pattern_cpt = smooth_and_verify(crackle_pattern_raw, "Crackle_Pattern")
    cpts["Crackle_Pattern"] = {
        "states": ["absent", "focal", "bilateral_basal", "diffuse"],
        "parents": ["Season"],
        "parent_states": [["monsoon", "winter", "other"]],
        "cpt": crackle_pattern_cpt,
        "note": "qualitative prior — weak evidence, update first with TRUPCR data"
    }

    # -----------------------------------------------------------------------
    # CRACKLE_PHASE
    # States: early_inspiratory / late_inspiratory / expiratory / mixed / absent
    # Parents: P_Consolidation (high/low)
    #
    # q_i (qualitative — no formal LR data for phase-resolved auscultation):
    #   late_insp → P_Consolidation: q_i = 0.55 (Pasterkamp 1997; Piirilä CHEST 1992)
    #   early_insp → P_Consolidation: q_i = 0.35 (Forgacs Lancet 1967 — modest decrease)
    #   velcro pattern (late_insp fine) → P_ILD_Flag: handled separately in ILD CPT
    #
    # ALL values qualitative — no published LR for phase-resolved crackles in LRTI.
    # Dirichlet smoothing REQUIRED.
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    # Shape: (5, 2) — 5 phase states × 2 P_Consolidation states (low, high)
    crackle_phase_raw = np.array([
        [0.25, 0.15],   # early_insp:  more common without consolidation
        [0.30, 0.45],   # late_insp:   increases with consolidation — q_i=0.55
        [0.15, 0.12],   # expiratory:  less common overall
        [0.10, 0.13],   # mixed
        [0.20, 0.15],   # absent:      decreases when consolidation present
    ])
    crackle_phase_cpt = smooth_and_verify(crackle_phase_raw, "Crackle_Phase")
    cpts["Crackle_Phase"] = {
        "states": ["early_inspiratory", "late_inspiratory", "expiratory", "mixed", "absent"],
        "parents": ["P_Consolidation"],
        "parent_states": [["low", "high"]],
        "cpt": crackle_phase_cpt,
        "note": "qualitative prior — weak evidence, update first with TRUPCR data"
    }

    # -----------------------------------------------------------------------
    # CRACKLE_SUBTYPE
    # States: fine / coarse / velcro / uncertain
    # Parents: P_Consolidation (low/high), P_ILD_Flag (low/high)
    #
    # q_i:
    #   coarse crackles → P_Consolidation: LR+ ~3.0 estimated (Pasterkamp 1997,
    #     Piirilä CHEST 1992 — flagged as estimate, no direct study)
    #     q_i = 3.0/4.0 = 0.75
    #   velcro → P_ILD_Flag: q_i = 0.70 (Bohadana NEJM 2014 — stronger evidence)
    #   fine → less specific: q_i = 0.40 estimated
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    # Parents: P_Consolidation (2 states) × P_ILD_Flag (2 states) = 4 columns
    # Rows: fine, coarse, velcro, uncertain
    # Cols: (consol=low,ILD=low), (consol=high,ILD=low),
    #       (consol=low,ILD=high), (consol=high,ILD=high)
    crackle_subtype_raw = np.array([
        [0.30, 0.25, 0.15, 0.15],   # fine: moderate in all — CHF pattern
        [0.25, 0.45, 0.10, 0.35],   # coarse: high when consolidation — q_i~0.75
        [0.10, 0.08, 0.55, 0.35],   # velcro: high when ILD flag — q_i=0.70
        [0.35, 0.22, 0.20, 0.15],   # uncertain: default when subtype not clear
    ])
    crackle_subtype_cpt = smooth_and_verify(crackle_subtype_raw, "Crackle_Subtype")
    cpts["Crackle_Subtype"] = {
        "states": ["fine", "coarse", "velcro", "uncertain"],
        "parents": ["P_Consolidation", "P_ILD_Flag"],
        "parent_states": [["low", "high"], ["low", "high"]],
        "cpt": crackle_subtype_cpt,
        "note": "coarse q_i ~0.75 estimated (Pasterkamp/Piirilä); velcro q_i=0.70 (Bohadana 2014)"
    }

    # -----------------------------------------------------------------------
    # WHEEZE_PATTERN
    # States: absent / focal / bilateral / diffuse
    # Parents: P_Obstructive_Exacerbation (low/high)
    #
    # LR for wheeze (any) in bacterial LRTI: ~1.0 (Metlay JAMA 1997)
    # → wheeze is NON-DISCRIMINATORY for bacterial vs viral
    # → feeds obstructive branch ONLY, not consolidation branch
    # q_i = 0.50 (effectively non-informative for bacterial inference)
    # -----------------------------------------------------------------------
    wheeze_pattern_raw = np.array([
        [0.55, 0.20],   # absent: common without obstruction
        [0.15, 0.25],   # focal: endobronchial or local bronchospasm
        [0.20, 0.40],   # bilateral: classic asthma/COPD
        [0.10, 0.15],   # diffuse
    ])
    wheeze_pattern_cpt = smooth_and_verify(wheeze_pattern_raw, "Wheeze_Pattern")
    cpts["Wheeze_Pattern"] = {
        "states": ["absent", "focal", "bilateral", "diffuse"],
        "parents": ["P_Obstructive_Exacerbation"],
        "parent_states": [["low", "high"]],
        "cpt": wheeze_pattern_raw,  # no Dirichlet needed — not phase CPT
        "note": "Wheeze LR ~1.0 for bacterial LRTI (Metlay JAMA 1997) — non-discriminatory"
    }
    verify_cpt_rows_sum_to_one(wheeze_pattern_raw, "Wheeze_Pattern")

    # -----------------------------------------------------------------------
    # WHEEZE_PHASE
    # States: inspiratory / expiratory / biphasic / absent
    # Parents: P_Obstructive_Exacerbation (low/high)
    #
    # q_i (qualitative):
    #   expiratory wheeze → P_Obstructive: q_i = 0.65 (Metlay 1997; Bohadana 2014)
    #   biphasic wheeze → P_Obstructive: q_i = 0.75 (Bohadana NEJM 2014 — stronger)
    #   inspiratory wheeze: q_i = 0.40 (less specific — upper airway possible)
    # ALL qualitative — Dirichlet smoothing applied.
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    wheeze_phase_raw = np.array([
        [0.15, 0.20],   # inspiratory: q_i~0.40 — less specific for obstruction
        [0.35, 0.50],   # expiratory: q_i=0.65 — classic obstructive
        [0.10, 0.20],   # biphasic: q_i=0.75 — severe obstruction
        [0.40, 0.10],   # absent: drops sharply with obstructive exacerbation
    ])
    wheeze_phase_cpt = smooth_and_verify(wheeze_phase_raw, "Wheeze_Phase")
    cpts["Wheeze_Phase"] = {
        "states": ["inspiratory", "expiratory", "biphasic", "absent"],
        "parents": ["P_Obstructive_Exacerbation"],
        "parent_states": [["low", "high"]],
        "cpt": wheeze_phase_cpt,
        "note": "qualitative prior — weak evidence, update first with TRUPCR data"
    }

    # -----------------------------------------------------------------------
    # WHEEZE_COMPOUND
    # States: 7 (see WHEEZE_COMPOUND_STATES in dag.py)
    # Parents: P_Obstructive_Exacerbation (low/high)
    #
    # ALL VALUES QUALITATIVE — no compound wheeze LR data in any clinical study.
    # This is the first encoding of morphology × frequency band in a diagnostic tool.
    # q_i values from sprint spec tables (Sovijarvi 2000, Pasterkamp 1997, Bohadana 2014,
    # clinical consensus). ALL flagged as estimates.
    # Dirichlet smoothing REQUIRED — these are the most fragile CPTs.
    # qualitative prior — weak evidence, update first with TRUPCR data
    #
    # State effects on P_Obstructive (from sprint spec):
    #   monophonic_low:   Moderate 0.55 — also suppresses P(bacterial)
    #   monophonic_mid:   Moderate 0.50
    #   monophonic_high:  Mild 0.45
    #   polyphonic_low:   Strong 0.72  — classic COPD
    #   polyphonic_mid:   Strong 0.70  — asthma small airway
    #   polyphonic_pan:   Very strong 0.78 — severe pan-airway
    #   uncertain:        Marginalize
    # -----------------------------------------------------------------------
    wheeze_compound_raw = np.array([
        # (obstructive=low, obstructive=high)
        # Arvind review: polyphonic_pan must be highest when obstruct=high (clinical reality)
        # monophonic_low must be lower than polyphonic_pan when obstruct=high (fixed vs dynamic obstruction)
        [0.08, 0.12],   # monophonic_low:  focal fixed obstruction — moderate, below polyphonic
        [0.07, 0.10],   # monophonic_mid:  medium airway — moderate
        [0.06, 0.08],   # monophonic_high: small airway — mild obstructive
        [0.12, 0.22],   # polyphonic_low:  classic COPD pattern — strong
        [0.10, 0.18],   # polyphonic_mid:  asthma — strong
        [0.07, 0.30],   # polyphonic_pan:  severe pan-airway — dominant when obstruct=high
        [0.50, 0.00],   # uncertain:       high when no obstruction, near-zero when obstructive
    ])
    wheeze_compound_cpt = smooth_and_verify(wheeze_compound_raw, "Wheeze_Compound")
    cpts["Wheeze_Compound"] = {
        "states": ["monophonic_low", "monophonic_mid", "monophonic_high",
                   "polyphonic_low", "polyphonic_mid", "polyphonic_pan", "uncertain"],
        "parents": ["P_Obstructive_Exacerbation"],
        "parent_states": [["low", "high"]],
        "cpt": wheeze_compound_cpt,
        "note": "qualitative prior — weak evidence, update first with TRUPCR data. "
                "NO compound wheeze LR data exists in any clinical study."
    }

    # -----------------------------------------------------------------------
    # RHONCHI_PRESENT
    # States: False (0) / True (1)
    # Parents: P_Consolidation (low/high)
    #
    # No published LR for rhonchi specifically in LRTI.
    # q_i = 0.30 — estimated, clinical consensus (secretions in airways)
    # Flagged as estimate.
    # -----------------------------------------------------------------------
    rhonchi_raw = np.array([
        [0.85, 0.60],   # absent: more common without consolidation
        [0.15, 0.40],   # present: increases with consolidation — q_i=0.30 estimated
    ])
    verify_cpt_rows_sum_to_one(rhonchi_raw, "Rhonchi_Present")
    cpts["Rhonchi_Present"] = {
        "states": ["absent", "present"],
        "parents": ["P_Consolidation"],
        "parent_states": [["low", "high"]],
        "cpt": rhonchi_raw,
        "note": "q_i=0.30 estimated — no published LR for rhonchi in LRTI"
    }

    # -----------------------------------------------------------------------
    # DIMINISHED_SOUNDS
    # States: absent / unilateral / bilateral
    # Parents: P_Consolidation (low/high), P_Severity (low/high)
    #
    # LR for decreased breath sounds:
    #   LR+ = 2.3–2.5 (Heckerling Ann Intern Med 1990)
    #   q_i = 2.4 / (1 + 2.4) = 0.71
    # -----------------------------------------------------------------------
    # (3 states) × (2 × 2 parent combos) = (3, 4)
    # Cols: (consol=low,sev=low), (consol=high,sev=low),
    #       (consol=low,sev=high), (consol=high,sev=high)
    diminished_raw = np.array([
        [0.85, 0.45, 0.70, 0.25],   # absent
        [0.12, 0.45, 0.22, 0.55],   # unilateral — q_i~0.71 (Heckerling 1990)
        [0.03, 0.10, 0.08, 0.20],   # bilateral — severe disease
    ])
    verify_cpt_rows_sum_to_one(diminished_raw, "Diminished_Sounds")
    cpts["Diminished_Sounds"] = {
        "states": ["absent", "unilateral", "bilateral"],
        "parents": ["P_Consolidation", "P_Severity"],
        "parent_states": [["low", "high"], ["low", "high"]],
        "cpt": diminished_raw,
        "note": "LR+ 2.3–2.5 (Heckerling 1990) → q_i=0.71"
    }

    # -----------------------------------------------------------------------
    # BILATERAL_ASYMMETRY_SCORE
    # States: none / mild / moderate / high
    # Parents: P_Consolidation (low/high)
    #
    # LR+ up to 44.1 for asymmetric chest signs (McGee/Diehr)
    # BUT: high specificity, very low sensitivity — use conservatively.
    # q_i for high asymmetry → consolidation: 0.85 (conservative given low sensitivity)
    # -----------------------------------------------------------------------
    asymmetry_raw = np.array([
        [0.55, 0.20],   # none: common without consolidation
        [0.25, 0.30],   # mild
        [0.12, 0.28],   # moderate
        [0.08, 0.22],   # high: LR+ up to 44.1 — very specific (McGee/Diehr)
    ])
    verify_cpt_rows_sum_to_one(asymmetry_raw, "Bilateral_Asymmetry_Score")
    cpts["Bilateral_Asymmetry_Score"] = {
        "states": ["none", "mild", "moderate", "high"],
        "parents": ["P_Consolidation"],
        "parent_states": [["low", "high"]],
        "cpt": asymmetry_raw,
        "note": "LR+ up to 44.1 (McGee/Diehr) — high specificity, low sensitivity. Used conservatively."
    }

    # -----------------------------------------------------------------------
    # CRANIOCAUDAL_GRADIENT
    # States: flat / apical / basal
    # Parents: P_ILD_Flag (low/high), P_Consolidation (low/high)
    #
    # Basal predominance → ILD: q_i = 0.65 estimated (clinical pattern recognition)
    # Apical predominance → TB/atypical: handled in consolidation pathway
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    # (3) × (2 × 2) = (3, 4)
    gradient_raw = np.array([
        [0.60, 0.40, 0.50, 0.30],   # flat: most common baseline
        [0.20, 0.30, 0.15, 0.15],   # apical: TB/atypical — increases with consolidation
        [0.20, 0.30, 0.35, 0.55],   # basal: ILD pattern — increases with ILD flag
    ])
    craniocaudal_cpt = smooth_and_verify(gradient_raw, "Craniocaudal_Gradient")
    cpts["Craniocaudal_Gradient"] = {
        "states": ["flat", "apical", "basal"],
        "parents": ["P_ILD_Flag", "P_Consolidation"],
        "parent_states": [["low", "high"], ["low", "high"]],
        "cpt": craniocaudal_cpt,
        "note": "qualitative prior — weak evidence, update first with TRUPCR data"
    }

    # -----------------------------------------------------------------------
    # ZONE_RELIABILITY
    # States: normal / reduced
    # Parents: none (set by input mapping from Layer 2 dual-zone logic)
    #
    # This is effectively an observation — set by map_layer2_to_bn().
    # CPT here is a prior for when Layer 2 doesn't specify.
    # dual_zone_only → reduced (computed in input mapper)
    # -----------------------------------------------------------------------
    zone_reliability_raw = np.array([
        [0.85],   # normal: most recordings
        [0.15],   # reduced: dual-zone-only findings
    ])
    verify_cpt_rows_sum_to_one(zone_reliability_raw, "Zone_Reliability")
    cpts["Zone_Reliability"] = {
        "states": ["normal", "reduced"],
        "parents": [],
        "parent_states": [],
        "cpt": zone_reliability_raw,
        "note": "Set by input mapper (dual_zone_only logic). Prior rarely used directly."
    }

    return cpts


# ---------------------------------------------------------------------------
# SANITY CHECKS — Day 2 required checks from sprint spec
# ---------------------------------------------------------------------------

def run_sanity_checks(cpts: dict):
    """
    4 required sanity checks from the sprint spec (Day 2).
    These check CPT structure and direction of influence — NOT full BN inference.
    Full inference sanity checks come Day 3 after intermediate nodes are built.
    """
    print("\n" + "=" * 55)
    print("Acoustic CPT Sanity Checks")
    print("=" * 55)

    all_passed = True

    # Check 1: Velcro → P_ILD_Flag high
    # In Crackle_Subtype CPT: when ILD_Flag=high, velcro row should be highest
    subtype_cpt = cpts["Crackle_Subtype"]["cpt"]
    # Col 2 = (consol=low, ILD=high)
    velcro_idx = 2  # velcro is index 2
    velcro_given_ild_high = subtype_cpt[velcro_idx, 2]
    velcro_given_ild_low  = subtype_cpt[velcro_idx, 0]
    check1 = velcro_given_ild_high > velcro_given_ild_low
    status = "✓" if check1 else "✗ FAIL"
    print(f"\nCheck 1 — Velcro crackles ↑ when P_ILD_Flag=high: {status}")
    print(f"  P(velcro|ILD=low)  = {velcro_given_ild_low:.3f}")
    print(f"  P(velcro|ILD=high) = {velcro_given_ild_high:.3f}")
    if not check1: all_passed = False

    # Check 2: monophonic_low wheeze → P_Obstructive moderate, NOT dominant
    # In Wheeze_Compound CPT: monophonic_low should be LOWER than polyphonic_pan
    # when obstructive=high
    wc_cpt = cpts["Wheeze_Compound"]["cpt"]
    mono_low_idx  = 0  # monophonic_low
    poly_pan_idx  = 5  # polyphonic_pan
    # Col 1 = obstructive=high
    mono_low_given_obstruct = wc_cpt[mono_low_idx, 1]
    poly_pan_given_obstruct  = wc_cpt[poly_pan_idx, 1]
    check2 = mono_low_given_obstruct < poly_pan_given_obstruct * 2.5
    status = "✓" if check2 else "✗ FAIL"
    print(f"\nCheck 2 — monophonic_low moderate (not dominant) vs polyphonic_pan: {status}")
    print(f"  P(monophonic_low|obstruct=high) = {mono_low_given_obstruct:.3f}")
    print(f"  P(polyphonic_pan|obstruct=high) = {poly_pan_given_obstruct:.3f}")
    if not check2: all_passed = False

    # Check 3: polyphonic_pan → P_Obstructive strong increase
    # P(polyphonic_pan|obstruct=high) > P(polyphonic_pan|obstruct=low)
    poly_pan_low  = wc_cpt[poly_pan_idx, 0]
    poly_pan_high = wc_cpt[poly_pan_idx, 1]
    check3 = poly_pan_high > poly_pan_low
    status = "✓" if check3 else "✗ FAIL"
    print(f"\nCheck 3 — polyphonic_pan ↑ when P_Obstructive=high: {status}")
    print(f"  P(polyphonic_pan|obstruct=low)  = {poly_pan_low:.3f}")
    print(f"  P(polyphonic_pan|obstruct=high) = {poly_pan_high:.3f}")
    if not check3: all_passed = False

    # Check 4: uncertain Wheeze_Compound → high when no obstruction (marginalization proxy)
    # P(uncertain|obstruct=low) should be highest among all states
    uncertain_idx = 6
    uncertain_low = wc_cpt[uncertain_idx, 0]
    max_non_uncertain = max(wc_cpt[i, 0] for i in range(6))
    check4 = uncertain_low > max_non_uncertain
    status = "✓" if check4 else "✗ FAIL"
    print(f"\nCheck 4 — uncertain state dominant when no obstruction: {status}")
    print(f"  P(uncertain|obstruct=low)   = {uncertain_low:.3f}")
    print(f"  Max other state|obstruct=low = {max_non_uncertain:.3f}")
    if not check4: all_passed = False

    # Check 5: All CPTs sum to 1.0 per column
    print(f"\nCheck 5 — All CPT columns sum to 1.0:")
    for name, data in cpts.items():
        col_sums = data["cpt"].sum(axis=0)
        ok = np.allclose(col_sums, 1.0, atol=1e-6)
        status = "✓" if ok else "✗ FAIL"
        print(f"  {name:35s}: {status}")
        if not ok: all_passed = False

    print()
    if all_passed:
        print("=" * 55)
        print("All acoustic CPT sanity checks PASSED ✓")
        print("=" * 55)
    else:
        print("=" * 55)
        print("SOME CHECKS FAILED — review CPT values above")
        print("=" * 55)

    return all_passed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Building acoustic CPTs...")
    cpts = build_acoustic_cpts()

    print(f"\nBuilt CPTs for {len(cpts)} acoustic nodes:")
    for name, data in cpts.items():
        parents = data["parents"] if data["parents"] else ["none (root)"]
        print(f"  {name:35s} | states: {len(data['states'])} | parents: {parents}")

    passed = run_sanity_checks(cpts)

    print("\nNOTE: 'qualitative prior' CPTs are Crackle_Phase, Wheeze_Phase,")
    print("Wheeze_Compound, Craniocaudal_Gradient — Dirichlet smoothed.")
    print("These are the FIRST to update when TRUPCR data arrives.")

    if not passed:
        exit(1)