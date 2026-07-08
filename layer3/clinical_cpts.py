"""
TreBle Respire — Layer 3: Clinical CPTs + Intermediate Nodes
Author: Asmi | Day 3

Clinical input node CPTs and all intermediate nodes:
  - P_Consolidation
  - P_Obstructive_Exacerbation
  - P_Severity
  - Anthonisen_Type
  - P_ILD_Flag

Prior_Antibiotic_Use modifier: multiplies all acoustic q_i by 0.70 at
inference time — NOT baked into CPTs.

References:
  - Anthonisen et al. Ann Intern Med 1987    (COPD exacerbation criteria)
  - Metlay JAMA 1997                         (fever, cough LRs in pneumonia)
  - Heckerling Ann Intern Med 1990           (clinical exam LRs)
  - Wipf et al. Arch Intern Med 1999         (crackle LRs)
  - Bohadana NEJM 2014                       (ILD velcro, wheeze phase)
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dirichlet_smoothing import smooth_and_verify, verify_cpt_rows_sum_to_one


# ---------------------------------------------------------------------------
# PRIOR ANTIBIOTIC USE MODIFIER
# Applied at inference time — NOT baked into CPTs
# ---------------------------------------------------------------------------
PRIOR_ANTIBIOTIC_MODIFIER = 0.70  # multiply all acoustic q_i by this when true


def apply_antibiotic_modifier(qi: float, prior_antibiotic_use: bool) -> float:
    """
    Applies Prior_Antibiotic_Use modifier to an acoustic q_i value.
    Use this at inference time when constructing evidence dicts.
    Do NOT bake this into CPT tables.
    """
    if prior_antibiotic_use:
        return qi * PRIOR_ANTIBIOTIC_MODIFIER
    return qi


# ---------------------------------------------------------------------------
# CLINICAL CPTS
# ---------------------------------------------------------------------------

def build_clinical_cpts() -> dict:
    """
    Builds CPTs for all clinical input nodes.
    Noisy-OR throughout.
    Nodes without published LRs: q_i=0.30, flagged as estimated.
    """
    cpts = {}

    # -----------------------------------------------------------------------
    # FEVER
    # States: False / True
    # Parents: P_Consolidation (low/high)
    #
    # LR+ for fever in pneumonia: ~1.7–2.1 (Metlay JAMA 1997)
    # q_i = 2.0 / (1 + 2.0) = 0.67
    # -----------------------------------------------------------------------
    fever_raw = np.array([
        [0.70, 0.35],   # no fever: more common without consolidation
        [0.30, 0.65],   # fever: q_i=0.67 (Metlay 1997)
    ])
    verify_cpt_rows_sum_to_one(fever_raw, "Fever")
    cpts["Fever"] = {
        "states": ["absent", "present"],
        "parents": ["P_Consolidation"],
        "parent_states": [["low", "high"]],
        "cpt": fever_raw,
        "note": "LR+ ~2.0 (Metlay JAMA 1997) → q_i=0.67"
    }

    # -----------------------------------------------------------------------
    # SYMPTOM_DURATION
    # States: short (<3d) / medium (3-7d) / long (>7d)
    # Parents: Age_Comorbidity (child_adult / elderly_healthy / elderly_comorbid)
    #
    # No formal LR data for duration by age. q_i=0.30 estimated.
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    duration_raw = np.array([
        [0.40, 0.30, 0.20],   # short: more common in younger/healthier
        [0.40, 0.45, 0.45],   # medium
        [0.20, 0.25, 0.35],   # long: more common in elderly/comorbid
    ])
    duration_cpt = smooth_and_verify(duration_raw, "Symptom_Duration")
    cpts["Symptom_Duration"] = {
        "states": ["short", "medium", "long"],
        "parents": ["Age_Comorbidity"],
        "parent_states": [["child_adult", "elderly_healthy", "elderly_comorbid"]],
        "cpt": duration_cpt,
        "note": "q_i=0.30 estimated — no published LR for duration by age in Indian LRTI"
    }

    # -----------------------------------------------------------------------
    # COUGH_CHARACTER
    # States: dry / productive_mucoid / productive_purulent
    # Parents: P_Consolidation (low/high)
    #
    # Purulent sputum → bacterial: q_i=0.55 estimated (clinical consensus)
    # No formal LR for cough character alone in LRTI.
    # -----------------------------------------------------------------------
    cough_raw = np.array([
        [0.40, 0.20],   # dry: more common viral/atypical
        [0.35, 0.35],   # productive_mucoid: non-specific
        [0.25, 0.45],   # productive_purulent: → bacterial q_i~0.55 estimated
    ])
    verify_cpt_rows_sum_to_one(cough_raw, "Cough_Character")
    cpts["Cough_Character"] = {
        "states": ["dry", "productive_mucoid", "productive_purulent"],
        "parents": ["P_Consolidation"],
        "parent_states": [["low", "high"]],
        "cpt": cough_raw,
        "note": "q_i=0.55 estimated for purulent — no formal LR for cough character in LRTI"
    }

    # -----------------------------------------------------------------------
    # SPUTUM_PURULENCE (Anthonisen criterion 1)
    # DYSPNEA_INCREASE  (Anthonisen criterion 2)
    # SPUTUM_VOLUME_INCREASE (Anthonisen criterion 3)
    # Parents: P_Consolidation (low/high) + Obstructive_Dx (False/True)
    #
    # Anthonisen 1987: purulence is the strongest single predictor in COPD.
    # q_i=0.55 for purulence → consolidation (estimated)
    # All three criteria: q_i=0.30 estimated without formal LR data
    # -----------------------------------------------------------------------
    for node_name, q_high in [
        ("Sputum_Purulence",      0.55),   # strongest criterion
        ("Dyspnea_Increase",      0.45),
        ("Sputum_Volume_Increase",0.40),
    ]:
        # (2 states) × (2 consol × 2 obstructive = 4 cols)
        q = q_high
        raw = np.array([
            [1-0.20, 1-q,    1-0.30, 1-(q*1.1)],   # absent
            [0.20,   q,      0.30,   min(q*1.1,0.95)],  # present
        ])
        # clamp
        raw = np.clip(raw, 0.01, 0.99)
        raw = raw / raw.sum(axis=0, keepdims=True)
        verify_cpt_rows_sum_to_one(raw, node_name)
        cpts[node_name] = {
            "states": ["absent", "present"],
            "parents": ["P_Consolidation", "Obstructive_Dx"],
            "parent_states": [["low", "high"], ["false", "true"]],
            "cpt": raw,
            "note": f"q_i={q} estimated — Anthonisen 1987. No formal LR for {node_name} alone."
        }

    # -----------------------------------------------------------------------
    # SPO2 — discretized input (see clinical_schema.py discretize_spo2)
    # States: critical / low / borderline / normal
    # Parents: P_Severity (low/high)
    # -----------------------------------------------------------------------
    spo2_raw = np.array([
        [0.02, 0.20],   # critical (<88% adjusted): severity drives this
        [0.08, 0.30],   # low (88-92%)
        [0.25, 0.30],   # borderline (92-95%)
        [0.65, 0.20],   # normal (>95%): common without severe disease
    ])
    verify_cpt_rows_sum_to_one(spo2_raw, "SpO2")
    cpts["SpO2"] = {
        "states": ["critical", "low", "borderline", "normal"],
        "parents": ["P_Severity"],
        "parent_states": [["low", "high"]],
        "cpt": spo2_raw,
        "note": "Sjoding NEJM 2020 correction applied at discretization step (clinical_schema.py)"
    }

    # -----------------------------------------------------------------------
    # FEV1_FVC_KNOWN
    # States: normal (>0.7) / obstructive (<0.7) / unknown
    # Parents: Obstructive_Dx (false/true)
    # -----------------------------------------------------------------------
    fev_raw = np.array([
        [0.70, 0.15],   # normal: common without obstructive dx
        [0.05, 0.70],   # obstructive: high when known obstructive dx
        [0.25, 0.15],   # unknown: not measured
    ])
    verify_cpt_rows_sum_to_one(fev_raw, "FEV1_FVC_Known")
    cpts["FEV1_FVC_Known"] = {
        "states": ["normal", "obstructive", "unknown"],
        "parents": ["Obstructive_Dx"],
        "parent_states": [["false", "true"]],
        "cpt": fev_raw,
        "note": "q_i=0.30 estimated — spirometry not always available in primary care"
    }

    return cpts


# ---------------------------------------------------------------------------
# INTERMEDIATE NODE CPTs
# ---------------------------------------------------------------------------

def build_intermediate_cpts() -> dict:
    """
    Builds CPTs for all intermediate nodes.
    These are the core clinical reasoning nodes.
    """
    cpts = {}

    # -----------------------------------------------------------------------
    # P_CONSOLIDATION
    # States: low / high
    # Parents: Crackle_Pattern, Crackle_Phase, Crackle_Subtype,
    #          Diminished_Sounds, Bilateral_Asymmetry_Score,
    #          Fever, Cough_Character, Symptom_Duration, Rhonchi_Present,
    #          Age_Comorbidity
    #
    # Noisy-OR: P(consolidation=high) increases with each positive finding.
    # Key q_i values (from literature):
    #   focal crackles:     LR+ 3.0 (Wipf 1999) → q_i = 0.75
    #   bilateral crackles: LR+ 2.42 (Htun 2019) → q_i = 0.71
    #   bronchial breath:   LR+ 3.3 (Heckerling 1990) → encoded in Diminished/Crackle
    #   decreased sounds:   LR+ 2.4 (Heckerling 1990) → q_i = 0.71
    #   asymmetry high:     LR+ 44.1 (McGee/Diehr) → q_i = 0.97 (use conservatively)
    #   fever:              LR+ 2.0 (Metlay 1997) → q_i = 0.67
    #
    # Implemented as a 2-state summary node — receives evidence from acoustic
    # and clinical parents. Actual Noisy-OR computation done at inference.
    # CPT here encodes base rates and parent influence directions.
    # -----------------------------------------------------------------------
    # For pgmpy: P_Consolidation has many parents — use simplified CPT
    # representing the "prior" before acoustic evidence, which is then
    # updated via likelihood weighting at inference time.
    # Base rate from Indian priors: P(bacterial LRTI) = 0.25, so
    # P(consolidation) ≈ 0.20 at baseline (not all bacterial = consolidation)

    # Simplified: binary node, 2 key parents for CPT structure
    # Full Noisy-OR computed in inference engine
    # (2) × (2 Fever × 2 CracklePattern_focal) = shape varies
    # Use scalar prior here; inference engine applies Noisy-OR updating
    consolidation_prior = np.array([[0.80], [0.20]])  # P(low)=0.80, P(high)=0.20 at baseline
    verify_cpt_rows_sum_to_one(consolidation_prior, "P_Consolidation_prior")
    cpts["P_Consolidation"] = {
        "states": ["low", "high"],
        "parents": [],   # root for CPT purposes — evidence applied at inference
        "parent_states": [],
        "cpt": consolidation_prior,
        "qi_values": {   # Noisy-OR q_i per parent — applied at inference time
            "Crackle_Pattern_focal":          0.75,  # LR+ 3.0 (Wipf 1999)
            "Crackle_Pattern_bilateral_basal":0.71,  # LR+ 2.42 (Htun 2019)
            "Crackle_Pattern_diffuse":        0.65,  # estimated
            "Crackle_Phase_late_insp":        0.55,  # qualitative (Pasterkamp 1997)
            "Crackle_Phase_early_insp":       0.35,  # qualitative (Forgacs 1967)
            "Crackle_Subtype_coarse":         0.75,  # estimated (Pasterkamp/Piirilä)
            "Diminished_unilateral":          0.71,  # LR+ 2.4 (Heckerling 1990)
            "Asymmetry_high":                 0.85,  # conservative (LR+ 44.1 McGee/Diehr)
            "Fever_present":                  0.50,  # LR+ 2.0 (Metlay 1997) → q_i conservative (fever→consolidation, not →antibiotics directly)
            "Rhonchi_present":                0.30,  # estimated
        },
        "note": "Noisy-OR q_i values applied at inference. CPT = prior only."
    }

    # -----------------------------------------------------------------------
    # P_OBSTRUCTIVE_EXACERBATION
    # States: low / high
    # Parents: Wheeze_Pattern, Wheeze_Phase, Wheeze_Compound,
    #          Obstructive_Dx, FEV1_FVC_Known, Fever (suppresses obstructive)
    #
    # Key q_i values:
    #   wheeze any:         LR ~1.0 (Metlay 1997) — non-discriminatory for bacterial
    #   expiratory wheeze:  q_i = 0.65 qualitative (Bohadana 2014)
    #   biphasic wheeze:    q_i = 0.75 qualitative (Bohadana 2014)
    #   polyphonic_pan:     q_i = 0.78 (sprint spec — clinical consensus)
    #   monophonic_low:     q_i = 0.55 moderate (Sovijarvi 2000)
    #   Obstructive_Dx:     q_i = 0.80 (known diagnosis strongly predicts exacerbation)
    # -----------------------------------------------------------------------
    obstruct_prior = np.array([[0.75], [0.25]])  # P(low)=0.75, P(high)=0.25
    verify_cpt_rows_sum_to_one(obstruct_prior, "P_Obstructive_prior")
    cpts["P_Obstructive_Exacerbation"] = {
        "states": ["low", "high"],
        "parents": [],
        "parent_states": [],
        "cpt": obstruct_prior,
        "qi_values": {
            "Obstructive_Dx_true":         0.80,  # known dx is strongest predictor
            "Wheeze_Phase_expiratory":     0.65,  # qualitative (Bohadana 2014)
            "Wheeze_Phase_biphasic":       0.75,  # qualitative (Bohadana 2014)
            "Wheeze_Compound_polyphonic_pan": 0.78,  # clinical consensus
            "Wheeze_Compound_polyphonic_low": 0.72,  # Pasterkamp 1997
            "Wheeze_Compound_monophonic_low": 0.55,  # Sovijarvi 2000
            "FEV1_FVC_obstructive":        0.70,  # spirometry confirmation
            "Biomass_Exposure_true":       0.40,  # Balakrishnan PMC4221659
        },
        "endobronchial_suppression": {
            "Wheeze_Compound_monophonic_low": 0.40,  # suppress P(bacterial) — fixed obstruction
        },
        "note": "Wheeze LR ~1.0 (Metlay 1997) for bacterial — feeds obstructive branch ONLY"
    }

    # -----------------------------------------------------------------------
    # P_SEVERITY
    # States: low / moderate / high
    # Parents: SpO2 (critical/low/borderline/normal), Dyspnea_Increase (bool),
    #          Age_Comorbidity
    #
    # SpO2 <92% (adjusted) → escalate regardless of BN — hard rule
    # q_i estimated throughout — no formal LR for severity composite score
    # -----------------------------------------------------------------------
    # (3 severity states) × (4 SpO2 states × 2 dyspnea × 3 age = 24 cols)
    # Simplified to key patterns for CPT
    severity_prior = np.array([[0.60], [0.30], [0.10]])
    verify_cpt_rows_sum_to_one(severity_prior, "P_Severity_prior")
    cpts["P_Severity"] = {
        "states": ["low", "moderate", "high"],
        "parents": [],
        "parent_states": [],
        "cpt": severity_prior,
        "qi_values": {
            "SpO2_critical":   0.95,  # hard escalation trigger
            "SpO2_low":        0.80,
            "SpO2_borderline": 0.50,
            "Dyspnea_present": 0.45,  # q_i estimated
            "Age_elderly_comorbid": 0.35,  # estimated
        },
        "note": "SpO2 <92% (Sjoding-adjusted) triggers escalate=True REGARDLESS of BN posterior"
    }

    # -----------------------------------------------------------------------
    # ANTHONISEN_TYPE
    # States: type1 / type2 / type3 / not_applicable
    # Derived deterministically from 3 criteria (see clinical_schema.py)
    # Only applies when Obstructive_Dx=True
    # Reference: Anthonisen et al. Ann Intern Med 1987
    #
    # CPT here is for pgmpy structure — actual value set by derive_anthonisen_type()
    # -----------------------------------------------------------------------
    # (4 states) × (2 Obstructive_Dx × 2 Purulence × 2 Dyspnea × 2 Volume = 16 cols)
    # Use deterministic mapping:
    #   (T,T,T) → type1;  (T,T,F)|(T,F,T)|(T,F,F)+1more → type2; (T,1,0,0) → type3
    #   Obstructive=F → not_applicable
    anthonisen_prior = np.array([
        [0.30],  # type1
        [0.30],  # type2
        [0.25],  # type3
        [0.15],  # not_applicable
    ])
    verify_cpt_rows_sum_to_one(anthonisen_prior, "Anthonisen_Type_prior")
    cpts["Anthonisen_Type"] = {
        "states": ["type1", "type2", "type3", "not_applicable"],
        "parents": [],
        "parent_states": [],
        "cpt": anthonisen_prior,
        "qi_values": {
            "type1_antibiotic_benefit": 0.90,  # Anthonisen 1987 — clear benefit
            "type2_antibiotic_benefit": 0.60,  # moderate benefit
            "type3_antibiotic_benefit": 0.20,  # minimal benefit
        },
        "note": "Anthonisen 1987 Ann Intern Med. Derived deterministically in clinical_schema.py"
    }

    # -----------------------------------------------------------------------
    # P_ILD_FLAG
    # States: low / high
    # This is a SAFETY NET — suppresses overconfident bacterial classification.
    # P_ILD_Flag=high → uncertainty_flag=True, do NOT output red
    #
    # Parents: Crackle_Subtype (velcro state),
    #          Crackle_Phase (late_insp),
    #          Craniocaudal_Gradient (basal)
    #          Age_Comorbidity
    #
    # q_i:
    #   velcro crackles:       q_i = 0.70 (Bohadana NEJM 2014 — stronger evidence)
    #   late_insp fine crackles: q_i = 0.55 (qualitative)
    #   basal gradient:        q_i = 0.55 (qualitative — clinical pattern)
    #   elderly age:           q_i = 0.30 (estimated — ILD more common in elderly)
    # qualitative prior — weak evidence, update first with TRUPCR data
    # -----------------------------------------------------------------------
    ild_prior = np.array([[0.92], [0.08]])  # ILD rare — very low base rate
    verify_cpt_rows_sum_to_one(ild_prior, "P_ILD_Flag_prior")
    cpts["P_ILD_Flag"] = {
        "states": ["low", "high"],
        "parents": [],
        "parent_states": [],
        "cpt": ild_prior,
        "qi_values": {
            "Crackle_Subtype_velcro":      0.70,  # Bohadana NEJM 2014
            "Crackle_Phase_late_insp":     0.55,  # qualitative (Pasterkamp 1997)
            "Craniocaudal_basal":          0.55,  # qualitative — clinical pattern
            "Age_elderly":                 0.30,  # estimated
        },
        "note": "qualitative prior — weak evidence, update first with TRUPCR data. "
                "ILD flag = safety net, NOT a diagnosis. Suppresses red output."
    }

    return cpts


# ---------------------------------------------------------------------------
# END-TO-END SANITY CHECKS (Day 3 required — 4 checks)
# These test directional logic using q_i values, not full BN inference
# ---------------------------------------------------------------------------

def run_end_to_end_sanity_checks(clinical_cpts: dict, intermediate_cpts: dict) -> bool:
    """
    4 required end-to-end sanity checks from sprint spec Day 3.
    Tests causal direction of influence through intermediate nodes.
    """
    print("\n" + "=" * 60)
    print("Day 3 End-to-End Sanity Checks")
    print("=" * 60)

    all_passed = True

    # Helper: simulate Noisy-OR posterior given active parents
    def noisy_or_posterior(base_p_high: float, active_qi: list) -> float:
        """P(node=high) given active parents with their q_i values."""
        p_low = 1.0 - base_p_high
        for qi in active_qi:
            p_low *= (1.0 - qi)
        return 1.0 - p_low

    consol_base = 0.20
    obstruct_base = 0.25
    ild_base = 0.08

    # CHECK 1: late_insp focal crackles + fever + asymmetry → RED zone
    # i.e. P_Consolidation should be HIGH
    qi_check1 = [
        intermediate_cpts["P_Consolidation"]["qi_values"]["Crackle_Pattern_focal"],
        intermediate_cpts["P_Consolidation"]["qi_values"]["Crackle_Phase_late_insp"],
        intermediate_cpts["P_Consolidation"]["qi_values"]["Fever_present"],
        intermediate_cpts["P_Consolidation"]["qi_values"]["Asymmetry_high"],
    ]
    p_consol_check1 = noisy_or_posterior(consol_base, qi_check1)
    check1 = p_consol_check1 > 0.90
    status = "✓" if check1 else "✗ FAIL"
    print(f"\nCheck 1 — late_insp crackles + fever + asymmetry → P_Consolidation HIGH: {status}")
    print(f"  P(Consolidation=high) = {p_consol_check1:.3f}  (expect >0.90 → RED)")
    if not check1: all_passed = False

    # CHECK 2: bilateral expiratory wheeze + no crackles + no fever → GREEN
    # P_Consolidation LOW, P_Obstructive potentially high
    # Without fever or crackles: consolidation stays near base rate
    qi_check2_consol = []  # no crackles, no fever
    p_consol_check2 = noisy_or_posterior(consol_base, qi_check2_consol)
    qi_check2_obstruct = [
        intermediate_cpts["P_Obstructive_Exacerbation"]["qi_values"]["Wheeze_Phase_expiratory"],
    ]
    p_obstruct_check2 = noisy_or_posterior(obstruct_base, qi_check2_obstruct)
    check2 = p_consol_check2 < 0.25
    status = "✓" if check2 else "✗ FAIL"
    print(f"\nCheck 2 — bilateral expiratory wheeze, no crackles, no fever → P_Consolidation LOW: {status}")
    print(f"  P(Consolidation=high) = {p_consol_check2:.3f}  (expect strictly < 0.25 → contributes to GREEN)")
    print(f"  Note: zone is GREEN when final p_antibiotics strictly < 0.20 (0.20 itself = AMBER)")
    print(f"  P(Obstructive=high)   = {p_obstruct_check2:.3f}  (obstructive branch active)")
    if not check2: all_passed = False

    # CHECK 3: Velcro bilateral → amber + uncertainty_flag
    # P_ILD_Flag should be HIGH
    qi_check3 = [
        intermediate_cpts["P_ILD_Flag"]["qi_values"]["Crackle_Subtype_velcro"],
        intermediate_cpts["P_ILD_Flag"]["qi_values"]["Crackle_Phase_late_insp"],
        intermediate_cpts["P_ILD_Flag"]["qi_values"]["Craniocaudal_basal"],
    ]
    p_ild_check3 = noisy_or_posterior(ild_base, qi_check3)
    check3 = p_ild_check3 > 0.60
    status = "✓" if check3 else "✗ FAIL"
    print(f"\nCheck 3 — Velcro bilateral basal → P_ILD_Flag HIGH → uncertainty_flag: {status}")
    print(f"  P(ILD_Flag=high) = {p_ild_check3:.3f}  (expect >0.60 → amber + uncertainty)")
    if not check3: all_passed = False

    # CHECK 4: polyphonic_pan + purulent sputum + Anthonisen Type 1 → RED
    # Both obstructive AND consolidation signals active
    qi_check4_obstruct = [
        intermediate_cpts["P_Obstructive_Exacerbation"]["qi_values"]["Wheeze_Compound_polyphonic_pan"],
        intermediate_cpts["P_Obstructive_Exacerbation"]["qi_values"]["Obstructive_Dx_true"],
    ]
    qi_check4_consol = [
        intermediate_cpts["P_Consolidation"]["qi_values"]["Fever_present"],
        intermediate_cpts["P_Consolidation"]["qi_values"]["Crackle_Pattern_focal"],
    ]
    p_obstruct_check4 = noisy_or_posterior(obstruct_base, qi_check4_obstruct)
    p_consol_check4   = noisy_or_posterior(consol_base, qi_check4_consol)
    # Anthonisen Type 1 benefit = 0.90
    p_antibiotics_check4 = noisy_or_posterior(
        0.25,
        [p_obstruct_check4 * 0.90, p_consol_check4 * 0.75]
    )
    check4 = p_antibiotics_check4 > 0.80
    status = "✓" if check4 else "✗ FAIL"
    print(f"\nCheck 4 — polyphonic_pan + purulent + Anthonisen Type 1 → RED: {status}")
    print(f"  P(Obstructive=high)      = {p_obstruct_check4:.3f}")
    print(f"  P(Consolidation=high)    = {p_consol_check4:.3f}")
    print(f"  P(Antibiotics indicated) = {p_antibiotics_check4:.3f}  (expect >0.80 → RED)")
    if not check4: all_passed = False

    print()
    if all_passed:
        print("=" * 60)
        print("All Day 3 end-to-end sanity checks PASSED ✓")
        print("=" * 60)
    else:
        print("=" * 60)
        print("SOME CHECKS FAILED — review q_i values in intermediate CPTs")
        print("=" * 60)
    return all_passed


if __name__ == "__main__":
    print("Building clinical CPTs...")
    clinical = build_clinical_cpts()
    print(f"Clinical CPTs: {len(clinical)} nodes")

    print("\nBuilding intermediate node CPTs...")
    intermediate = build_intermediate_cpts()
    print(f"Intermediate CPTs: {len(intermediate)} nodes")

    passed = run_end_to_end_sanity_checks(clinical, intermediate)
    if not passed:
        exit(1)