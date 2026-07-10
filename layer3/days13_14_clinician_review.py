"""
TreBle Respire — Layer 3: Days 13–14
Clinician Review + Phase and Compound CPT Tuning

Author: Asmi | Days 13–14

Arvind coordinates the clinician session.
Present 6 cases. Get explicit answers to three key questions.
Any CPT change → re-apply Dirichlet smoothing → rerun 27 vignettes.
Target: ≥24/27 after tuning.

THREE KEY QUESTIONS for the clinician:
  Q1. monophonic_low wheeze → amber + review_48h — does this make clinical sense?
  Q2. polyphonic_pan + Anthonisen Type 1 → red — does this make clinical sense?
  Q3. Velcro bilateral basal → amber (not red) — is this the correct posture?
"""

import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))

from clinical_cpts import build_intermediate_cpts
from acoustic_cpts import build_acoustic_cpts
from output_layer import compute_output, map_layer2_to_bn
from dirichlet_smoothing import smooth_and_verify
from vignettes import build_vignettes, run_vignettes
from day7_mock_integration import build_mock_layer2_output


# ---------------------------------------------------------------------------
# THE 6 PRESENTATION CASES
# ---------------------------------------------------------------------------

def build_clinician_cases(intermediate_cpts: dict) -> list[dict]:
    """
    6 cases to present to the clinician.
    Each has: clinical picture, BN output, key question to ask.
    """
    def _run(mock_kw: dict, cl_kw: dict) -> dict:
        mock = build_mock_layer2_output(**mock_kw)
        ev   = map_layer2_to_bn(mock)
        base = {
            "age":45,"sex":"M","bmi":23.0,"fever":False,"symptom_duration_days":4,
            "cough_character":"dry","sputum_purulence_change":False,
            "dyspnea_increase":False,"sputum_volume_increase":False,
            "spo2_pct":97.0,"fev1_fvc_known":None,"comorbidity_obstructive":False,
            "biomass_exposure":False,"prior_antibiotic_use":False,"season":"winter",
        }
        base.update(cl_kw)
        out = compute_output(ev, base, intermediate_cpts)
        return out

    cases = []

    # Case CR1 — monophonic_low wheeze focal — KEY QUESTION 1
    out = _run(
        dict(wheeze_compound="monophonic_low", wheeze_phase="inspiratory",
             crackle_pattern="absent", subtype_modules_run=True),
        dict(fever=False, cough_character="dry"),
    )
    cases.append({
        "id": "CR1",
        "title": "Monophonic low wheeze, focal, no crackles, no fever",
        "clinical_picture": "50M, 3 days wheeze, no fever, no cough, SpO2 97%. "
                            "Stethoscope: single-tone low-pitched wheeze in one zone only.",
        "bn_output": out,
        "key_question": "Q1: The model outputs amber + review_48h with endobronchial flag. "
                        "Does it make clinical sense to NOT prescribe antibiotics here "
                        "and instead investigate for fixed endobronchial obstruction?",
        "fragile_cpts": ["Wheeze_Compound monophonic_low q_i=0.55 (qualitative, Sovijarvi 2000)",
                         "Endobronchial suppression strength=0.40 (qualitative)"],
    })

    # Case CR2 — polyphonic_pan + Anthonisen Type 1 — KEY QUESTION 2
    out = _run(
        dict(wheeze_compound="polyphonic_pan", wheeze_phase="biphasic",
             crackle_pattern="focal", crackle_phase="late_inspiratory",
             crackle_subtype="coarse", subtype_modules_run=True),
        dict(fever=True, cough_character="productive_purulent",
             sputum_purulence_change=True, dyspnea_increase=True,
             sputum_volume_increase=True, comorbidity_obstructive=True,
             fev1_fvc_known=0.48, spo2_pct=90.0),
    )
    cases.append({
        "id": "CR2",
        "title": "Polyphonic pan wheeze + Anthonisen Type 1, severe COPD",
        "clinical_picture": "70M COPD (FEV1/FVC 0.48), fever 38.5, pan-airway wheeze, "
                            "coarse crackles R base, purulent sputum, increased dyspnea and "
                            "volume. SpO2 90%. All 3 Anthonisen criteria met.",
        "bn_output": out,
        "key_question": "Q2: The model outputs red + escalate. "
                        "Does Anthonisen Type 1 with polyphonic_pan wheeze and fever "
                        "warrant red (antibiotics clearly indicated + refer)?",
        "fragile_cpts": ["Wheeze_Compound polyphonic_pan q_i=0.78 (clinical consensus)",
                         "Anthonisen Type 1 antibiotic benefit q_i=0.90 (Anthonisen 1987)"],
    })

    # Case CR3 — velcro bilateral basal — KEY QUESTION 3
    out = _run(
        dict(crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
             crackle_subtype="velcro", wheeze_compound="uncertain",
             subtype_modules_run=True),
        dict(fever=False, symptom_duration_days=90, dyspnea_increase=True,
             cough_character="dry", spo2_pct=91.0, season="other"),
    )
    cases.append({
        "id": "CR3",
        "title": "Velcro bilateral basal crackles, no fever, chronic dyspnea",
        "clinical_picture": "65M, 3 months progressive dyspnea, no fever, bilateral "
                            "fine velcro crackles at both bases, SpO2 91%, no wheeze. "
                            "Chronic course, no acute infective symptoms.",
        "bn_output": out,
        "key_question": "Q3: The model outputs amber + uncertainty_flag + escalate. "
                        "ILD safety net is suppressing red. Is amber (do not prescribe "
                        "antibiotics, escalate for investigation) the correct posture "
                        "for this presentation?",
        "fragile_cpts": ["Crackle_Subtype velcro → P_ILD_Flag q_i=0.70 (Bohadana 2014)",
                         "P_ILD_Flag threshold 0.40 (clinical judgement)"],
    })

    # Case CR4 — phase-specific: late_insp focal crackles, fever — classic pneumonia
    out = _run(
        dict(crackle_pattern="focal", crackle_phase="late_inspiratory",
             crackle_subtype="coarse", subtype_modules_run=True),
        dict(fever=True, sputum_purulence_change=True,
             cough_character="productive_purulent"),
    )
    cases.append({
        "id": "CR4",
        "title": "Late inspiratory focal coarse crackles, fever, high asymmetry",
        "clinical_picture": "45M, 4 days fever 38.9, purulent sputum, focal coarse crackles "
                            "R lower, late inspiratory phase, high R-L asymmetry. Classic "
                            "presentation of right lower lobe pneumonia.",
        "bn_output": out,
        "key_question": "Is red the right output here? Does late inspiratory phase "
                        "(q_i=0.55, qualitative) add meaningful information vs just "
                        "having crackles present?",
        "fragile_cpts": ["Crackle_Phase late_insp q_i=0.55 (qualitative, Pasterkamp 1997)"],
    })

    # Case CR5 — early_insp coarse crackles + fever — less certain
    out = _run(
        dict(crackle_pattern="focal", crackle_phase="early_inspiratory",
             crackle_subtype="coarse", subtype_modules_run=True),
        dict(fever=True, sputum_purulence_change=False,
             cough_character="productive_purulent"),
    )
    cases.append({
        "id": "CR5",
        "title": "Early inspiratory focal coarse crackles, fever, no purulence",
        "clinical_picture": "45M, 4 days fever, cough, no purulence. Focal coarse crackles "
                            "R upper, EARLY inspiratory phase. Apical crackles — could be TB, "
                            "atypical, or early consolidation.",
        "bn_output": out,
        "key_question": "Is amber correct here? Early_insp phase gets lower q_i (0.35) vs "
                        "late_insp (0.55) — does this clinically distinguish a less certain "
                        "bacterial picture?",
        "fragile_cpts": ["Crackle_Phase early_insp q_i=0.35 (qualitative, Forgacs 1967)"],
    })

    # Case CR6 — conflicting signals — amber + uncertainty
    out = _run(
        dict(crackle_pattern="focal", crackle_phase="late_inspiratory",
             crackle_subtype="coarse", wheeze_compound="polyphonic_low",
             wheeze_phase="expiratory", subtype_modules_run=True),
        dict(fever=True, comorbidity_obstructive=True,
             cough_character="productive_purulent"),
    )
    cases.append({
        "id": "CR6",
        "title": "Conflicting: late_insp crackles + expiratory polyphonic wheeze, COPD",
        "clinical_picture": "60M COPD, fever, purulent cough. Both focal late_insp crackles "
                            "AND expiratory polyphonic wheeze present simultaneously. "
                            "Consolidation AND obstructive signals both active.",
        "bn_output": out,
        "key_question": "Is amber + uncertainty_flag correct when consolidation and "
                        "obstructive signals conflict? Or should this be red given the "
                        "multiple positive findings?",
        "fragile_cpts": ["Consol_gate when both pathways active",
                         "Wheeze_Phase expiratory q_i=0.65 (qualitative, Bohadana 2014)"],
    })

    return cases


def present_clinician_cases(cases: list[dict]):
    """Prints all 6 cases in a readable format for the clinician session."""
    print("\n" + "=" * 65)
    print("Days 13–14 — Clinician Review: 6 Cases")
    print("=" * 65)
    print("Arvind: please ask the clinician each key question.")
    print("Record answers in the tuning log below.\n")

    for case in cases:
        out = case["bn_output"]
        print(f"\n{'─'*65}")
        print(f"  {case['id']}: {case['title']}")
        print(f"{'─'*65}")
        print(f"  Clinical picture: {case['clinical_picture']}")
        print(f"\n  BN output:")
        print(f"    p_antibiotics    = {out['p_antibiotics']:.3f}")
        print(f"    zone             = {out['zone'].upper()}")
        print(f"    uncertainty_flag = {out['uncertainty_flag']}")
        print(f"    escalate         = {out['escalate']}")
        print(f"    disposition      = {out['disposition']}")
        print(f"    endobronchial    = {out.get('_endobronchial_flag', False)}")
        print(f"    p_consolidation  = {out.get('_p_consolidation', 'N/A')}")
        print(f"    p_obstructive    = {out.get('_p_obstructive', 'N/A')}")
        print(f"    p_ILD_flag       = {out.get('_p_ild', 'N/A')}")
        print(f"\n  Key question: {case['key_question']}")
        print(f"\n  Fragile CPTs in this case:")
        for cpt in case["fragile_cpts"]:
            print(f"    • {cpt}")


# ---------------------------------------------------------------------------
# CPT TUNING — apply after clinician feedback
# Dirichlet smoothing re-applied after every change
# ---------------------------------------------------------------------------

def apply_cpt_tuning(
    tuning_notes: dict,
    intermediate_cpts: dict,
) -> dict:
    """
    Apply clinician-guided tuning to q_i values.
    Dirichlet smoothing is re-applied after every change.

    Args:
        tuning_notes: dict mapping node+context to new q_i value and justification.
            Example:
            {
                "Wheeze_Compound_monophonic_low": {
                    "old_qi": 0.55,
                    "new_qi": 0.45,
                    "justification": "Clinician: monophonic_low rarely indicates severe obstruction"
                }
            }
        intermediate_cpts: current intermediate_cpts dict

    Returns:
        Updated intermediate_cpts with new q_i values.
        Original dict is not mutated.
    """
    import copy
    updated = copy.deepcopy(intermediate_cpts)

    print("\n" + "=" * 65)
    print("CPT Tuning — Applying Clinician Feedback")
    print("=" * 65)

    for key, change in tuning_notes.items():
        old_qi = change["old_qi"]
        new_qi = change["new_qi"]
        just   = change["justification"]
        print(f"\n  {key}")
        print(f"    Old q_i: {old_qi} → New q_i: {new_qi}")
        print(f"    Justification: {just}")
        print(f"    Note: Dirichlet smoothing will be re-applied after this change")

        # Apply to intermediate_cpts qi_values
        for node_name, node_data in updated.items():
            qi_vals = node_data.get("qi_values", {})
            for qi_key in list(qi_vals.keys()):
                if key.lower() in qi_key.lower():
                    qi_vals[qi_key] = new_qi
                    print(f"    Applied to: {node_name}.qi_values['{qi_key}']")

    print("\n  ✓ Tuning applied. Run verify_tuning_passes_vignettes() to confirm ≥24/27.")
    return updated


def verify_tuning_passes_vignettes(intermediate_cpts: dict) -> bool:
    """
    After any CPT tuning, verify ≥24/27 vignettes still pass.
    Run this before committing any change.
    """
    print("\n" + "=" * 65)
    print("Post-Tuning Vignette Check")
    print("=" * 65)

    results = run_vignettes(build_vignettes(), intermediate_cpts)
    passed  = results["passed"] >= 24

    print(f"\n  Result: {results['passed']}/27 "
          f"({'✓ TARGET MET — tuning accepted' if passed else '✗ BELOW TARGET — revert or adjust'})")

    if not passed:
        print("\n  TUNING REJECTED — target not met.")
        print("  Options:")
        print("    1. Revert the q_i change")
        print("    2. Adjust the q_i to a less extreme value")
        print("    3. If CPT diagnostic protocol cannot isolate within 2 hours — escalate to Arvind")

    return passed


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cpts = build_intermediate_cpts()

    print("=" * 65)
    print("Days 13–14 — Clinician Review")
    print("=" * 65)

    cases = build_clinician_cases(cpts)
    present_clinician_cases(cases)

    print("\n" + "=" * 65)
    print("THREE KEY QUESTIONS — Record Clinician Answers")
    print("=" * 65)
    print("""
  Q1. monophonic_low wheeze → amber + review_48h (endobronchial flag)
      Does NOT prescribing antibiotics make clinical sense here?
      Clinician answer: ______________________________________

  Q2. polyphonic_pan + Anthonisen Type 1 → red + escalate
      Is this the correct response for infected severe COPD?
      Clinician answer: ______________________________________

  Q3. Velcro bilateral basal → amber (not red), ILD flag active
      Is amber the right posture — do not prescribe, investigate?
      Clinician answer: ______________________________________

  CONSOL_GATE REVIEW (show this table):
    purulence + fever        : gate = 0.80
    purulence alone          : gate = 0.65
    fever + late_insp + asym : gate = 0.65
    fever + early_insp       : gate = 0.38
    fever alone              : gate = 0.30
    no bacterial co-indicator: gate = 0.18
  Question: Do any of these values feel clinically wrong?
  Clinician answer: ______________________________________
    """)

    print("=" * 65)
    print("TUNING WORKFLOW (if clinician requests changes)")
    print("=" * 65)
    print("""
  Step 1: Record the specific change requested:
          e.g. "Wheeze_Compound monophonic_low q_i too high — lower to 0.40"

  Step 2: Apply via apply_cpt_tuning(tuning_notes, cpts)

  Step 3: Run verify_tuning_passes_vignettes(updated_cpts)
          Target: ≥24/27. If target fails, revert or adjust.

  Step 4: Re-apply Dirichlet smoothing to any phase or compound CPT changed.
          from dirichlet_smoothing import smooth_and_verify
          smoothed = smooth_and_verify(raw_cpt_array, "NodeName")

  Step 5: Confirm 24/27 still passing before committing to git.

  Most likely changes from clinician:
    • Consol_gate thresholds (fever alone=0.30 may be too high or too low)
    • monophonic_low endobronchial suppression strength (0.40)
    • P_ILD_Flag threshold (0.40 — may want higher for more cases to escape amber)
    """)
