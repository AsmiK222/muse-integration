"""
TreBle Respire — Layer 3: Day 8
Joint End-to-End Test Cases + Edge Case Hardening

Author: Asmi | Day 8

5 joint test cases agreed with Nishevithaa.
Share this file with her — she builds Layer 2 inputs that produce
exactly these Layer 3 outputs.

Edge cases E1–E5 stress-test the model at boundaries.
All must produce well-formed output with no crashes.
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from clinical_cpts import build_intermediate_cpts
from output_layer import compute_output, map_layer2_to_bn
from day7_mock_integration import build_mock_layer2_output, validate_layer2_output


# ---------------------------------------------------------------------------
# BASE CLINICAL — reused across cases
# ---------------------------------------------------------------------------

def _base_cl(**overrides) -> dict:
    cl = {
        "age": 45, "sex": "M", "bmi": 23.0,
        "fever": False, "symptom_duration_days": 4,
        "cough_character": "dry",
        "sputum_purulence_change": False, "dyspnea_increase": False,
        "sputum_volume_increase": False, "spo2_pct": 97.0,
        "fev1_fvc_known": None, "comorbidity_obstructive": False,
        "biomass_exposure": False, "prior_antibiotic_use": False,
        "season": "winter",
    }
    cl.update(overrides)
    return cl


# ---------------------------------------------------------------------------
# 5 JOINT TEST CASES
# ---------------------------------------------------------------------------

def build_joint_test_cases() -> list[dict]:
    """
    5 joint cases with agreed expected Layer 3 outputs.
    Share with Nishevithaa before Day 11.
    She verifies her Layer 2 output produces these results.
    """
    return [
        # J1 — bilateral late_insp fine crackles — CHF/pneumonia differential
        {
            "id": "J1",
            "name": "Late insp bilateral crackles — CHF/pneumonia differential",
            "layer2_mock": build_mock_layer2_output(
                crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
                crackle_subtype="fine", wheeze_compound="uncertain",
                subtype_modules_run=True,
            ),
            "clinical": _base_cl(
                age=68, sex="F", bmi=26.0, fever=False,
                symptom_duration_days=5, dyspnea_increase=True, spo2_pct=94.0,
            ),
            "expected": {
                "zone": "amber", "uncertainty_flag": True,
                "escalate": True,         # SpO2=94 → Sjoding-adjusted=91 → escalate
                "disposition": "refer",
                "note": "CHF/pneumonia differential. SpO2 94→91 after Sjoding correction.",
            },
        },

        # J2 — velcro bilateral basal — ILD safety net
        {
            "id": "J2",
            "name": "Velcro bilateral basal — ILD safety net active",
            "layer2_mock": build_mock_layer2_output(
                crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
                crackle_subtype="velcro", wheeze_compound="uncertain",
                subtype_modules_run=True,
            ),
            "clinical": _base_cl(
                age=65, sex="M", bmi=22.0, fever=False,
                symptom_duration_days=90, dyspnea_increase=True, spo2_pct=91.0,
                season="other",
            ),
            "expected": {
                "zone": "amber", "uncertainty_flag": True,
                "escalate": True,         # SpO2=91 → Sjoding-adjusted=88 → escalate
                "disposition": "refer",
                "note": "ILD flag active from velcro+basal. Must NOT be red.",
            },
        },

        # J3 — monophonic_low wheeze focal — endobronchial pathway
        {
            "id": "J3",
            "name": "Monophonic low wheeze focal — endobronchial, not LRTI",
            "layer2_mock": build_mock_layer2_output(
                wheeze_compound="monophonic_low", wheeze_phase="inspiratory",
                crackle_pattern="absent", subtype_modules_run=True,
            ),
            "clinical": _base_cl(age=50, sex="M", bmi=24.0),
            "expected": {
                "zone": "amber", "uncertainty_flag": True,
                "escalate": False, "disposition": "review_48h",
                "endobronchial_flag": True,
                "note": "Fixed endobronchial obstruction — NOT bacterial LRTI. Investigation needed.",
            },
        },

        # J4 — polyphonic_pan + Anthonisen Type 1 — infected severe COPD
        {
            "id": "J4",
            "name": "Polyphonic pan + Anthonisen Type 1 — infected severe COPD",
            "layer2_mock": build_mock_layer2_output(
                wheeze_compound="polyphonic_pan", wheeze_phase="biphasic",
                crackle_pattern="focal", crackle_phase="late_inspiratory",
                crackle_subtype="coarse", subtype_modules_run=True,
            ),
            "clinical": _base_cl(
                age=70, sex="M", bmi=21.0, fever=True,
                cough_character="productive_purulent",
                sputum_purulence_change=True, dyspnea_increase=True,
                sputum_volume_increase=True, spo2_pct=90.0,
                fev1_fvc_known=0.48, comorbidity_obstructive=True,
            ),
            "expected": {
                "zone": "red", "uncertainty_flag": True,
                "escalate": True,         # SpO2=90 → Sjoding-adjusted=87 → escalate
                "disposition": "refer",
                "note": "Anthonisen Type 1 + polyphonic_pan + fever → clear antibiotic indication.",
            },
        },

        # J5 — early insp coarse crackles + fever — apical TB differential
        {
            "id": "J5",
            "name": "Early insp coarse crackles + fever — apical/TB pattern",
            "layer2_mock": build_mock_layer2_output(
                crackle_pattern="focal", crackle_phase="early_inspiratory",
                crackle_subtype="coarse", wheeze_compound="uncertain",
                subtype_modules_run=True,
            ),
            "clinical": _base_cl(
                age=35, sex="M", bmi=20.0, fever=True,
                symptom_duration_days=14,
                cough_character="productive_purulent", season="other",
            ),
            "expected": {
                "zone": "amber", "uncertainty_flag": True,
                "escalate": False, "disposition": "review_48h",
                "note": "Early_insp + apical gradient → TB differential. Refer for CXR first.",
            },
        },
    ]


def run_joint_test_cases(cases: list[dict], intermediate_cpts: dict) -> bool:
    print("\n" + "=" * 65)
    print("Day 8 — Joint End-to-End Test Cases (share with Nishevithaa)")
    print("=" * 65)

    all_passed = True
    for case in cases:
        print(f"\n  {case['id']}: {case['name']}")
        valid, _ = validate_layer2_output(case["layer2_mock"], source=case["id"])
        if not valid:
            all_passed = False
            continue

        evidence = map_layer2_to_bn(case["layer2_mock"])
        clinical = case["clinical"]
        clinical["bmi"] = case["layer2_mock"]["body_habitus"]["bmi"]
        clinical["age"] = case["layer2_mock"]["body_habitus"]["age"]
        out = compute_output(evidence, clinical, intermediate_cpts)

        exp   = case["expected"]
        fails = []
        if out["zone"]             != exp["zone"]:            fails.append(f"zone: expected {exp['zone']}, got {out['zone']}")
        if out["uncertainty_flag"] != exp["uncertainty_flag"]:fails.append(f"uncertainty_flag: expected {exp['uncertainty_flag']}")
        if out["escalate"]         != exp["escalate"]:        fails.append(f"escalate: expected {exp['escalate']}, got {out['escalate']}")
        if "endobronchial_flag" in exp and out.get("_endobronchial_flag") != exp["endobronchial_flag"]:
            fails.append(f"endobronchial_flag: expected {exp['endobronchial_flag']}")

        passed = len(fails) == 0
        if not passed: all_passed = False
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"    {status} | zone={out['zone']} p={out['p_antibiotics']:.3f} "
              f"unc={out['uncertainty_flag']} esc={out['escalate']}")
        for f in fails:
            print(f"      ✗ {f}")
        print(f"    {exp['note']}")

    print()
    print("=" * 65)
    print("All 5 joint cases PASSED ✓" if all_passed else "SOME JOINT CASES FAILED")
    print("=" * 65)
    return all_passed


# ---------------------------------------------------------------------------
# EDGE CASE HARDENING
# ---------------------------------------------------------------------------

def run_edge_case_hardening(intermediate_cpts: dict) -> bool:
    print("\n" + "=" * 65)
    print("Day 8 — Edge Case Hardening")
    print("=" * 65)

    base = _base_cl(spo2_pct=96.0)

    edge_cases = [
        (
            "E1: Velcro + fever + minimal inputs — ILD suppresses confidence",
            map_layer2_to_bn(build_mock_layer2_output(
                crackle_subtype="velcro", crackle_phase="late_inspiratory",
                crackle_pattern="bilateral_basal", wheeze_compound="uncertain",
            )),
            _base_cl(fever=True),
            {"zone": "amber", "uncertainty_flag": True},
        ),
        (
            "E2: Phase nodes marginalized + strong consolidation — must NOT collapse to green",
            map_layer2_to_bn(build_mock_layer2_output(
                crackle_pattern="focal", crackle_phase=None,
                crackle_subtype=None, wheeze_compound=None, subtype_modules_run=False,
            )),
            _base_cl(fever=True, sputum_purulence_change=True,
                     cough_character="productive_purulent"),
            {"zone_not": "green"},
        ),
        (
            "E3: Dual-zone-only (Z06/Z08) — Zone_Reliability=reduced, inference continues",
            map_layer2_to_bn(build_mock_layer2_output(
                crackle_pattern="focal", crackle_phase="late_inspiratory",
                dual_zone_only=True,
            )),
            _base_cl(fever=True),
            {"zone_reliability": "reduced", "uncertainty_flag": True},
        ),
        (
            "E4: Conflicting signals — late_insp crackles + expiratory polyphonic wheeze",
            map_layer2_to_bn(build_mock_layer2_output(
                crackle_pattern="focal", crackle_phase="late_inspiratory",
                crackle_subtype="coarse", wheeze_compound="polyphonic_low",
                wheeze_phase="expiratory",
            )),
            _base_cl(fever=True, comorbidity_obstructive=True,
                     cough_character="productive_purulent"),
            {"uncertainty_flag": True},
        ),
        (
            "E5: No SpO2 + all acoustic absent + poor data — must not crash",
            map_layer2_to_bn(build_mock_layer2_output(
                crackle_pattern="absent", wheeze_compound="uncertain",
                subtype_modules_run=False, data_quality="poor",
            )),
            _base_cl(spo2_pct=None),
            {"uncertainty_flag": True, "escalate": False},
        ),
    ]

    all_passed = True
    for name, evidence, clinical, checks in edge_cases:
        print(f"\n  {name}")
        try:
            out = compute_output(evidence, clinical, intermediate_cpts)
        except Exception as e:
            print(f"    ✗ CRASH: {e}"); all_passed = False; continue

        ok = True
        if "zone"             in checks and out["zone"]              != checks["zone"]:
            print(f"    ✗ zone: expected {checks['zone']}, got {out['zone']}"); ok = False
        if "zone_not"         in checks and out["zone"]              == checks["zone_not"]:
            print(f"    ✗ zone must NOT be {checks['zone_not']}"); ok = False
        if "uncertainty_flag" in checks and out["uncertainty_flag"]  != checks["uncertainty_flag"]:
            print(f"    ✗ uncertainty_flag: expected {checks['uncertainty_flag']}"); ok = False
        if "escalate"         in checks and out["escalate"]          != checks["escalate"]:
            print(f"    ✗ escalate: expected {checks['escalate']}, got {out['escalate']}"); ok = False
        if "zone_reliability" in checks and evidence.get("Zone_Reliability") != checks["zone_reliability"]:
            print(f"    ✗ Zone_Reliability: expected {checks['zone_reliability']}"); ok = False

        if not ok: all_passed = False
        print(f"    {'✓' if ok else '✗ FAIL'} zone={out['zone']} "
              f"p={out['p_antibiotics']:.3f} unc={out['uncertainty_flag']} esc={out['escalate']}")

    print()
    print("=" * 65)
    print("All edge cases PASSED ✓" if all_passed else "SOME EDGE CASES FAILED")
    print("=" * 65)
    return all_passed


if __name__ == "__main__":
    cpts = build_intermediate_cpts()
    jp = run_joint_test_cases(build_joint_test_cases(), cpts)
    ep = run_edge_case_hardening(cpts)

    print("\n" + "=" * 65)
    print("DAY 8 SUMMARY")
    print("=" * 65)
    print(f"  Joint cases J1–J5 : {'✓ all passed' if jp else '✗ some failed'}")
    print(f"  Edge cases E1–E5  : {'✓ all passed' if ep else '✗ some failed'}")
    print(f"\n  ACTION: Share this file with Nishevithaa.")
    print(f"  She must produce Layer 2 outputs that yield these exact Layer 3 results.")
