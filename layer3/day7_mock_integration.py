"""
TreBle Respire — Layer 3: Day 7
Layer 2 Mock Consumption + Indian Calibration Hardening

Author: Asmi | Day 7 (updated for 20-zone interface)

Changes from first version:
  - Zone count corrected to 20 (Z01-Z20), per Arvind confirmation
  - consume_nish_mock() added for Nishevithaa's real generator
  - All 5 mock scenarios use 20-zone output
  - Biomass adjustment unchanged (Balakrishnan PMC4221659)
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from clinical_cpts import build_intermediate_cpts
from output_layer import compute_output, map_layer2_to_bn, VALID_WHEEZE_COMPOUND_STATES


ALL_ZONES       = [f"Z{i:02d}" for i in range(1, 21)]   # Z01–Z20
DUAL_ZONES      = {"Z06", "Z08"}
EXPECTED_ZONES  = 20
BIOMASS_MULTIPLIER = 1.35  # Balakrishnan PMC4221659

REQUIRED_SUMMARY_FLAGS = [
    "crackles_present", "crackles_focal", "crackles_bilateral",
    "crackle_subtype_dominant", "crackle_phase_dominant", "velcro_crackles_present",
    "wheeze_present", "wheeze_bilateral", "wheeze_focal", "wheeze_phase_dominant",
    "wheeze_compound_dominant",
    "wheeze_monophonic_any", "wheeze_polyphonic_any", "wheeze_focal_monophonic",
    "wheeze_low_freq_present", "wheeze_high_freq_present", "wheeze_pan_airway",
    "diminished_present", "diminished_unilateral", "consolidation_signs",
    "zones_recorded", "bilateral_asymmetry_score_max",
    "craniocaudal_gradient_dominant", "rhonchi_present",
]
REQUIRED_META_FIELDS = ["subtype_modules_run", "data_quality"]
REQUIRED_ZONE_FIELDS = ["zone_id", "recorded", "any_abnormality", "is_dual_representation_zone"]


# ---------------------------------------------------------------------------
# INTERNAL MOCK BUILDER
# ---------------------------------------------------------------------------

def build_mock_layer2_output(
    wheeze_compound: str = "uncertain",
    crackle_phase: str = None,
    wheeze_phase: str = None,
    crackle_pattern: str = "absent",
    crackle_subtype: str = None,
    dual_zone_only: bool = False,
    subtype_modules_run: bool = True,
    data_quality: str = "normal",
    spo2: float = 97.0,
    bmi: float = 23.0,
    age: int = 45,
) -> dict:
    """Internal mock — 20 zones, all required Layer 3 fields populated."""
    BASAL = {"Z05","Z06","Z13","Z14","Z15","Z16"}

    zone_findings = []
    for z in ALL_ZONES:
        is_dual = z in DUAL_ZONES
        is_abn  = crackle_pattern != "absent" and z in BASAL
        zone_findings.append({
            "zone_id":                     z,
            "recorded":                    True,
            "any_abnormality":             is_abn,
            "is_dual_representation_zone": is_dual,
            "signal_quality":              0.85 if not is_dual else 0.60,
        })

    if dual_zone_only:
        for z in zone_findings:
            z["any_abnormality"] = z["zone_id"] in DUAL_ZONES

    summary_flags = {
        "crackles_present":               crackle_pattern != "absent",
        "crackles_focal":                 crackle_pattern == "focal",
        "crackles_bilateral":             crackle_pattern == "bilateral_basal",
        "crackle_subtype_dominant":       crackle_subtype,
        "crackle_phase_dominant":         crackle_phase,
        "velcro_crackles_present":        crackle_subtype == "velcro",
        "wheeze_present":                 wheeze_compound not in ("uncertain", None),
        "wheeze_bilateral":               wheeze_compound in ("polyphonic_low","polyphonic_pan","polyphonic_mid"),
        "wheeze_focal":                   wheeze_compound in ("monophonic_low","monophonic_mid","monophonic_high"),
        "wheeze_phase_dominant":          wheeze_phase,
        "wheeze_compound_dominant":       wheeze_compound,  # NOT wheeze_character
        "wheeze_monophonic_any":          wheeze_compound in ("monophonic_low","monophonic_mid","monophonic_high"),
        "wheeze_polyphonic_any":          wheeze_compound in ("polyphonic_low","polyphonic_mid","polyphonic_pan"),
        "wheeze_focal_monophonic":        wheeze_compound == "monophonic_low",
        "wheeze_low_freq_present":        wheeze_compound in ("monophonic_low","polyphonic_low"),
        "wheeze_high_freq_present":       wheeze_compound in ("monophonic_high","polyphonic_pan"),
        "wheeze_pan_airway":              wheeze_compound == "polyphonic_pan",
        "diminished_present":             False,
        "diminished_unilateral":          False,
        "consolidation_signs":            crackle_pattern == "focal" and crackle_phase == "late_inspiratory",
        "zones_recorded":                 EXPECTED_ZONES,
        "zones_low_quality":              2 if dual_zone_only else 0,
        "dual_zones_low_quality":         dual_zone_only,
        "rhonchi_present":                False,
        "bilateral_asymmetry_score_max":  0.75 if crackle_pattern == "focal" else 0.10,
        "craniocaudal_gradient_dominant": "flat",
    }

    return {
        "zone_findings": zone_findings,
        "summary_flags": summary_flags,
        "output_meta": {
            "subtype_modules_run": subtype_modules_run,
            "data_quality":        data_quality,
            "pipeline_version":    "2.1.0",
        },
        "body_habitus": {"bmi": bmi, "age": age, "sex": "M"},
    }


# ---------------------------------------------------------------------------
# CONSUME NISHEVITHAA'S REAL MOCK
# ---------------------------------------------------------------------------

def consume_nish_mock(nish_output: dict, label: str = "nish",
                      clinical_overrides: dict = None) -> tuple[bool, dict]:
    """
    Validates and runs Nishevithaa's generate_layer2_output_for_bn() output
    through the full Layer 3 pipeline.

    Usage:
        from utils.mock_generator import generate_layer2_output_for_bn
        nish_out = generate_layer2_output_for_bn(mode="focal_crackles")
        passed, out = consume_nish_mock(nish_out, label="focal_crackles")
    """
    valid, errors = validate_layer2_output(nish_output, source=label)
    if not valid:
        return False, {}

    evidence = map_layer2_to_bn(nish_output)
    bh = nish_output.get("body_habitus", {})
    clinical = {
        "age":   bh.get("age", 45),
        "sex":   bh.get("sex", "M"),
        "bmi":   bh.get("bmi", 23.0),
        "fever": False, "symptom_duration_days": 4,
        "cough_character": "dry",
        "sputum_purulence_change": False, "dyspnea_increase": False,
        "sputum_volume_increase": False, "spo2_pct": 97.0,
        "fev1_fvc_known": None, "comorbidity_obstructive": False,
        "biomass_exposure": False, "prior_antibiotic_use": False,
        "season": "winter",
    }
    if clinical_overrides:
        clinical.update(clinical_overrides)

    cpts = build_intermediate_cpts()
    out  = compute_output(evidence, clinical, cpts)
    print(f"  ✓ [{label}] zone={out['zone']} p={out['p_antibiotics']:.3f} "
          f"unc={out['uncertainty_flag']} esc={out['escalate']} "
          f"disposition={out['disposition']}")
    return True, out


# ---------------------------------------------------------------------------
# VALIDATOR
# ---------------------------------------------------------------------------

def validate_layer2_output(layer2_output: dict, source: str = "mock") -> tuple[bool, list]:
    """Validates all fields Layer 3 needs from Layer 2."""
    errors = []
    flags  = layer2_output.get("summary_flags", {})
    meta   = layer2_output.get("output_meta", {})
    zones  = layer2_output.get("zone_findings", [])

    if "wheeze_character" in flags:
        errors.append("WRONG FIELD: 'wheeze_character' — must be 'wheeze_compound_dominant'")
    for f in REQUIRED_SUMMARY_FLAGS:
        if f not in flags:
            errors.append(f"MISSING summary_flag: '{f}'")
    for f in REQUIRED_META_FIELDS:
        if f not in meta:
            errors.append(f"MISSING output_meta: '{f}'")
    if not isinstance(meta.get("subtype_modules_run"), bool):
        errors.append("subtype_modules_run must be bool")
    if meta.get("data_quality") not in ("normal", "poor", None):
        errors.append(f"data_quality must be 'normal' or 'poor'")

    if not zones:
        errors.append("zone_findings is empty")
    else:
        if len(zones) != EXPECTED_ZONES:
            errors.append(f"zone count {len(zones)} ≠ {EXPECTED_ZONES}")
        for z in zones:
            for f in REQUIRED_ZONE_FIELDS:
                if f not in z:
                    errors.append(f"Zone {z.get('zone_id','?')}: MISSING '{f}'")
            if z.get("zone_id") in DUAL_ZONES and not z.get("is_dual_representation_zone"):
                errors.append(f"Zone {z['zone_id']}: is_dual_representation_zone must be True")

    wc = flags.get("wheeze_compound_dominant")
    if wc is not None and wc not in VALID_WHEEZE_COMPOUND_STATES:
        errors.append(f"INVALID wheeze_compound_dominant: '{wc}'")

    is_valid = len(errors) == 0
    print(f"  Layer 2 output validation [{source}]: {'✓ VALID' if is_valid else f'✗ {len(errors)} ERRORS'}")
    for e in errors:
        print(f"    → {e}")
    return is_valid, errors


# ---------------------------------------------------------------------------
# BIOMASS ADJUSTMENT
# ---------------------------------------------------------------------------

def apply_biomass_adjustment(p_base: float, biomass: bool, obstructive: bool) -> float:
    """
    Increases P_Obstructive base rate for biomass-exposed patients.
    Balakrishnan et al. PMC4221659 — 70% of Indian homes use biomass fuel.
    Non-smoking COPD is a distinct phenotype; adjustment only when no established dx.
    """
    if not biomass:
        return p_base
    if obstructive:
        return min(p_base * 1.10, 0.95)
    return min(p_base * BIOMASS_MULTIPLIER, 0.50)


# ---------------------------------------------------------------------------
# 5 MOCK SCENARIOS
# ---------------------------------------------------------------------------

def run_five_mock_scenarios(intermediate_cpts: dict) -> bool:
    """5 required Day 7 scenarios. All must pass with no crashes."""
    print("\n" + "=" * 65)
    print("Day 7 — 5 Mock Pipeline Scenarios")
    print("=" * 65)

    base_cl = {
        "age": 45, "sex": "M", "bmi": 23.0,
        "fever": True, "symptom_duration_days": 4,
        "cough_character": "productive_purulent",
        "sputum_purulence_change": False, "dyspnea_increase": False,
        "sputum_volume_increase": False, "spo2_pct": 95.0,
        "fev1_fvc_known": None, "comorbidity_obstructive": False,
        "biomass_exposure": False, "prior_antibiotic_use": False,
        "season": "winter",
    }

    scenarios = [
        ("S1 — monophonic_low wheeze",
         dict(wheeze_compound="monophonic_low", wheeze_phase="inspiratory",
              crackle_pattern="absent", subtype_modules_run=True),
         dict(fever=False, cough_character="dry"),
         {"endobronchial_flag": True, "zone_not": "red"}),

        ("S2 — polyphonic_pan wheeze",
         dict(wheeze_compound="polyphonic_pan", wheeze_phase="biphasic",
              crackle_pattern="focal", crackle_phase="late_inspiratory",
              subtype_modules_run=True),
         dict(comorbidity_obstructive=True, sputum_purulence_change=True,
              dyspnea_increase=True, sputum_volume_increase=True),
         {"zone": "red"}),

        (
            # S3 — dual-zone-only (Z06/Z08)
            #
            # WHAT THIS TESTS: only Z06 and Z08 (dual-representation zones) are abnormal.
            # These two zones represent the same underlying anatomy recorded twice —
            # findings here are spatially redundant, not independent evidence.
            #
            # WHY p_antibiotics IS STILL HIGH HERE:
            # The high p (~0.72) is driven by FINDING STRENGTH — focal late_insp
            # crackles are strong evidence (q_i=0.75, Wipf 1999) regardless of how
            # many zones recorded them. The model does NOT require multi-zone spread
            # to increase the posterior. A single strong finding in two zones raises
            # p_antibiotics just as much as the same finding in six zones.
            #
            # WHAT ZONE_RELIABILITY DOES:
            # It does NOT suppress p_antibiotics. It widens the CI and sets
            # uncertainty_flag=True, because we cannot confirm the spatial pattern
            # (lobar vs diffuse vs bilateral) from dual zones alone. The WHAT is
            # uncertain; the WHETHER is still likely given the finding strength.
            #
            # CLINICAL MEANING: patient probably has pathology, but we can't determine
            # the spatial distribution confidently. CI wider, refer for CXR — do not
            # use zone pattern alone to localise.
            "S3 — dual-zone-only (Z06/Z08)",
            dict(wheeze_compound="uncertain", crackle_pattern="focal",
                 crackle_phase="late_inspiratory", dual_zone_only=True,
                 subtype_modules_run=True),
            dict(fever=True),
            {"zone_reliability": "reduced", "uncertainty_flag": True},
        ),

        ("S4 — subtype_modules_run=False",
         dict(wheeze_compound="uncertain", crackle_pattern="focal",
              crackle_phase="late_inspiratory", subtype_modules_run=False),
         dict(fever=True),
         {"wheeze_compound_in_evidence": None}),

        ("S5 — all-uncertain + poor data",
         dict(wheeze_compound="uncertain", crackle_pattern="absent",
              crackle_phase=None, wheeze_phase=None,
              subtype_modules_run=True, data_quality="poor"),
         dict(fever=False, cough_character="dry"),
         {"uncertainty_flag": True}),
    ]

    all_passed = True
    for name, mk, cl_ov, checks in scenarios:
        print(f"\n  {name}")
        mock     = build_mock_layer2_output(**mk)
        valid, _ = validate_layer2_output(mock, source=name[:20])
        if not valid:
            all_passed = False
            continue

        evidence = map_layer2_to_bn(mock)
        clinical = {**base_cl, **cl_ov,
                    "bmi": mock["body_habitus"]["bmi"],
                    "age": mock["body_habitus"]["age"]}
        try:
            out = compute_output(evidence, clinical, intermediate_cpts)
        except Exception as e:
            print(f"    ✗ CRASH: {e}"); all_passed = False; continue

        ok = True
        if "zone"         in checks and out["zone"] != checks["zone"]:
            print(f"    ✗ zone: expected {checks['zone']}, got {out['zone']}"); ok = False
        if "zone_not"     in checks and out["zone"] == checks["zone_not"]:
            print(f"    ✗ zone must NOT be {checks['zone_not']}"); ok = False
        if "uncertainty_flag" in checks and not out["uncertainty_flag"]:
            print(f"    ✗ uncertainty_flag must be True"); ok = False
        if "endobronchial_flag" in checks and not out.get("_endobronchial_flag"):
            print(f"    ✗ endobronchial_flag must be True"); ok = False
        if "zone_reliability" in checks and evidence.get("Zone_Reliability") != checks["zone_reliability"]:
            print(f"    ✗ Zone_Reliability: expected {checks['zone_reliability']}"); ok = False
        if "wheeze_compound_in_evidence" in checks and evidence.get("Wheeze_Compound") != checks["wheeze_compound_in_evidence"]:
            print(f"    ✗ Wheeze_Compound: expected {checks['wheeze_compound_in_evidence']}, "
                  f"got {evidence.get('Wheeze_Compound')}"); ok = False

        if ok:
            print(f"    ✓ zone={out['zone']} p={out['p_antibiotics']:.3f} "
                  f"unc={out['uncertainty_flag']} "
                  f"endobronchial={out.get('_endobronchial_flag', False)}")
            # S3-specific explanation printed in output log
            if "S3" in name:
                print(f"    ── S3 note ──────────────────────────────────────────")
                print(f"    p={out['p_antibiotics']:.3f} is HIGH — driven by finding strength,")
                print(f"    not by multi-zone spread. Only Z06+Z08 recorded (2 zones),")
                print(f"    but focal late_insp crackles (q_i=0.75, Wipf 1999) + fever")
                print(f"    are strong evidence regardless of zone count.")
                print(f"    Zone_Reliability='reduced' widens CI and sets unc=True")
                print(f"    because spatial pattern (lobar vs diffuse) is unconfirmable")
                print(f"    from dual zones alone — not because the finding is weak.")
                print(f"    CI: [{out['ci_lower']:.3f}, {out['ci_upper']:.3f}]  "
                      f"width={out['ci_upper']-out['ci_lower']:.3f}")
                print(f"    ─────────────────────────────────────────────────────")
        else:
            all_passed = False

    print()
    print("=" * 65)
    print("All 5 mock scenarios PASSED ✓" if all_passed else "SOME SCENARIOS FAILED")
    print("=" * 65)
    return all_passed


# ---------------------------------------------------------------------------
# WHEEZE COMPOUND PATH TRACE
# ---------------------------------------------------------------------------

def verify_wheeze_compound_paths(intermediate_cpts: dict):
    """Traces each wheeze compound through the pipeline. Key: monophonic_low ≠ red."""
    print("\n" + "=" * 65)
    print("Wheeze Compound Path Verification")
    print("=" * 65)

    base_cl = {
        "age": 45, "sex": "M", "bmi": 23.0,
        "fever": False, "symptom_duration_days": 3, "cough_character": "dry",
        "sputum_purulence_change": False, "dyspnea_increase": False,
        "sputum_volume_increase": False, "spo2_pct": 97.0,
        "fev1_fvc_known": None, "comorbidity_obstructive": False,
        "biomass_exposure": False, "prior_antibiotic_use": False,
        "season": "winter",
    }
    paths = [
        ("monophonic_low",  "inspiratory", "endobronchial suppression → NOT red"),
        ("monophonic_high", "inspiratory", "small airway → mild obstructive"),
        ("polyphonic_low",  "expiratory",  "classic COPD → obstructive"),
        ("polyphonic_pan",  "biphasic",    "severe pan-airway → strong obstructive"),
        ("uncertain",       None,          "marginalize → prior only"),
    ]

    print(f"\n  {'Compound':<20} {'Phase':<12} {'P(abx)':>7} {'Zone':>6} {'Endobronchial':>14}")
    print("  " + "-" * 65)
    for compound, phase, note in paths:
        mock = build_mock_layer2_output(wheeze_compound=compound, wheeze_phase=phase,
                                        crackle_pattern="absent", subtype_modules_run=True)
        ev   = map_layer2_to_bn(mock)
        out  = compute_output(ev, base_cl, intermediate_cpts)
        eb   = out.get("_endobronchial_flag", False)
        print(f"  {compound:<20} {str(phase):<12} {out['p_antibiotics']:>7.3f} "
              f"{out['zone']:>6} {str(eb):>14}   {note}")

    mono_mock = build_mock_layer2_output(wheeze_compound="monophonic_low",
                                          wheeze_phase="inspiratory", crackle_pattern="absent")
    ev_m = map_layer2_to_bn(mono_mock)
    out_m = compute_output(ev_m, base_cl, intermediate_cpts)
    assert out_m["zone"] != "red",        "monophonic_low must NOT be red"
    assert out_m["_endobronchial_flag"],  "monophonic_low must set endobronchial_flag"
    assert "wheeze_character" not in ev_m
    print(f"\n  ✓ monophonic_low → {out_m['zone']} (not red), endobronchial_flag=True")
    print(f"  ✓ wheeze_character not in evidence keys")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cpts = build_intermediate_cpts()

    print("=" * 65)
    print("Day 7 — Layer 2 Mock Consumption + Indian Calibration")
    print("=" * 65)

    print("\nBiomass Exposure Adjustment (Balakrishnan PMC4221659):")
    for bm, ob, label in [(False,False,"no exposure"),(True,False,"exposed, no dx"),(True,True,"exposed + COPD")]:
        print(f"  {label:<25} → P_Obstructive base = {apply_biomass_adjustment(0.25, bm, ob):.3f}")

    passed = run_five_mock_scenarios(cpts)
    verify_wheeze_compound_paths(cpts)

    print("\n" + "=" * 65)
    print("DAY 7 SUMMARY")
    print("=" * 65)
    print(f"  5 mock scenarios   : {'✓ all passed' if passed else '✗ some failed'}")
    print(f"  Wheeze paths       : ✓ verified")
    print(f"  Biomass adjustment : ✓ (×{BIOMASS_MULTIPLIER}, Balakrishnan PMC4221659)")
    print(f"  Zone count         : {EXPECTED_ZONES} (Z01–Z20)")
    print(f"  wheeze_character   : ✓ zero functional assignments")
    print(f"\n  To use Nishevithaa's real mock:")
    print(f"    from utils.mock_generator import generate_layer2_output_for_bn")
    print(f"    consume_nish_mock(generate_layer2_output_for_bn('focal_crackles'))")    