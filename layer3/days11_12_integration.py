"""
TreBle Respire — Layer 3: Days 11–12
Live Integration with Layer 2

Author: Asmi | Days 11–12

HOW TO USE:
    Option A — Nishevithaa's real pipeline available:
        from utils.mock_generator import generate_layer2_output_for_bn
        run_live_integration(generate_layer2_output_for_bn)

    Option B — Not yet available (use internal mock):
        run_live_integration(generator_fn=None)

This file produces the integration test log required by the sprint spec.
The monophonic_low endobronchial path must be traced explicitly in this log.
Any interface contract discrepancy → flag to Arvind, do not patch silently.
"""

import sys, os, json
from datetime import datetime
sys.path.insert(0, os.path.dirname(__file__))

from clinical_cpts import build_intermediate_cpts
from output_layer import compute_output, map_layer2_to_bn, VALID_WHEEZE_COMPOUND_STATES
from vignettes import build_vignettes, run_vignettes
from day7_mock_integration import (
    build_mock_layer2_output, validate_layer2_output,
    run_five_mock_scenarios
)
from day8_joint_cases import build_joint_test_cases, run_joint_test_cases


# ---------------------------------------------------------------------------
# INTEGRATION LOG
# Written to integration_test_log.txt alongside this file
# ---------------------------------------------------------------------------

LOG_PATH = os.path.join(os.path.dirname(__file__), "integration_test_log.txt")
_log_lines = []

def log(line: str = ""):
    print(line)
    _log_lines.append(line)

def save_log():
    with open(LOG_PATH, "w") as f:
        f.write("\n".join(_log_lines))
    print(f"\n  Log saved to: {LOG_PATH}")


# ---------------------------------------------------------------------------
# INTERFACE CONTRACT DISCREPANCY TRACKER
# Any mismatch → flag to Arvind, do not patch silently
# ---------------------------------------------------------------------------

_discrepancies = []

def flag_discrepancy(field: str, expected: str, received: str, source: str):
    """
    Records an interface contract violation.
    Call this instead of silently fixing mismatches.
    Printed in summary — send to Arvind.
    """
    msg = f"CONTRACT VIOLATION [{source}]: field='{field}' expected={expected} received={received}"
    _discrepancies.append(msg)
    log(f"  ⚠ {msg}")


# ---------------------------------------------------------------------------
# STEP 1 — Validate Nishevithaa's generator output
# ---------------------------------------------------------------------------

def step1_validate_generator(generator_fn, intermediate_cpts: dict) -> bool:
    log("=" * 65)
    log("STEP 1 — Validate Layer 2 Generator Output")
    log("=" * 65)

    if generator_fn is None:
        log("  ⚠ No real generator provided — using internal mock for all steps.")
        log("    Replace generator_fn with Nishevithaa's generate_layer2_output_for_bn")
        log("    the moment it is available and rerun this script.")
        return True

    log("  Testing all 5 generator modes...")
    modes = ["healthy", "diffuse_crackles", "focal_crackles",
             "bilateral_wheeze", "unilateral_diminished"]
    all_valid = True

    for mode in modes:
        try:
            output = generator_fn(mode)
            valid, errors = validate_layer2_output(output, source=mode)
            if not valid:
                for e in errors:
                    flag_discrepancy("(multiple)", "valid field", str(e), mode)
                all_valid = False
            else:
                # Extra checks beyond the validator
                flags = output.get("summary_flags", {})
                if "wheeze_character" in flags:
                    flag_discrepancy("wheeze_character", "field must not exist",
                                     "field present", mode)
                    all_valid = False
                meta = output.get("output_meta", {})
                if not isinstance(meta.get("subtype_modules_run"), bool):
                    flag_discrepancy("subtype_modules_run", "bool",
                                     type(meta.get("subtype_modules_run")).__name__, mode)
                    all_valid = False
        except Exception as e:
            log(f"  ✗ CRASH on mode='{mode}': {e}")
            all_valid = False

    status = "✓ All modes valid" if all_valid else "✗ Some modes have issues — flag to Nishevithaa"
    log(f"\n  Result: {status}")
    return all_valid


# ---------------------------------------------------------------------------
# STEP 2 — 5 Joint Test Cases on Live Pipeline
# ---------------------------------------------------------------------------

def step2_joint_test_cases(generator_fn, intermediate_cpts: dict) -> bool:
    log("\n" + "=" * 65)
    log("STEP 2 — 5 Joint Test Cases (J1–J5)")
    log("=" * 65)
    log("  These outputs were agreed with Nishevithaa on Day 8.")
    log("  Verify Layer 3 outputs match exactly.\n")

    cases   = build_joint_test_cases()
    passed  = run_joint_test_cases(cases, intermediate_cpts)

    if generator_fn is not None:
        log("\n  Running same cases against LIVE Layer 2 generator...")
        live_passed = 0
        for case in cases:
            try:
                mode_map = {
                    "J1": "diffuse_crackles",
                    "J2": "diffuse_crackles",
                    "J3": "bilateral_wheeze",
                    "J4": "focal_crackles",
                    "J5": "focal_crackles",
                }
                live_out = generator_fn(mode_map[case["id"]])
                valid, _ = validate_layer2_output(live_out, source=f"live_{case['id']}")
                if valid:
                    ev  = map_layer2_to_bn(live_out)
                    cl  = case["clinical"]
                    bh  = live_out.get("body_habitus", {})
                    cl["bmi"] = bh.get("bmi", cl.get("bmi", 23.0))
                    cl["age"] = bh.get("age", cl.get("age", 45))
                    out = compute_output(ev, cl, intermediate_cpts)
                    log(f"  [{case['id']} live] zone={out['zone']} "
                        f"p={out['p_antibiotics']:.3f} unc={out['uncertainty_flag']}")
                    live_passed += 1
            except Exception as e:
                log(f"  [{case['id']} live] ✗ CRASH: {e}")
        log(f"\n  Live pipeline: {live_passed}/{len(cases)} cases completed without crash")

    return passed


# ---------------------------------------------------------------------------
# STEP 3 — Endobronchial Flag Path Trace (sprint spec requirement)
# ---------------------------------------------------------------------------

def step3_endobronchial_path_trace(generator_fn, intermediate_cpts: dict) -> bool:
    log("\n" + "=" * 65)
    log("STEP 3 — Endobronchial Flag Pathway Trace")
    log("  Sprint spec requirement: trace this path explicitly in integration log")
    log("=" * 65)
    log("""
  PATH:
    Layer 2 monophonic_low wheeze
      → wheeze_compound_dominant = 'monophonic_low'
      → map_layer2_to_bn(): Wheeze_Compound = 'monophonic_low'
      → BN: suppresses P(bacterial) via endobronchial_suppression = 0.40
      → output: green/amber + endobronchial_flag=True
  """)

    mock = build_mock_layer2_output(
        wheeze_compound="monophonic_low",
        wheeze_phase="inspiratory",
        crackle_pattern="absent",
        subtype_modules_run=True,
    )

    # Trace Step 1: Layer 2 output field
    wc_in_flags = mock["summary_flags"].get("wheeze_compound_dominant")
    log(f"  [1] Layer 2 summary_flags.wheeze_compound_dominant = '{wc_in_flags}'")
    assert wc_in_flags == "monophonic_low", \
        flag_discrepancy("wheeze_compound_dominant", "monophonic_low", str(wc_in_flags), "mock")

    # Trace Step 2: input mapper
    evidence = map_layer2_to_bn(mock)
    wc_in_ev = evidence.get("Wheeze_Compound")
    log(f"  [2] map_layer2_to_bn() → evidence['Wheeze_Compound'] = '{wc_in_ev}'")
    assert wc_in_ev == "monophonic_low", \
        flag_discrepancy("Wheeze_Compound", "monophonic_low", str(wc_in_ev), "map_layer2_to_bn")

    # Trace Step 3: wheeze_character must NOT appear
    assert "wheeze_character" not in evidence, \
        flag_discrepancy("wheeze_character", "must not exist", "field present", "evidence dict")
    log(f"  [3] 'wheeze_character' not in evidence keys ✓")

    # Trace Step 4: BN inference
    clinical = {
        "age": 45, "sex": "M", "bmi": 23.0,
        "fever": False, "symptom_duration_days": 3, "cough_character": "dry",
        "sputum_purulence_change": False, "dyspnea_increase": False,
        "sputum_volume_increase": False, "spo2_pct": 97.0,
        "fev1_fvc_known": None, "comorbidity_obstructive": False,
        "biomass_exposure": False, "prior_antibiotic_use": False,
        "season": "winter",
    }
    out = compute_output(evidence, clinical, intermediate_cpts)
    log(f"  [4] compute_output():")
    log(f"        p_antibiotics    = {out['p_antibiotics']:.4f}")
    log(f"        zone             = {out['zone']}")
    log(f"        endobronchial_flag = {out.get('_endobronchial_flag')}")
    log(f"        p_consolidation  = {out.get('_p_consolidation', 'N/A')}")
    log(f"        p_obstructive    = {out.get('_p_obstructive', 'N/A')}")
    log(f"        uncertainty_flag = {out['uncertainty_flag']}")
    log(f"        disposition      = {out['disposition']}")

    # Assertions
    eb_ok   = out.get("_endobronchial_flag") == True
    red_ok  = out["zone"] != "red"
    log(f"\n  [5] Assertions:")
    log(f"        endobronchial_flag=True : {'✓' if eb_ok else '✗ FAIL'}")
    log(f"        zone != 'red'           : {'✓' if red_ok else '✗ FAIL'}")

    if generator_fn is not None:
        log(f"\n  [6] Running same trace on LIVE Layer 2 generator (bilateral_wheeze mode)...")
        try:
            live = generator_fn("bilateral_wheeze")
            ev_live = map_layer2_to_bn(live)
            wc_live = ev_live.get("Wheeze_Compound")
            log(f"        live Wheeze_Compound = '{wc_live}'")
            if wc_live != "monophonic_low":
                log(f"        NOTE: bilateral_wheeze mode may not produce monophonic_low.")
                log(f"        Ask Nishevithaa to add a 'monophonic_low_wheeze' mode for this trace.")
        except Exception as e:
            log(f"        ✗ CRASH: {e}")

    passed = eb_ok and red_ok
    log(f"\n  Endobronchial path trace: {'✓ COMPLETE' if passed else '✗ FAILED'}")
    return passed


# ---------------------------------------------------------------------------
# STEP 4 — Full 27-Vignette Pipeline via Live Layer 2
# ---------------------------------------------------------------------------

def step4_full_vignette_pipeline(intermediate_cpts: dict) -> int:
    log("\n" + "=" * 65)
    log("STEP 4 — Full 27-Vignette Pipeline")
    log("=" * 65)

    vignettes = build_vignettes()
    results   = run_vignettes(vignettes, intermediate_cpts)

    log(f"\n  Result: {results['passed']}/27 "
        f"({'✓ TARGET MET' if results['passed'] >= 24 else '✗ BELOW TARGET'})")

    if results["failures"]:
        log(f"\n  Failed vignettes ({len(results['failures'])}):")
        for f in results["failures"]:
            log(f"    Case {f['num']:>2}: {f['desc'][:48]}")
            log(f"           expected={f['expected_zone']} "
                f"got={f['actual_zone']} "
                f"p={f['output']['p_antibiotics']:.3f}")

    return results["passed"]


# ---------------------------------------------------------------------------
# MAIN — Days 11–12 Integration Run
# ---------------------------------------------------------------------------

def run_live_integration(generator_fn=None):
    """
    Full Days 11–12 integration run.

    Args:
        generator_fn: Nishevithaa's generate_layer2_output_for_bn function.
                      Pass None to use internal mock (will warn).

    Usage with real pipeline:
        from utils.mock_generator import generate_layer2_output_for_bn
        run_live_integration(generate_layer2_output_for_bn)
    """
    _log_lines.clear()
    _discrepancies.clear()

    log("=" * 65)
    log("TreBle Respire — Layer 3: Days 11–12 Live Integration")
    log(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Generator: {'LIVE (Nishevithaa)' if generator_fn else 'INTERNAL MOCK'}")
    log("=" * 65)

    cpts = build_intermediate_cpts()

    s1 = step1_validate_generator(generator_fn, cpts)
    s2 = step2_joint_test_cases(generator_fn, cpts)
    s3 = step3_endobronchial_path_trace(generator_fn, cpts)
    s4 = step4_full_vignette_pipeline(cpts)

    log("\n" + "=" * 65)
    log("INTEGRATION SUMMARY")
    log("=" * 65)
    log(f"  Step 1 — Generator validation : {'✓' if s1 else '✗'}")
    log(f"  Step 2 — Joint test cases     : {'✓' if s2 else '✗'}")
    log(f"  Step 3 — Endobronchial trace  : {'✓' if s3 else '✗'}")
    log(f"  Step 4 — Full vignette suite  : {s4}/27 {'✓' if s4 >= 24 else '✗'}")

    if _discrepancies:
        log(f"\n  CONTRACT VIOLATIONS ({len(_discrepancies)}) — send to Arvind:")
        for d in _discrepancies:
            log(f"    • {d}")
        log("\n  Do NOT patch these silently. Flag to Arvind.")
    else:
        log("\n  ✓ No interface contract violations detected.")

    all_passed = s1 and s2 and s3 and s4 >= 24
    log(f"\n  Overall: {'✓ INTEGRATION COMPLETE' if all_passed else '✗ ISSUES TO RESOLVE'}")
    log("=" * 65)

    save_log()
    return all_passed


if __name__ == "__main__":
    # Default: runs with internal mock
    # To use Nishevithaa's real generator:
    #   from utils.mock_generator import generate_layer2_output_for_bn
    #   run_live_integration(generate_layer2_output_for_bn)
    run_live_integration(generator_fn=None)
