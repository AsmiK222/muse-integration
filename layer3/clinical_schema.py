"""
TreBle Respire — Layer 3: Clinical Input Schema + Validator
Author: Asmi | Day 2

This dict is your TEST HARNESS — every vignette is an instance of this schema.
Clinical inputs are collected DIRECTLY by Layer 3 (not from Layer 2).
Body habitus (bmi, age) come from Layer 2 body_habitus field — do not re-collect.

Reference: Sprint spec clinical input schema (locked)
"""

from typing import Optional
import json


# ---------------------------------------------------------------------------
# VALID VALUES FOR EACH FIELD
# ---------------------------------------------------------------------------

VALID_SEX              = {"M", "F", "unknown"}
VALID_COUGH_CHARACTER  = {"dry", "productive_mucoid", "productive_purulent"}
VALID_SEASON           = {"monsoon", "winter", "other"}


# ---------------------------------------------------------------------------
# SCHEMA TEMPLATE
# Every vignette must be an instance of this structure.
# ---------------------------------------------------------------------------

CLINICAL_INPUT_TEMPLATE = {
    # --- Demographics (from Layer 2 body_habitus) ---
    "age":                    None,   # int, years
    "sex":                    None,   # "M" | "F" | "unknown"
    "bmi":                    None,   # float, from Layer 2 body_habitus

    # --- Symptom history ---
    "fever":                  None,   # bool — temperature > 37.8°C
    "symptom_duration_days":  None,   # int

    # --- Cough & sputum ---
    "cough_character":        None,   # "dry" | "productive_mucoid" | "productive_purulent"
    "sputum_purulence_change":None,   # bool — Anthonisen criterion 1
    "dyspnea_increase":       None,   # bool — Anthonisen criterion 2
    "sputum_volume_increase": None,   # bool — Anthonisen criterion 3

    # --- Objective measurements ---
    "spo2_pct":               None,   # float | None  (None = not measured)
    "fev1_fvc_known":         None,   # float | None  (None = not available)

    # --- Background / risk factors ---
    "comorbidity_obstructive":None,   # bool — known COPD or asthma
    "biomass_exposure":       None,   # bool — cooks with biomass fuel
    "prior_antibiotic_use":   None,   # bool — antibiotics in past 3 months
    "season":                 None,   # "monsoon" | "winter" | "other"
}


# ---------------------------------------------------------------------------
# VALIDATOR
# ---------------------------------------------------------------------------

def validate_clinical_input(data: dict) -> tuple[bool, list[str]]:
    """
    Validates a clinical input dict against the schema.

    Returns:
        (is_valid: bool, errors: list[str])
        If is_valid is True, errors is empty.
        If is_valid is False, errors lists every problem found.

    Usage:
        ok, errors = validate_clinical_input(vignette)
        if not ok:
            for e in errors:
                print(e)
    """
    errors = []

    # --- Required fields that must never be None ---
    required_fields = [
        "age", "sex", "fever", "symptom_duration_days",
        "cough_character", "sputum_purulence_change",
        "dyspnea_increase", "sputum_volume_increase",
        "comorbidity_obstructive", "biomass_exposure",
        "prior_antibiotic_use", "season",
    ]
    for field in required_fields:
        if field not in data:
            errors.append(f"MISSING field: '{field}'")
        elif data[field] is None:
            errors.append(f"NULL required field: '{field}' must not be None")

    # --- Optional fields that may be None but must be correct type if present ---
    optional_typed = {
        "spo2_pct":       (float, int),
        "fev1_fvc_known": (float, int),
        "bmi":            (float, int),
    }
    for field, types in optional_typed.items():
        val = data.get(field)
        if val is not None and not isinstance(val, types):
            errors.append(f"TYPE ERROR: '{field}' must be float or None, got {type(val).__name__}")

    # --- Type checks for required fields ---
    if "age" in data and data["age"] is not None:
        if not isinstance(data["age"], int):
            errors.append(f"TYPE ERROR: 'age' must be int, got {type(data['age']).__name__}")
        elif not (0 < data["age"] < 120):
            errors.append(f"RANGE ERROR: 'age' = {data['age']} is outside plausible range (0–120)")

    if "sex" in data and data["sex"] is not None:
        if data["sex"] not in VALID_SEX:
            errors.append(f"VALUE ERROR: 'sex' must be one of {VALID_SEX}, got '{data['sex']}'")

    if "symptom_duration_days" in data and data["symptom_duration_days"] is not None:
        if not isinstance(data["symptom_duration_days"], int):
            errors.append(f"TYPE ERROR: 'symptom_duration_days' must be int")
        elif data["symptom_duration_days"] < 0:
            errors.append(f"RANGE ERROR: 'symptom_duration_days' cannot be negative")

    if "cough_character" in data and data["cough_character"] is not None:
        if data["cough_character"] not in VALID_COUGH_CHARACTER:
            errors.append(f"VALUE ERROR: 'cough_character' must be one of {VALID_COUGH_CHARACTER}")

    if "season" in data and data["season"] is not None:
        if data["season"] not in VALID_SEASON:
            errors.append(f"VALUE ERROR: 'season' must be one of {VALID_SEASON}")

    # --- Bool fields ---
    bool_fields = [
        "fever", "sputum_purulence_change", "dyspnea_increase",
        "sputum_volume_increase", "comorbidity_obstructive",
        "biomass_exposure", "prior_antibiotic_use",
    ]
    for field in bool_fields:
        val = data.get(field)
        if val is not None and not isinstance(val, bool):
            errors.append(f"TYPE ERROR: '{field}' must be bool, got {type(val).__name__}")

    # --- SpO2 range check ---
    spo2 = data.get("spo2_pct")
    if spo2 is not None and not (50.0 <= spo2 <= 100.0):
        errors.append(f"RANGE ERROR: 'spo2_pct' = {spo2} is outside plausible range (50–100)")

    # --- FEV1/FVC range check ---
    fev = data.get("fev1_fvc_known")
    if fev is not None and not (0.1 <= fev <= 1.0):
        errors.append(f"RANGE ERROR: 'fev1_fvc_known' = {fev} should be ratio between 0.1 and 1.0")

    # --- Anthonisen completeness warning (not an error, just a flag) ---
    anthonisen_fields = ["sputum_purulence_change", "dyspnea_increase", "sputum_volume_increase"]
    anthonisen_vals = [data.get(f) for f in anthonisen_fields]
    if all(v is False for v in anthonisen_vals):
        # All three false = Anthonisen Type 3 — valid, just note it
        pass

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_and_raise(data: dict, vignette_name: str = ""):
    """Validates and raises ValueError if invalid. Use in tests."""
    ok, errors = validate_clinical_input(data)
    if not ok:
        label = f"Vignette '{vignette_name}'" if vignette_name else "Clinical input"
        raise ValueError(f"{label} failed validation:\n" + "\n".join(f"  • {e}" for e in errors))
    return True


# ---------------------------------------------------------------------------
# ANTHONISEN TYPE DERIVATION
# Reference: Anthonisen et al. Ann Intern Med 1987
# ---------------------------------------------------------------------------

def derive_anthonisen_type(data: dict) -> int:
    """
    Derives Anthonisen Type from the three cardinal criteria.
    Type 1: all 3 present → antibiotic benefit clear
    Type 2: any 2 present → moderate benefit
    Type 3: only 1 present → minimal benefit

    Returns 1, 2, or 3.
    Returns None if patient has no obstructive comorbidity (not applicable).
    """
    if not data.get("comorbidity_obstructive", False):
        return None  # Anthonisen only applies to COPD exacerbations

    criteria = [
        data.get("sputum_purulence_change", False),  # criterion 1
        data.get("dyspnea_increase", False),          # criterion 2
        data.get("sputum_volume_increase", False),    # criterion 3
    ]
    n_present = sum(bool(c) for c in criteria)

    if n_present == 3:   return 1
    elif n_present == 2: return 2
    else:                return 3  # 0 or 1


# ---------------------------------------------------------------------------
# SPO2 DISCRETIZER
# Includes ±3% Indian skin tone uncertainty (Sjoding NEJM 2020)
# ---------------------------------------------------------------------------

def discretize_spo2(spo2_pct: Optional[float]) -> Optional[str]:
    """
    Discretizes SpO2 float into BN states.
    Applies ±3% uncertainty for Indian skin tones (Sjoding NEJM 2020).
    Returns None if SpO2 not measured.

    States: "critical" / "low" / "borderline" / "normal"
    Escalation trigger: "critical" (<88%) or "low" (88–92%) → escalate=True
    """
    if spo2_pct is None:
        return None

    # Sjoding NEJM 2020: pulse ox may over-read by ~3% in darker skin tones.
    # Applied UNIFORMLY as a conservative safety measure.
    #
    # CAVEAT (Arvind, pre-freeze): This uniform correction may OVER-CORRECT
    # for patients NOT in the high-melanin risk group. A lighter-skinned patient
    # with SpO2=94% is treated as 91% — potentially triggering unnecessary escalation.
    # Current decision: accept over-escalation (safer) over under-escalation
    # in a CHW setting with limited clinical backup.
    # Review when: (a) skin tone field added to clinical schema, or
    #              (b) TRUPCR data allows calibration by skin tone.
    adjusted = spo2_pct - 3.0  # conservative, uniform — see caveat above

    if adjusted < 88.0:   return "critical"    # escalate regardless of BN output
    elif adjusted < 92.0: return "low"         # escalate
    elif adjusted < 95.0: return "borderline"
    else:                 return "normal"


# ---------------------------------------------------------------------------
# SELF-TEST
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Clinical Schema Validator — Self Test")
    print("=" * 55)

    # --- Test 1: Valid complete input ---
    valid_case = {
        "age": 45, "sex": "F", "bmi": 22.5,
        "fever": True, "symptom_duration_days": 5,
        "cough_character": "productive_purulent",
        "sputum_purulence_change": True,
        "dyspnea_increase": True,
        "sputum_volume_increase": True,
        "spo2_pct": 94.0,
        "fev1_fvc_known": None,
        "comorbidity_obstructive": False,
        "biomass_exposure": True,
        "prior_antibiotic_use": False,
        "season": "winter",
    }
    ok, errors = validate_clinical_input(valid_case)
    assert ok, f"Valid case failed: {errors}"
    print("\nTest 1 — Valid complete input: ✓")

    # --- Test 2: Missing required field ---
    bad_case = valid_case.copy()
    del bad_case["fever"]
    ok, errors = validate_clinical_input(bad_case)
    assert not ok and any("fever" in e for e in errors)
    print("Test 2 — Missing 'fever' caught: ✓")

    # --- Test 3: Wrong type for bool field ---
    bad_case2 = valid_case.copy()
    bad_case2["fever"] = 1  # int, not bool
    ok, errors = validate_clinical_input(bad_case2)
    assert not ok
    print("Test 3 — Bool type error caught: ✓")

    # --- Test 4: SpO2 out of range ---
    bad_case3 = valid_case.copy()
    bad_case3["spo2_pct"] = 110.0
    ok, errors = validate_clinical_input(bad_case3)
    assert not ok and any("spo2_pct" in e for e in errors)
    print("Test 4 — SpO2 range error caught: ✓")

    # --- Test 5: Anthonisen Type derivation ---
    # Type 1: all 3 criteria
    t1 = valid_case.copy()
    t1["comorbidity_obstructive"] = True
    assert derive_anthonisen_type(t1) == 1
    # Type 3: only 1 criterion
    t3 = valid_case.copy()
    t3["comorbidity_obstructive"] = True
    t3["sputum_purulence_change"] = False
    t3["dyspnea_increase"] = False
    t3["sputum_volume_increase"] = True
    assert derive_anthonisen_type(t3) == 3
    # Not applicable (no obstructive comorbidity)
    t_na = valid_case.copy()
    t_na["comorbidity_obstructive"] = False
    assert derive_anthonisen_type(t_na) is None
    print("Test 5 — Anthonisen Type 1/3/None: ✓")

    # --- Test 6: SpO2 discretizer with skin tone correction ---
    assert discretize_spo2(None)    is None
    assert discretize_spo2(99.0)    == "normal"     # 99-3=96 → normal
    assert discretize_spo2(97.0)    == "borderline"  # 97-3=94 → borderline
    assert discretize_spo2(94.0)    == "low"         # 94-3=91 → low
    assert discretize_spo2(90.0)    == "critical"    # 90-3=87 → critical
    print("Test 6 — SpO2 discretizer + Sjoding correction: ✓")

    # --- Test 7: Invalid cough_character ---
    bad_case4 = valid_case.copy()
    bad_case4["cough_character"] = "wet"  # not a valid state
    ok, errors = validate_clinical_input(bad_case4)
    assert not ok
    print("Test 7 — Invalid cough_character caught: ✓")

    print()
    print("=" * 55)
    print("All schema validator tests passed ✓")
    print("=" * 55)
    print()
    print("USAGE:")
    print("  from clinical_schema import validate_clinical_input, derive_anthonisen_type")
    print("  ok, errors = validate_clinical_input(my_vignette)")