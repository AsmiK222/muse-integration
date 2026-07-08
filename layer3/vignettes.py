"""
TreBle Respire — Layer 3: 27-Case Vignette Suite + Sensitivity Analysis
Author: Asmi | Day 5

All 27 vignettes from sprint spec.
Each case: full clinical input dict + Layer 2 mock evidence + expected zone
           + expected uncertainty_flag + pass/fail result.

Target: ≥24/27 passing.

CPT diagnostic protocol: for each failing vignette, remove each input node
one at a time. Node whose removal most reduces error = responsible CPT.
Run this before any CPT tuning.
"""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from clinical_cpts import build_intermediate_cpts
from output_layer import compute_output

# ---------------------------------------------------------------------------
# MOCK LAYER 2 EVIDENCE BUILDER
# Used until Nishevithaa delivers her mock generator on Day 6
# ---------------------------------------------------------------------------

def make_evidence(
    crackle_pattern="absent", crackle_phase=None, crackle_subtype=None,
    wheeze_pattern="absent", wheeze_phase=None, wheeze_compound=None,
    rhonchi="absent", diminished="absent",
    asymmetry="none", gradient="flat",
    zone_reliability="normal", data_quality="normal",
    subtype_modules_run=True, dual_zone_only=False,
):
    return {
        "Crackle_Pattern":            crackle_pattern,
        "Crackle_Phase":              crackle_phase,
        "Crackle_Subtype":            crackle_subtype,
        "Wheeze_Pattern":             wheeze_pattern,
        "Wheeze_Phase":               wheeze_phase,
        "Wheeze_Compound":            wheeze_compound,
        "Rhonchi_Present":            rhonchi,
        "Diminished_Sounds":          diminished,
        "Bilateral_Asymmetry_Score":  asymmetry,
        "Craniocaudal_Gradient":      gradient,
        "Zone_Reliability":           zone_reliability,
        "_data_quality":              data_quality,
        "_subtype_modules_run":       subtype_modules_run,
        "_dual_zone_only":            dual_zone_only,
    }


def make_clinical(
    age=45, sex="M", bmi=23.0,
    fever=False, duration=4,
    cough="dry",
    purulence=False, dyspnea=False, volume=False,
    spo2=97.0, fev1_fvc=None,
    obstructive=False, biomass=False, prior_abx=False,
    season="winter",
):
    return {
        "age": age, "sex": sex, "bmi": bmi,
        "fever": fever, "symptom_duration_days": duration,
        "cough_character": cough,
        "sputum_purulence_change": purulence,
        "dyspnea_increase": dyspnea,
        "sputum_volume_increase": volume,
        "spo2_pct": spo2,
        "fev1_fvc_known": fev1_fvc,
        "comorbidity_obstructive": obstructive,
        "biomass_exposure": biomass,
        "prior_antibiotic_use": prior_abx,
        "season": season,
    }


# ---------------------------------------------------------------------------
# VIGNETTE SUITE — 27 cases
# ---------------------------------------------------------------------------

def build_vignettes() -> list[dict]:
    vignettes = []

    def v(num, desc, evidence, clinical, expected_zone,
          expected_uncertainty=False, expected_escalate=False, notes=""):
        vignettes.append({
            "num": num, "desc": desc,
            "evidence": evidence, "clinical": clinical,
            "expected_zone": expected_zone,
            "expected_uncertainty": expected_uncertainty,
            "expected_escalate": expected_escalate,
            "notes": notes,
        })

    # ------------------------------------------------------------------
    # CASE 1: Classic bacterial — focal coarse crackles R lower, late insp,
    #         fever, asymmetry 0.75, no prior antibiotics
    # Expected: RED
    # ------------------------------------------------------------------
    v(1, "Classic bacterial pneumonia",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype="coarse", asymmetry="high"),
      make_clinical(fever=True, cough="productive_purulent", purulence=True),
      expected_zone="red")

    # ------------------------------------------------------------------
    # CASE 2: Viral — bilateral expiratory wheeze polyphonic, no fever, symmetric
    # Expected: GREEN
    # ------------------------------------------------------------------
    v(2, "Viral LRTI — polyphonic bilateral wheeze, no fever",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="expiratory",
                    wheeze_compound="polyphonic_low"),
      make_clinical(fever=False, cough="dry"),
      expected_zone="green")

    # ------------------------------------------------------------------
    # CASE 3: COPD + bacterial — expiratory wheeze + purulent sputum, Anthonisen Type 1
    # Expected: RED
    # ------------------------------------------------------------------
    v(3, "COPD exacerbation + bacterial — Anthonisen Type 1",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="expiratory",
                    wheeze_compound="polyphonic_low", crackle_pattern="focal",
                    crackle_phase="late_inspiratory"),
      make_clinical(fever=True, cough="productive_purulent", purulence=True,
                    dyspnea=True, volume=True, obstructive=True, fev1_fvc=0.58),
      expected_zone="red")

    # ------------------------------------------------------------------
    # CASE 4: COPD no infection — wheeze only, no purulence, Anthonisen Type 3
    # Expected: GREEN or AMBER
    # ------------------------------------------------------------------
    v(4, "COPD no infection — wheeze only, Anthonisen Type 3",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="expiratory",
                    wheeze_compound="polyphonic_low"),
      make_clinical(obstructive=True, fev1_fvc=0.60, cough="productive_mucoid"),
      expected_zone="green",  # accepts amber too — see check logic
      notes="acceptable: green or amber")

    # ------------------------------------------------------------------
    # CASE 5: SpO2 88% — escalate regardless of findings
    # Expected: any zone + escalate=True
    # ------------------------------------------------------------------
    v(5, "SpO2 88% — escalate regardless",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory"),
      make_clinical(fever=True, spo2=88.0),
      expected_zone=None,  # any zone acceptable
      expected_escalate=True,
      notes="zone irrelevant — escalate=True is the key check")

    # ------------------------------------------------------------------
    # CASE 6: Atypical bacterial (Mycoplasma) — focal, low fever, young adult
    # Expected: AMBER or RED
    # ------------------------------------------------------------------
    v(6, "Atypical bacterial — Mycoplasma pattern",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype="fine"),
      make_clinical(age=28, fever=True, cough="dry", duration=7, season="winter"),
      expected_zone="amber")

    # ------------------------------------------------------------------
    # CASE 7: Bilateral basal late_insp fine crackles, no fever, elderly
    # Expected: AMBER + uncertainty_flag (CHF/pneumonia differential)
    # ------------------------------------------------------------------
    v(7, "Bilateral basal late_insp fine crackles, elderly, no fever",
      make_evidence(crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
                    crackle_subtype="fine", gradient="basal"),
      make_clinical(age=72, fever=False, cough="dry"),
      expected_zone="amber", expected_uncertainty=True,
      notes="CHF/pneumonia differential — uncertainty expected")

    # ------------------------------------------------------------------
    # CASE 8: Same as case 1 but prior_antibiotic=True
    # Expected: AMBER (wider CI vs case 1, lower p_antibiotics)
    # ------------------------------------------------------------------
    v(8, "Classic bacterial + prior antibiotics (same as case 1)",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype="coarse", asymmetry="high"),
      make_clinical(fever=True, cough="productive_purulent", purulence=True, prior_abx=True),
      expected_zone="amber",
      notes="Prior antibiotics reduce q_i by 0.70 — expect lower than case 1")

    # ------------------------------------------------------------------
    # CASE 9: Biomass-exposed female, wheezy, no purulence
    # Expected: GREEN or AMBER
    # ------------------------------------------------------------------
    v(9, "Biomass-exposed female, wheeze, no purulence",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="expiratory",
                    wheeze_compound="polyphonic_low"),
      make_clinical(sex="F", biomass=True, cough="productive_mucoid"),
      expected_zone="green",
      notes="acceptable: green or amber")

    # ------------------------------------------------------------------
    # CASE 10: All inputs missing except SpO2=96%
    # Expected: AMBER + uncertainty_flag
    # ------------------------------------------------------------------
    v(10, "All inputs missing except SpO2=96%",
      make_evidence(),  # all absent/None
      make_clinical(spo2=96.0, fever=False, cough="dry"),
      expected_zone="amber", expected_uncertainty=True,
      notes=">2 key inputs missing → uncertainty_flag")

    # ------------------------------------------------------------------
    # CASES 11–15: Missing data variants — CI widens progressively
    # ------------------------------------------------------------------
    v(11, "No SpO2",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory"),
      make_clinical(fever=True, spo2=None),
      expected_zone="amber",
      notes="CI wider than full data")

    v(12, "No spirometry",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="expiratory"),
      make_clinical(obstructive=True, fev1_fvc=None),
      expected_zone="green",
      notes="Minimal CI change vs full data")

    v(13, "4 zones missing — partial recording",
      make_evidence(crackle_pattern=None, wheeze_pattern=None,
                    asymmetry=None, gradient=None),
      make_clinical(fever=True, cough="productive_purulent"),
      expected_zone="amber", expected_uncertainty=True,
      notes="CI widens proportionally with missing acoustic data")

    v(14, "8 zones missing — poor recording",
      make_evidence(crackle_pattern=None, crackle_phase=None,
                    wheeze_pattern=None, wheeze_phase=None,
                    asymmetry=None, gradient=None,
                    data_quality="poor"),
      make_clinical(fever=True),
      expected_zone="amber", expected_uncertainty=True)

    v(15, "subtype_modules_run=False — no compound wheeze info",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype=None, wheeze_compound=None,
                    subtype_modules_run=False),
      make_clinical(fever=True, cough="productive_purulent"),
      expected_zone="amber",
      notes="subtype_modules_run=False → marginalize, not uncertain")

    # ------------------------------------------------------------------
    # CASE 16: Velcro crackles bilateral, no fever, chronic dyspnea
    # Expected: AMBER + uncertainty_flag — P_ILD_Flag high, do NOT output red
    # ------------------------------------------------------------------
    v(16, "Velcro bilateral crackles — ILD differential",
      make_evidence(crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
                    crackle_subtype="velcro", gradient="basal"),
      make_clinical(fever=False, cough="dry", dyspnea=True),
      expected_zone="amber", expected_uncertainty=True,
      notes="ILD flag suppresses red — must not output red")

    # ------------------------------------------------------------------
    # CASE 17: Monophonic_low wheeze focal Z05 only, no crackles, no fever
    # Expected: GREEN + investigation flag (endobronchial, not bacterial)
    # ------------------------------------------------------------------
    v(17, "Monophonic low wheeze focal — endobronchial pattern",
      make_evidence(wheeze_pattern="focal", wheeze_phase="inspiratory",
                    wheeze_compound="monophonic_low"),
      make_clinical(fever=False, cough="dry"),
      expected_zone="green",
      notes="Endobronchial suppression — green + investigation flag")

    # ------------------------------------------------------------------
    # CASE 18: polyphonic_pan + purulent sputum + fever + Anthonisen Type 1
    # Expected: RED — Anthonisen overrides, infected severe COPD
    # ------------------------------------------------------------------
    v(18, "Severe COPD exacerbation — polyphonic_pan + Anthonisen Type 1",
      make_evidence(wheeze_pattern="bilateral", wheeze_phase="biphasic",
                    wheeze_compound="polyphonic_pan", crackle_pattern="focal",
                    crackle_phase="late_inspiratory"),
      make_clinical(fever=True, cough="productive_purulent", purulence=True,
                    dyspnea=True, volume=True, obstructive=True, fev1_fvc=0.48),
      expected_zone="red")

    # ------------------------------------------------------------------
    # CASE 19: Early insp coarse crackles R upper, fever, high R-L asymmetry
    # Expected: AMBER or RED
    # ------------------------------------------------------------------
    v(19, "Early insp coarse crackles R upper + fever + asymmetry",
      make_evidence(crackle_pattern="focal", crackle_phase="early_inspiratory",
                    crackle_subtype="coarse", asymmetry="high", gradient="apical"),
      make_clinical(fever=True, cough="productive_purulent"),
      expected_zone="amber",
      notes="Early insp less specific than late — amber expected")

    # ------------------------------------------------------------------
    # CASE 20: Conflicting signals — late_insp crackles + expiratory polyphonic wheeze
    # Expected: AMBER + uncertainty_flag
    # ------------------------------------------------------------------
    v(20, "Conflicting signals — crackles + polyphonic wheeze simultaneously",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype="coarse",
                    wheeze_pattern="bilateral", wheeze_phase="expiratory",
                    wheeze_compound="polyphonic_low"),
      make_clinical(fever=True, cough="productive_purulent", obstructive=True),
      expected_zone="amber", expected_uncertainty=True,
      notes="Both consolidation and obstructive signals — uncertainty expected")

    # ------------------------------------------------------------------
    # CASE 21: All abnormal findings Z06/Z08 only (dual zones)
    # Expected: AMBER — Zone_Reliability=reduced, CI widens
    # ------------------------------------------------------------------
    v(21, "All abnormal findings in dual-representation zones only",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    zone_reliability="reduced", dual_zone_only=True),
      make_clinical(fever=True),
      expected_zone="amber", expected_uncertainty=True,
      notes="Zone_Reliability=reduced → CI widens → amber")

    # ------------------------------------------------------------------
    # CASES 22–27: Edge cases
    # ------------------------------------------------------------------
    v(22, "All inputs missing",
      make_evidence(),
      make_clinical(fever=False, cough="dry", spo2=97.0),
      expected_zone="amber", expected_uncertainty=True,
      notes="No evidence → prior only → amber + uncertainty")

    v(23, "Maximally conflicting signals",
      make_evidence(crackle_pattern="focal", crackle_phase="late_inspiratory",
                    crackle_subtype="velcro",
                    wheeze_pattern="bilateral", wheeze_phase="biphasic",
                    wheeze_compound="polyphonic_pan",
                    asymmetry="high", gradient="basal"),
      make_clinical(fever=True, purulence=True, dyspnea=True, obstructive=True),
      expected_zone="amber", expected_uncertainty=True,
      notes="ILD + consolidation + obstructive all active — amber + uncertainty")

    v(24, "Single input only — fever alone",
      make_evidence(),
      make_clinical(fever=True, cough="dry"),
      expected_zone="amber", expected_uncertainty=True,
      notes="One clinical input — wide CI, uncertainty")

    v(25, "Post-antibiotic + ambiguous acoustics",
      make_evidence(crackle_pattern="focal", crackle_phase="early_inspiratory",
                    crackle_subtype=None, wheeze_compound=None,
                    subtype_modules_run=False),
      make_clinical(fever=False, prior_abx=True, cough="productive_mucoid"),
      expected_zone="amber", expected_uncertainty=True,
      notes="Prior antibiotics + no subtype data → wide CI")

    v(26, "Elderly + all comorbidities",
      make_evidence(crackle_pattern="bilateral_basal", crackle_phase="late_inspiratory",
                    crackle_subtype="fine", gradient="basal"),
      make_clinical(age=80, obstructive=True, biomass=True,
                    fever=True, cough="productive_purulent",
                    purulence=True, dyspnea=True, volume=True,
                    fev1_fvc=0.55),
      expected_zone="red",
      notes="Multiple risk factors converge — red")

    v(27, "Apical crackles — TB pattern",
      make_evidence(crackle_pattern="focal", crackle_phase="early_inspiratory",
                    crackle_subtype="coarse", gradient="apical", asymmetry="mild"),
      make_clinical(age=35, fever=True, cough="productive_purulent",
                    duration=14, season="other"),
      expected_zone="amber", expected_uncertainty=True,
      notes="Apical pattern — TB differential raises uncertainty")

    return vignettes


# ---------------------------------------------------------------------------
# VIGNETTE RUNNER
# ---------------------------------------------------------------------------

def run_vignettes(vignettes: list[dict], intermediate_cpts: dict) -> dict:
    passed = 0
    failed = 0
    failures = []

    print("\n" + "=" * 70)
    print(f"{'#':>3}  {'Description':<42} {'Zone':>6} {'Exp':>6}  {'Unc':>4} {'Esc':>4}  {'Result'}")
    print("-" * 70)

    for v in vignettes:
        out = compute_output(v["evidence"], v["clinical"], intermediate_cpts)
        actual_zone = out["zone"]
        actual_unc  = out["uncertainty_flag"]
        actual_esc  = out["escalate"]

        # Zone check — some cases accept two zones
        exp_zone = v["expected_zone"]
        notes = v.get("notes", "")
        if exp_zone is None:
            zone_ok = True  # any zone acceptable
        elif "green or amber" in notes or "acceptable: green or amber" in notes:
            zone_ok = actual_zone in ("green", "amber")
        elif "amber or red" in notes:
            zone_ok = actual_zone in ("amber", "red")
        elif "amber/red" in notes:
            zone_ok = actual_zone in ("amber", "red")
        else:
            zone_ok = actual_zone == exp_zone

        # Escalate check
        esc_ok = (not v["expected_escalate"]) or actual_esc

        # Uncertainty check — if expected True, actual must be True
        unc_ok = (not v["expected_uncertainty"]) or actual_unc

        # Special rule: case 16 (velcro) must NOT be red
        if v["num"] == 16 and actual_zone == "red":
            zone_ok = False

        case_passed = zone_ok and esc_ok and unc_ok

        if case_passed:
            passed += 1
            result = "✓ PASS"
        else:
            failed += 1
            result = "✗ FAIL"
            failures.append({
                "num": v["num"], "desc": v["desc"],
                "expected_zone": exp_zone, "actual_zone": actual_zone,
                "expected_unc": v["expected_uncertainty"], "actual_unc": actual_unc,
                "expected_esc": v["expected_escalate"], "actual_esc": actual_esc,
                "output": out,
            })

        exp_z = exp_zone or "any"
        print(f"  {v['num']:>2}  {v['desc']:<42} {actual_zone:>6} {exp_z:>6}  "
              f"{'T' if actual_unc else 'F':>4} {'T' if actual_esc else 'F':>4}  {result}")

    print("-" * 70)
    print(f"\nTotal: {passed}/{len(vignettes)} passing  "
          f"({'✓ TARGET MET' if passed >= 24 else '✗ BELOW TARGET — need 24/27'})")

    return {"passed": passed, "failed": failed, "failures": failures,
            "total": len(vignettes)}


# ---------------------------------------------------------------------------
# CPT DIAGNOSTIC PROTOCOL
# For failing vignettes: remove each input one at a time,
# find which removal most reduces the error
# ---------------------------------------------------------------------------

def diagnose_failures(failures: list[dict], intermediate_cpts: dict):
    if not failures:
        print("\nNo failures to diagnose.")
        return

    print("\n" + "=" * 60)
    print("CPT Diagnostic Protocol")
    print("=" * 60)

    all_inputs = [
        "Crackle_Pattern", "Crackle_Phase", "Crackle_Subtype",
        "Wheeze_Pattern", "Wheeze_Phase", "Wheeze_Compound",
        "Bilateral_Asymmetry_Score", "Diminished_Sounds",
        "Craniocaudal_Gradient", "Zone_Reliability",
        "fever", "spo2_pct", "comorbidity_obstructive",
        "sputum_purulence_change", "dyspnea_increase",
    ]

    for f in failures:
        print(f"\nCase {f['num']}: {f['desc']}")
        print(f"  Expected zone: {f['expected_zone']} | Actual: {f['actual_zone']}")
        print(f"  P(antibiotics) = {f['output']['p_antibiotics']}")

        target_p = {"red": 0.65, "amber": 0.40, "green": 0.15}
        target = target_p.get(f["expected_zone"], 0.40)
        current_error = abs(f["output"]["p_antibiotics"] - target)

        deltas = []
        for inp in all_inputs:
            if "evidence" not in f:
                print(f"  (evidence dict not available)")
                continue
            ev_mod = dict(f["evidence"])
            cl_mod = dict(f["clinical"])
            if inp in ev_mod:
                ev_mod[inp] = None
            elif inp in cl_mod:
                cl_mod[inp] = None

            out_mod = compute_output(ev_mod, cl_mod, intermediate_cpts)
            new_error = abs(out_mod["p_antibiotics"] - target)
            delta_error = current_error - new_error  # positive = removal helps
            deltas.append((inp, delta_error, out_mod["p_antibiotics"]))

        deltas.sort(key=lambda x: x[1], reverse=True)
        print(f"  Responsible CPT (removal most reduces error):")
        for inp, delta, p_new in deltas[:3]:
            direction = "↓ error" if delta > 0 else "↑ error"
            print(f"    {inp:<35} {direction} by {abs(delta):.3f} → p={p_new:.3f}")


# ---------------------------------------------------------------------------
# SENSITIVITY ANALYSIS — tornado diagram data
# ---------------------------------------------------------------------------

def sensitivity_analysis(intermediate_cpts: dict) -> dict:
    """
    Vary each q_i ±20%, measure output change.
    Flag parameters where ±20% flips output zone.
    """
    print("\n" + "=" * 60)
    print("Sensitivity Analysis — q_i ±20%")
    print("=" * 60)

    # Base case: classic bacterial (vignette 1 equivalent)
    base_evidence = make_evidence(
        crackle_pattern="focal", crackle_phase="late_inspiratory",
        crackle_subtype="coarse", asymmetry="high"
    )
    base_clinical = make_clinical(
        fever=True, cough="productive_purulent", purulence=True
    )
    base_out = compute_output(base_evidence, base_clinical, intermediate_cpts)
    base_p = base_out["p_antibiotics"]
    base_zone = base_out["zone"]

    # All q_i values to test
    qi_params = {
        **intermediate_cpts["P_Consolidation"]["qi_values"],
        **intermediate_cpts["P_Obstructive_Exacerbation"]["qi_values"],
        **intermediate_cpts["P_ILD_Flag"]["qi_values"],
    }

    results = []
    fragile = []

    for param_name, base_qi in qi_params.items():
        sensitivities = []
        for direction, multiplier in [("up", 1.20), ("down", 0.80)]:
            mod_qi = float(np.clip(base_qi * multiplier, 0.01, 0.99))
            # Patch the q_i and re-run
            intermediate_mod = {
                "P_Consolidation": dict(intermediate_cpts["P_Consolidation"]),
                "P_Obstructive_Exacerbation": dict(intermediate_cpts["P_Obstructive_Exacerbation"]),
                "P_ILD_Flag": dict(intermediate_cpts["P_ILD_Flag"]),
                "P_Severity": intermediate_cpts["P_Severity"],
                "Anthonisen_Type": intermediate_cpts["Anthonisen_Type"],
            }
            for node in ["P_Consolidation", "P_Obstructive_Exacerbation", "P_ILD_Flag"]:
                if param_name in intermediate_mod[node].get("qi_values", {}):
                    intermediate_mod[node]["qi_values"] = dict(intermediate_cpts[node]["qi_values"])
                    intermediate_mod[node]["qi_values"][param_name] = mod_qi

            out_mod = compute_output(base_evidence, base_clinical, intermediate_mod)
            p_mod = out_mod["p_antibiotics"]
            zone_flip = out_mod["zone"] != base_zone
            sensitivities.append((direction, mod_qi, p_mod, zone_flip))
            if zone_flip:
                fragile.append((param_name, direction, base_qi, mod_qi, base_zone, out_mod["zone"]))

        delta_up   = sensitivities[0][2] - base_p
        delta_down = sensitivities[1][2] - base_p
        results.append((param_name, base_qi, delta_up, delta_down))

    # Sort by max absolute delta
    results.sort(key=lambda x: max(abs(x[2]), abs(x[3])), reverse=True)

    print(f"\nBase P(antibiotics) = {base_p:.3f}  |  Base zone = {base_zone}")
    print(f"\n{'Parameter':<40} {'Base q_i':>8}  {'Δ +20%':>8}  {'Δ -20%':>8}")
    print("-" * 70)
    for param, base_qi, d_up, d_down in results[:15]:
        print(f"  {param:<38} {base_qi:>8.3f}  {d_up:>+8.3f}  {d_down:>+8.3f}")

    print(f"\nFRAGILE PARAMETERS (±20% flips output zone):")
    if fragile:
        for param, direction, qi_old, qi_new, z_old, z_new in fragile:
            print(f"  {param} ({direction} {qi_old:.2f}→{qi_new:.2f}): {z_old}→{z_new}")
        print("\n  NOTE: Phase and Wheeze_Compound q_i values are most likely fragile.")
        print("  This is expected given their qualitative basis.")
        print("  These are the FIRST to update when TRUPCR data arrives.")
    else:
        print("  No parameters flip zone at ±20% on the base case tested.")
        print("  CAVEAT: Sensitivity was tested on a strong-signal case (classic bacterial).")
        print("  Amber boundary cases (p_antibiotics 0.20–0.45) are likely more sensitive.")
        print("  Consol_gate thresholds were NOT swept — these are the most clinically")
        print("  loaded parameters and were not part of the ±20% analysis.")
        print("  Phase and Wheeze_Compound q_i values remain qualitative regardless")
        print("  of this result — they are still FIRST to update with TRUPCR data.")

    return {"results": results, "fragile": fragile}


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    intermediate_cpts = build_intermediate_cpts()
    vignettes = build_vignettes()

    print("=" * 70)
    print(f"TreBle Respire — Layer 3 Vignette Suite ({len(vignettes)} cases)")
    print("=" * 70)

    results = run_vignettes(vignettes, intermediate_cpts)

    if results["failures"]:
        diagnose_failures(results["failures"], intermediate_cpts)

    sensitivity = sensitivity_analysis(intermediate_cpts)

    print("\n" + "=" * 70)
    print("DAY 5 SUMMARY")
    print("=" * 70)
    print(f"  Vignettes passing : {results['passed']}/{results['total']}")
    print(f"  Target            : 24/27")
    print(f"  Status            : {'✓ TARGET MET' if results['passed'] >= 24 else '⚠ BELOW TARGET'}")
    print(f"  Fragile params    : {len(sensitivity['fragile'])}")

    # List failed vignettes explicitly
    if results["failures"]:
        print(f"\n  FAILED VIGNETTES ({len(results['failures'])}):")
        for f in results["failures"]:
            print(f"    Case {f['num']:>2}: {f['desc'][:50]}")
            print(f"           Expected zone={f['expected_zone']} | "
                  f"Got zone={f['actual_zone']} | "
                  f"p_abx={f['output']['p_antibiotics']:.3f}")
            print(f"           Expected unc={f['expected_unc']} | Got unc={f['actual_unc']}")
    else:
        print(f"\n  No failed vignettes.")

    print(f"\n  → Send fragile parameters list to Arvind.")
    print(f"  → Phase and Wheeze_Compound q_i values flagged for TRUPCR update.")
    print(f"  → Consol_gate thresholds need clinician review Days 13-14 (not swept in sensitivity).")