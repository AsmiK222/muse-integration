"""
TreBle Respire — Layer 3: Output Layer + Layer 2 Input Mapping
Author: Asmi | Day 4

Implements:
  1. Full output layer (p_antibiotics, CI, zone, uncertainty_flag,
     escalation overlay, disposition, contributing_factors)
  2. map_layer2_to_bn() — bridge from Nishevithaa's Layer 2 output
  3. Missing data degradation hierarchy
  4. Contributing factors via node marginalization

Output schema (locked at bn_version 2.1.0):
  p_antibiotics       float [0,1]
  ci_lower, ci_upper  float [0,1]   95% credible interval
  zone                str            green / amber / red
  uncertainty_flag    bool
  escalate            bool           True if SpO2 <92% regardless of p_antibiotics
  disposition         str            home_safety_net / review_48h / refer
  contributing_factors list[(name, delta)]
  bn_version          str            "2.1.0"

CRITICAL field name: wheeze_compound (NOT wheeze_character — removed from Layer 2)
"""

import numpy as np
from typing import Optional
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from clinical_schema import discretize_spo2
from clinical_cpts import build_intermediate_cpts, apply_antibiotic_modifier

# ---------------------------------------------------------------------------
# VALID WHEEZE COMPOUND STATES — must match Layer 2 contract exactly
# ---------------------------------------------------------------------------
VALID_WHEEZE_COMPOUND_STATES = [
    "monophonic_low", "monophonic_mid", "monophonic_high",
    "polyphonic_low", "polyphonic_mid", "polyphonic_pan", "uncertain",
]

BN_VERSION = "2.1.0"

# ---------------------------------------------------------------------------
# ZONE THRESHOLDS
# ---------------------------------------------------------------------------
ZONE_GREEN_UPPER = 0.20
ZONE_AMBER_UPPER = 0.60
# >0.60 = red


def compute_zone(p_antibiotics: float) -> str:
    if p_antibiotics < ZONE_GREEN_UPPER:
        return "green"
    elif p_antibiotics <= ZONE_AMBER_UPPER:
        return "amber"
    else:
        return "red"


# ---------------------------------------------------------------------------
# LAYER 2 INPUT MAPPING
# Consumes Nishevithaa's output tuple → BN evidence dict
# CRITICAL: use wheeze_compound NOT wheeze_character (field removed)
# ---------------------------------------------------------------------------

def discretize_asymmetry(flags: dict) -> Optional[str]:
    """
    Discretizes bilateral asymmetry score from Layer 2 flags.
    Layer 2 outputs a float; BN needs discrete states.
    """
    score = flags.get("bilateral_asymmetry_score_max")
    if score is None:
        return None
    if score < 0.2:   return "none"
    elif score < 0.5: return "mild"
    elif score < 0.75:return "moderate"
    else:             return "high"


def discretize_gradient(flags: dict) -> Optional[str]:
    """
    Discretizes craniocaudal gradient from Layer 2 flags.
    """
    gradient = flags.get("craniocaudal_gradient_dominant")
    if gradient is None:
        return None
    valid = ["flat", "apical", "basal"]
    return gradient if gradient in valid else "flat"


def map_layer2_to_bn(layer2_output: dict) -> dict:
    """
    Maps Layer 2 output tuple to BN evidence dict.

    Key rules:
      - wheeze_compound field: use wheeze_compound_dominant, NOT wheeze_character
      - subtype_modules_run=False → pass None (marginalize), NOT "uncertain"
        None = absence of evidence. "uncertain" = evidence of ambiguity. DIFFERENT.
      - dual-zone-only findings → Zone_Reliability = "reduced"
      - is_dual_representation_zone is a static flag (always true for Z06, Z08)

    Args:
        layer2_output: Full Layer 2 output dict from Nishevithaa

    Returns:
        evidence dict for BN inference (None values = marginalized nodes)
    """
    flags = layer2_output.get("summary_flags", {})
    meta  = layer2_output.get("output_meta", {})
    zones = layer2_output.get("zone_findings", [])

    # --- Wheeze compound ---
    # CRITICAL: field is wheeze_compound_dominant, NOT wheeze_character
    wheeze_compound = flags.get("wheeze_compound_dominant", "uncertain")
    if wheeze_compound not in VALID_WHEEZE_COMPOUND_STATES:
        wheeze_compound = "uncertain"

    # --- Dual-zone handling ---
    # If ALL abnormal recorded zones are dual-representation zones → reduced reliability
    abnormal_recorded = [
        z for z in zones
        if z.get("recorded", False) and z.get("any_abnormality", False)
    ]
    if abnormal_recorded:
        dual_zone_only = all(z.get("is_dual_representation_zone", False) for z in abnormal_recorded)
    else:
        dual_zone_only = False
    zone_reliability = "reduced" if dual_zone_only else "normal"

    # --- Subtype modules ---
    # subtype_modules_run=False → marginalize (None), NOT set to "uncertain"
    # None passed to BN = absence of evidence
    # "uncertain" passed to BN = evidence of ambiguity — completely different meaning
    subtype_modules_run = meta.get("subtype_modules_run", False)
    crackle_subtype = (
        flags.get("crackle_subtype_dominant") if subtype_modules_run else None
    )
    wheeze_compound_val = (
        wheeze_compound if subtype_modules_run else None
    )

    # --- Data quality ---
    data_quality = meta.get("data_quality", "normal")

    # --- Build evidence dict ---
    evidence = {
        # Acoustic nodes from Layer 2
        "Crackle_Pattern":           _map_crackle_pattern(flags),
        "Crackle_Phase":             flags.get("crackle_phase_dominant"),
        "Crackle_Subtype":           crackle_subtype,
        "Wheeze_Pattern":            _map_wheeze_pattern(flags),
        "Wheeze_Phase":              flags.get("wheeze_phase_dominant"),
        "Wheeze_Compound":           wheeze_compound_val,
        "Rhonchi_Present":           "present" if flags.get("rhonchi_present") else "absent",
        "Diminished_Sounds":         _map_diminished(flags),
        "Bilateral_Asymmetry_Score": discretize_asymmetry(flags),
        "Craniocaudal_Gradient":     discretize_gradient(flags),
        "Zone_Reliability":          zone_reliability,

        # Metadata
        "_data_quality":             data_quality,
        "_subtype_modules_run":      subtype_modules_run,
        "_dual_zone_only":           dual_zone_only,
    }

    return evidence


def _map_crackle_pattern(flags: dict) -> Optional[str]:
    if not flags.get("crackles_present"):
        return "absent"
    if flags.get("crackles_focal"):
        return "focal"
    if flags.get("crackles_bilateral"):
        return "bilateral_basal"
    return "diffuse"


def _map_wheeze_pattern(flags: dict) -> Optional[str]:
    if not flags.get("wheeze_present"):
        return "absent"
    if flags.get("wheeze_focal"):
        return "focal"
    if flags.get("wheeze_bilateral"):
        return "bilateral"
    return "diffuse"


def _map_diminished(flags: dict) -> str:
    if not flags.get("diminished_present"):
        return "absent"
    if flags.get("diminished_unilateral"):
        return "unilateral"
    return "bilateral"


# ---------------------------------------------------------------------------
# NOISY-OR INFERENCE ENGINE
# Lightweight posterior computation without full pgmpy query
# (Full pgmpy integration in attach_cpts.py once all CPTs confirmed)
# ---------------------------------------------------------------------------

def compute_posterior(
    evidence: dict,
    clinical_input: dict,
    intermediate_cpts: dict,
) -> dict:
    """
    Computes P(Antibiotics_Indicated) using Noisy-OR inference.

    This is a lightweight implementation for Days 4–5.
    Full pgmpy VariableElimination attached Day 9 after vignette validation.

    Returns raw posterior dict before output layer processing.
    """
    consol_qi = intermediate_cpts["P_Consolidation"]["qi_values"]
    obstruct_qi = intermediate_cpts["P_Obstructive_Exacerbation"]["qi_values"]
    ild_qi = intermediate_cpts["P_ILD_Flag"]["qi_values"]

    prior_abx = clinical_input.get("prior_antibiotic_use", False)

    def noisy_or(base: float, active_qis: list) -> float:
        p_low = 1.0 - base
        for qi in active_qis:
            qi_mod = apply_antibiotic_modifier(qi, prior_abx)
            p_low *= (1.0 - qi_mod)
        return 1.0 - p_low

    # --- P_Consolidation ---
    active_consol = []
    cp = evidence.get("Crackle_Pattern")
    if cp == "focal":           active_consol.append(consol_qi["Crackle_Pattern_focal"])
    elif cp == "bilateral_basal":active_consol.append(consol_qi["Crackle_Pattern_bilateral_basal"])
    elif cp == "diffuse":       active_consol.append(consol_qi["Crackle_Pattern_diffuse"])
    if evidence.get("Crackle_Phase") == "late_inspiratory":
        active_consol.append(consol_qi["Crackle_Phase_late_insp"])
    if evidence.get("Crackle_Phase") == "early_inspiratory":
        active_consol.append(consol_qi["Crackle_Phase_early_insp"])
    if evidence.get("Crackle_Subtype") == "coarse":
        active_consol.append(consol_qi["Crackle_Subtype_coarse"])
    if evidence.get("Diminished_Sounds") == "unilateral":
        active_consol.append(consol_qi["Diminished_unilateral"])
    elif evidence.get("Diminished_Sounds") == "bilateral":
        active_consol.append(0.40)   # qualitative — bilateral diminished, less specific than unilateral
    asym_score = evidence.get("Bilateral_Asymmetry_Score")
    if asym_score == "high":
        active_consol.append(consol_qi["Asymmetry_high"])     # 0.85 — Wipf 1999 LR+ 44.1
    elif asym_score == "moderate":
        active_consol.append(0.45)   # qualitative — moderate asymmetry (LR ~1.8 estimated)
    elif asym_score == "mild":
        active_consol.append(0.23)   # qualitative — mild asymmetry (LR ~1.3 estimated)
    if clinical_input.get("fever"):
        active_consol.append(consol_qi["Fever_present"])
    if clinical_input.get("cough_character") == "productive_purulent":
        active_consol.append(0.55)  # estimated

    p_consolidation = noisy_or(0.10, active_consol)  # Indian prior: ~10% baseline consolidation
    # Dampen when evidence is correlated (crackle pattern + phase share same auscultation event)
    # Noisy-OR overcounts correlated features — Heckerman 1995 known limitation
    # Only apply dampening when multiple crackle sub-features are active (not independent)
    n_crackle_features = sum([
        1 if evidence.get("Crackle_Pattern") in ("focal","bilateral_basal","diffuse") else 0,
        1 if evidence.get("Crackle_Phase") in ("late_inspiratory","early_inspiratory") else 0,
        1 if evidence.get("Crackle_Subtype") in ("coarse","fine","velcro") else 0,
    ])
    if n_crackle_features >= 3:
        p_consolidation = min(p_consolidation, 0.85)   # 3 correlated features — dampen slightly
    elif n_crackle_features == 2:
        p_consolidation = min(p_consolidation, 0.90)   # 2 correlated features — minimal dampening

    # --- P_Obstructive ---
    # Wheeze_Pattern wired in per Arvind decision — feeds P_Obstructive AND
    # the focal monophonic suppression path. Not wiring this in was a bug.
    active_obstruct = []
    if clinical_input.get("comorbidity_obstructive"):
        active_obstruct.append(obstruct_qi["Obstructive_Dx_true"])
    wp_phase = evidence.get("Wheeze_Phase")
    if wp_phase == "expiratory":  active_obstruct.append(obstruct_qi["Wheeze_Phase_expiratory"])
    elif wp_phase == "biphasic":  active_obstruct.append(obstruct_qi["Wheeze_Phase_biphasic"])
    wc = evidence.get("Wheeze_Compound")
    wc_map = {
        "polyphonic_pan": obstruct_qi["Wheeze_Compound_polyphonic_pan"],
        "polyphonic_low": obstruct_qi["Wheeze_Compound_polyphonic_low"],
        "monophonic_low": obstruct_qi["Wheeze_Compound_monophonic_low"],
    }
    if wc in wc_map:
        active_obstruct.append(wc_map[wc])

    # Wheeze_Pattern — fallback obstructive evidence when Wheeze_Compound is
    # unavailable (subtype_modules_run=False). Wheeze_Pattern is coarser
    # (focal/bilateral/diffuse spatial spread) than Wheeze_Compound
    # (morphology x freq_band), but is the only wheeze signal available
    # when phase/subtype modules have not run. qualitative — update with TRUPCR.
    wp_pattern = evidence.get("Wheeze_Pattern")
    if wc is None:   # only use Wheeze_Pattern when Wheeze_Compound unavailable
        if wp_pattern == "bilateral":
            active_obstruct.append(0.55)  # qualitative — bilateral wheeze, obstructive pattern
        elif wp_pattern == "diffuse":
            active_obstruct.append(0.60)  # qualitative — diffuse wheeze, strong obstructive signal
        elif wp_pattern == "focal":
            active_obstruct.append(0.35)  # qualitative — focal wheeze, weaker obstructive signal

    if clinical_input.get("biomass_exposure"):
        active_obstruct.append(obstruct_qi["Biomass_Exposure_true"])

    p_obstructive = noisy_or(0.25, active_obstruct)

    # --- Endobronchial suppression ---
    # Focal monophonic suppression path (Arvind decision 1):
    # When Wheeze_Compound is unavailable but Wheeze_Pattern=focal is present,
    # apply a partial suppression — focal wheeze without polyphonic/diffuse
    # spread is more consistent with fixed endobronchial obstruction than
    # diffuse small-airway disease. Weaker than full monophonic_low suppression
    # (0.40) since Wheeze_Pattern carries less morphological information.
    endobronchial_suppression = 0.0
    if evidence.get("Wheeze_Compound") == "monophonic_low":
        endobronchial_suppression = 0.40  # Sovijarvi 2000
    elif wc is None and wp_pattern == "focal":
        endobronchial_suppression = 0.20  # qualitative — partial, Wheeze_Pattern only

    # --- P_ILD_Flag ---
    # ILD requires VELCRO crackles as necessary condition (Bohadana NEJM 2014)
    # Late_insp phase alone is non-specific — only counts when velcro is also present
    # Clinical: late_insp fine crackles in pneumonia are common, not ILD-specific
    active_ild = []
    has_velcro = evidence.get("Crackle_Subtype") == "velcro"
    has_late_insp_crackle = evidence.get("Crackle_Phase") == "late_inspiratory"
    has_basal_gradient = evidence.get("Craniocaudal_Gradient") == "basal"

    if has_velcro:
        active_ild.append(ild_qi["Crackle_Subtype_velcro"])
        # Late_insp only contributes when velcro is also present (corroborating pattern)
        if has_late_insp_crackle:
            active_ild.append(ild_qi["Crackle_Phase_late_insp"])
    if has_basal_gradient:
        active_ild.append(ild_qi["Craniocaudal_basal"])

    p_ild = noisy_or(0.08, active_ild)

    # --- Anthonisen contribution ---
    from clinical_schema import derive_anthonisen_type
    anthonisen = derive_anthonisen_type(clinical_input)
    anthonisen_qi = {1: 0.90, 2: 0.60, 3: 0.20, None: 0.0}.get(anthonisen, 0.0)

    # --- P_Antibiotics_Indicated ---
    # KEY CLINICAL LOGIC:
    # Consolidation pathway → antibiotics: direct (bacterial pneumonia)
    # Obstructive pathway → antibiotics: ONLY with Anthonisen criteria
    #   wheeze alone (no purulence/fever/dyspnea) does NOT indicate antibiotics
    #   Metlay JAMA 1997: wheeze LR ~1.0 for bacterial — non-discriminatory
    # ILD flag suppresses overconfident bacterial classification

    ild_suppression = p_ild * 0.40
    effective_consol = p_consolidation * (1.0 - ild_suppression) * (1.0 - endobronchial_suppression)

    # Obstructive contribution to antibiotics:
    # Gate by Anthonisen — if no obstructive dx, wheeze alone gives low gate (0.15)
    # This ensures viral wheeze without purulence/dyspnea stays GREEN
    if anthonisen is not None:
        # Has obstructive dx — Anthonisen applicable
        obstruct_gate = anthonisen_qi  # 0.90 / 0.60 / 0.20 by type
    else:
        # No obstructive dx — wheeze might be incidental; low gate
        # Only bacterial evidence (fever, purulence) should drive antibiotics
        has_bacterial_evidence = (
            clinical_input.get("fever", False) or
            clinical_input.get("sputum_purulence_change", False)
        )
        obstruct_gate = 0.20 if has_bacterial_evidence else 0.05

    effective_obstruct_abx = p_obstructive * obstruct_gate

    # Bacterial co-indicator scaling:
    # Consolidation alone (no fever, no purulence) → CHF/viral possible → scale down
    # Consolidation + fever/purulence → bacterial likely → full weight
    # Prior antibiotics → already treated → scale consol down further
    # Gate consolidation→antibiotics on bacterial co-indicators
    # Purulence (Anthonisen criterion 1) is the strongest single predictor
    # Gate uses strict Anthonisen criterion 1 (sputum_purulence_change),
    # NOT cough_character — cough feeds consolidation probability separately.
    # asymmetry_high is an independent spatial signal that boosts confidence.
    has_purulence     = clinical_input.get("sputum_purulence_change", False)
    has_fever         = clinical_input.get("fever", False)
    has_high_asymmetry= evidence.get("Bilateral_Asymmetry_Score") == "high"
    phase_is_late     = evidence.get("Crackle_Phase") == "late_inspiratory"
    phase_is_early    = evidence.get("Crackle_Phase") == "early_inspiratory"

    if has_purulence and has_fever:
        consol_gate = 0.80   # Anthonisen criterion + fever: strong bacterial
    elif has_purulence:
        consol_gate = 0.65   # purulence alone: bacterial likely
    elif has_fever and has_high_asymmetry and phase_is_late:
        consol_gate = 0.65   # late_insp + fever + asymmetry: classic pneumonia pattern
    elif has_fever and has_high_asymmetry and phase_is_early:
        consol_gate = 0.38   # early_insp less specific (Forgacs 1967) → amber range
    elif has_fever and has_high_asymmetry:
        consol_gate = 0.45   # fever + asymmetry, phase unknown
    elif has_fever and phase_is_late:
        consol_gate = 0.40   # late_insp + fever only
    elif has_fever:
        consol_gate = 0.30   # fever alone: could be viral
    else:
        consol_gate = 0.18   # no bacterial co-indicators

    if clinical_input.get("prior_antibiotic_use", False):
        consol_gate *= 0.55  # prior antibiotics: partially treated, lower confidence

    active_abx = []
    if effective_consol > 0.05:       active_abx.append(effective_consol * consol_gate)
    if effective_obstruct_abx > 0.05: active_abx.append(effective_obstruct_abx * 0.70)
    if anthonisen == 1:               active_abx.append(0.85)
    # High asymmetry (LR+ up to 44.1, McGee/Diehr) adds minor direct signal
    if evidence.get("Bilateral_Asymmetry_Score") == "high" and has_fever:
        active_abx.append(0.25)   # conservative — high specificity, low sensitivity

    p_antibiotics = noisy_or(0.15, active_abx)  # base prior 0.15 — Indian epidemiology
    p_antibiotics = float(np.clip(p_antibiotics, 0.01, 0.99))

    return {
        "p_antibiotics":        p_antibiotics,
        "p_consolidation":      p_consolidation,
        "p_obstructive":        p_obstructive,
        "p_ild":                p_ild,
        "anthonisen_type":      anthonisen,
        "endobronchial_flag":   endobronchial_suppression > 0,
    }


# ---------------------------------------------------------------------------
# CONFIDENCE INTERVAL
# Bootstrap-style CI based on input completeness and zone_reliability
# ---------------------------------------------------------------------------

def compute_ci(
    p_antibiotics: float,
    evidence: dict,
    clinical_input: dict,
    n_missing_key_inputs: int,
) -> tuple[float, float]:
    """
    Computes 95% credible interval.
    CI widens with:
      - missing SpO2
      - missing spirometry
      - Zone_Reliability = reduced
      - Layer 2 data_quality = poor
      - Many missing inputs
    """
    base_half_width = 0.10  # ±10% baseline

    # Widen for missing inputs
    if clinical_input.get("spo2_pct") is None:
        base_half_width += 0.05
    if clinical_input.get("fev1_fvc_known") is None:
        base_half_width += 0.02
    if evidence.get("Zone_Reliability") == "reduced":
        base_half_width += 0.05
    if evidence.get("_data_quality") == "poor":
        base_half_width += 0.08
    if n_missing_key_inputs >= 4:
        base_half_width += 0.08
    elif n_missing_key_inputs >= 2:
        base_half_width += 0.04

    ci_lower = float(np.clip(p_antibiotics - base_half_width, 0.0, 1.0))
    ci_upper = float(np.clip(p_antibiotics + base_half_width, 0.0, 1.0))
    return ci_lower, ci_upper


# ---------------------------------------------------------------------------
# CONTRIBUTING FACTORS
# Marginalization delta per node — sorted by |delta| descending
# ---------------------------------------------------------------------------

def compute_contributing_factors(
    evidence: dict,
    clinical_input: dict,
    intermediate_cpts: dict,
    base_p_antibiotics: float,
) -> list[tuple[str, float]]:
    """
    For each input node: compute posterior WITH node observed vs WITH node marginalized.
    Delta = difference. Sort by |delta| descending.
    This is correct for all node cardinalities including multi-state nodes.
    """
    factors = []

    def p_with_node_removed(node_name: str) -> float:
        """Re-run inference with one node marginalized."""
        evidence_mod = dict(evidence)
        clinical_mod = dict(clinical_input)

        # Remove from evidence or clinical input
        if node_name in evidence_mod:
            evidence_mod[node_name] = None
        elif node_name in clinical_mod:
            clinical_mod[node_name] = None

        result = compute_posterior(evidence_mod, clinical_mod, intermediate_cpts)
        return result["p_antibiotics"]

    # Key nodes to assess
    nodes_to_assess = [
        # Acoustic
        "Crackle_Pattern", "Crackle_Phase", "Crackle_Subtype",
        "Wheeze_Compound", "Wheeze_Phase",
        "Bilateral_Asymmetry_Score", "Diminished_Sounds",
        "Craniocaudal_Gradient",
        # Clinical
        "fever", "spo2_pct", "comorbidity_obstructive",
        "sputum_purulence_change", "dyspnea_increase", "sputum_volume_increase",
        "biomass_exposure", "prior_antibiotic_use",
    ]

    for node in nodes_to_assess:
        p_without = p_with_node_removed(node)
        delta = base_p_antibiotics - p_without
        factors.append((node, round(delta, 4)))

    # Sort by absolute delta descending
    factors.sort(key=lambda x: abs(x[1]), reverse=True)
    return factors


# ---------------------------------------------------------------------------
# FULL OUTPUT LAYER
# ---------------------------------------------------------------------------

def compute_output(
    evidence: dict,
    clinical_input: dict,
    intermediate_cpts: dict,
) -> dict:
    """
    Full output layer. Takes BN evidence + clinical input → complete output dict.

    Escalation overlay: SpO2 <92% (Sjoding-adjusted) → escalate=True REGARDLESS
    of p_antibiotics or zone.
    """
    # Count missing key inputs
    key_inputs = [
        "Crackle_Pattern", "Wheeze_Pattern", "Crackle_Phase",
        "Wheeze_Compound", "Bilateral_Asymmetry_Score",
    ]
    n_missing = sum(1 for k in key_inputs if evidence.get(k) is None)
    # Note: missing fev1_fvc alone does not count toward n_missing (sprint spec:
    # "Remove spirometry → minimal CI change"). It's already captured in CI width.

    # --- Posterior ---
    posterior = compute_posterior(evidence, clinical_input, intermediate_cpts)
    p_antibiotics = posterior["p_antibiotics"]

    # --- CI ---
    ci_lower, ci_upper = compute_ci(p_antibiotics, evidence, clinical_input, n_missing)

    # --- Zone ---
    zone = compute_zone(p_antibiotics)

    # Zone_Reliability=reduced → cap at amber (dual-zone findings lack spatial certainty)
    if evidence.get("Zone_Reliability") == "reduced" and zone == "red":
        zone = "amber"

    # P_ILD_Flag > 0.40 → cap at amber (ILD differential suppresses red)
    # EXCEPTION: Anthonisen Type 1 + purulence overrides ILD cap (clear bacterial indication)
    anthonisen_overrides = (
        posterior.get("anthonisen_type") == 1 and
        clinical_input.get("sputum_purulence_change", False)
    )
    if posterior["p_ild"] > 0.40 and zone == "red" and not anthonisen_overrides:
        zone = "amber" 
    zone_lower = compute_zone(ci_lower)
    zone_upper = compute_zone(ci_upper)

    # --- Uncertainty flag ---
    ci_spans_two_zones = zone_lower != zone_upper
    data_quality_poor  = evidence.get("_data_quality") == "poor"
    uncertainty_flag = (
        ci_spans_two_zones
        or n_missing > 2
        or data_quality_poor
        or posterior["p_ild"] > 0.40                           # ILD differential
        or evidence.get("Craniocaudal_Gradient") == "apical"   # apical pattern → TB differential
        or (posterior["p_consolidation"] > 0.50               # conflicting pathways:
            and posterior["p_obstructive"] > 0.50)            # both consolidation AND obstructive active
    )

    # When uncertainty is high, minimum zone = amber (never confidently green)
    # Sprint spec: uncertainty_flag means CI spans zones — green is overconfident
    if uncertainty_flag and zone == "green":
        zone = "amber" 

    # --- Escalation overlay (hard rule — overrides everything) ---
    spo2 = clinical_input.get("spo2_pct")
    spo2_state = discretize_spo2(spo2)
    escalate = spo2_state in ("critical", "low")  # <92% Sjoding-adjusted

    # --- Disposition ---
    if escalate:
        disposition = "refer"
    elif zone == "red":
        disposition = "review_48h" if not escalate else "refer"
    elif zone == "amber" or uncertainty_flag:
        disposition = "review_48h"
    else:
        disposition = "home_safety_net"

    # Endobronchial flag changes disposition
    if posterior.get("endobronchial_flag") and zone == "green":
        disposition = "review_48h"  # monophonic_low wheeze warrants investigation

    # --- Contributing factors ---
    contributing_factors = compute_contributing_factors(
        evidence, clinical_input, intermediate_cpts, p_antibiotics
    )

    return {
        "p_antibiotics":         round(p_antibiotics, 4),
        "ci_lower":              round(ci_lower, 4),
        "ci_upper":              round(ci_upper, 4),
        "zone":                  zone,
        "uncertainty_flag":      uncertainty_flag,
        "escalate":              escalate,
        "disposition":           disposition,
        "contributing_factors":  contributing_factors[:8],  # top 8
        "bn_version":            BN_VERSION,
        # Debug fields
        "_p_consolidation":      round(posterior["p_consolidation"], 4),
        "_p_obstructive":        round(posterior["p_obstructive"], 4),
        "_p_ild":                round(posterior["p_ild"], 4),
        "_anthonisen_type":      posterior["anthonisen_type"],
        "_endobronchial_flag":   posterior["endobronchial_flag"],
        "_n_missing_key_inputs": n_missing,
    }


# ---------------------------------------------------------------------------
# MISSING DATA DEGRADATION TESTS
# ---------------------------------------------------------------------------

def test_degradation_hierarchy(intermediate_cpts: dict):
    """
    Tests all degradation levels — must produce no errors or crashes.
    From sprint spec Day 4.
    """
    print("\n" + "=" * 60)
    print("Missing Data Degradation Tests")
    print("=" * 60)

    base_clinical = {
        "age": 55, "sex": "M", "bmi": 24.0,
        "fever": True, "symptom_duration_days": 4,
        "cough_character": "productive_purulent",
        "sputum_purulence_change": True,
        "dyspnea_increase": True,
        "sputum_volume_increase": False,
        "spo2_pct": 95.0,
        "fev1_fvc_known": None,
        "comorbidity_obstructive": False,
        "biomass_exposure": False,
        "prior_antibiotic_use": False,
        "season": "winter",
    }
    base_evidence = {
        "Crackle_Pattern": "focal", "Crackle_Phase": "late_inspiratory",
        "Crackle_Subtype": "coarse", "Wheeze_Pattern": "absent",
        "Wheeze_Phase": None, "Wheeze_Compound": None,
        "Rhonchi_Present": "absent", "Diminished_Sounds": "absent",
        "Bilateral_Asymmetry_Score": "high", "Craniocaudal_Gradient": "flat",
        "Zone_Reliability": "normal",
        "_data_quality": "normal", "_subtype_modules_run": True, "_dual_zone_only": False,
    }

    results = []

    # Level 1: Full data
    out = compute_output(base_evidence, base_clinical, intermediate_cpts)
    results.append(("Full data", out["p_antibiotics"], out["ci_upper"] - out["ci_lower"]))

    # Level 2: Remove SpO2
    c2 = dict(base_clinical); c2["spo2_pct"] = None
    out2 = compute_output(base_evidence, c2, intermediate_cpts)
    results.append(("No SpO2", out2["p_antibiotics"], out2["ci_upper"] - out2["ci_lower"]))
    assert not out2["escalate"], "Should not escalate without SpO2"

    # Level 3: Remove spirometry (already None in base — add fev1)
    c3 = dict(base_clinical); c3["spo2_pct"] = None; c3["fev1_fvc_known"] = None
    out3 = compute_output(base_evidence, c3, intermediate_cpts)
    results.append(("No SpO2+FEV1", out3["p_antibiotics"], out3["ci_upper"] - out3["ci_lower"]))

    # Level 4: 4 acoustic zones missing
    e4 = dict(base_evidence)
    e4["Crackle_Pattern"] = None; e4["Wheeze_Pattern"] = None
    e4["Bilateral_Asymmetry_Score"] = None; e4["Craniocaudal_Gradient"] = None
    out4 = compute_output(e4, base_clinical, intermediate_cpts)
    results.append(("4 acoustic missing", out4["p_antibiotics"], out4["ci_upper"] - out4["ci_lower"]))

    # Level 5: data_quality = poor
    e5 = dict(base_evidence); e5["_data_quality"] = "poor"
    out5 = compute_output(e5, base_clinical, intermediate_cpts)
    assert out5["uncertainty_flag"], "Poor data quality must set uncertainty_flag"
    results.append(("data_quality=poor", out5["p_antibiotics"], out5["ci_upper"] - out5["ci_lower"]))

    # Level 6: SpO2 = 88 → escalate
    c6 = dict(base_clinical); c6["spo2_pct"] = 88.0
    out6 = compute_output(base_evidence, c6, intermediate_cpts)
    assert out6["escalate"], "SpO2=88 must trigger escalate=True"
    assert out6["disposition"] == "refer"
    results.append(("SpO2=88 escalate", out6["p_antibiotics"], out6["ci_upper"] - out6["ci_lower"]))

    # Level 7: subtype_modules_run=False
    e7 = dict(base_evidence)
    e7["Crackle_Subtype"] = None; e7["Wheeze_Compound"] = None
    e7["_subtype_modules_run"] = False
    out7 = compute_output(e7, base_clinical, intermediate_cpts)
    results.append(("subtype_modules_run=False", out7["p_antibiotics"], out7["ci_upper"] - out7["ci_lower"]))

    print(f"\n{'Level':<30} {'P(abx)':>8} {'CI width':>10}")
    print("-" * 52)
    for label, p, ci_w in results:
        print(f"  {label:<28} {p:>8.3f} {ci_w:>10.3f}")

    # Verify CI widens progressively (levels 1→4)
    ci_widths = [r[2] for r in results[:4]]
    widening = all(ci_widths[i] <= ci_widths[i+1] for i in range(len(ci_widths)-1))
    status = "✓" if widening else "✗ FAIL — CI not widening progressively"
    print(f"\nCI widening check (levels 1→4): {status}")

    print("\n✓ All degradation levels completed without errors")
    return True


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from clinical_cpts import build_intermediate_cpts
    intermediate_cpts = build_intermediate_cpts()

    # Quick smoke test
    evidence = {
        "Crackle_Pattern": "focal",
        "Crackle_Phase": "late_inspiratory",
        "Crackle_Subtype": "coarse",
        "Wheeze_Pattern": "absent",
        "Wheeze_Phase": None,
        "Wheeze_Compound": None,
        "Rhonchi_Present": "absent",
        "Diminished_Sounds": "unilateral",
        "Bilateral_Asymmetry_Score": "high",
        "Craniocaudal_Gradient": "flat",
        "Zone_Reliability": "normal",
        "_data_quality": "normal",
        "_subtype_modules_run": True,
        "_dual_zone_only": False,
    }
    clinical = {
        "age": 60, "sex": "M", "bmi": 23.0,
        "fever": True, "symptom_duration_days": 5,
        "cough_character": "productive_purulent",
        "sputum_purulence_change": True,
        "dyspnea_increase": False,
        "sputum_volume_increase": False,
        "spo2_pct": 94.0,
        "fev1_fvc_known": None,
        "comorbidity_obstructive": False,
        "biomass_exposure": False,
        "prior_antibiotic_use": False,
        "season": "winter",
    }

    print("=" * 60)
    print("Day 4 — Output Layer Smoke Test")
    print("=" * 60)
    out = compute_output(evidence, clinical, intermediate_cpts)
    print(f"\n  p_antibiotics    : {out['p_antibiotics']}")
    print(f"  CI               : [{out['ci_lower']}, {out['ci_upper']}]")
    print(f"  zone             : {out['zone']}")
    print(f"  uncertainty_flag : {out['uncertainty_flag']}")
    print(f"  escalate         : {out['escalate']}")
    print(f"  disposition      : {out['disposition']}")
    print(f"  bn_version       : {out['bn_version']}")
    print(f"\n  Top contributing factors:")
    for name, delta in out["contributing_factors"][:5]:
        direction = "↑" if delta > 0 else "↓"
        print(f"    {name:<35} {direction} {abs(delta):.3f}")

    test_degradation_hierarchy(intermediate_cpts)
    print("\n✓ Day 4 complete")
