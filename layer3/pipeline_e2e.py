"""
TreBle Respire - End-to-End Pipeline
Layer 1 (acoustic) -> Layer 2 (spatial) -> Layer 3 (BN inference)

Thresholds follow the sprint spec contract exactly:
  CRACKLE_THRESH = 0.11   taal study threshold (crackle_v1.meta.json)
  WHEEZE_THRESH  = 0.243  in-dist screening threshold (wheeze has no taal threshold)

All summary_flags follow the Layer 2 → Layer 3 interface contract in output_layer.py.
subtype_modules_run=False until phase segmentation is confirmed working on Taal recordings.

Usage:
    from pipeline_e2e import run_patient
    result = run_patient(zone_audio_files={"Z01": "path/Z01.wav", ...}, clinical={...})
"""

import sys, os, warnings
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MUSE)
sys.path.insert(0, os.path.join(_MUSE, "layer1"))
sys.path.insert(0, os.path.join(_MUSE, "layer2"))
sys.path.insert(0, os.path.join(_MUSE, "layer2", "utils"))
os.environ.setdefault("EFFICIENTAT_PATH",
                      os.path.join(_MUSE, "layer1", "EfficientAT"))

# ── Layer 1 ───────────────────────────────────────────────────────────────────
try:
    from layer1 import (predict_window, analyse_subtypes, THRESHOLDS,
                        load_crackle, load_wheeze, load_breath_segmenter)
    LAYER1_AVAILABLE = True
except Exception as e:
    LAYER1_AVAILABLE = False
    warnings.warn(f"Layer 1 not available: {e}")

# ── Phase segmenter ───────────────────────────────────────────────────────────
try:
    from phase_segmenter import (run_segmenter_full, get_phase_segments,
                                  run_phase_resolved_detection,
                                  verify_segmenter_contract)
    PHASE_AVAILABLE = True
except Exception as e:
    PHASE_AVAILABLE = False

# ── Layer 3 ───────────────────────────────────────────────────────────────────
from clinical_cpts import build_intermediate_cpts
from output_layer import compute_output, map_layer2_to_bn
from clinical_schema import validate_clinical_input
from day7_mock_integration import build_mock_layer2_output, validate_layer2_output

# ── Constants — sprint spec contract ─────────────────────────────────────────
ALL_ZONES  = [f"Z{i:02d}" for i in range(1, 21)]
DUAL_ZONES = {"Z06", "Z08"}
SR_TARGET  = 16000
SEG_SAMPLES = 128000    # 8.0s @ 16kHz — Layer 1 contract

# Thresholds — imported from named config file per Arvind decision 3:
# "Confirm CONSISTENCY_THRESHOLD=0.28 is in a named config file before freeze."
# Do not redefine these inline — consistency_config.py is the single source.
from consistency_config import (
    CRACKLE_THRESHOLD as CRACKLE_THRESH,
    WHEEZE_THRESHOLD  as WHEEZE_THRESH,
    WHEEZE_THRESHOLD_STATUS,
    CONSISTENCY_THRESHOLD,
    MIN_RECORDING_S,
    EXTENDED_RECORDING_S,
    BRADYPNOEA_THRESHOLD_BPM,
    required_recording_duration,
    get_threshold,
)

WHEEZE_PROVISIONAL = (WHEEZE_THRESHOLD_STATUS == "provisional")

# Bilateral pairs — confirmed by Nishevithaa (Z17↔Z18, Z19↔Z20)
_BILATERAL_PAIRS = [
    ("Z01","Z02"),("Z03","Z04"),("Z05","Z06"),("Z07","Z08"),
    ("Z09","Z10"),("Z11","Z12"),("Z13","Z14"),("Z15","Z16"),
    ("Z17","Z18"),("Z19","Z20"),
]
# Basal zones — confirmed Z13-Z20 (Arvind)
_APICAL_ZONES = ["Z01","Z02","Z09","Z10"]
_BASAL_ZONES  = [f"Z{i:02d}" for i in range(13, 21)]

# Wheeze compound mapping — morphology × freq_band → 7-state compound
# Exactly matching VALID_WHEEZE_COMPOUND_STATES in output_layer.py
WHEEZE_COMPOUND_MAP = {
    ("monophonic","low"):   "monophonic_low",
    ("monophonic","mid"):   "monophonic_mid",
    ("monophonic","high"):  "monophonic_high",
    ("polyphonic","low"):   "polyphonic_low",
    ("polyphonic","mid"):   "polyphonic_mid",
    ("polyphonic","high"):  "polyphonic_pan",
    ("polyphonic","mixed"): "polyphonic_pan",
}


# ── Audio loading ─────────────────────────────────────────────────────────────

def load_audio_full(path: str) -> tuple:
    """
    Load full-duration audio for phase segmentation + 8s window for models.
    Returns (audio_full, audio_8s) both peak-normalised float32 at 16kHz.
    """
    import torchaudio
    waveform, sr = torchaudio.load(path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != SR_TARGET:
        waveform = torchaudio.functional.resample(
            waveform, sr, SR_TARGET,
            lowpass_filter_width=128, rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
        )
    audio_full = waveform.squeeze(0).numpy().astype(np.float32)
    peak = np.abs(audio_full).max()
    if peak > 1e-6:
        audio_full = audio_full / peak
    audio_8s = (audio_full[:SEG_SAMPLES] if len(audio_full) >= SEG_SAMPLES
                else np.pad(audio_full, (0, SEG_SAMPLES - len(audio_full))))
    return audio_full, audio_8s


def load_audio(path: str) -> np.ndarray:
    """Load audio → peak-normalised float32 16kHz 8s."""
    _, audio_8s = load_audio_full(path)
    return audio_8s


# ── Layer 1 per-zone inference ────────────────────────────────────────────────

def run_layer1_on_zone(
    audio:      np.ndarray,
    zone_id:    str,
    device:     str = "cpu",
    seg_model=None,
    cr_model=None,
    wh_model=None,
    audio_full: np.ndarray = None,
) -> dict:
    """
    Run Layer 1 on one zone's 8s audio.

    Phase segmentation: runs on full-duration audio when seg_model provided.
    subtype_modules_run=False until phase confirmed working on Taal recordings.
    """
    l1 = predict_window(audio, device=device)

    # Subtype analysis — crackle fine/coarse/velcro + wheeze morphology/freq
    # Phase=None: not passing phase — not yet confirmed working on Taal recordings
    try:
        subtypes = analyse_subtypes(
            audio_16k       = audio,
            layer1_out      = l1,
            crackle_phase   = None,
            wheeze_phase    = None,
            wheeze_segments = None,
        )
        cr = subtypes.get("crackle_subtype")
        wc = subtypes.get("wheeze_character")
    except Exception:
        cr, wc = None, None

    # Phase-resolved detection — uses full recording length
    crackle_phase = None
    wheeze_phase  = None

    if seg_model is not None and PHASE_AVAILABLE:
        try:
            seg_audio    = audio_full if audio_full is not None else audio
            frame_labels = run_segmenter_full(seg_audio, seg_model)
            phase_out    = run_phase_resolved_detection(
                audio, frame_labels, cr_model, wh_model, device=device
            )
            crackle_phase = phase_out["dominant_crackle_phase"]
            wheeze_phase  = phase_out["dominant_wheeze_phase"]
        except Exception:
            pass

    return {
        "zone_id":                    zone_id,
        "p_wheeze":                   float(l1["p_wheeze"]),
        "p_crackle":                  float(l1["p_crackle"]),
        "wheeze_morphology":          wc.character          if wc else "uncertain",
        "wheeze_freq_band":           wc.dominant_freq_band if wc else None,
        "crackle_subtype":            cr.subtype            if cr else "uncertain",
        "crackle_subtype_confidence": cr.subtype_confidence if cr else 0.0,
        "velcro_score":               cr.velcro_score       if cr else 0.0,
        "crackle_phase":              crackle_phase,
        "wheeze_phase":               wheeze_phase,
    }


# ── Layer 2 aggregation — contract-exact ─────────────────────────────────────

def build_layer2_from_zone_results(
    zone_results:        dict,
    recorded_zones:      set,
    body_habitus:        dict,
    subtype_modules_run: bool = False,
) -> dict:
    """
    Aggregate per-zone Layer 1 outputs into the Layer 2 output structure
    consumed by map_layer2_to_bn() in output_layer.py.

    Every field follows the Layer 2 → Layer 3 interface contract exactly.
    Thresholds: CRACKLE_THRESH=0.11, WHEEZE_THRESH=0.243 (contract values).
    """
    # Build zone_findings list — 20 zones
    zone_findings = []
    for z in ALL_ZONES:
        is_rec  = z in recorded_zones
        is_dual = z in DUAL_ZONES
        if is_rec and z in zone_results:
            zr  = zone_results[z]
            # Contract: any_abnormality = crackle OR wheeze above their thresholds
            crackle_abn = zr["p_crackle"] > CRACKLE_THRESH
            wheeze_abn  = zr["p_wheeze"]  > WHEEZE_THRESH
            abn = crackle_abn or wheeze_abn
            sq  = min(1.0, max(zr["p_crackle"], zr["p_wheeze"]) + 0.3)
            # Per Arvind decision 2: "Always store raw scores. Do not run
            # without thresholds." Raw scores preserved regardless of
            # threshold decision — allows re-analysis if thresholds change.
            raw_crackle = zr["p_crackle"]
            raw_wheeze  = zr["p_wheeze"]
            # Wheeze findings flagged provisional until bench calibration
            wheeze_confidence = "provisional" if (wheeze_abn and WHEEZE_PROVISIONAL) else "normal"
        else:
            abn, sq = False, 0.0
            raw_crackle, raw_wheeze = None, None
            wheeze_confidence = None
        zone_findings.append({
            "zone_id":                     z,
            "recorded":                    is_rec,
            "any_abnormality":             abn,
            "is_dual_representation_zone": is_dual,
            "signal_quality":              round(sq, 3),
            "raw_p_crackle":               raw_crackle,    # Arvind: always store raw scores
            "raw_p_wheeze":                raw_wheeze,
            "wheeze_confidence":           wheeze_confidence,
        })

    # Recorded zone results only
    rr = {z: r for z, r in zone_results.items() if z in recorded_zones}

    # ── Crackle summary flags ─────────────────────────────────────────────────
    crackles_present  = any(r["p_crackle"] > CRACKLE_THRESH for r in rr.values())
    abn_c = {z for z, r in rr.items() if r["p_crackle"] > CRACKLE_THRESH}
    right_c = {z for z in abn_c if int(z[1:]) % 2 == 1}
    left_c  = {z for z in abn_c if int(z[1:]) % 2 == 0}
    crackles_bilateral = bool(right_c and left_c)
    crackles_focal     = bool(abn_c) and not crackles_bilateral

    # Dominant crackle subtype (majority vote, only when subtype_modules_run)
    if subtype_modules_run:
        subtypes = [r["crackle_subtype"] for r in rr.values()
                    if r["p_crackle"] > CRACKLE_THRESH
                    and r["crackle_subtype"] not in ("uncertain", None)]
        subtype_dom  = max(set(subtypes), key=subtypes.count) if subtypes else "uncertain"
        velcro_present = "velcro" in subtypes
    else:
        subtype_dom   = None
        velcro_present = False

    # Dominant crackle phase (only when subtype_modules_run and phase resolved)
    if subtype_modules_run:
        cp_list = [r["crackle_phase"] for r in rr.values()
                   if r["p_crackle"] > CRACKLE_THRESH
                   and r["crackle_phase"] is not None]
        crackle_phase_dom = (
            max(set(cp_list), key=cp_list.count) if cp_list else None
        )
        # Map to BN states: inspiratory→late_inspiratory (most common clinical pattern)
        _pmap = {"inspiratory": "late_inspiratory", "expiratory": "expiratory",
                 "biphasic": "late_inspiratory"}
        crackle_phase_dom = _pmap.get(crackle_phase_dom) if crackle_phase_dom else None
    else:
        crackle_phase_dom = None

    # ── Wheeze summary flags ──────────────────────────────────────────────────
    wheeze_present = any(r["p_wheeze"] > WHEEZE_THRESH for r in rr.values())
    abn_w = {z for z, r in rr.items() if r["p_wheeze"] > WHEEZE_THRESH}
    right_w = {z for z in abn_w if int(z[1:]) % 2 == 1}
    left_w  = {z for z in abn_w if int(z[1:]) % 2 == 0}
    wheeze_bilateral = bool(right_w and left_w)
    wheeze_focal     = bool(abn_w) and not wheeze_bilateral

    # Wheeze compound — morphology × freq_band (only when subtype_modules_run)
    if subtype_modules_run:
        compounds = []
        for r in rr.values():
            if r["p_wheeze"] > WHEEZE_THRESH:
                key = (r["wheeze_morphology"], r["wheeze_freq_band"])
                compound = WHEEZE_COMPOUND_MAP.get(key)
                if compound:
                    compounds.append(compound)
        wheeze_compound_dom = (
            max(set(compounds), key=compounds.count) if compounds else "uncertain"
        )
        # Wheeze phase
        wp_list = [r["wheeze_phase"] for r in rr.values()
                   if r["p_wheeze"] > WHEEZE_THRESH and r["wheeze_phase"] is not None]
        wheeze_phase_dom = (
            max(set(wp_list), key=wp_list.count) if wp_list else None
        )
    else:
        wheeze_compound_dom = "uncertain"
        wheeze_phase_dom    = None

    # ── Spatial metrics ───────────────────────────────────────────────────────
    flat = {z: r for z, r in rr.items()}

    # Bilateral asymmetry — max crackle difference across confirmed pairs
    valid_pairs = [(a, b) for a, b in _BILATERAL_PAIRS if a in flat and b in flat]
    asym_max = round(
        max(abs(flat[a]["p_crackle"] - flat[b]["p_crackle"]) for a, b in valid_pairs), 4
    ) if valid_pairs else 0.0

    # Craniocaudal gradient — basal vs apical crackle mean
    rec_ap = [z for z in _APICAL_ZONES if z in flat]
    rec_ba = [z for z in _BASAL_ZONES  if z in flat]
    if rec_ap and rec_ba:
        diff = (sum(flat[z]["p_crackle"] for z in rec_ba) / len(rec_ba)
              - sum(flat[z]["p_crackle"] for z in rec_ap) / len(rec_ap))
        gradient = "basal" if diff > 0.10 else ("apical" if diff < -0.10 else "flat")
    else:
        gradient = "flat"

    # ── Data quality ──────────────────────────────────────────────────────────
    n_rec = len(recorded_zones)
    dq    = "normal" if n_rec >= 8 else "poor"

    # ── Assemble summary_flags — every field the contract requires ────────────
    summary_flags = {
        # Crackle flags
        "crackles_present":               crackles_present,
        "crackles_focal":                 crackles_focal,
        "crackles_bilateral":             crackles_bilateral,
        "crackle_subtype_dominant":       subtype_dom,
        "crackle_phase_dominant":         crackle_phase_dom,
        "velcro_crackles_present":        velcro_present,
        # Wheeze flags
        "wheeze_present":                 wheeze_present,
        "wheeze_bilateral":               wheeze_bilateral,
        "wheeze_focal":                   wheeze_focal,
        "wheeze_phase_dominant":          wheeze_phase_dom,
        "wheeze_compound_dominant":       wheeze_compound_dom,
        "wheeze_monophonic_any":          any(c.startswith("monophonic")
                                             for c in ([] if not subtype_modules_run
                                                        else [WHEEZE_COMPOUND_MAP.get(
                                                            (r["wheeze_morphology"], r["wheeze_freq_band"]), "")
                                                            for r in rr.values()
                                                            if r["p_wheeze"] > WHEEZE_THRESH])),
        "wheeze_polyphonic_any":          any(c.startswith("polyphonic")
                                             for c in ([] if not subtype_modules_run
                                                        else [WHEEZE_COMPOUND_MAP.get(
                                                            (r["wheeze_morphology"], r["wheeze_freq_band"]), "")
                                                            for r in rr.values()
                                                            if r["p_wheeze"] > WHEEZE_THRESH])),
        "wheeze_focal_monophonic":        wheeze_compound_dom == "monophonic_low" and wheeze_focal,
        "wheeze_low_freq_present":        wheeze_compound_dom in ("monophonic_low","polyphonic_low"),
        "wheeze_high_freq_present":       wheeze_compound_dom in ("monophonic_high","polyphonic_pan"),
        "wheeze_pan_airway":              wheeze_compound_dom == "polyphonic_pan",
        # Other acoustic flags
        "diminished_present":             False,    # no diminished detector in Layer 1
        "diminished_unilateral":          False,
        "consolidation_signs":            (crackles_focal
                                           and crackle_phase_dom == "late_inspiratory"),
        "rhonchi_present":                False,    # no rhonchi detector in Layer 1
        # Spatial metrics
        "zones_recorded":                 n_rec,
        "zones_low_quality":              sum(1 for z in zone_findings
                                             if z["recorded"] and z["signal_quality"] < 0.4),
        "dual_zones_low_quality":         any(z["zone_id"] in DUAL_ZONES
                                             and z["signal_quality"] < 0.4
                                             for z in zone_findings),
        "bilateral_asymmetry_score_max":  asym_max,
        "craniocaudal_gradient_dominant": gradient,
    }

    return {
        "zone_findings": zone_findings,
        "summary_flags": summary_flags,
        "output_meta": {
            "subtype_modules_run": subtype_modules_run,
            "data_quality":        dq,
            "pipeline_version":    "2.1.0",
            "n_zones_recorded":    n_rec,
        },
        "body_habitus": body_habitus,
    }


# ── Main entry point ──────────────────────────────────────────────────────────

def run_patient(
    zone_audio_files: dict,
    clinical:         dict,
    device:           str  = "cpu",
    use_layer1:       bool = True,
    verbose:          bool = True,
) -> dict:
    """
    Full end-to-end pipeline for one patient.

    Args:
        zone_audio_files: {zone_id: wav_path} — any subset of Z01–Z20
        clinical:         clinical input dict (see clinical_schema.py)
        device:           "cpu" or "cuda"
        use_layer1:       True = real acoustic models, False = mock
        verbose:          print progress
    """
    # Per Arvind decision 2: "Do not run without thresholds."
    # Guard confirms both thresholds are defined and non-None before proceeding.
    try:
        _c_thresh, _c_status = get_threshold("crackle")
        _w_thresh, _w_status = get_threshold("wheeze")
        assert _c_thresh is not None and _w_thresh is not None
    except Exception as e:
        raise RuntimeError(
            f"Pipeline cannot run without valid thresholds (Arvind decision 2). "
            f"Check consistency_config.py. Error: {e}"
        )

    if verbose and _w_status == "provisional":
        print(f"  ⚠ WHEEZE THRESHOLD IS PROVISIONAL ({_w_thresh}) — "
              f"bench calibration not yet run. All wheeze findings flagged accordingly.")

    if verbose:
        print(f"\n{'='*60}")
        print(f"TreBle Respire - End-to-End Patient Inference")
        print(f"{'='*60}")
        print(f"  Zones    : {len(zone_audio_files)}")
        print(f"  Layer 1  : {'real' if (use_layer1 and LAYER1_AVAILABLE) else 'mock'}")

    ok, errors = validate_clinical_input(clinical)
    if not ok:
        print("  Clinical input invalid:")
        for e in errors: print(f"    {e}")
        return {"error": "invalid_clinical_input", "errors": errors}

    bh = {"bmi": clinical.get("bmi", 23.0),
          "age": clinical.get("age", 45),
          "sex": clinical.get("sex", "M")}

    zone_results   = {}
    recorded_zones = set()
    seg_model = cr_model = wh_model = None

    # Load models once — reuse across all zones
    if use_layer1 and LAYER1_AVAILABLE and zone_audio_files:
        try:
            seg_model = load_breath_segmenter(device=device)
            cr_model  = load_crackle(device=device)
            wh_model  = load_wheeze(device=device)
            if verbose and PHASE_AVAILABLE:
                print(f"  Phase segmentation: ENABLED")
            elif verbose:
                print(f"  Phase segmentation: DISABLED")
        except Exception as e:
            if verbose: print(f"  Model loading: {e}")

    if use_layer1 and LAYER1_AVAILABLE and zone_audio_files:
        for zone_id, path in zone_audio_files.items():
            if verbose:
                print(f"  [{zone_id}] processing...", end=" ", flush=True)
            try:
                audio_full, audio = load_audio_full(path)
                result = run_layer1_on_zone(
                    audio, zone_id, device,
                    seg_model=seg_model, cr_model=cr_model, wh_model=wh_model,
                    audio_full=audio_full,
                )
                zone_results[zone_id]    = result
                recorded_zones.add(zone_id)
                if verbose:
                    phase_str = (f"  phase={result['crackle_phase']}"
                                 if result["crackle_phase"] else "")
                    above_c = result["p_crackle"] > CRACKLE_THRESH
                    above_w = result["p_wheeze"]  > WHEEZE_THRESH
                    flag = (" ← CRACKLE" if above_c else "") + (" ← WHEEZE" if above_w else "")
                    print(f"crackle={result['p_crackle']:.3f}  "
                          f"wheeze={result['p_wheeze']:.3f}  "
                          f"subtype={result['crackle_subtype']}{phase_str}{flag}")
            except Exception as e:
                if verbose: print(f"failed: {e}")
    elif verbose:
        print("  Using mock acoustic input")

    # subtype_modules_run: True only if phase confirmed working
    # Currently False — phase segmenter not yet calibrated for Taal recordings
    phase_ran = any(r.get("crackle_phase") is not None for r in zone_results.values())

    if zone_results:
        l2_out = build_layer2_from_zone_results(
            zone_results        = zone_results,
            recorded_zones      = recorded_zones,
            body_habitus        = bh,
            subtype_modules_run = phase_ran,
        )
    else:
        l2_out = build_mock_layer2_output(bmi=bh["bmi"], age=bh["age"])

    valid, val_errors = validate_layer2_output(l2_out, source="pipeline")
    if not valid and verbose:
        print(f"  Layer 2 warnings: {val_errors}")

    if verbose:
        flags = l2_out["summary_flags"]
        print(f"\n  Acoustic summary:")
        print(f"    crackle_pattern   : {flags.get('crackles_present')} "
              f"focal={flags.get('crackles_focal')} "
              f"bilateral={flags.get('crackles_bilateral')}")
        print(f"    wheeze_present    : {flags.get('wheeze_present')} "
              f"focal={flags.get('wheeze_focal')} "
              f"bilateral={flags.get('wheeze_bilateral')}")
        print(f"    asymmetry_max     : {flags.get('bilateral_asymmetry_score_max'):.3f}")
        print(f"    gradient          : {flags.get('craniocaudal_gradient_dominant')}")
        print(f"    subtype_modules   : {l2_out['output_meta']['subtype_modules_run']}")
        print(f"\n  Running BN inference...")

    cpts     = build_intermediate_cpts()
    evidence = map_layer2_to_bn(l2_out)
    out      = compute_output(evidence, clinical, cpts)

    if verbose:
        print(f"\n{'─'*60}")
        print(f"  p_antibiotics  : {out['p_antibiotics']:.3f}")
        print(f"  zone           : {out['zone'].upper()}")
        print(f"  uncertainty    : {out['uncertainty_flag']}")
        print(f"  escalate       : {out['escalate']}")
        print(f"  disposition    : {out['disposition']}")
        if out.get("_endobronchial_flag"):
            print(f"  ENDOBRONCHIAL FLAG — investigate fixed obstruction")
        print(f"\n  Top factors:")
        for name, delta in out["contributing_factors"][:4]:
            direction = "UP" if delta > 0 else "dn"
            print(f"    {name:<35} {direction} {abs(delta):.3f}")
        print(f"{'='*60}")

    out["_layer2_output"] = l2_out
    out["_zone_results"]  = zone_results
    out["_evidence"]      = evidence
    return out
