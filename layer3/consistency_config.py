"""
TreBle Respire â€” Consistency Configuration

Single source of truth for CONSISTENCY_THRESHOLD and related aggregation
constants used across Layer 1/2/3 wheeze and crackle detection.

Per Arvind decision 3: "Confirm CONSISTENCY_THRESHOLD=0.28 is in a named
config file before freeze." This file is that named config.

Import this constant everywhere it is needed â€” do not redefine inline.
"""

# â”€â”€ Consistency threshold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Used by Nishevithaa's Layer 2 pipeline to weight multi-cycle aggregation.
# With n=1 cycle (single 8s window), the rescue clause fires
# (consistency = 1/n = 1.0), which can overstate confidence on short
# single-cycle recordings (e.g. ICBHI reference files).
#
# Value confirmed by Nishevithaa's validation report (Section 1.2/5.1).
CONSISTENCY_THRESHOLD = 0.28

# â”€â”€ Recording duration â€” per Arvind decision 3 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# "15 seconds is sufficient at normal respiratory rates. Extend to 20 seconds
# if the patient is breathing at fewer than 12 breaths per minute."
MIN_RECORDING_S          = 15.0
EXTENDED_RECORDING_S     = 20.0
BRADYPNOEA_THRESHOLD_BPM = 12


def required_recording_duration(respiratory_rate_bpm: float = None) -> float:
    """
    Returns the minimum recording duration in seconds for a given
    respiratory rate, per Arvind decision 3.

    Args:
        respiratory_rate_bpm: measured or estimated breaths/minute.
                              If None, returns the conservative default
                              (extended duration) since rate is unknown.

    Returns:
        Required recording duration in seconds.
    """
    if respiratory_rate_bpm is None:
        return EXTENDED_RECORDING_S   # unknown rate â€” be conservative
    if respiratory_rate_bpm < BRADYPNOEA_THRESHOLD_BPM:
        return EXTENDED_RECORDING_S
    return MIN_RECORDING_S


# â”€â”€ Threshold provenance â€” per Arvind decision 2 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# CRACKLE: Nishevithaa-validated in-distribution screening threshold.
# WHEEZE:  provisional until bench calibration produces a Taal-specific value.
CRACKLE_THRESHOLD       = 0.491
WHEEZE_THRESHOLD        = 0.0276   # Taal calibrated (preliminary, n=4+4)
WHEEZE_THRESHOLD_STATUS = "taal_preliminary"   # "provisional" | "calibrated"

THRESHOLD_PROVENANCE = {
    "crackle": {
        "value":  CRACKLE_THRESHOLD,
        "source": "Nishevithaa-validated in-distribution screening threshold",
        "status": "calibrated",
    },
    "wheeze": {
        "value":  WHEEZE_THRESHOLD,
        "source": "in-distribution screening threshold â€” no Taal-specific value exists",
        "status": WHEEZE_THRESHOLD_STATUS,
        "action_required": "Run bench calibration with Taal stethoscope before study start. "
                           "Until then, all wheeze findings must carry confidence=provisional.",
    },
}


def get_threshold(finding_type: str) -> tuple:
    """
    Returns (threshold_value, status) for a given finding type.

    Args:
        finding_type: "crackle" or "wheeze"

    Returns:
        (threshold: float, status: str)
    """
    if finding_type not in THRESHOLD_PROVENANCE:
        raise ValueError(
            f"Unknown finding_type '{finding_type}'. "
            f"Must be one of: {list(THRESHOLD_PROVENANCE.keys())}. "
            f"Per Arvind decision 2: do not run without thresholds."
        )
    entry = THRESHOLD_PROVENANCE[finding_type]
    return entry["value"], entry["status"]


if __name__ == "__main__":
    print("=" * 60)
    print("Consistency & Threshold Configuration")
    print("=" * 60)
    print(f"\nCONSISTENCY_THRESHOLD = {CONSISTENCY_THRESHOLD}")
    print(f"\nRecording duration (Arvind decision 3):")
    print(f"  Normal rate (â‰¥12 bpm) : {MIN_RECORDING_S}s")
    print(f"  Bradypnoea (<12 bpm)  : {EXTENDED_RECORDING_S}s")
    print(f"\nThresholds (Arvind decision 2):")
    for finding, entry in THRESHOLD_PROVENANCE.items():
        print(f"  {finding:<10} = {entry['value']}  [{entry['status']}]")
        print(f"             source: {entry['source']}")
        if "action_required" in entry:
            print(f"             ACTION: {entry['action_required']}")

