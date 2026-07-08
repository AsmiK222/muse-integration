# icbhi_wheeze_reference.py
#
# ICBHI 2017 wheeze reference calibration — SANITY CHECK ONLY
#
# Per Arvind Decision 2: use ICBHI for reference verification only.
# Thresholds derived here are NOT valid for the TRUPCR study —
# ICBHI was recorded with Meditron/AKGC/Littmann stethoscopes, not Taal.
# This script confirms Layer 1 responds correctly to known wheeze content.
# Status in consistency_config.py will be "icbhi_reference_only", not "calibrated".
#
# HOW TO USE:
#   1. Download ICBHI 2017 from Kaggle:
#      https://www.kaggle.com/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database
#
#   2. Extract to a folder, e.g.:
#      D:\Desktop\INTERNSHIP\icbhi_2017\
#      Should contain:
#        *.wav  (920 audio files)
#        *.txt  (annotation files, same name as wav)
#
#   3. Run:
#      python icbhi_wheeze_reference.py --icbhi-dir "D:\Desktop\INTERNSHIP\icbhi_2017"
#
# OUTPUT:
#   icbhi_reference_report.txt   — full results
#   icbhi_reference_report.json  — raw scores for re-analysis
#   Prints the reference threshold block to paste into consistency_config.py

import os
import sys
import json
import argparse
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MUSE)
sys.path.insert(0, os.path.join(_MUSE, "layer1"))
os.environ.setdefault("EFFICIENTAT_PATH",
                      os.path.join(_MUSE, "layer1", "EfficientAT"))

try:
    from layer1 import predict_window
    LAYER1_AVAILABLE = True
except Exception as e:
    LAYER1_AVAILABLE = False
    print(f"⚠ Layer 1 not available: {e}")

from pipeline_e2e import load_audio_full
from wheeze_bench_calibration import compute_operating_points


SR_TARGET   = 16000
SEG_SAMPLES = 128000   # 8s @ 16kHz


# ── Step 1: Parse ICBHI annotations ──────────────────────────────────────────

def parse_annotations(annotation_path: str) -> list:
    """
    Parse one ICBHI .txt annotation file.
    Returns list of dicts: {start, end, crackle, wheeze}
    """
    cycles = []
    with open(annotation_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                cycles.append({
                    "start":   float(parts[0]),
                    "end":     float(parts[1]),
                    "crackle": int(parts[2]),
                    "wheeze":  int(parts[3]),
                })
            except ValueError:
                continue
    return cycles


def scan_icbhi_directory(icbhi_dir: str) -> dict:
    """
    Scans ICBHI directory. Returns:
    {
        "wheeze_files":  [(wav_path, annotation, cycles_with_wheeze)],
        "clear_files":   [(wav_path, annotation, all_cycles)],
        "skipped":       [path],
    }
    """
    result = {"wheeze_files": [], "clear_files": [], "skipped": []}

    wav_files = sorted(f for f in os.listdir(icbhi_dir) if f.lower().endswith(".wav"))
    print(f"  Found {len(wav_files)} .wav files in {icbhi_dir}")

    for fname in wav_files:
        wav_path = os.path.join(icbhi_dir, fname)
        ann_path = wav_path.replace(".wav", ".txt").replace(".WAV", ".txt")

        if not os.path.exists(ann_path):
            result["skipped"].append(fname)
            continue

        cycles = parse_annotations(ann_path)
        if not cycles:
            result["skipped"].append(fname)
            continue

        wheeze_cycles = [c for c in cycles if c["wheeze"] == 1]
        clear_cycles  = [c for c in cycles if c["wheeze"] == 0 and c["crackle"] == 0]

        if wheeze_cycles:
            result["wheeze_files"].append((wav_path, cycles, wheeze_cycles))
        elif clear_cycles and not any(c["wheeze"] == 1 for c in cycles):
            # Only use as "absent" if NO wheeze anywhere in the recording
            result["clear_files"].append((wav_path, cycles, clear_cycles))

    print(f"  Recordings with wheeze   : {len(result['wheeze_files'])}")
    print(f"  Recordings fully clear   : {len(result['clear_files'])}")
    print(f"  Skipped (no annotation)  : {len(result['skipped'])}")
    return result


# ── Step 2: Extract audio cycles and run Layer 1 ──────────────────────────────

def extract_cycle_audio(wav_path: str, start_s: float, end_s: float) -> np.ndarray:
    """
    Extracts a single respiratory cycle from a recording.
    Pads or trims to exactly 8 seconds (Layer 1 contract).
    Peak-normalises the extracted segment.
    """
    import torchaudio
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample to 16kHz if needed
    if sr != SR_TARGET:
        waveform = torchaudio.functional.resample(
            waveform, sr, SR_TARGET,
            lowpass_filter_width=128, rolloff=0.99,
            resampling_method="sinc_interp_kaiser",
        )

    # Extract cycle
    start_sample = int(start_s * SR_TARGET)
    end_sample   = int(end_s   * SR_TARGET)
    cycle = waveform.squeeze(0).numpy()[start_sample:end_sample].astype("float32")

    # Pad to 8s so the mel extraction is stable
    if len(cycle) < SEG_SAMPLES:
        cycle = np.pad(cycle, (0, SEG_SAMPLES - len(cycle)))
    else:
        cycle = cycle[:SEG_SAMPLES]

    # Peak normalise
    peak = np.abs(cycle).max()
    if peak > 1e-6:
        cycle = cycle / peak

    return cycle


def run_layer1_on_cycles(
    file_list: list,
    label: str,
    max_cycles_per_recording: int = 3,
    max_recordings: int = 50,
    device: str = "cpu",
) -> list:
    """
    Runs Layer 1 on individual respiratory cycles from ICBHI recordings.
    Returns list of raw p_wheeze scores.

    max_cycles_per_recording: avoid over-representing a single recording
    max_recordings: cap total recordings processed (speed vs thoroughness)
    """
    scores = []
    files_processed = 0

    for wav_path, all_cycles, target_cycles in file_list[:max_recordings]:
        fname = os.path.basename(wav_path)
        cycle_scores = []

        for cycle in target_cycles[:max_cycles_per_recording]:
            try:
                audio = extract_cycle_audio(wav_path, cycle["start"], cycle["end"])
                out   = predict_window(audio, device=device)
                p_w   = float(out["p_wheeze"])
                cycle_scores.append(p_w)
            except Exception as e:
                pass

        if cycle_scores:
            mean_score = round(sum(cycle_scores) / len(cycle_scores), 4)
            scores.extend(cycle_scores)
            files_processed += 1
            if files_processed % 10 == 0:
                print(f"    [{label}] {files_processed} recordings processed...")

    print(f"  {label}: {len(scores)} cycle scores from {files_processed} recordings")
    return scores


# ── Step 3: Generate report ───────────────────────────────────────────────────

def generate_icbhi_report(
    scores_wheeze: list,
    scores_clear:  list,
    operating_points: dict,
    output_path: str,
):
    lines = [
        "=" * 65,
        "ICBHI 2017 WHEEZE REFERENCE CALIBRATION — SANITY CHECK ONLY",
        "",
        "!! IMPORTANT: This is NOT a valid Taal threshold.",
        "  ICBHI was recorded with Meditron/AKGC/Littmann stethoscopes.",
        "  Use these values for sanity checking Layer 1 behaviour only.",
        "  For TRUPCR study: collect Taal-specific labelled recordings.",
        "=" * 65,
        "",
        f"Wheeze cycles analysed : {len(scores_wheeze)}",
        f"Clear  cycles analysed : {len(scores_clear)}",
        "",
    ]

    if "error" in operating_points:
        lines.append(f"ERROR: {operating_points['error']}")
    else:
        lines.extend([
            f"AUROC : {operating_points['auroc']}",
            f"  (>0.70 = Layer 1 is discriminating wheeze in ICBHI data)",
            f"  (0.50  = random — model not generalising to this stethoscope)",
            "",
            "Reference operating points (ICBHI — NOT for study use):",
            f"  screening : {operating_points['screening']}  "
            f"(sens={operating_points['screening_sensitivity']}  "
            f"spec={operating_points['screening_specificity']})",
            f"  study     : {operating_points['study']}  "
            f"(sens={operating_points['study_sensitivity']}  "
            f"spec={operating_points['study_specificity']})",
            f"  demo      : {operating_points['demo']}  "
            f"(sens={operating_points['demo_sensitivity']}  "
            f"spec={operating_points['demo_specificity']})",
            "",
            "─" * 65,
            "IF AUROC > 0.70: Layer 1 is detecting wheeze reliably.",
            "  The study threshold from ICBHI will be approximately correct",
            "  but must be verified with Taal recordings before use.",
            "",
            "IF AUROC < 0.70: Layer 1 is not generalising to ICBHI stethoscopes.",
            "  This does NOT mean the model is broken — it may be well-calibrated",
            "  for Taal specifically but not for Meditron/AKGC/Littmann.",
            "  Flag to Prabhu with the AUROC number.",
            "─" * 65,
            "",
            "Paste into consistency_config.py for reference tracking",
            "(status='icbhi_reference_only' — NOT 'calibrated'):",
            "",
            "  ICBHI_REFERENCE_THRESHOLDS = {",
            f'    "screening": {operating_points["screening"]},',
            f'    "study":     {operating_points["study"]},',
            f'    "demo":      {operating_points["demo"]},',
            f'    "auroc":     {operating_points["auroc"]},',
            '    "status":    "icbhi_reference_only",',
            '    "note":      "ICBHI stethoscopes — not Taal. Sanity check only.",',
            "  }",
            "",
            "─" * 65,
            "Score distributions:",
            f"  Wheeze: mean={round(sum(scores_wheeze)/len(scores_wheeze),3)}  "
            f"p25={round(sorted(scores_wheeze)[len(scores_wheeze)//4],3)}  "
            f"p75={round(sorted(scores_wheeze)[3*len(scores_wheeze)//4],3)}",
            f"  Clear:  mean={round(sum(scores_clear)/len(scores_clear),3)}  "
            f"p25={round(sorted(scores_clear)[len(scores_clear)//4],3)}  "
            f"p75={round(sorted(scores_clear)[3*len(scores_clear)//4],3)}",
        ])

    report = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\n  Report saved to: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ICBHI wheeze reference calibration")
    parser.add_argument("--icbhi-dir", type=str, required=True,
                        help="Path to ICBHI 2017 folder containing .wav and .txt files")
    parser.add_argument("--max-recordings", type=int, default=50,
                        help="Max recordings to process per class (default 50 for speed)")
    parser.add_argument("--output", type=str, default="icbhi_reference_report.txt")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not LAYER1_AVAILABLE:
        print("✗ Layer 1 not available — cannot run calibration.")
        return

    if not os.path.isdir(args.icbhi_dir):
        print(f"✗ Directory not found: {args.icbhi_dir}")
        print(f"  Download ICBHI 2017 from:")
        print(f"  https://www.kaggle.com/datasets/nimalanparameshwaran/icbhi-2017-challenge-respiratory-sound-database")
        return

    print("=" * 65)
    print("ICBHI 2017 Wheeze Reference Calibration")
    print("SANITY CHECK ONLY — not a Taal-specific threshold")
    print("=" * 65)

    print(f"\nScanning {args.icbhi_dir} ...")
    file_index = scan_icbhi_directory(args.icbhi_dir)

    if not file_index["wheeze_files"] or not file_index["clear_files"]:
        print("\n✗ Could not find both wheeze and clear recordings.")
        print("  Check that the ICBHI folder contains both .wav and .txt files.")
        return

    print(f"\nRunning Layer 1 on wheeze cycles ...")
    scores_wheeze = run_layer1_on_cycles(
        file_index["wheeze_files"], "wheeze",
        max_recordings=args.max_recordings,
        device=args.device,
    )

    print(f"\nRunning Layer 1 on clear cycles ...")
    scores_clear = run_layer1_on_cycles(
        file_index["clear_files"], "clear",
        max_recordings=args.max_recordings,
        device=args.device,
    )

    if not scores_wheeze or not scores_clear:
        print("\n✗ No scores produced. Check Layer 1 is loading correctly.")
        return

    operating_points = compute_operating_points(scores_wheeze, scores_clear)

    output_path = os.path.join(_HERE, args.output)
    generate_icbhi_report(scores_wheeze, scores_clear, operating_points, output_path)

    # Save raw scores
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "scores_wheeze":   scores_wheeze,
            "scores_clear":    scores_clear,
            "operating_points": operating_points,
            "status":          "icbhi_reference_only",
        }, f, indent=2)
    print(f"\n  Raw scores (JSON): {json_path}")
    print(f"\n  ⚠ Reminder: these thresholds are ICBHI-specific.")
    print(f"  For TRUPCR study, collect Taal-recorded labelled data.")


if __name__ == "__main__":
    main()