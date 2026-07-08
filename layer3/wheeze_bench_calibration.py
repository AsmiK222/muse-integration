# wheeze_bench_calibration.py
#
# Per Arvind Decision 2: "Run a brief bench calibration before study start
# to get a TAAL-specific [wheeze] threshold."
#
# This script takes a folder of labelled recordings (known wheeze present /
# known wheeze absent), runs them through Layer 1, and computes the
# Taal-specific operating points: screening / study / demo — exactly
# mirroring the structure already validated for crackle in crackle_v1.meta.json.
#
# Usage:
#   1. Collect recordings into two folders:
#        bench_data/wheeze_present/*.wav   (confirmed wheeze on auscultation)
#        bench_data/wheeze_absent/*.wav    (confirmed clear breath sounds)
#      Minimum 15-20 recordings per class for a usable ROC curve.
#      More is better — this mirrors how crackle_v1's taal_thresholds were derived.
#
#   2. Run:
#        python wheeze_bench_calibration.py --bench-dir bench_data
#
#   3. Script outputs a taal_thresholds block to paste into wheeze_v1.meta.json,
#      plus a full calibration report for Arvind's review.

import os
import sys
import argparse
import json
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MUSE)
sys.path.insert(0, os.path.join(_MUSE, "layer1"))
os.environ.setdefault("EFFICIENTAT_PATH",
                      os.path.join(_MUSE, "layer1", "EfficientAT"))

try:
    from layer1 import predict_window, load_wheeze
    LAYER1_AVAILABLE = True
except Exception as e:
    LAYER1_AVAILABLE = False
    print(f"⚠ Layer 1 not available: {e}")

from pipeline_e2e import load_audio_full


# ── Step 1: collect raw scores from labelled recordings ──────────────────────

def collect_scores(bench_dir: str, device: str = "cpu") -> dict:
    """
    Runs every .wav file in bench_dir/wheeze_present and bench_dir/wheeze_absent
    through Layer 1's wheeze detector. Returns raw p_wheeze scores per class.

    Per Arvind decision 2: "Always store raw scores." This function stores
    every individual score, not just summary statistics.
    """
    present_dir = os.path.join(bench_dir, "wheeze_present")
    absent_dir  = os.path.join(bench_dir, "wheeze_absent")

    scores = {"present": [], "absent": []}
    files  = {"present": [], "absent": []}

    for label, d in [("present", present_dir), ("absent", absent_dir)]:
        if not os.path.isdir(d):
            print(f"  ⚠ Folder not found: {d}")
            continue
        wav_files = sorted(f for f in os.listdir(d) if f.lower().endswith(".wav"))
        print(f"\n  {label.upper()} ({len(wav_files)} files):")
        for fname in wav_files:
            path = os.path.join(d, fname)
            try:
                _, audio_8s = load_audio_full(path)
                out = predict_window(audio_8s, device=device)
                p_wheeze = float(out["p_wheeze"])
                scores[label].append(p_wheeze)
                files[label].append(fname)
                print(f"    {fname:<40} p_wheeze={p_wheeze:.4f}")
            except Exception as e:
                print(f"    {fname:<40} FAILED: {e}")

    return {
        "scores_present": scores["present"],
        "scores_absent":  scores["absent"],
        "files_present":  files["present"],
        "files_absent":   files["absent"],
    }


# ── Step 2: compute ROC-style operating points ────────────────────────────────

def compute_operating_points(scores_present: list, scores_absent: list) -> dict:
    """
    Computes three operating points mirroring crackle_v1.meta.json structure:
        screening — maximise sensitivity (catch everything, accept false positives)
        study     — balanced sensitivity/specificity (recommended default)
        demo      — maximise specificity (fewer false positives)

    Method: scan candidate thresholds, compute sensitivity and specificity
    at each, select the threshold matching each target operating point.
    """
    if not scores_present or not scores_absent:
        return {"error": "Insufficient data — need both present and absent recordings"}

    all_scores = sorted(set(scores_present + scores_absent))
    candidates = np.linspace(min(all_scores), max(all_scores), 200)

    results = []
    for thresh in candidates:
        tp = sum(1 for s in scores_present if s > thresh)
        fn = sum(1 for s in scores_present if s <= thresh)
        tn = sum(1 for s in scores_absent  if s <= thresh)
        fp = sum(1 for s in scores_absent  if s > thresh)

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        results.append({"threshold": thresh, "sensitivity": sens, "specificity": spec})

    # Screening: highest threshold achieving sensitivity >= 0.95
    screening_candidates = [r for r in results if r["sensitivity"] >= 0.95]
    screening = max(screening_candidates, key=lambda r: r["threshold"]) \
                if screening_candidates else min(results, key=lambda r: r["threshold"])

    # Study: maximise sensitivity + specificity (Youden's J statistic)
    for r in results:
        r["youden_j"] = r["sensitivity"] + r["specificity"] - 1.0
    study = max(results, key=lambda r: r["youden_j"])

    # Demo: highest threshold achieving specificity >= 0.90
    demo_candidates = [r for r in results if r["specificity"] >= 0.90]
    demo = min(demo_candidates, key=lambda r: r["threshold"]) \
           if demo_candidates else max(results, key=lambda r: r["threshold"])

    # AUROC via trapezoidal rule on sensitivity/specificity pairs
    sorted_results = sorted(results, key=lambda r: r["threshold"])
    fprs = [1 - r["specificity"] for r in sorted_results]
    tprs = [r["sensitivity"]     for r in sorted_results]
    auroc = abs(np.trapezoid(tprs, fprs)) if hasattr(np, 'trapezoid') else abs(np.trapz(tprs, fprs))

    return {
        "screening": round(float(screening["threshold"]), 4),
        "study":     round(float(study["threshold"]), 4),
        "demo":      round(float(demo["threshold"]), 4),
        "auroc":     round(float(auroc), 4),
        "screening_sensitivity": round(screening["sensitivity"], 4),
        "screening_specificity": round(screening["specificity"], 4),
        "study_sensitivity":     round(study["sensitivity"], 4),
        "study_specificity":     round(study["specificity"], 4),
        "demo_sensitivity":      round(demo["sensitivity"], 4),
        "demo_specificity":      round(demo["specificity"], 4),
    }


# ── Step 3: generate report + meta.json patch ─────────────────────────────────

def generate_report(bench_results: dict, operating_points: dict, output_path: str):
    """Writes a full calibration report for Arvind's review."""
    n_present = len(bench_results["scores_present"])
    n_absent  = len(bench_results["scores_absent"])

    lines = [
        "=" * 65,
        "WHEEZE BENCH CALIBRATION REPORT",
        "Per Arvind Decision 2 — Taal-specific wheeze threshold",
        "=" * 65,
        "",
        f"Sample sizes:",
        f"  wheeze_present : n={n_present}",
        f"  wheeze_absent  : n={n_absent}",
        "",
    ]

    if n_present < 15 or n_absent < 15:
        lines.append("⚠ WARNING: sample size below recommended minimum (n=15-20 per class).")
        lines.append("  Results below should be treated as preliminary, not final calibration.")
        lines.append("")

    if "error" in operating_points:
        lines.append(f"ERROR: {operating_points['error']}")
    else:
        lines.extend([
            f"AUROC: {operating_points['auroc']}",
            "",
            "Operating points (mirrors crackle_v1.meta.json structure):",
            "",
            f"  screening: {operating_points['screening']}",
            f"    sensitivity={operating_points['screening_sensitivity']}  "
            f"specificity={operating_points['screening_specificity']}",
            "",
            f"  study:     {operating_points['study']}",
            f"    sensitivity={operating_points['study_sensitivity']}  "
            f"specificity={operating_points['study_specificity']}",
            "",
            f"  demo:      {operating_points['demo']}",
            f"    sensitivity={operating_points['demo_sensitivity']}  "
            f"specificity={operating_points['demo_specificity']}",
            "",
            "─" * 65,
            "Paste into wheeze_v1.meta.json:",
            "─" * 65,
            "",
            '  "taal_thresholds": {',
            f'    "screening": {operating_points["screening"]},',
            f'    "study": {operating_points["study"]},',
            f'    "demo": {operating_points["demo"]}',
            "  },",
            "",
            "─" * 65,
            "Raw scores (Arvind decision 2: always store raw scores)",
            "─" * 65,
        ])
        lines.append("\nwheeze_present:")
        for f, s in zip(bench_results["files_present"], bench_results["scores_present"]):
            lines.append(f"  {f:<40} {s:.4f}")
        lines.append("\nwheeze_absent:")
        for f, s in zip(bench_results["files_absent"], bench_results["scores_absent"]):
            lines.append(f"  {f:<40} {s:.4f}")

    report = "\n".join(lines)
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\n{report}")
    print(f"\n  Report saved to: {output_path}")


# ── Step 4: update consistency_config.py with calibrated status ──────────────

def update_calibration_status(new_threshold: float, output_path: str = None):
    """
    Prints the exact line change needed in consistency_config.py to mark
    the wheeze threshold as calibrated rather than provisional.
    Does NOT auto-edit the file — Arvind should review and approve first.
    """
    print("\n" + "=" * 65)
    print("To mark wheeze threshold as CALIBRATED (after Arvind approval):")
    print("=" * 65)
    print(f"""
  Edit consistency_config.py:

    WHEEZE_THRESHOLD        = {new_threshold}     # was 0.243
    WHEEZE_THRESHOLD_STATUS = "calibrated"      # was "provisional"

  Also update THRESHOLD_PROVENANCE["wheeze"]:
    "status": "calibrated",
    "source": "Taal bench calibration, n={{n_present}}+{{n_absent}}, AUROC={{auroc}}"

  Do NOT make this change without Arvind's explicit sign-off — this
  threshold drives the endobronchial suppression and obstructive pathways.
    """)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Wheeze bench calibration (Arvind decision 2)")
    parser.add_argument("--bench-dir", type=str, required=True,
                        help="Folder containing wheeze_present/ and wheeze_absent/ subfolders")
    parser.add_argument("--output", type=str, default="wheeze_calibration_report.txt",
                        help="Output report path")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if not LAYER1_AVAILABLE:
        print("✗ Layer 1 not available — cannot run bench calibration.")
        print("  Ensure layer1 package and checkpoints are accessible.")
        return

    print("=" * 65)
    print("Wheeze Bench Calibration — Arvind Decision 2")
    print("=" * 65)
    print(f"\nScanning: {args.bench_dir}")

    bench_results = collect_scores(args.bench_dir, device=args.device)

    n_present = len(bench_results["scores_present"])
    n_absent  = len(bench_results["scores_absent"])

    if n_present == 0 or n_absent == 0:
        print(f"\n✗ Insufficient data: present={n_present}, absent={n_absent}")
        print(f"  Both folders need at least one recording each.")
        print(f"  Expected structure:")
        print(f"    {args.bench_dir}/wheeze_present/*.wav")
        print(f"    {args.bench_dir}/wheeze_absent/*.wav")
        return

    operating_points = compute_operating_points(
        bench_results["scores_present"], bench_results["scores_absent"]
    )

    output_path = os.path.join(_HERE, args.output)
    generate_report(bench_results, operating_points, output_path)

    if "error" not in operating_points:
        update_calibration_status(operating_points["study"])

    # Save raw scores as JSON too, for re-analysis
    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w") as f:
        json.dump({
            "bench_results": bench_results,
            "operating_points": operating_points,
        }, f, indent=2)
    print(f"\n  Raw data (JSON): {json_path}")


if __name__ == "__main__":
    main()
