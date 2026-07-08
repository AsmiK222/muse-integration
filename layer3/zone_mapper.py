"""
TreBle Respire - Taal Recording Zone Mapper
Handles all three naming conventions found in taal recordings:

  Convention A (101_1):    aal_raw.wav, aar_raw.wav
  Convention B (101_2/3):  01_aal.wav, 01_aar.wav
  Convention C (102_1):    leftapex_raw, rightsuperiorld_raw (no .wav)

Usage:
    python zone_mapper.py                         -- list all patients
    python zone_mapper.py --map                   -- print zone mapping
    python zone_mapper.py --patient "taal recordings/101_1"
    python zone_mapper.py --all
    python zone_mapper.py --all --no-layer1       -- skip acoustic models
"""

import os, sys, argparse, re

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MUSE)
sys.path.insert(0, os.path.join(_MUSE, "layer1"))
sys.path.insert(0, os.path.join(_MUSE, "layer2"))
sys.path.insert(0, os.path.join(_MUSE, "layer2", "utils"))
os.environ.setdefault("EFFICIENTAT_PATH",
                      os.path.join(_MUSE, "layer1", "EfficientAT"))


# ── Mapping tables ────────────────────────────────────────────────────────────

# Convention A & B — short prefix codes
SHORT_PREFIX_TO_ZONE = {
    "aar":  "Z01", "aal":  "Z02",
    "asr":  "Z03", "asl":  "Z04",
    "amr":  "Z05", "aml":  "Z06",   # Z06 = dual zone
    "air":  "Z07", "ail":  "Z08",   # Z08 = dual zone
    "par":  "Z09", "pal":  "Z10",
    "psr":  "Z11", "psl":  "Z12",
    "plbr": "Z13", "plbl": "Z14",
    "pir":  "Z15", "pil":  "Z16",
    # Convention B variants (sometimes seen)
    "ailr": "Z07", "aill": "Z08",   # alternate spellings
    "pmll": "Z14", "pmlr": "Z13",
    "pill": "Z16", "pilr": "Z15",
    "amll": "Z06", "amlr": "Z05",
    "asll": "Z04", "aslr": "Z03",
    "asrl": "Z03", "asrr": "Z03",
    "psll": "Z12", "pslr": "Z11",
    "pmrl": "Z14", "pmrr": "Z13",
}

# Convention C — full anatomical names (102_1 style)
FULL_NAME_TO_ZONE = {
    # Right anterior
    "rightapex":              "Z01",
    "rightsuperiorld":        "Z03",
    "rightmiddlelobe":        "Z05",
    "rightinferiorlobe":      "Z07",
    # Left anterior
    "leftapex":               "Z02",
    "leftsuperiorlobe":         "Z04",
    "leftsuperiorld2":          "Z04",
    "leftsuperiorld":         "Z04",
    "leftmiddlelobe":         "Z06",
    "leftinferiorlobe":       "Z08",
    # Right posterior
    "post_rightapex":         "Z09",
    "post_rightsuperiorld":   "Z11",
    "post_rightlungbase":     "Z13",
    "post_rightinferiorlobe": "Z15",
    # Left posterior
    "post_leftapices":        "Z10",
    "post_leftsuperiorld":    "Z12",
    "post_leftlungbase":      "Z14",
    "post_leftinferiorlobe":  "Z16",
    # Alternate forms seen in 102_1
    "post_apices":            "Z09",   # ambiguous — assign right apex
    "rightsuperiorlobe":      "Z03",
    "leftsuperiorld2":        "Z04",
}


def _normalise(name: str) -> str:
    """Lowercase, remove spaces, underscores, hyphens for fuzzy matching."""
    return re.sub(r"[\s_\-]+", "", name.lower())


def _stem(filename: str) -> str:
    """Remove _raw suffix and extension, normalise."""
    s = filename
    for ext in (".wav", ".WAV", ".mp3", ".flac", ".ogg"):
        s = s.replace(ext, "")
    s = re.sub(r"_raw$", "", s)
    s = re.sub(r"^0\d_", "", s)   # strip leading number prefix (01_, 02_)
    return s.strip("_")


def _match_zone(stem: str) -> str | None:
    """Try all mapping strategies. Returns zone_id or None."""
    norm = _normalise(stem)

    # 1. Exact short prefix match
    if stem in SHORT_PREFIX_TO_ZONE:
        return SHORT_PREFIX_TO_ZONE[stem]

    # 2. Exact full name match
    if norm in {_normalise(k): v for k, v in FULL_NAME_TO_ZONE.items()}:
        for k, v in FULL_NAME_TO_ZONE.items():
            if _normalise(k) == norm:
                return v

    # 3. Fuzzy: check if any full name key is contained in the normalised stem
    for k, v in FULL_NAME_TO_ZONE.items():
        if _normalise(k) in norm or norm in _normalise(k):
            return v

    # 4. Fuzzy: check short prefixes against stem without number prefix
    clean = re.sub(r"^0\d", "", stem).strip("_")
    if clean in SHORT_PREFIX_TO_ZONE:
        return SHORT_PREFIX_TO_ZONE[clean]

    return None


def get_zone_audio_files(folder: str) -> dict:
    """
    Returns {zone_id: full_path} for all audio files in a patient folder.
    Handles all three naming conventions automatically.
    """
    files = {}
    unmatched = []

    if not os.path.isdir(folder):
        return files

    for fname in sorted(os.listdir(folder)):
        # Accept .wav files and files with _raw suffix but no extension
        is_audio = (fname.lower().endswith(".wav") or
                    fname.endswith("_raw") or
                    fname.lower().endswith(".flac"))
        if not is_audio:
            continue

        stem    = _stem(fname)
        zone_id = _match_zone(stem)

        if zone_id:
            # If multiple files map to same zone, keep the first
            if zone_id not in files:
                files[zone_id] = os.path.join(folder, fname)
        else:
            unmatched.append(fname)

    if unmatched:
        print(f"  ⚠ Unmatched files (zone unknown): {unmatched}")

    return files


def detect_convention(folder: str) -> str:
    """Detect which naming convention a folder uses."""
    files = os.listdir(folder) if os.path.isdir(folder) else []
    for f in files:
        if re.match(r"^\d+_[a-z]", f):     return "B (01_aal.wav)"
        if re.match(r"^[a-z]{2,4}_raw", f): return "A (aal_raw.wav)"
        if "lobe" in f.lower() or "apex" in f.lower(): return "C (rightapex_raw)"
    return "unknown"


def list_patients(root: str) -> list:
    if not os.path.isdir(root):
        return []
    return [os.path.join(root, d) for d in sorted(os.listdir(root))
            if os.path.isdir(os.path.join(root, d))]


def default_clinical() -> dict:
    """
    REPLACE these with real patient values before interpreting results.
    Collect: age, sex, BMI, fever, SpO2, symptom duration, cough character,
             purulence change, dyspnea increase, sputum volume, COPD, biomass.
    """
    return {
        "age":                    21,
        "sex":                    "F",
        "bmi":                    22.0,
        "fever":                  False,
        "symptom_duration_days":  0,
        "cough_character":        "dry",
        "sputum_purulence_change":False,
        "dyspnea_increase":       False,
        "sputum_volume_increase": False,
        "spo2_pct":               98.0,
        "fev1_fvc_known":         None,
        "comorbidity_obstructive":False,
        "biomass_exposure":       False,
        "prior_antibiotic_use":   False,
        "season":                 "other",
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--patient",   type=str, default=None)
    parser.add_argument("--all",       action="store_true")
    parser.add_argument("--map",       action="store_true")
    parser.add_argument("--no-layer1", action="store_true")
    args = parser.parse_args()

    recordings_root = os.path.join(_MUSE, "taal recordings")

    if args.map:
        print("\nShort prefix -> Zone:")
        for pfx, z in sorted(SHORT_PREFIX_TO_ZONE.items(), key=lambda x: x[1]):
            print(f"  {pfx:<8} -> {z}")
        print("\nFull name -> Zone:")
        for name, z in sorted(FULL_NAME_TO_ZONE.items(), key=lambda x: x[1]):
            print(f"  {name:<30} -> {z}")
        return

    if args.patient:
        folders = [args.patient]
    elif args.all:
        folders = list_patients(recordings_root)
        if not folders:
            print(f"No patient folders found in: {recordings_root}")
            return
    else:
        patients = list_patients(recordings_root)
        if not patients:
            print(f"No recordings found in: {recordings_root}")
            return
        print(f"\nPatients in '{recordings_root}':")
        print(f"  {'Folder':<15} {'Convention':<25} {'Zones'}")
        print(f"  {'-'*50}")
        for p in patients:
            zf   = get_zone_audio_files(p)
            conv = detect_convention(p)
            print(f"  {os.path.basename(p):<15} {conv:<25} {len(zf)} zones found")
        print(f"\nRun one:  python zone_mapper.py --patient \"taal recordings/101_1\"")
        print(f"Run all:  python zone_mapper.py --all")
        print(f"No models: python zone_mapper.py --all --no-layer1")
        return

    from pipeline_e2e import run_patient

    all_results = []
    for folder in folders:
        pid = os.path.basename(folder)
        zf  = get_zone_audio_files(folder)

        if not zf:
            print(f"  [{pid}] no audio files matched")
            continue

        conv    = detect_convention(folder)
        missing = [f"Z{i:02d}" for i in range(1, 17) if f"Z{i:02d}" not in zf]

        print(f"\n{'='*60}")
        print(f"Patient : {pid}   Convention: {conv}")
        print(f"Zones   : {len(zf)} found  |  Missing: {' '.join(missing) or 'none'}")
        print(f"WARNING : Using DEFAULT clinical values.")
        print(f"          Edit default_clinical() in zone_mapper.py with real data.")

        out = run_patient(
            zone_audio_files = zf,
            clinical         = default_clinical(),
            use_layer1       = not args.no_layer1,
            verbose          = True,
        )

        if "error" not in out:
            all_results.append({
                "pid": pid, "zone": out["zone"],
                "p": out["p_antibiotics"],
                "unc": out["uncertainty_flag"],
                "esc": out["escalate"],
                "disp": out["disposition"],
            })

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"SUMMARY — {len(all_results)} patients")
        print(f"{'='*60}")
        print(f"  {'Patient':<12} {'Zone':<8} {'P(abx)':>7}  Disposition")
        print(f"  {'-'*45}")
        for r in all_results:
            esc = " ESCALATE" if r["esc"] else ""
            print(f"  {r['pid']:<12} {r['zone']:<8} {r['p']:>7.3f}  {r['disp']}{esc}")


if __name__ == "__main__":
    main()
