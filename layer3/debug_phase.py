"""
Debug: check what the breath segmenter sees on real audio.
Run:
  python layer3/debug_phase.py
"""
import sys, os
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, _HERE)
sys.path.insert(0, _MUSE)
sys.path.insert(0, os.path.join(_MUSE, "layer1"))
os.environ.setdefault("EFFICIENTAT_PATH",
                      os.path.join(_MUSE, "layer1", "EfficientAT"))

from phase_segmenter import extract_193ch_features, run_segmenter, SEG_SR, L1_SR, SEG_N_FFT, SEG_HOP
from layer1.models import load_breath_segmenter

import torchaudio
import torchaudio.functional as F

# ── Load a real zone file ─────────────────────────────────────────────────────
zone_file = os.path.join(_MUSE, "taal recordings", "101_1", "aal_raw.wav")
if not os.path.exists(zone_file):
    # Try to find any wav file
    for root, dirs, files in os.walk(os.path.join(_MUSE, "taal recordings")):
        for f in files:
            if f.endswith(".wav"):
                zone_file = os.path.join(root, f)
                break
        if zone_file:
            break

print(f"Testing on: {zone_file}")

wf_raw, sr_raw = torchaudio.load(zone_file)
print(f"Raw audio:  sr={sr_raw}Hz  shape={tuple(wf_raw.shape)}  "
      f"duration={wf_raw.shape[1]/sr_raw:.1f}s")

# Resample to 16kHz
if sr_raw != L1_SR:
    wf_16k = torchaudio.functional.resample(wf_raw, sr_raw, L1_SR,
                                             lowpass_filter_width=128,
                                             rolloff=0.99,
                                             resampling_method="sinc_interp_kaiser")
else:
    wf_16k = wf_raw

audio_16k = wf_16k.squeeze(0).numpy().astype(np.float32)
# Pad/trim to 8s
TARGET = 128000
if len(audio_16k) < TARGET:
    audio_16k = np.pad(audio_16k, (0, TARGET - len(audio_16k)))
else:
    audio_16k = audio_16k[:TARGET]

print(f"16kHz audio: shape={audio_16k.shape}  "
      f"peak={np.abs(audio_16k).max():.4f}  "
      f"rms={np.sqrt(np.mean(audio_16k**2)):.4f}")

# ── Downsample to 4kHz and check signal ──────────────────────────────────────
wf_4k = torchaudio.functional.resample(
    torch.from_numpy(audio_16k).unsqueeze(0), L1_SR, SEG_SR,
    lowpass_filter_width=64, rolloff=0.99,
    resampling_method="sinc_interp_kaiser"
)
audio_4k = wf_4k.squeeze(0).numpy()
print(f"\n4kHz audio:  shape={audio_4k.shape}  "
      f"peak={np.abs(audio_4k).max():.4f}  "
      f"rms={np.sqrt(np.mean(audio_4k**2)):.4f}")

# ── Extract features ──────────────────────────────────────────────────────────
features = extract_193ch_features(audio_16k)
print(f"\nFeatures:    shape={tuple(features.shape)}")
print(f"  min={features.min():.4f}  max={features.max():.4f}  "
      f"mean={features.mean():.4f}  nonzero={( features > 0.01).float().mean():.2%}")

# Check for all-zero channels
zero_channels = (features.squeeze(0).max(dim=1).values < 0.01).sum().item()
print(f"  Zero/near-zero channels: {zero_channels}/193")

# ── Run segmenter ─────────────────────────────────────────────────────────────
print(f"\nLoading segmenter...")
seg = load_breath_segmenter(device="cpu")

labels = run_segmenter(audio_16k, seg)
counts = {0: (labels==0).sum().item(),
          1: (labels==1).sum().item(),
          2: (labels==2).sum().item()}
total  = len(labels)

print(f"Frame labels ({total} frames at 62.5Hz = {total/62.5:.1f}s):")
print(f"  I (insp) : {counts[0]:4d} frames  ({100*counts[0]//max(total,1)}%)")
print(f"  E (exp)  : {counts[1]:4d} frames  ({100*counts[1]//max(total,1)}%)")
print(f"  P (pause): {counts[2]:4d} frames  ({100*counts[2]//max(total,1)}%)")

if counts[0] == 0 and counts[1] == 0:
    print(f"\n  ALL FRAMES = PAUSE")
    print(f"  Possible causes:")
    print(f"  1. Audio level too low — peak={np.abs(audio_16k).max():.4f}")
    print(f"     If peak < 0.01 the recording may be near-silent")
    print(f"  2. Recording does not contain audible breathing")
    print(f"     (stethoscope positioned incorrectly or not on body)")
    print(f"  3. Segmenter threshold is calibrated for a different recording setup")
    print(f"\n  Raw logits from segmenter (first 5 frames):")
    with torch.no_grad():
        out = seg(features)
    print(f"  shape={tuple(out.shape)}")
    print(f"  logits[:,  :5]:")
    for c, name in [(0,"I"),(1,"E"),(2,"P")]:
        vals = out[0, c, :5].tolist()
        print(f"    {name}: {[f'{v:.3f}' for v in vals]}")
    print(f"\n  Softmax probabilities (first 5 frames):")
    probs = torch.softmax(out[0], dim=0)
    for c, name in [(0,"I"),(1,"E"),(2,"P")]:
        vals = probs[c, :5].tolist()
        print(f"    {name}: {[f'{v:.3f}' for v in vals]}")
else:
    print(f"\n  Phase detection working on real audio")
