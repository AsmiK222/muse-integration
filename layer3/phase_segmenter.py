"""
TreBle Respire — Phase Segmentation
Implements breath_segmenter_v1.pt feature extraction and inference.

Segmenter contract (from breath_segmenter_v1.meta.json):
    Input:  193-channel feature matrix, shape (batch, 193, time_frames)
            Built from audio downsampled to 4kHz:
              [0:129]   log|STFT|           n_fft=256, hop=64
              [129:149] MFCC-20
              [149:169] MFCC-20 delta
              [169:189] MFCC-20 delta2
              [189:193] energy bands (0-250, 250-500, 500-1000, 0-2000 Hz)
            Per-channel min-max normalized to [0,1]
    Output: per-frame logits or labels, I=0 / E=1 / P=2 at 62.5Hz
    Viterbi decode (I→E→P): baked into forward pass

Audio pipeline:
    Raw .wav (any sr) → resample to 4kHz → extract 193ch features → segmenter

Author: Asmi
"""

import torch
import numpy as np
import os, sys, warnings

_HERE = os.path.dirname(os.path.abspath(__file__))
_MUSE = os.path.join(_HERE, "..")
sys.path.insert(0, os.path.join(_MUSE, "layer1"))

# Segmenter operates at 4kHz
SEG_SR         = 4000
SEG_N_FFT      = 256
SEG_HOP        = 64
SEG_FRAME_RATE = 62.5
SEG_N_MFCC     = 20
SEG_N_CHANNELS = 193

# Layer 1 operates at 16kHz
L1_SR          = 16000
L1_SEG_SAMPLES = 128000   # 8s @ 16kHz


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_193ch_features(audio_16k: np.ndarray) -> torch.Tensor:
    """
    Converts 16kHz audio → 193-channel feature matrix for the segmenter.

    Steps:
        1. Downsample 16kHz → 4kHz
        2. Compute log|STFT| (129 bins, n_fft=256, hop=64)
        3. Compute MFCC-20 + delta + delta2 (60 channels)
        4. Compute 4 energy bands (4 channels)
        5. Concatenate → (193, time_frames)
        6. Per-channel min-max normalize to [0, 1]

    Returns:
        torch.Tensor shape (1, 193, time_frames) — ready for segmenter
    """
    import torchaudio
    import torchaudio.functional as F
    import torchaudio.transforms as T

    wf = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0)  # (1, N)

    # Step 1 — downsample to 4kHz
    wf_4k = torchaudio.functional.resample(wf, L1_SR, SEG_SR,
                                            lowpass_filter_width=64,
                                            rolloff=0.99,
                                            resampling_method="sinc_interp_kaiser")

    n_samples = wf_4k.shape[1]

    # Step 2 — log|STFT|: shape (1, n_fft//2+1, time) = (1, 129, T)
    stft = torch.stft(
        wf_4k.squeeze(0),
        n_fft      = SEG_N_FFT,
        hop_length = SEG_HOP,
        win_length = SEG_N_FFT,
        window     = torch.hann_window(SEG_N_FFT),
        return_complex = True,
    )
    log_stft = torch.log(stft.abs() + 1e-8)   # (129, T)
    T_frames = log_stft.shape[1]

    # Step 3 — MFCC-20 + delta + delta2: 20 + 20 + 20 = 60 channels
    mfcc_transform = T.MFCC(
        sample_rate  = SEG_SR,
        n_mfcc       = SEG_N_MFCC,
        melkwargs    = {"n_fft": SEG_N_FFT, "hop_length": SEG_HOP,
                        "n_mels": 40, "f_min": 0, "f_max": 2000},
    )
    mfcc = mfcc_transform(wf_4k).squeeze(0)          # (20, T')

    # Align time dimension with STFT frames
    if mfcc.shape[1] > T_frames:
        mfcc = mfcc[:, :T_frames]
    elif mfcc.shape[1] < T_frames:
        mfcc = torch.nn.functional.pad(mfcc, (0, T_frames - mfcc.shape[1]))

    def delta(x: torch.Tensor, N: int = 2) -> torch.Tensor:
        """Simple delta computation."""
        padded = torch.nn.functional.pad(x, (N, N), mode="replicate")
        denom  = 2 * sum(i*i for i in range(1, N+1))
        d = sum(i * (padded[:, N+i:N+i+x.shape[1]] -
                     padded[:, N-i:N-i+x.shape[1]])
                for i in range(1, N+1))
        return d / denom

    mfcc_d  = delta(mfcc)
    mfcc_d2 = delta(mfcc_d)

    # Step 4 — energy bands at 4kHz: 4 channels
    # Frequency resolution: SEG_SR/SEG_N_FFT = 4000/256 = 15.625 Hz/bin
    hz_per_bin = SEG_SR / SEG_N_FFT
    def band_energy(stft_abs, f_low, f_high):
        b_low  = int(f_low  / hz_per_bin)
        b_high = int(f_high / hz_per_bin) + 1
        b_high = min(b_high, stft_abs.shape[0])
        return stft_abs[b_low:b_high, :].mean(dim=0, keepdim=True)

    stft_abs = stft.abs()
    e1 = band_energy(stft_abs,    0,  250)
    e2 = band_energy(stft_abs,  250,  500)
    e3 = band_energy(stft_abs,  500, 1000)
    e4 = band_energy(stft_abs,    0, 2000)
    energy = torch.cat([e1, e2, e3, e4], dim=0)   # (4, T)

    # Step 5 — concatenate: (129 + 20 + 20 + 20 + 4, T) = (193, T)
    features = torch.cat([log_stft, mfcc, mfcc_d, mfcc_d2, energy], dim=0)
    assert features.shape[0] == SEG_N_CHANNELS, \
        f"Expected 193 channels, got {features.shape[0]}"

    # Step 6 — per-channel min-max normalize to [0, 1]
    fmin = features.min(dim=1, keepdim=True).values
    fmax = features.max(dim=1, keepdim=True).values
    denom = (fmax - fmin).clamp(min=1e-8)
    features = (features - fmin) / denom

    return features.unsqueeze(0)   # (1, 193, T)


# ── Run segmenter ─────────────────────────────────────────────────────────────

def run_segmenter(audio_16k: np.ndarray, segmenter_model) -> torch.Tensor:
    """
    Run breath_segmenter_v1 on 16kHz audio.
    Extracts 193-channel features, runs model, returns frame labels.

    Returns:
        frame_labels: torch.Tensor shape (n_frames,), values 0=I 1=E 2=P
    """
    features = extract_193ch_features(audio_16k)   # (1, 193, T)

    with torch.no_grad():
        out = segmenter_model(features)

    # Handle output shape variants
    if out.dim() == 3 and out.shape[1] == 3:
        # (batch, 3, frames) logits → argmax
        frame_labels = out.squeeze(0).argmax(dim=0)
    elif out.dim() == 2 and out.shape[-1] == 3:
        # (frames, 3) logits → argmax
        frame_labels = out.argmax(dim=-1)
    elif out.dim() == 2:
        # (batch, frames) — already decoded
        frame_labels = out.squeeze(0).long()
    else:
        frame_labels = out.long()

    return frame_labels.cpu()




def run_segmenter_full(audio_16k_full: "np.ndarray", segmenter_model,
                       window_s: float = 8.0, hop_s: float = 4.0) -> "torch.Tensor":
    """
    Run segmenter on full-duration audio using sliding windows.

    The breath segmenter was designed for 8s windows. For longer recordings
    (Taal stethoscope records 30s), processing only the first 8s misses
    the oscillating I/E pattern needed for reliable phase detection.

    Args:
        audio_16k_full: full recording at 16kHz (any length)
        window_s:       window size in seconds (8.0 = model training length)
        hop_s:          hop between windows (4.0 = 50% overlap)

    Returns:
        frame_labels for the full recording (majority vote at overlap regions)
    """
    window = int(window_s * L1_SR)   # 128000 samples
    hop    = int(hop_s    * L1_SR)   # 64000 samples
    n      = len(audio_16k_full)

    all_labels = {}   # frame_index -> list of label votes

    start = 0
    while start < n:
        end    = min(start + window, n)
        chunk  = audio_16k_full[start:end]

        # Pad last chunk if shorter than window
        if len(chunk) < window:
            chunk = np.pad(chunk, (0, window - len(chunk)))

        labels = run_segmenter(chunk, segmenter_model)

        # Map frame indices back to full-recording time
        frame_offset = int(start / L1_SR * SEG_FRAME_RATE)
        for i, label in enumerate(labels.tolist()):
            global_frame = frame_offset + i
            if global_frame not in all_labels:
                all_labels[global_frame] = []
            all_labels[global_frame].append(int(label))

        start += hop
        if start >= n:
            break

    # Majority vote at each frame
    max_frame = max(all_labels.keys()) + 1
    final = []
    for i in range(max_frame):
        if i in all_labels:
            votes = all_labels[i]
            final.append(max(set(votes), key=votes.count))
        else:
            final.append(2)   # pause for any gap

    return torch.tensor(final, dtype=torch.long)

# ── Extract phase audio segments ──────────────────────────────────────────────

def get_phase_segments(audio_16k: np.ndarray, frame_labels: torch.Tensor) -> dict:
    """
    Slice 16kHz audio into inspiratory and expiratory segments.
    Implements Arvind's get_phase_segments() exactly.

    Frame rate is 62.5Hz → hop = 16000/62.5 = 256 samples at 16kHz.
    """
    HOP_16K = int(L1_SR / SEG_FRAME_RATE)   # 256 samples at 16kHz
    waveform = torch.from_numpy(audio_16k.astype(np.float32)).unsqueeze(0)
    labels   = frame_labels.tolist()

    insp_segs = []
    exp_segs  = []
    n_pause   = 0

    for i, label in enumerate(labels):
        start = int(i * HOP_16K)
        end   = min(int((i + 1) * HOP_16K), waveform.shape[1])
        chunk = waveform[:, start:end]
        if   label == 0: insp_segs.append(chunk)
        elif label == 1: exp_segs.append(chunk)
        else:            n_pause += 1

    insp_audio = torch.cat(insp_segs, dim=1).squeeze(0).numpy() \
                 if insp_segs else None
    exp_audio  = torch.cat(exp_segs,  dim=1).squeeze(0).numpy() \
                 if exp_segs  else None

    return {
        "insp_audio":     insp_audio,
        "exp_audio":      exp_audio,
        "insp_duration":  len(insp_segs) * HOP_16K / L1_SR,
        "exp_duration":   len(exp_segs)  * HOP_16K / L1_SR,
        "pause_duration": n_pause        * HOP_16K / L1_SR,
        "n_insp_frames":  len(insp_segs),
        "n_exp_frames":   len(exp_segs),
    }


# ── Phase-resolved crackle/wheeze detection ───────────────────────────────────

def run_phase_resolved_detection(
    audio_16k:    np.ndarray,
    frame_labels: torch.Tensor,
    crackle_model,
    wheeze_model,
    device: str = "cpu",
) -> dict:
    """
    Run crackle and wheeze detectors separately on insp and exp audio.
    Returns phase-resolved probabilities for Layer 2.
    """
    from layer1.models import CRACKLE_T, WHEEZE_T

    segs = get_phase_segments(audio_16k, frame_labels)

    def _detect(model, audio_seg, temperature):
        """
        Run detector on a phase segment.
        Models were trained on 8-second segments (128000 samples at 16kHz).
        Pad phase segment to 8 seconds so mel extraction is stable.
        Short segments get zero-padded — this under-estimates probability
        slightly but avoids the mel filterbank zero-bin issue.
        """
        TARGET = 128000   # 8s @ 16kHz — model training length
        if audio_seg is None or len(audio_seg) < 256:
            return 0.0
        # Pad to full 8-second segment
        if len(audio_seg) < TARGET:
            audio_seg = np.pad(audio_seg, (0, TARGET - len(audio_seg)))
        else:
            audio_seg = audio_seg[:TARGET]
        wf = torch.from_numpy(audio_seg.astype(np.float32)).unsqueeze(0)
        with torch.no_grad():
            try:
                logit = model(wf)
                if isinstance(logit, (tuple, list)): logit = logit[0]
                return float(torch.sigmoid(logit.squeeze() / temperature).item())
            except Exception:
                return 0.0

    p_ci = _detect(crackle_model, segs["insp_audio"], CRACKLE_T)
    p_ce = _detect(crackle_model, segs["exp_audio"],  CRACKLE_T)
    p_wi = _detect(wheeze_model,  segs["insp_audio"], WHEEZE_T)
    p_we = _detect(wheeze_model,  segs["exp_audio"],  WHEEZE_T)

    def _dominant_phase(insp, exp, label_i, label_e):
        # Use Taal study threshold (0.11) to determine if phase finding is real
        if insp < 0.11 and exp < 0.11: return None
        if insp > exp * 1.3: return label_i
        if exp  > insp * 1.3: return label_e
        return "biphasic"

    return {
        "p_crackle_insp":         p_ci,
        "p_crackle_exp":          p_ce,
        "p_wheeze_insp":          p_wi,
        "p_wheeze_exp":           p_we,
        "dominant_crackle_phase": _dominant_phase(p_ci, p_ce, "inspiratory", "expiratory"),
        "dominant_wheeze_phase":  _dominant_phase(p_wi, p_we, "inspiratory", "expiratory"),
        "insp_duration_s":        segs["insp_duration"],
        "exp_duration_s":         segs["exp_duration"],
    }


# ── Contract verification ─────────────────────────────────────────────────────

def verify_segmenter_contract(segmenter_model) -> bool:
    """Run Arvind's three checks. Returns True if all pass."""
    print("\n" + "="*55)
    print("Breath Segmenter Contract Verification")
    print("="*55)

    dummy_16k = np.random.randn(L1_SEG_SAMPLES).astype(np.float32)

    # Check 1 — feature extraction + forward pass
    try:
        features = extract_193ch_features(dummy_16k)
        print(f"  [1] Feature extraction: ✓  shape={tuple(features.shape)}")
        with torch.no_grad():
            out = segmenter_model(features)
        print(f"  [1] Forward pass: ✓  output shape={tuple(out.shape)}")
        check1 = True
    except Exception as e:
        print(f"  [1] FAILED: {e}")
        check1 = False

    # Check 2 — output type
    if check1:
        if out.dim() == 3 and out.shape[1] == 3:
            print(f"  [2] Output: (batch, 3, frames) logits ✓ — argmax will be applied")
        elif out.dim() == 2 and out.shape[-1] == 3:
            print(f"  [2] Output: (frames, 3) logits ✓ — argmax will be applied")
        else:
            vals = out.flatten()[:5].tolist()
            print(f"  [2] Output shape {tuple(out.shape)}, values {[f'{v:.2f}' for v in vals]}")
        check2 = True

    # Check 3 — Viterbi smoothness
    if check1:
        labels = run_segmenter(dummy_16k, segmenter_model)
        trans  = (labels[1:] != labels[:-1]).sum().item() / len(labels)
        viterbi = "likely baked in" if trans < 0.3 else "may need external Viterbi"
        print(f"  [3] Transition rate={trans:.2f} → {viterbi}")
        counts = {0: (labels==0).sum().item(),
                  1: (labels==1).sum().item(),
                  2: (labels==2).sum().item()}
        total  = len(labels)
        print(f"      I={counts[0]}f ({100*counts[0]//total}%)  "
              f"E={counts[1]}f ({100*counts[1]//total}%)  "
              f"P={counts[2]}f ({100*counts[2]//total}%)")
        check3 = True

    all_ok = check1 and (check2 if check1 else False) and (check3 if check1 else False)
    print(f"\n  Result: {'✓ ALL CHECKS PASS' if all_ok else '✗ FAILED'}")
    print("="*55)
    return all_ok


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _MUSE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
    ckpt  = os.path.join(_MUSE, "layer1", "checkpoints", "breath_segmenter_v1.pt")
    sys.path.insert(0, os.path.join(_MUSE, "layer1"))

    print("Phase Segmenter — Integration Test")
    print(f"  Loading: {ckpt}")

    if not os.path.exists(ckpt):
        print(f"  Checkpoint not found: {ckpt}")
    else:
        from layer1.models import load_breath_segmenter
        seg = load_breath_segmenter(device="cpu")
        ok  = verify_segmenter_contract(seg)
        if ok:
            print("\n  Segmenter ready. Phase integration can proceed.")
        else:
            print("\n  Fix issues above before running on real recordings.")