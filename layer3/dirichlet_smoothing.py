"""
TreBle Respire — Layer 3: Dirichlet Smoothing Utility

Apply to ALL phase-related and Wheeze_Compound CPTs immediately after setting q_i values.
Do NOT apply to CPTs with hard likelihood-ratio evidence (acoustic LR CPTs) — only phase and compound.

Formula (from sprint spec):
    smoothed = (raw * N_eff + alpha) / (N_eff + alpha * n_states)
    where N_eff = 20 (effective sample size), alpha = 2 (pseudocount)

Every CPT cell smoothed by this function must have the comment:
    # qualitative prior — weak evidence, update first with TRUPCR data

Reference: Heckerman 1995 (MS-TR-95-06) — pseudocount smoothing for sparse CPTs
"""

import numpy as np
from typing import Union


def apply_dirichlet_smoothing(
    cpt_array: np.ndarray,
    alpha: float = 2.0,
    n_effective: int = 20,
) -> np.ndarray:
    """
    Apply Dirichlet smoothing to a CPT array.

    Args:
        cpt_array : numpy array of shape (n_states_child, n_states_parents...)
                    Each COLUMN must sum to 1.0 (pgmpy convention).
        alpha     : Dirichlet pseudocount. Default 2 per sprint spec.
        n_effective: Effective sample size. Default 20 per sprint spec.

    Returns:
        Smoothed CPT array of the same shape.
        Columns are renormalized to sum to 1.0.

    Example:
        raw = np.array([[0.7, 0.3], [0.2, 0.5], [0.1, 0.2]])  # 3 child states, 2 parent configs
        smoothed = apply_dirichlet_smoothing(raw)
        # Each column sums to 1.0 after smoothing
    """
    cpt = np.array(cpt_array, dtype=float)
    n_states = cpt.shape[0]  # number of child states

    # Formula from sprint spec
    smoothed = (cpt * n_effective + alpha) / (n_effective + alpha * n_states)

    # Renormalise each column to sum to exactly 1.0 (floating point safety)
    col_sums = smoothed.sum(axis=0, keepdims=True)
    smoothed = smoothed / col_sums

    return smoothed


def verify_cpt_rows_sum_to_one(cpt_array: np.ndarray, node_name: str = "") -> bool:
    """
    Verify that every column of a CPT sums to 1.0 (within floating point tolerance).
    Raises AssertionError if any column is off.
    Use this on every CPT after smoothing.
    """
    cpt = np.array(cpt_array, dtype=float)
    col_sums = cpt.sum(axis=0)
    ok = np.allclose(col_sums, 1.0, atol=1e-6)
    if not ok:
        bad_cols = np.where(~np.isclose(col_sums, 1.0, atol=1e-6))[0]
        raise AssertionError(
            f"CPT '{node_name}': columns {bad_cols} do not sum to 1.0. "
            f"Sums: {col_sums[bad_cols]}"
        )
    print(f"  ✓ CPT '{node_name}' — all columns sum to 1.0")
    return True


def smooth_and_verify(
    cpt_array: np.ndarray,
    node_name: str,
    alpha: float = 2.0,
    n_effective: int = 20,
) -> np.ndarray:
    """
    Convenience wrapper: smooth then verify. Use this everywhere.

    Usage:
        raw_cpt = np.array([[0.55, 0.35], [0.45, 0.65]])
        smoothed = smooth_and_verify(raw_cpt, "Crackle_Phase")
    """
    smoothed = apply_dirichlet_smoothing(cpt_array, alpha, n_effective)
    verify_cpt_rows_sum_to_one(smoothed, node_name)
    return smoothed


# ---------------------------------------------------------------------------
# SELF-TEST — run to verify the smoother works correctly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("Dirichlet Smoothing — Self Test")
    print("=" * 55)

    # Test 1: basic smoothing flattens extreme values
    raw = np.array([[0.9], [0.1]])
    smoothed = apply_dirichlet_smoothing(raw)
    print(f"\nTest 1 — Extreme prior [0.9, 0.1]:")
    print(f"  Raw:      {raw.flatten()}")
    print(f"  Smoothed: {smoothed.flatten().round(4)}")
    assert smoothed[0, 0] < 0.9, "Should pull away from extreme"
    assert smoothed[1, 0] > 0.1, "Should pull toward uniform"
    print("  ✓ Smoothing pulls toward uniform — correct")

    # Test 2: columns sum to 1.0
    raw2 = np.array([[0.55, 0.35, 0.20],
                     [0.30, 0.45, 0.50],
                     [0.15, 0.20, 0.30]])
    smoothed2 = smooth_and_verify(raw2, "Test_Node_3states")
    print(f"\nTest 2 — 3-state CPT with 3 parent configs:")
    print(f"  Column sums after smoothing: {smoothed2.sum(axis=0).round(6)}")

    # Test 3: uniform input stays uniform
    raw3 = np.array([[0.25], [0.25], [0.25], [0.25]])
    smoothed3 = apply_dirichlet_smoothing(raw3)
    assert np.allclose(smoothed3, raw3, atol=1e-6), "Uniform should stay uniform"
    print(f"\nTest 3 — Uniform prior stays uniform: ✓")

    # Test 4: simulate a Crackle_Phase CPT (qualitative q_i values from sprint spec)
    # Parent: P_Consolidation (2 states: low, high)
    # Child: Crackle_Phase (4 states: early_insp, late_insp, expiratory, mixed)
    # qualitative prior — weak evidence, update first with TRUPCR data
    raw_phase = np.array([
        [0.35, 0.15],   # early_insp:  modest decrease when consolidation present
        [0.45, 0.55],   # late_insp:   modest increase when consolidation present
        [0.10, 0.15],   # expiratory
        [0.10, 0.15],   # mixed
    ])
    smoothed_phase = smooth_and_verify(raw_phase, "Crackle_Phase_mock")
    print(f"\nTest 4 — Mock Crackle_Phase CPT (qualitative prior, update with TRUPCR):")
    print(f"  Raw:\n{raw_phase}")
    print(f"  Smoothed:\n{smoothed_phase.round(4)}")

    # Test 5: Wheeze_Compound (7 states) — single parent config
    # qualitative prior — weak evidence, update first with TRUPCR data
    raw_wc = np.array([
        [0.15],  # monophonic_low
        [0.12],  # monophonic_mid
        [0.10],  # monophonic_high
        [0.20],  # polyphonic_low
        [0.18],  # polyphonic_mid
        [0.15],  # polyphonic_pan
        [0.10],  # uncertain
    ])
    smoothed_wc = smooth_and_verify(raw_wc, "Wheeze_Compound_mock_7states")
    print(f"\nTest 5 — Mock Wheeze_Compound CPT (7 states, qualitative prior, update with TRUPCR):")
    print(f"  Smoothed: {smoothed_wc.flatten().round(4)}")

    print()
    print("=" * 55)
    print("All Dirichlet smoothing tests passed ✓")
    print("=" * 55)
    print()
    print("USAGE REMINDER:")
    print("  from dirichlet_smoothing import smooth_and_verify")
    print("  smoothed_cpt = smooth_and_verify(raw_array, 'NodeName')")
    print("  # Add comment: # qualitative prior — weak evidence, update first with TRUPCR data")