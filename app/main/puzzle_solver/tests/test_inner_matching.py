"""
Tests for Step 5: Inner-Matching (1D Profile + NCC).

Test Coverage:
- Profile extraction (straight, curved, degenerate)
- NCC computation (identical, reversed, different)
- Length cost (various ratios)
- Prefiltering (length, piece ID)
- Candidate generation (multi-piece, top-k)
- ICP stub (placeholder returns 0.0)
"""

import numpy as np
from solver.models import ContourSegment
from solver.config import MatchingConfig
from solver.inner_matching import extract_1d_profile, compute_ncc, compute_ncc_with_flip, generate_inner_candidates
from solver.inner_matching.candidates import _compute_length_cost, _prefilter_candidates


def create_test_segment(
    piece_id: int,
    segment_id: int,
    points: np.ndarray,
    flatness_error: float = 0.5
) -> ContourSegment:
    """Helper to create test segment from points."""
    M = len(points)
    if M < 2:
        raise ValueError("Need at least 2 points")

    # Compute length
    diffs = np.diff(points, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    length_mm = float(np.sum(seg_lengths))

    # Chord
    chord_start = points[0].copy()
    chord_end = points[-1].copy()
    chord = (chord_start, chord_end)

    # Direction angle
    chord_vec = chord_end - chord_start
    direction_angle_deg = float(np.rad2deg(np.arctan2(chord_vec[1], chord_vec[0])))

    return ContourSegment(
        piece_id=piece_id,
        segment_id=segment_id,
        points_mm=points,
        length_mm=length_mm,
        chord=chord,
        direction_angle_deg=direction_angle_deg,
        flatness_error=flatness_error,
        profile_1d=None
    )


# ========== Test 1: Profile Extraction (Straight) ==========

def test_profile_extraction_straight():
    """Test profile extraction for straight horizontal segment."""
    # Straight horizontal line: y=0, x from 0 to 20
    points = np.array([[float(x), 0.0] for x in range(21)])  # 21 points, 20mm length
    seg = create_test_segment(piece_id=1, segment_id=0, points=points, flatness_error=0.0)

    config = MatchingConfig()
    profile = extract_1d_profile(seg, config)

    # Assertions
    assert profile.shape == (config.profile_samples_N,), "Profile shape should be (N,)"
    assert seg.profile_1d is not None, "profile_1d should be set on segment"
    assert np.array_equal(profile, seg.profile_1d), "Returned profile should match segment.profile_1d"

    # Straight line → all points on chord → profile ≈ 0
    assert np.max(np.abs(profile)) < 0.01, f"Straight segment should have near-zero profile, got max {np.max(np.abs(profile))}"

    print(f"Test 1: profile_extraction_straight... ✓ (max_abs={np.max(np.abs(profile)):.4f})")


# ========== Test 2: Profile Extraction (Curved) ==========

def test_profile_extraction_curved():
    """Test profile extraction for curved segment (circular arc)."""
    # Circular arc: radius=10mm, 90° arc from (0,0) to (10,10)
    angles = np.linspace(0, np.pi/2, 21)  # 21 points
    radius = 10.0
    points = np.array([[radius * np.cos(a), radius * np.sin(a)] for a in angles])

    seg = create_test_segment(piece_id=1, segment_id=0, points=points, flatness_error=2.0)

    config = MatchingConfig()
    profile = extract_1d_profile(seg, config)

    # Assertions
    assert profile.shape == (config.profile_samples_N,), "Profile shape should be (N,)"

    # Curved segment → points deviate from chord → non-zero profile
    # 90° Arc, r=10mm → Sagitta = r(1-cos(θ/2)) = 10(1-cos(45°)) ≈ 2.93mm
    max_abs = np.max(np.abs(profile))
    assert 2.5 < max_abs < 3.3, \
        f"Curved profile sagitta expected ~2.93mm (±0.4mm for resampling), got {max_abs:.2f}mm"

    # Profile should be smooth (no huge jumps)
    diffs = np.abs(np.diff(profile))
    assert np.max(diffs) < 2.0, f"Profile should be smooth, got max diff {np.max(diffs):.4f}"

    print(f"Test 2: profile_extraction_curved... ✓ (max_abs={max_abs:.4f})")


# ========== Test 3: Profile Extraction (Degenerate) ==========

def test_profile_degenerate():
    """Test profile extraction for degenerate segment (zero length)."""
    # Degenerate: all points identical
    points = np.array([[5.0, 5.0], [5.0, 5.0]])
    seg = create_test_segment(piece_id=1, segment_id=0, points=points, flatness_error=0.0)

    config = MatchingConfig()
    profile = extract_1d_profile(seg, config)

    # Degenerate → zero profile
    assert np.allclose(profile, 0.0), f"Degenerate segment should have zero profile, got {profile[:5]}"

    print("Test 3: profile_degenerate... ✓")


# ========== Test 4: NCC (Identical) ==========

def test_ncc_identical():
    """Test NCC for identical profiles."""
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    ncc = compute_ncc(profile_a, profile_b)

    # Identical → NCC = 1.0
    assert abs(ncc - 1.0) < 1e-6, f"Identical profiles should have NCC=1.0, got {ncc}"

    print(f"Test 4: ncc_identical... ✓ (NCC={ncc:.6f})")


# ========== Test 5: NCC (Reversed) ==========

def test_ncc_reversed():
    """Test NCC for reversed profiles."""
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b_reversed = np.array([0.0, 1.0, 2.0, 1.0, 0.0])  # Symmetric, but test reversal

    # Use asymmetric profile
    profile_a = np.array([0.0, 1.0, 2.0, 3.0, 1.0])
    profile_b = np.array([1.0, 3.0, 2.0, 1.0, 0.0])  # Reversed

    ncc_forward = compute_ncc(profile_a, profile_b)
    ncc_reversed = compute_ncc(profile_a, profile_b[::-1])

    # Reversed should match original
    assert abs(ncc_reversed - 1.0) < 1e-6, f"Reversed should have NCC=1.0, got {ncc_reversed}"
    assert ncc_reversed > ncc_forward, f"Reversed NCC should be > forward NCC"

    print(f"Test 5: ncc_reversed... ✓ (forward={ncc_forward:.4f}, reversed={ncc_reversed:.4f})")


# ========== Test 17: NCC (Sign-Flip Detection) ==========

def test_ncc_sign_flip():
    """Test NCC with sign-flip detection for opposite-side profiles (FIX 1)."""
    # Case A: Same-side profile
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b_same = np.array([0.0, 1.0, 2.0, 1.0, 0.0])

    ncc_same, rev_same, flip_same = compute_ncc_with_flip(profile_a, profile_b_same, allow_reversal=False)

    # Same-side: NCC ≈ 1.0, no flip
    assert abs(ncc_same - 1.0) < 1e-6, f"Same-side: NCC should be 1.0, got {ncc_same:.6f}"
    assert not rev_same, "Same-side: reversal_used should be False"
    assert not flip_same, "Same-side: sign_flip_used should be False"

    # Case B: Opposite-side profile (negated)
    profile_b_opposite = -profile_b_same

    ncc_opp, rev_opp, flip_opp = compute_ncc_with_flip(profile_a, profile_b_opposite, allow_reversal=False)

    # Opposite-side: |NCC| ≈ 1.0, sign_flip=True
    assert abs(abs(ncc_opp) - 1.0) < 1e-6, f"Opposite-side: |NCC| should be 1.0, got {abs(ncc_opp):.6f}"
    assert not rev_opp, "Opposite-side: reversal_used should be False"
    assert flip_opp, "Opposite-side: sign_flip_used should be True"

    # Case C: Reversed + sign-flipped
    profile_c = np.array([0.0, 1.0, 2.0, 3.0, 1.0])  # Asymmetric
    profile_d = -profile_c[::-1]  # Reversed + negated

    ncc_both, rev_both, flip_both = compute_ncc_with_flip(profile_c, profile_d, allow_reversal=True)

    # Should detect both reversal and sign-flip
    assert abs(abs(ncc_both) - 1.0) < 1e-6, f"Rev+flip: |NCC| should be 1.0, got {abs(ncc_both):.6f}"
    assert rev_both, "Rev+flip: reversal_used should be True"
    assert flip_both, "Rev+flip: sign_flip_used should be True"

    print(f"Test 17: ncc_sign_flip... ✓ (same={ncc_same:.4f}, opp={ncc_opp:.4f}, both={ncc_both:.4f})")


# ========== Test 6: NCC (Different) ==========

def test_ncc_different():
    """Test NCC for different profiles."""
    profile_a = np.array([0.0, 1.0, 2.0, 1.0, 0.0])
    profile_b = np.array([0.0, 0.5, 0.2, 0.3, 0.0])  # Different shape

    ncc = compute_ncc(profile_a, profile_b)

    # Different → NCC < threshold (not perfect match)
    assert ncc < 0.7, f"Different signals should have low NCC (<0.7), got {ncc:.3f}"

    print(f"Test 6: ncc_different... ✓ (NCC={ncc:.4f})")


# ========== Test 7: Length Cost ==========

def test_length_cost():
    """Test length cost computation."""
    tolerance = 0.15

    # Perfect match
    cost1 = _compute_length_cost(10.0, 10.0, tolerance)
    assert abs(cost1 - 0.0) < 1e-6, f"Perfect match should have cost=0, got {cost1}"

    # 10% difference
    cost2 = _compute_length_cost(10.0, 11.0, tolerance)
    expected2 = (1 - 10.0/11.0) / tolerance  # ≈ 0.606
    assert abs(cost2 - expected2) < 0.01, f"10% diff: expected {expected2:.3f}, got {cost2:.3f}"

    # 13% difference
    cost3 = _compute_length_cost(10.0, 11.3, tolerance)
    expected3 = (1 - 10.0/11.3) / tolerance  # ≈ 0.87
    assert abs(cost3 - expected3) < 0.01, f"13% diff: expected {expected3:.3f}, got {cost3:.3f}"

    # >15% difference (clamped)
    cost4 = _compute_length_cost(10.0, 13.0, tolerance)
    assert abs(cost4 - 1.0) < 1e-6, f">15% diff should clamp to 1.0, got {cost4}"

    print(f"Test 7: length_cost... ✓ (0%={cost1:.3f}, 10%={cost2:.3f}, 13%={cost3:.3f}, >15%={cost4:.3f})")


# ========== Test 8: Prefilter (Length) ==========

def test_prefilter_length():
    """Test prefiltering by length tolerance."""
    # Reference segment: 20mm
    points_ref = np.array([[0.0, 0.0], [20.0, 0.0]])
    seg_ref = create_test_segment(piece_id=1, segment_id=0, points=points_ref)

    # Candidate segments from different pieces
    # Seg A: 21mm (5% diff, within tolerance)
    points_a = np.array([[0.0, 0.0], [21.0, 0.0]])
    seg_a = create_test_segment(piece_id=2, segment_id=0, points=points_a)

    # Seg B: 25mm (20% diff, exceeds tolerance 15%)
    points_b = np.array([[0.0, 0.0], [25.0, 0.0]])
    seg_b = create_test_segment(piece_id=3, segment_id=0, points=points_b)

    all_segments = [seg_ref, seg_a, seg_b]
    config = MatchingConfig(length_tolerance_ratio=0.15)

    candidates = _prefilter_candidates(seg_ref, all_segments, config)

    # Should include seg_a (5%), exclude seg_ref (same piece), exclude seg_b (20%)
    candidate_ids = [(c.piece_id, c.segment_id) for c in candidates]
    assert (2, 0) in candidate_ids, "Should include seg_a (5% diff)"
    assert (3, 0) not in candidate_ids, "Should exclude seg_b (20% diff)"
    assert (1, 0) not in candidate_ids, "Should exclude seg_ref (same piece)"

    print(f"Test 8: prefilter_length... ✓ (candidates={len(candidates)})")


# ========== Test 9: Prefilter (Piece ID) ==========

def test_prefilter_piece_id():
    """Test prefiltering excludes same-piece segments."""
    points = np.array([[0.0, 0.0], [20.0, 0.0]])
    seg_ref = create_test_segment(piece_id=1, segment_id=0, points=points)

    # Same piece, different segment
    seg_same_piece = create_test_segment(piece_id=1, segment_id=1, points=points)

    # Different piece
    seg_diff_piece = create_test_segment(piece_id=2, segment_id=0, points=points)

    all_segments = [seg_ref, seg_same_piece, seg_diff_piece]
    config = MatchingConfig()

    candidates = _prefilter_candidates(seg_ref, all_segments, config)

    # Should only include seg_diff_piece
    assert len(candidates) == 1, f"Should have 1 candidate, got {len(candidates)}"
    assert candidates[0].piece_id == 2, "Should only include different piece"

    print(f"Test 9: prefilter_piece_id... ✓ (candidates={len(candidates)})")


# ========== Test 10: Generate Candidates (Basic) ==========

def test_generate_candidates_basic():
    """Test candidate generation for multi-piece scenario."""
    # Create 2 pieces, each with 2 segments
    # Piece 1: segments 0, 1
    points_1_0 = np.array([[0.0, 0.0], [20.0, 0.0]])
    points_1_1 = np.array([[20.0, 0.0], [40.0, 0.0]])
    seg_1_0 = create_test_segment(piece_id=1, segment_id=0, points=points_1_0, flatness_error=0.5)
    seg_1_1 = create_test_segment(piece_id=1, segment_id=1, points=points_1_1, flatness_error=0.8)

    # Piece 2: segments 0, 1 (similar to piece 1)
    points_2_0 = np.array([[0.0, 0.0], [21.0, 0.0]])  # Slightly different length
    points_2_1 = np.array([[21.0, 0.0], [42.0, 0.0]])
    seg_2_0 = create_test_segment(piece_id=2, segment_id=0, points=points_2_0, flatness_error=0.6)
    seg_2_1 = create_test_segment(piece_id=2, segment_id=1, points=points_2_1, flatness_error=0.9)

    all_segments = [seg_1_0, seg_1_1, seg_2_0, seg_2_1]
    config = MatchingConfig(topk_per_segment=3)

    candidates = generate_inner_candidates(all_segments, config)

    # Each segment should have candidates (only from other piece)
    assert len(candidates) == 4, f"Should have 4 segment keys, got {len(candidates)}"

    # Check seg_1_0 candidates
    cands_1_0 = candidates.get((1, 0), [])
    assert len(cands_1_0) <= 3, f"Should have ≤3 candidates, got {len(cands_1_0)}"
    assert len(cands_1_0) == 2, f"Should have 2 candidates from piece 2, got {len(cands_1_0)}"

    # Check sorting (ascending cost)
    for i in range(len(cands_1_0) - 1):
        assert cands_1_0[i].cost_inner <= cands_1_0[i+1].cost_inner, "Candidates should be sorted by cost"

    # Check no self-matches
    for seg_ref, cand_list in candidates.items():
        for cand in cand_list:
            assert cand.seg_a_ref == seg_ref, "seg_a_ref should match key"
            assert cand.seg_b_ref[0] != seg_ref[0], "Should not match same piece"

    # Validate cost ranges (FIX 2: costs clamped to [0, 1])
    for seg_ref, cand_list in candidates.items():
        for cand in cand_list:
            assert 0.0 <= cand.cost_inner <= 1.0, \
                f"cost_inner out of range: {cand.cost_inner}"
            assert 0.0 <= cand.profile_cost <= 1.0, \
                f"profile_cost out of range: {cand.profile_cost}"
            assert 0.0 <= cand.length_cost <= 1.0, \
                f"length_cost out of range: {cand.length_cost}"
            assert cand.fit_cost == 0.0, \
                f"fit_cost should be 0.0 (stub), got {cand.fit_cost}"

    # Validate debug fields (ncc_best, best_variant)
    for seg_ref, cand_list in candidates.items():
        for cand in cand_list:
            # ncc_best should be in [-1, 1]
            assert -1.0 <= cand.ncc_best <= 1.0, \
                f"ncc_best out of range: {cand.ncc_best}"

            # best_variant should be one of 4 variants
            assert cand.best_variant in ["fwd", "fwd_flip", "rev", "rev_flip"], \
                f"Invalid best_variant: {cand.best_variant}"

            # Consistency: sign_flip_used → variant contains "flip"
            if cand.sign_flip_used:
                assert "flip" in cand.best_variant, \
                    f"sign_flip_used=True but variant={cand.best_variant}"

            # Consistency: reversal_used → variant contains "rev"
            if cand.reversal_used:
                assert "rev" in cand.best_variant, \
                    f"reversal_used=True but variant={cand.best_variant}"

    print(f"Test 10: generate_candidates_basic... ✓ (keys={len(candidates)}, cands_1_0={len(cands_1_0)})")


# ========== Test 11: ICP Stub ==========

def test_icp_stub():
    """Test ICP fit cost placeholder (should return 0.0)."""
    from solver.inner_matching.candidates import _compute_icp_fit_cost

    points = np.array([[0.0, 0.0], [20.0, 0.0]])
    seg_a = create_test_segment(piece_id=1, segment_id=0, points=points)
    seg_b = create_test_segment(piece_id=2, segment_id=0, points=points)

    config_disabled = MatchingConfig(enable_icp=False)
    config_enabled = MatchingConfig(enable_icp=True)

    cost_disabled = _compute_icp_fit_cost(seg_a, seg_b, config_disabled)
    cost_enabled = _compute_icp_fit_cost(seg_a, seg_b, config_enabled)

    # Both should return 0.0 (placeholder)
    assert cost_disabled == 0.0, f"ICP disabled should return 0.0, got {cost_disabled}"
    assert cost_enabled == 0.0, f"ICP enabled (placeholder) should return 0.0, got {cost_enabled}"

    print(f"Test 11: icp_stub... ✓ (disabled={cost_disabled}, enabled={cost_enabled})")


# ========== Test 12: Profile Invariance (Rigid Transform) ==========

def test_profile_invariance_rigid_transform():
    """Test profile is invariant to rigid transformations (translation + rotation)."""
    # Reference: straight line 0..20mm
    points_ref = np.array([[float(i), 0.0] for i in range(21)])
    seg_ref = create_test_segment(piece_id=1, segment_id=0, points=points_ref, flatness_error=0.0)

    # Rotated 30° + translated (dx=5, dy=-3)
    angle_rad = np.deg2rad(30)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    points_rot = (R @ points_ref.T).T + np.array([5.0, -3.0])

    seg_rot = create_test_segment(piece_id=2, segment_id=0, points=points_rot, flatness_error=0.0)

    config = MatchingConfig()
    profile_ref = extract_1d_profile(seg_ref, config)
    profile_rot = extract_1d_profile(seg_rot, config)

    # Profiles should be identical (invariant to rigid transform)
    assert np.allclose(profile_ref, profile_rot, atol=1e-6), \
        f"Profile should be invariant to rigid transforms, max diff: {np.max(np.abs(profile_ref - profile_rot)):.6f}"

    print(f"Test 12: profile_invariance_rigid_transform... ✓ (max_diff={np.max(np.abs(profile_ref - profile_rot)):.6f})")


# ========== Test 13: NCC Constant Signal (No NaN) ==========

def test_ncc_constant_signal_no_nan():
    """Test NCC handles constant signals (std=0) without producing NaN."""
    # Constant signals (std=0)
    profile_a = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    profile_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])

    ncc = compute_ncc(profile_a, profile_b)

    # Must not produce NaN
    assert np.isfinite(ncc), f"NCC must not produce NaN for constant signals, got {ncc}"

    # Policy: constant signals → NCC should be 0.0 (degenerate case) or high
    assert ncc == 0.0 or ncc >= 0.99, \
        f"Expected NCC=0.0 (degenerate) or ≥0.99 (identical constant), got {ncc:.4f}"

    print(f"Test 13: ncc_constant_signal_no_nan... ✓ (NCC={ncc:.4f})")


# ========== Test 14: Prefilter Flatness ==========

def test_prefilter_flatness():
    """Test flatness tolerance filters candidates."""
    # Flat segment
    points_flat = np.array([[0.0, 0.0], [20.0, 0.0]])
    seg_flat = create_test_segment(piece_id=1, segment_id=0, points=points_flat, flatness_error=0.0)

    # Very curved segment (high flatness error)
    points_curved = np.array([[0.0, 0.0], [10.0, 5.0], [20.0, 0.0]])
    seg_curved = create_test_segment(piece_id=2, segment_id=0, points=points_curved, flatness_error=10.0)

    all_segments = [seg_flat, seg_curved]
    config = MatchingConfig()  # flatness_tolerance_mm=2.0

    candidates = generate_inner_candidates(all_segments, config)

    # seg_flat → seg_curved should be filtered (flatness diff = 10mm > 2mm tolerance)
    cands_flat = candidates.get((1, 0), [])

    # Should have 0 candidates (filtered) or very high cost
    if len(cands_flat) > 0:
        # If not filtered, cost should be high
        assert cands_flat[0].cost_inner > 0.8, \
            f"High flatness mismatch should result in high cost, got {cands_flat[0].cost_inner:.3f}"
    else:
        # Preferred: filtered completely
        pass

    print(f"Test 14: prefilter_flatness... ✓ (candidates={len(cands_flat)})")


# ========== Test 15: Profile > Length (Weighting) ==========

def test_generate_candidates_prefers_profile_over_length():
    """Test profile weighting: With equal length, better profile match wins (weight=0.6)."""
    # Reference segment: slightly wavy (to avoid std=0), 20mm, flatness=0.5
    # y = 0.02 * sin(2π * x / 20)
    L = 20.0
    x_vals = np.linspace(0, L, 21)
    y_vals_a = 0.02 * np.sin(2 * np.pi * x_vals / L)
    points_a = np.column_stack([x_vals, y_vals_a])
    seg_a = create_test_segment(piece_id=1, segment_id=0, points=points_a, flatness_error=0.5)

    # Candidate 1: Same waveform (good profile match), SAME length 20mm, flatness=0.6
    y_vals_b1 = 0.02 * np.sin(2 * np.pi * x_vals / L)  # Identical waveform
    points_b1 = np.column_stack([x_vals, y_vals_b1])
    seg_b1 = create_test_segment(piece_id=2, segment_id=0, points=points_b1, flatness_error=0.6)

    # Candidate 2: Different waveform (90° phase shift), SAME length 20mm, flatness=0.8
    y_vals_b2 = 0.02 * np.cos(2 * np.pi * x_vals / L)  # 90° phase shift → orthogonal
    points_b2 = np.column_stack([x_vals, y_vals_b2])
    seg_b2 = create_test_segment(piece_id=3, segment_id=0, points=points_b2, flatness_error=0.8)

    all_segments = [seg_a, seg_b1, seg_b2]
    config = MatchingConfig()  # inner_weights: profile=0.6, length=0.2, fit=0.2

    candidates = generate_inner_candidates(all_segments, config)

    cands_a = candidates.get((1, 0), [])
    assert len(cands_a) == 2, f"Should have 2 candidates, got {len(cands_a)}"

    # Find costs and verify lengths are equal
    cand_b1 = next((c for c in cands_a if c.seg_b_ref[0] == 2), None)
    cand_b2 = next((c for c in cands_a if c.seg_b_ref[0] == 3), None)

    assert cand_b1 is not None and cand_b2 is not None, "Both candidates should be present"

    # Both should have length_cost ≈ 0 (same length)
    assert cand_b1.length_cost < 0.05, f"length_cost should be ≈0 for same length, got {cand_b1.length_cost:.3f}"
    assert cand_b2.length_cost < 0.05, f"length_cost should be ≈0 for same length, got {cand_b2.length_cost:.3f}"

    # Profile costs should differ significantly (straight vs curved)
    assert cand_b1.profile_cost < cand_b2.profile_cost, \
        f"Straight should have lower profile_cost than curved: {cand_b1.profile_cost:.3f} vs {cand_b2.profile_cost:.3f}"

    # With equal length_cost, profile_cost dominates → b1 (straight) wins
    assert cand_b1.cost_inner < cand_b2.cost_inner, \
        f"Better profile match should win with equal length: cost_straight={cand_b1.cost_inner:.3f} vs cost_curved={cand_b2.cost_inner:.3f}"

    print(f"Test 15: generate_candidates_prefers_profile_over_length... ✓ (straight={cand_b1.cost_inner:.3f}, curved={cand_b2.cost_inner:.3f})")


# ========== Test 16: Cost Decomposition ==========

def test_candidate_cost_decomposition():
    """Test that cost_inner = weighted sum of components."""
    # Simple segments
    points_a = np.array([[0.0, 0.0], [20.0, 0.0]])
    points_b = np.array([[0.0, 0.0], [21.0, 0.0]])  # 5% longer

    seg_a = create_test_segment(piece_id=1, segment_id=0, points=points_a, flatness_error=0.5)
    seg_b = create_test_segment(piece_id=2, segment_id=0, points=points_b, flatness_error=0.6)

    all_segments = [seg_a, seg_b]
    config = MatchingConfig()  # weights: profile=0.6, length=0.2, fit=0.2

    candidates = generate_inner_candidates(all_segments, config)

    cands_a = candidates.get((1, 0), [])
    assert len(cands_a) == 1, f"Should have 1 candidate, got {len(cands_a)}"

    cand = cands_a[0]

    # Verify cost decomposition
    weights = config.inner_weights
    expected_cost = (
        weights["profile"] * cand.profile_cost +
        weights["length"] * cand.length_cost +
        weights["fit"] * cand.fit_cost
    )

    assert abs(cand.cost_inner - expected_cost) < 1e-6, \
        f"cost_inner should equal weighted sum: {cand.cost_inner:.6f} != {expected_cost:.6f}"

    print(f"Test 16: candidate_cost_decomposition... ✓ (cost_inner={cand.cost_inner:.4f}, expected={expected_cost:.4f})")


if __name__ == "__main__":
    print("Running Inner-Matching Tests (Step 5)...\n")

    test_profile_extraction_straight()
    test_profile_extraction_curved()
    test_profile_degenerate()
    test_ncc_identical()
    test_ncc_reversed()
    test_ncc_different()
    test_length_cost()
    test_prefilter_length()
    test_prefilter_piece_id()
    test_generate_candidates_basic()
    test_icp_stub()
    test_profile_invariance_rigid_transform()
    test_ncc_constant_signal_no_nan()
    test_prefilter_flatness()
    test_generate_candidates_prefers_profile_over_length()
    test_candidate_cost_decomposition()

    print("\n✅ All 16 tests passed!")
