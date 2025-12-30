# Implementation Schritt 8: Fallback-Modul (Many-to-One)

**Status:** ✅ Abgeschlossen (2025-12-29)
**Tests:** 25/25 passing (test_fallback.py), 32/32 beam_solver (no regression)
**Config:** 12 fallback-related parameters in MatchingConfig (k_conf, many_to_one_enable, beam_width_fallback, etc.)

---

## 1. Übersicht

### Zweck

Implementierung von **Confidence-Mapping** und **Many-to-One Fallback** zur Robustheit bei feiner Segmentierung.

**Design-Basis:**
- `docs/design/08_fallback.md` (Fallback-Konzept, Many-to-One Matching)
- `docs/test_spec/08_fallback_test_spec.md` (Test-Spezifikation, 25 Tests)

**Metriken:**
- `compute_confidence(cost_total, config)`: Confidence = exp(-k_conf × cost_total)
- `should_trigger_fallback(solution, config)`: Trigger wenn confidence < threshold
- `create_composite_segments(segments, config)`: Adjazente Segment-Ketten (k=2, k=3)
- `extend_inner_candidates(...)`: Composite ↔ Atomic Matches

**Verwendung:**
- **Fallback-Trigger** nach Beam-Search: Wenn confidence < 0.5 → Rerun mit composites
- **Rerun** via `_attempt_fallback_rerun()` in beam_solver/solver.py (max 1 Iteration)

---

### Module

| Modul | Zeilen | Funktionen | Zweck |
|-------|--------|-----------|-------|
| `solver/fallback/many_to_one.py` | 454 | 8 | Confidence, Composites, Candidate Extension |
| `solver/fallback/__init__.py` | 17 | - | API Exports (5 Funktionen) |
| **Integration:** `solver/beam_solver/solver.py` | +96 | 1 | Fallback Rerun (_attempt_fallback_rerun) |
| **Config:** `solver/config.py` | +73 | - | Fallback Parameters (Zeilen 372-444) |

**Exports (`__init__.py`):**
```python
from .many_to_one import (
    compute_confidence,
    create_composite_segments,
    extend_inner_candidates,
    should_trigger_fallback,
    run_fallback_iteration  # Stub for compatibility
)
__all__ = ["compute_confidence", "create_composite_segments",
           "extend_inner_candidates", "should_trigger_fallback",
           "run_fallback_iteration"]
```

**Helper-Funktionen** (nicht exportiert, intern):
- `_build_composite_segment`, `_concatenate_points`, `_length_filter`

---

## 2. many_to_one.py Implementation

### 2.1 Confidence Computation

**Funktion:** `compute_confidence(cost_total: float, config: MatchingConfig) -> float`
**Zeilen:** 31-57

**Formel:** `confidence = exp(-k_conf × cost_total)`

**Algorithmus:**
1. **Input Validation** (Zeile 51-52):
   - Falls `cost_total < 0`: ValueError
   - Falls `cost_total = inf`: return 0.0

2. **Confidence Berechnung** (Zeile 57):
   - `np.exp(-config.k_conf * cost_total)`

**Edge Cases:**
- cost = 0: returns 1.0 (perfekte Lösung)
- cost = inf: returns 0.0 (unmögliche Lösung)
- cost < 0: ValueError (invalid cost)

**Tests:** test_fallback.py Tests 1-6

---

### 2.2 Fallback Trigger

**Funktion:** `should_trigger_fallback(solution, config: MatchingConfig) -> tuple[bool, Optional[str]]`
**Zeilen:** 60-91

**Logik:**
```python
# 1. Check if fallback enabled (Zeile 78)
if not config.many_to_one_enable:
    return False, None

# 2. Extract confidence value (Zeilen 82-86)
if isinstance(solution, float):
    confidence = solution
else:
    confidence = solution.confidence  # PuzzleSolution object

# 3. Trigger decision (Zeilen 88-91)
if confidence < config.fallback_conf_threshold:
    return True, "confidence_below_threshold"

return False, None
```

**Flexible Input:**
- Accepts `float` confidence directly (unit tests)
- Accepts `PuzzleSolution` object (integration)

**Tests:** test_fallback.py Tests 13-17

---

### 2.3 Composite Segment Creation

**Funktion:** `create_composite_segments(segments: list[ContourSegment], config: MatchingConfig) -> list[ContourSegment]`
**Zeilen:** 94-186

**Algorithmus:**

1. **Group by Piece** (Zeilen 127-136):
   - Gruppiere Segmente nach `piece_id`
   - Sortiere innerhalb jeder Gruppe nach `segment_id` (Konturreihenfolge)

2. **k=2 Composites** (Zeilen 146-163):
   - Generiere adjazente Paare: (seg_i + seg_{i+1})
   - Apply cap: `config.max_composites_per_piece_k2` (first N)
   - Build via `_build_composite_segment()`

3. **k=3 Composites** (Zeilen 166-184, optional):
   - Falls `config.many_to_one_max_chain_len >= 3`
   - Generiere Triples: (seg_i + seg_{i+1} + seg_{i+2})
   - Apply cap: `config.max_composites_per_piece_k3`

4. **Segment ID Encoding** (Zeile 370):
   - k=2 composites: `segment_id = 2000 + composite_index`
   - k=3 composites: `segment_id = 3000 + composite_index`

**Output:** List of composite ContourSegments with recomputed features

**Tests:** test_fallback.py Tests 7-12

---

### 2.4 Candidate Extension

**Funktion:** `extend_inner_candidates(original_candidates, atomic_segments, composite_segments, config) -> dict`
**Zeilen:** 189-329

**Signatur (nach Rework):**
```python
def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]],  # ONLY dict
    atomic_segments: list[ContourSegment],
    composite_segments: list[ContourSegment],
    config: MatchingConfig
) -> dict[tuple, list[InnerMatchCandidate]]:
```

**Algorithmus:**

1. **Deep Copy Original** (Zeile 228):
   - `extended = copy.deepcopy(original_candidates)`

2. **Direction 1: Composite ↔ Atomic** (Zeilen 234-266):
   - Für jedes composite-atomic Paar:
     - Skip same piece (Zeile 237-238)
     - Length filter ±20% (Zeilen 241-242)
     - Compute `cost_inner` via `_compute_inner_cost()`
     - Add `penalty_composite_used` (Zeile 250)
     - Create `InnerMatchCandidate`

3. **Direction 2: Atomic ↔ Composite** (Zeilen 269-301):
   - Symmetrische Richtung (atomic als seg_a)

4. **Merge Candidates** (Zeilen 304-308):
   - Append to `extended[seg_a_ref]`

5. **Apply Caps** (Zeilen 311-327):
   - **Per-segment cap:** `topk_per_segment_fallback` (Zeile 313)
   - **Global cap:** `max_total_candidates_fallback` (Zeile 317-327)

**Filters:**
- Same-piece blocking: `seg_a.piece_id == seg_b.piece_id` → skip
- Length tolerance: ±20% via `_length_filter()`

**Tests:** test_fallback.py Tests 18-21

---

### 2.5 Helper Functions

| Funktion | Zeilen | Zweck |
|----------|--------|-------|
| `_build_composite_segment` | 334-388 | Build single composite from chain (concatenate, recompute features) |
| `_concatenate_points` | 391-418 | Join point arrays, remove duplicate junction points |
| `_length_filter` | 421-440 | Check length compatibility (±20% tolerance) |
| `run_fallback_iteration` | 443-453 | Stub (raises NotImplementedError, points to _attempt_fallback_rerun) |

**_build_composite_segment Details:**
- Concatenate points via `_concatenate_points()` (duplicate junction removal)
- Recompute features: `length_mm`, `flatness_error`, `direction_angle_deg`
- Assign offset segment_id: `1000 × chain_len + composite_index`
- Extract profile via `extract_1d_profile(composite, config)`

**_concatenate_points Logic (Zeilen 409-416):**
```python
# Check if last point of result == first point of next
if np.linalg.norm(result[-1] - points_next[0]) < 1e-6:
    # Duplicate junction → skip first point of next
    result = np.vstack([result, points_next[1:]])
else:
    # No duplicate → concat directly
    result = np.vstack([result, points_next])
```

**_length_filter Formula (Zeilen 439):**
```python
ratio = min(len_a, len_b) / max(len_a, len_b)
return ratio >= (1 - tolerance)  # tolerance=0.2 → ±20%
```

---

## 3. Design Decisions

### D1: Confidence Formula (Exponential)

**Problem:** Map cost_total → confidence [0, 1]

**Optionen:**
- **Option A:** Linear `conf = 1 - cost_total / max_cost` (hard to normalize)
- **Option B:** Exponential `conf = exp(-k_conf × cost_total)` (self-normalizing)

**Entscheidung:** **Option B (Exponential)**

**Implementierung (Zeile 57):**
```python
return float(np.exp(-config.k_conf * cost_total))
```

**Begründung:**
- Self-normalizing: conf ∈ [0, 1] für alle cost_total ≥ 0
- Monotonic decreasing: Höherer cost → niedrigere confidence
- Tunable via `k_conf` parameter (steepness)
- Standard in probabilistic systems

**Tests:** Test 1-6 (edge cases: cost=0, cost=inf, k_conf variation)

---

### D2: Fallback Trigger (Confidence Threshold)

**Problem:** Wann soll Many-to-One Fallback aktiviert werden?

**Entscheidung:** `confidence < fallback_conf_threshold` (default 0.5)

**Implementierung (Zeilen 88-91):**
```python
if confidence < config.fallback_conf_threshold:
    return True, "confidence_below_threshold"
return False, None
```

**Begründung:**
- Simple threshold rule (robust, interpretable)
- threshold=0.5 corresponds to cost_total ≈ 0.693 (ln(2))
- Config parameter allows tuning
- Strict `<` (not `<=`) → deterministic edge case

**Tests:** Test 13-17 (above/below/equal threshold, enable flag)

---

### D3: Composite Segment Capping

**Problem:** Wie viele composite Segmente pro Teil generieren? (Kombinatorik-Risiko)

**Entscheidung:** **First N** (no smart ranking in V1)

**Implementierung:**
- k=2 cap: `config.max_composites_per_piece_k2 = 6` (Zeilen 151-152)
- k=3 cap: `config.max_composites_per_piece_k3 = 3` (Zeilen 172-173)

**Begründung (aus design/08_fallback.md):**
- V1: Simple first-N selection (adjacent pair order)
- Future: Ranking by flatness/length variance (split-likelihood)
- Typical piece: ~8 segments → 6 k=2 composites = 75% coverage

**Trade-offs:**
- **First N:** Fast, deterministic, but may miss best composites
- **Smart Ranking (future):** Better quality, but slower + more complex

**Tests:** Test 7-8 (composite count validation)

---

### D4: Candidate Format (dict vs list)

**Problem:** Welches Format für inner_candidates? (SSOT violation in initial implementation)

**Initial Implementation (PROBLEMATIC):**
```python
def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]] | list[InnerMatchCandidate],
    ...
) -> list[InnerMatchCandidate]:  # Returns list
```

**Rework (Design-compliant):**
```python
def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]],  # ONLY dict
    ...
) -> dict[tuple, list[InnerMatchCandidate]]:  # Returns dict
```

**Entscheidung:** **dict format as SSOT** (Single Source of Truth)

**Begründung:**
- dict format: `{seg_a_ref: [candidates]}` allows efficient per-segment operations
- SSOT principle: ONE canonical format throughout codebase
- Caller converts to list if needed (explicit conversion, no type unions)

**Integration Point (solver.py Zeilen 238-241):**
```python
# Flatten dict to list for beam_search (explicit conversion)
inner_candidates_flat = []
for cand_list in extended_candidates_dict.values():
    inner_candidates_flat.extend(cand_list)
```

**Tests:** Test 18-21 (dict input/output)

---

### D5: Filter Requirements (Same-Piece + Length)

**Problem:** Welche Filters sind Pflicht für composite ↔ atomic matching?

**Initial Implementation (BROKEN):**
```python
# Filters commented out for "test compatibility"
# if comp_seg.piece_id == atom_seg.piece_id:
#     continue
# if not _length_filter(...):
#     continue
```

**Rework (Design-compliant):**
```python
# Same-piece filter (Zeilen 237-238)
if comp_seg.piece_id == atom_seg.piece_id:
    continue

# Length filter ±20% (Zeilen 241-242)
if not _length_filter(comp_seg.length_mm, atom_seg.length_mm, tolerance=0.2):
    continue
```

**Entscheidung:** **Both filters MANDATORY** (design requirement)

**Begründung:**
- Same-piece filter: Prevents invalid self-matching
- Length filter: Reduces candidate space explosion (±20% tolerance from design)
- Design rule: "Code folgt Design/Spec, NICHT Tests"

**Test Fixture Rework:**
- Changed from 1 piece to 2 pieces (enable same-piece filter)
- Adjusted lengths to be compatible (enable length filter)
- Tests 7-8 expectations updated (5 composites total, not 3)

**Tests:** Test 18-21 (filter validation implicit)

---

### D6: Fallback Parameters (1.5× Multiplier)

**Problem:** Wie beam_width und topk während Fallback anpassen?

**Entscheidung:** **1.5× Multiplier** (Conservative increase)

**Implementierung (config.py Zeilen 398-412):**
```python
beam_width_fallback: int = 30     # 1.5× normal (20 → 30)
topk_per_segment_fallback: int = 15  # 1.5× normal (10 → 15)
```

**Begründung:**
- Composite space größer → mehr Exploration nötig
- 1.5× balanciert Quality vs Performance
- V1 conservative choice (tuning später via data)

**Alternative (rejected):** 2× multiplier (zu teuer für V1)

**Tests:** Integration tests 22-25 (rerun with fallback config)

---

## 4. Code-Fixes (Rework nach Initial Implementation)

### Rework Context

**Problem:** Initial implementation hatte 3 Workarounds (Fixes 2, 3, 5) die Design verletzen

**User-Anforderung:** "Sauberer Code, KEINE Workarounds, Code folgt Design/Spec"

**Rework-Umfang:** Fixes 2, 3, 5 komplett neu implementiert + Test-Fixtures angepasst

**Design-LLM Review Score:** 70% initial → 100% nach Rework

---

### Fix 2: Type Compatibility (dict vs list)

**Initial Problem:**
- `extend_inner_candidates()` expected dict input
- Test passed list (from old implementation)
- Workaround: Accept both via `isinstance()` check

**Initial Code (PROBLEMATIC):**
```python
def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]] | list[InnerMatchCandidate],
    ...
) -> list[InnerMatchCandidate]:
    # Workaround: Handle both formats
    if isinstance(original_candidates, list):
        all_candidates = copy.deepcopy(original_candidates)
    else:
        all_candidates = []
        for cand_list in original_candidates.values():
            all_candidates.extend(copy.deepcopy(cand_list))
    # ... rest
```

**Rework (Design-compliant, Zeilen 189-329):**
```python
def extend_inner_candidates(
    original_candidates: dict[tuple, list[InnerMatchCandidate]],  # ONLY dict
    atomic_segments: list[ContourSegment],
    composite_segments: list[ContourSegment],
    config: MatchingConfig
) -> dict[tuple, list[InnerMatchCandidate]]:  # Returns dict
    # 1. Copy original candidates (dict format - SSOT)
    extended = copy.deepcopy(original_candidates)

    # ... generate composite candidates ...

    # 3. Merge into extended dict
    for cand in new_candidates:
        key = cand.seg_a_ref
        if key not in extended:
            extended[key] = []
        extended[key].append(cand)

    # ... apply caps ...

    return extended
```

**Test Changes:**
```python
# BEFORE:
original_candidates = [InnerMatchCandidate(...), ...]

# AFTER:
original_candidates = {
    (1, 0): [InnerMatchCandidate(seg_a_ref=(1, 0), seg_b_ref=(2, 0), ...)],
    (1, 1): [InnerMatchCandidate(seg_a_ref=(1, 1), seg_b_ref=(2, 1), ...)]
}
```

**Impact:** SSOT restored, type system consistent, no workarounds

---

### Fix 3: Config Field Duplication

**Initial Problem:**
- Tests used `config.many_to_one_enable`
- Code checked for `enable_many_to_one_fallback` (improvised dual-field)
- SSOT violation: Two fields for same concept

**Initial Code (PROBLEMATIC, config.py):**
```python
many_to_one_enable: bool = True
enable_many_to_one_fallback: bool = True  # Alias for compatibility

# In many_to_one.py:
if not getattr(config, 'enable_many_to_one_fallback',
               getattr(config, 'many_to_one_enable', False)):
    return False, None
```

**Rework (Design-compliant):**

**config.py (Zeile 382-383):**
```python
many_to_one_enable: bool = True
"""Enable many-to-one matching fallback"""
# Removed: enable_many_to_one_fallback field
```

**many_to_one.py (Zeile 78):**
```python
# Direct access, no getattr fallback
if not config.many_to_one_enable:
    return False, None
```

**Impact:** SSOT restored, single field name, no dual-field confusion

---

### Fix 5: Disabled Filters

**Initial Problem:**
- Same-piece filter disabled: `# if comp_seg.piece_id == atom_seg.piece_id: continue`
- Length filter disabled: `# if not _length_filter(...): continue`
- Reason: Test fixture incompatible (only 1 piece, incompatible lengths)
- Workaround: Comment out filters → DESIGN VIOLATION

**Initial Code (PROBLEMATIC, Zeilen ~237 + ~275):**
```python
# Direction 1: composite ↔ atomic
for comp_seg in composite_segments:
    for atom_seg in atomic_segments:
        # DISABLED for test compatibility
        # if comp_seg.piece_id == atom_seg.piece_id:
        #     continue
        # if not _length_filter(comp_seg.length_mm, atom_seg.length_mm, tolerance=0.2):
        #     continue

        # ... compute cost and create candidate
```

**Rework (Design-compliant, Zeilen 237-242 + 271-276):**
```python
# Direction 1: composite ↔ atomic
for comp_seg in composite_segments:
    for atom_seg in atomic_segments:
        # Skip same piece (RE-ENABLED)
        if comp_seg.piece_id == atom_seg.piece_id:
            continue

        # Length filter ±20% (RE-ENABLED)
        if not _length_filter(comp_seg.length_mm, atom_seg.length_mm, tolerance=0.2):
            continue

        # ... compute cost and create candidate
```

**Test Fixture Rework (test_fallback.py Zeilen 53-126):**
```python
@pytest.fixture
def atomic_segments():
    """Atomic segments from 2 pieces with lengths compatible for ±20% filter.

    Piece 1: lengths [10, 12, 8, 15] → composites k=2: [22, 20, 23]
    Piece 2: lengths [20, 22, 24] → compatible with piece 1 composites (±20%)
    """
    return [
        # Piece 1 segments (4 segments)
        ContourSegment(piece_id=1, segment_id=0, length_mm=10.0, ...),
        ContourSegment(piece_id=1, segment_id=1, length_mm=12.0, ...),
        ContourSegment(piece_id=1, segment_id=2, length_mm=8.0, ...),
        ContourSegment(piece_id=1, segment_id=3, length_mm=15.0, ...),
        # Piece 2 segments (3 segments, ADDED in rework)
        ContourSegment(piece_id=2, segment_id=0, length_mm=20.0, ...),
        ContourSegment(piece_id=2, segment_id=1, length_mm=22.0, ...),
        ContourSegment(piece_id=2, segment_id=2, length_mm=24.0, ...),
    ]
```

**Test Expectation Changes (Tests 7-8):**
```python
# BEFORE (1 piece):
assert len(composites) == 3  # Only piece 1: 4 segments → 3 pairs

# AFTER (2 pieces):
assert len(composites) == 5  # Piece 1: 3 pairs + Piece 2: 2 pairs
```

**Impact:** Filters re-enabled (design-compliant), fixtures corrected, no workarounds

---

## 5. Integration (beam_solver/solver.py)

### 5.1 Fallback Rerun Function

**Funktion:** `_attempt_fallback_rerun(...) -> tuple[SolverState | None, dict]`
**Zeilen:** 177-272

**Signatur:**
```python
def _attempt_fallback_rerun(
    original_state: SolverState,
    all_pieces: dict[int, PuzzlePiece],
    all_segments: dict[int, list],
    frame_hypotheses: dict[int, list[FrameHypothesis]],
    config: MatchingConfig,
    frame: FrameModel
) -> tuple[SolverState | None, dict]:
```

**Algorithmus:**

1. **Flatten Segments** (Zeilen 213-215):
   - Extract all segments from `all_segments.values()`

2. **Create Composites** (Zeile 218):
   - `composite_segments = create_composite_segments(segments_flat, config)`

3. **Generate Original Candidates** (Zeile 227):
   - `original_candidates = generate_inner_candidates(segments_flat, config)`

4. **Extend with Composites** (Zeilen 230-235):
   - `extended_candidates_dict = extend_inner_candidates(...)`

5. **Flatten to List** (Zeilen 238-241):
   - beam_search expects `list[InnerMatchCandidate]`
   - Flatten dict values to list

6. **Create Fallback Config** (Zeilen 244-246):
   - `fallback_config = copy.deepcopy(config)`
   - Override `beam_width` → `beam_width_fallback`
   - Override `topk_per_segment` → `topk_per_segment_fallback`

7. **Rerun Beam Search** (Zeilen 249-256):
   - `fallback_states = beam_search(..., inner_candidates=inner_candidates_flat, config=fallback_config, ...)`

8. **Return Best State + Debug** (Zeilen 259-272):
   - Extract best fallback state (min cost_total)
   - Build debug_info dict (composites created, candidates extended)

**Max Iterations:** 1 (no recursive fallback)

**Tests:** test_fallback.py Tests 22-25 (integration-near)

---

## 6. Tests (test_fallback.py)

### Test Coverage: 25/25 passing

#### Tests 1-6: Confidence Computation

| # | Test | Beschreibung | Erwartung |
|---|------|--------------|-----------|
| 1 | `test_01_confidence_cost_zero` | cost=0 → conf=1.0 | abs(conf-1.0) ≤ 1e-3 |
| 2 | `test_02_confidence_cost_ln2` | cost=ln(2) → conf≈0.5 | 0.499 ≤ conf ≤ 0.501 |
| 3 | `test_03_confidence_k_conf_variation` | k_conf=0.5 wirkt | 0.706 ≤ conf ≤ 0.708 |
| 4 | `test_04_confidence_high_cost` | cost=10 → conf≈0 | conf ≤ 5e-4 |
| 5 | `test_05_confidence_inf_cost` | cost=inf → conf=0 | conf == 0.0 |
| 6 | `test_06_confidence_negative_cost` | cost<0 → ValueError | Exception raised |

#### Tests 7-12: Composite Segments

| # | Test | Beschreibung | Erwartung |
|---|------|--------------|-----------|
| 7 | `test_07_composite_chain_len_2_pairs` | k=2 erzeugt Paare | 5 composites (3+2 from 2 pieces) |
| 8 | `test_08_composite_chain_len_3_triples` | k=3 erzeugt Triples | 8 composites (3+2 k=2 + 2+1 k=3) |
| 9 | `test_09_composite_non_adjacent_excluded` | Nicht-adjazent verboten | Kein (s0+s2) composite |
| 10 | `test_10_composite_points_junction_removed` | Duplikat-Punkt entfernt | len(composite.points) = sum(seg.points) - 1 |
| 11 | `test_11_composite_length_recalculated` | Länge additiv | 21.999 ≤ composite.length ≤ 22.001 |
| 12 | `test_12_composite_wraparound_disabled` | Wrap-around V1 aus | Kein (s_last+s0) composite |

#### Tests 13-17: Fallback Trigger

| # | Test | Beschreibung | Erwartung |
|---|------|--------------|-----------|
| 13 | `test_13_trigger_above_threshold` | conf=0.6, threshold=0.5 | trigger=False |
| 14 | `test_14_trigger_below_threshold` | conf=0.49, threshold=0.5 | trigger=True, reason="confidence_below_threshold" |
| 15 | `test_15_trigger_equal_threshold` | conf=0.5, threshold=0.5 | trigger=False (strict `<`) |
| 16 | `test_16_trigger_disabled` | many_to_one_enable=False | trigger=False |
| 17 | `test_17_trigger_puzzle_solution_input` | PuzzleSolution object input | trigger basiert auf solution.confidence |

#### Tests 18-21: Candidate Extension

| # | Test | Beschreibung | Erwartung |
|---|------|--------------|-----------|
| 18 | `test_18_extend_candidates_original_preserved` | Original-Kandidaten erhalten | extended ⊇ original (dict format) |
| 19 | `test_19_extend_bidirectional` | Composite↔Atomic bidirektional | (comp,atom) und (atom,comp) vorhanden |
| 20 | `test_20_extend_composite_composite_excluded` | Composite↔Composite V1 aus | Kein (comp_a, comp_b) |
| 21 | `test_21_extend_length_prefilter` | Längen-Prefilter ±20% | Unplausible Paare blockiert |

#### Tests 22-25: Rerun Integration

| # | Test | Beschreibung | Erwartung |
|---|------|--------------|-----------|
| 22 | `test_22_rerun_improves` | Fallback besser → gewählt | final cost < original cost |
| 23 | `test_23_rerun_worse` | Fallback schlechter → original bleibt | final cost == original cost |
| 24 | `test_24_max_one_iteration` | Keine Schleife | Max 1 Rerun |
| 25 | `test_25_debug_output` | Debug info vollständig | composites_created, candidates_extended keys vorhanden |

### Test-Strategie

**Properties (garantiert):**
- Confidence monotonic decreasing mit cost_total
- Composite segment count deterministic (first N)
- Filters aktiv (same-piece + length ±20%)

**Edge Cases:**
- cost=0/inf/negative (T1, 4-6)
- conf equal threshold (T15, strict `<`)
- Disabled fallback (T16, enable flag)
- Duplicate junction points (T10)
- Wraparound disabled (T12, V1 limitation)

---

## 7. Statistik

### Module

| Datei | Zeilen | Funktionen | Tests | Status |
|-------|--------|-----------|-------|--------|
| `many_to_one.py` | 454 | 8 | 21 | ✅ 21/21 |
| `__init__.py` | 17 | - | - | ✅ |
| **Integration:** `solver.py` | +96 | 1 | 4 | ✅ 4/4 |
| **Config:** `config.py` | +73 | - | - | ✅ |
| **Tests:** `test_fallback.py` | 571 | - | 25 | ✅ 25/25 |

### Funktionen (many_to_one.py)

| # | Funktion | Zeilen | Type | Tests |
|---|----------|--------|------|-------|
| 1 | `compute_confidence` | 31-57 | **Main API** | Direct (T1-T6) |
| 2 | `should_trigger_fallback` | 60-91 | **Main API** | Direct (T13-T17) |
| 3 | `create_composite_segments` | 94-186 | **Main API** | Direct (T7-T12) |
| 4 | `extend_inner_candidates` | 189-329 | **Main API** | Direct (T18-T21) |
| 5 | `_build_composite_segment` | 334-388 | Helper | Indirect (T7-T12) |
| 6 | `_concatenate_points` | 391-418 | Helper | Indirect (T10) |
| 7 | `_length_filter` | 421-440 | Helper | Indirect (T21) |
| 8 | `run_fallback_iteration` | 443-453 | Stub | - |

**Total Functions:** 8 (4 API, 3 Helpers, 1 Stub)

### Test Coverage

**Unit Tests (many_to_one.py):** 21 Tests (T1-T21)
- Confidence: 6 Tests
- Composites: 6 Tests
- Trigger: 5 Tests
- Candidate Extension: 4 Tests

**Integration Tests (_attempt_fallback_rerun):** 4 Tests (T22-T25)

**Total:** 25/25 passing ✅

---

## 8. Validierung

### Test-Coverage

**Command:**
```bash
pytest tests/test_fallback.py -v
```

**Result:** 25/25 passing ✅

**Breakdown:**
- **Confidence (T1-T6):** 6/6
- **Composites (T7-T12):** 6/6
- **Trigger (T13-T17):** 5/5
- **Candidates (T18-T21):** 4/4
- **Rerun (T22-T25):** 4/4

**Regression:**
```bash
pytest tests/test_beam_solver.py -v
```
**Result:** 32/32 passing ✅ (no regressions from fallback integration)

---

### Config-Parameter (verwendet)

**Aus `solver/config.py` (Zeilen 372-444):**

**Confidence:**
- `k_conf: float = 1.0` (Confidence mapping steepness)
- `fallback_conf_threshold: float = 0.5` (Trigger threshold)

**Many-to-One:**
- `many_to_one_enable: bool = True` (Enable fallback)
- `many_to_one_enable_only_if_triggered: bool = True` (Not in initial run)
- `many_to_one_max_chain_len: int = 2` (k=2 or k=3)

**Scoring:**
- `penalty_composite_used: float = 5.0` (Composite match penalty)

**Fallback Search:**
- `beam_width_fallback: int = 30` (1.5× normal beam_width)
- `topk_per_segment_fallback: int = 15` (1.5× normal topk_per_segment)

**Capping:**
- `max_composites_per_piece_k2: int = 6` (~75% of typical 8 segments)
- `max_composites_per_piece_k3: int = 3` (Sparse sampling)
- `max_total_candidates_fallback: int = 2000` (Hard cap)

**V1 Limitation:**
- `composite_match_frame: bool = False` (Composites only for inner matching)

---

### Kritische Properties (geprüft)

- ✅ **Confidence Monotonic** (T1-T6): Higher cost → lower confidence
- ✅ **Trigger Deterministic** (T13-T17): Strict `<` threshold, enable flag respected
- ✅ **Composites Deterministic** (T7-T12): First N selection, no wraparound V1
- ✅ **Filters Active** (T18-T21): Same-piece + Length ±20% enforced
- ✅ **SSOT Restored** (Rework): dict format canonical, single config field
- ✅ **Max 1 Rerun** (T24): No recursive fallback loop

---

### Numerische Erwartungen

**Toleranzen (test_fallback.py Zeilen 22-24):**
- `TOL_CONF = 1e-3` (Confidence assertions)
- `TOL_COST = 1e-6` (Cost comparisons)
- `TOL_LEN_MM = 1e-3` (Length assertions)

**Spezifische Test-Werte:**
- **T1 (cost=0):** `abs(conf-1.0) ≤ 1e-3`
- **T2 (cost=ln(2)):** `0.499 ≤ conf ≤ 0.501`
- **T3 (k_conf=0.5):** `0.706 ≤ conf ≤ 0.708`
- **T11 (composite length):** `21.999 ≤ length ≤ 22.001`

---

### Abhängigkeiten

**Gelöst/Implementiert:**
- ✅ `solver/models.py`: ContourSegment, InnerMatchCandidate (Schritt 1)
- ✅ `solver/config.py`: MatchingConfig (Schritt 1)
- ✅ `solver/inner_matching/candidates.py`: generate_inner_candidates, _compute_inner_cost (Schritt 5)
- ✅ `solver/beam_solver/state.py`: SolverState (Schritt 6)
- ✅ `solver/beam_solver/solver.py`: beam_search (Schritt 6)

**Nicht benötigt:**
- Keine externen Dependencies (nur NumPy, bereits vorhanden)

---

### Determinismus

**Garantiert reproduzierbar:**
- Composite generation deterministisch (first N, sorted by segment_id)
- Confidence formula deterministisch (keine Randomness)
- Candidate generation deterministisch (nested loop order)
- Capping deterministisch (sorted by cost_inner)

**Seed:** Nicht relevant (kein RNG)

---

## 9. Offene Punkte

### V1 Limitations

**Composite Ranking (Zeilen 151-152, 172-173):**
```python
# V1: First N composites (no smart ranking)
k2_chains = k2_chains[:k2_cap]
k3_chains = k3_chains[:k3_cap]
```

**Future:** Ranking by split-likelihood:
- Flatness (low flatness → likely split)
- Length variance (similar lengths → likely split)
- Curvature (high curvature → intentional split)

**Grund:** V1 focuses on simple, deterministic selection. Optimization later.

---

**Composite ↔ Composite Matching (Zeile 223):**
```python
# Exclusions (design/08_fallback.md):
# - composite ↔ composite (V1)
```

**V1:** Disabled (combinatorial explosion risk)

**Future:** Optional enable via `config.composite_match_composite` flag

**Grund:** V1 risk mitigation. Evaluate necessity based on data.

---

**Wraparound Merge (T12):**
```python
# Test 12: Wraparound disabled in V1
# Expected: No (s_last+s0) composite
```

**V1:** Disabled (simple linear chain generation)

**Future:** Config flag `many_to_one_allow_wraparound` (for cyclic contours)

**Grund:** Edge case complexity. Add if needed.

---

**Frame-First Composites (config.py Zeile 437-443):**
```python
composite_match_frame: bool = False
"""
Enable composite segments for frame-first matching.
V1: False (composites only for inner matching).
"""
```

**V1:** Disabled (design decision)

**Begründung:** Frame edges typically well-segmented (straight, long). Composites for inner edges (curved, split-prone).

---

### source_segment_ids Tracking (V1 Limitation)

**Design-Anforderung:** Test-Spec (`docs/test_spec/08_fallback_test_spec.md` Zeile 61) erwähnt `source_segment_ids` für Composite-Traceability:
```markdown
- Referenz auf Quelle: `source_segment_ids`
```

**V1 Implementation:** Offset-basierte segment_id Encoding (many_to_one.py Zeile 370):
```python
# Assign segment_id (offset-based)
chain_len = len(indices)
segment_id_offset = 1000 * chain_len  # k=2 → 2000, k=3 → 3000
segment_id = segment_id_offset + composite_index
```

**Limitation:** Kein explizites Field `source_segment_ids: list[int]` in ContourSegment dataclass (models.py L92-124).

**Implikation:**
- ✅ Eindeutige IDs verfügbar (via Offset 2000+/3000+)
- ❌ Provenance-Tracking fehlt (welche atomic segments bilden composite?)
- ❌ Debug-Traceability eingeschränkt (cannot reconstruct chain from composite alone)

**V2 Enhancement:**
```python
@dataclass
class ContourSegment:
    # ... existing fields
    source_segment_ids: Optional[list[int]] = None
    """For composites: IDs of atomic segments that form this composite.
       Example: [0, 1] for (seg_0 + seg_1), [1, 2, 3] for (seg_1 + seg_2 + seg_3)
       None for atomic segments (not composites)."""
```

**Benefit:** Full debug traceability, composite decomposition, provenance tracking.

**Begründung V1:** Offset-Encoding ausreichend für Uniqueness und Composite-Detection. Full source tracking deferred für Implementation simplicity.

---

### Recursive Fallback (T24)

**Current:** Max 1 rerun (simple, deterministic)

**Limitation:** If fallback still low-confidence → no further action

**Future:** Multi-level fallback:
1. k=2 composites
2. k=3 composites (if k=2 insufficient)
3. Relaxed filters (if k=3 insufficient)

**Grund:** V1 avoids complexity. Evaluate necessity via data.

---

### Composite Penalty Tuning

**Current:** `penalty_composite_used = 5.0` (arbitrary start value)

**Limitation:** No data-driven tuning yet

**Future:** Tune based on:
- False-positive rate (unnecessary composites)
- False-negative rate (missed valid composites)
- A/B testing (penalty=0 vs penalty=5 vs penalty=10)

**TODO:** Schritt 10 (Integration + Tests) includes penalty tuning

---

## 10. Status

**Implementation:** ✅ Abgeschlossen (2025-12-29)

**Tests:**
- ✅ 25/25 fallback tests passing (test_fallback.py)
- ✅ 32/32 beam_solver tests passing (no regression)

**Integration:** ✅ solver.py (_attempt_fallback_rerun, Zeilen 177-272)

**Dokumentation:** ✅ Vollständig (08_fallback_impl.md)

**Rework:** ✅ Fixes 2, 3, 5 komplett neu implementiert (Design-compliant)

**Design-LLM Score:** 100/100 (all SSOT violations fixed, filters active, dict format canonical)

---

### Nächste Schritte

**Schritt 9 (Pose-Refinement):**
- Numerische Optimierung (scipy.optimize.minimize)
- Cost function: J_frame + J_inner + J_overlap
- Final overlap check (depth ≤ 0.1mm)
- Refinement after fallback (if fallback triggered)

**Schritt 10 (Integration + Tests):**
- A/B Vergleich mit/ohne Fallback
- Penalty tuning (penalty_composite_used)
- Performance metrics (composite generation time, rerun overhead)
- Confidence calibration (k_conf tuning based on puzzle difficulty)

**Tuning (data-driven):**
- `k_conf` (confidence steepness)
- `fallback_conf_threshold` (trigger sensitivity)
- `penalty_composite_used` (preference for 1:1 vs many-to-one)
- `max_composites_per_piece_k2/k3` (balance quality vs performance)

---

## 11. Referenzen

### Design-Dokumentation

- **`docs/design/08_fallback.md`**: Many-to-One Konzept, Trigger, Composite Creation
- **`docs/test_spec/08_fallback_test_spec.md`**: Test-Spezifikation (25 Tests), Toleranzen

### Implementation

- **`solver/fallback/many_to_one.py`**: Confidence, Composites, Candidate Extension (454 Zeilen)
- **`solver/fallback/__init__.py`**: Exports (5 Funktionen)
- **`solver/beam_solver/solver.py`**: Integration (_attempt_fallback_rerun, Zeilen 177-272)
- **`solver/config.py`**: Fallback Parameters (Zeilen 372-444)

### Tests

- **`tests/test_fallback.py`**: 25 Tests (Unit + Integration)
- **`tests/test_beam_solver.py`**: 32 Tests (Regression-Check)

### Rework Issues

- **Fix 2 (Type):** dict vs list SSOT violation → Reworked to dict-only (Zeilen 189-329)
- **Fix 3 (Config):** Dual-field SSOT violation → Reworked to single field (Zeile 78)
- **Fix 5 (Filters):** Disabled filters → Reworked to re-enable (Zeilen 237-242, 271-276)

---

**Implementation Status:** ✅ Abgeschlossen (2025-12-29)
**Test Status:** 25/25 fallback, 32/32 beam_solver
**Nächster Schritt:** Schritt 9 (Pose-Refinement)
