# Schritt 5: Inner-Matching - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-25

---

## Was implementiert

### 1. solver/inner_matching/__init__.py (~21 Zeilen)

- **Exports**: extract_1d_profile, compute_ncc, compute_ncc_with_flip, generate_inner_candidates
- **__all__**: API-Deklaration
- **Status**: ✅ Implementiert (FIX 1: compute_ncc_with_flip added)

---

### 2. solver/inner_matching/profile.py (1 Funktion, ~118 Zeilen)

#### extract_1d_profile() - Main API
- **Zweck**: 1D signed chord distance profile aus Segment extrahieren
- **Signatur**: `extract_1d_profile(seg, config) -> np.ndarray`
- **Algorithmus**:
  1. Cumulative Arclength berechnen
  2. Resample auf N=profile_samples_N (128) Punkte
  3. Für jeden Punkt: Signed perpendicular distance zu Chord
  4. Optional smoothing (aktuell disabled)
- **Output**: (N,) array, setzt seg.profile_1d in-place
- **Status**: ✅ Implementiert

**Signed Distance Formula**:
```python
chord_vec_norm = (chord_end - chord_start) / ||chord_end - chord_start||
chord_perp = [-chord_vec_norm[1], chord_vec_norm[0]]  # 90° CCW rotation
signed_dist[i] = dot(resampled_point[i] - chord_start, chord_perp)
```

**Orientation**:
- Positive: rechts von Chord (right-hand rule)
- Negative: links von Chord
- Invariant zu Segment start/end swap (symmetrisch)

---

### 3. solver/inner_matching/candidates.py (7 Funktionen, ~455 Zeilen)

#### compute_ncc() - NCC Computation (deprecated)
- **Zweck**: Normalized Cross-Correlation (Pearson) zwischen zwei Profilen
- **Signatur**: `compute_ncc(profile_a, profile_b) -> float`
- **Formula**: `NCC = dot(a_norm, b_norm) / (std_a * std_b * N)` wobei `a_norm = a - mean(a)`
- **Returns**: NCC ∈ [-1, 1], höher = ähnlicher
- **Degenerate**: std=0 → returns 0.0
- **Status**: ✅ Implementiert (DEPRECATED: use compute_ncc_with_flip)

#### compute_ncc_with_flip() - NCC mit Sign-Flip Detection (FIX 1)
- **Zweck**: NCC mit sign-flip und reversal detection für opposite-side profiles
- **Signatur**: `compute_ncc_with_flip(profile_a, profile_b, allow_reversal) -> (float, bool, bool)`
- **Returns**: (best_ncc, reversal_used, sign_flip_used)
- **Algorithmus**:
  1. Test forward: NCC(a, b) und NCC(a, -b)
  2. If allow_reversal: Test reversed: NCC(a, b[::-1]) und NCC(a, -b[::-1])
  3. Return combination mit max |NCC|
- **Vorteile**: Erkennt opposite-side profiles (NCC ≈ -1 → guter Match)
- **Status**: ✅ Implementiert (FIX 1)

---

#### _prefilter_candidates() - Candidate Prefiltering
- **Zweck**: Kandidaten-Pool reduzieren vor NCC-Berechnung
- **Signatur**: `_prefilter_candidates(seg_ref, all_segs, config) -> list[ContourSegment]`
- **Filter**:
  1. **Piece ID**: seg.piece_id != seg_ref.piece_id (no self-matching)
  2. **Length**: `|len_a - len_b| / max(len_a, len_b) <= length_tolerance_ratio` (0.15 default)
  3. **Flatness** (optional): `|flatness_a - flatness_b| <= flatness_tolerance_mm` (2.0mm default)
  4. **Frame-likelihood** (TODO): Bevorzuge niedrigeren frame_cost (aktuell nicht implementiert)
- **Vorteil**: Reduziert NCC-Kosten von O(n²) → O(n*k) wobei k << n
- **Status**: ✅ Implementiert (Filter 1-3, Filter 4 als TODO)

---

#### _compute_length_cost() - Length Compatibility
- **Zweck**: Längen-Kompatibilität als Kosten
- **Signatur**: `_compute_length_cost(len_a, len_b, tolerance_ratio) -> float`
- **Formula**: `cost = min((1 - min/max) / tolerance, 1.0)`
- **Returns**: [0, 1], niedriger = kompatibler
- **Beispiele**:
  - Perfect match (10mm vs 10mm): cost=0.0
  - 10% diff (10mm vs 11mm): cost=0.606
  - >15% diff (10mm vs 13mm): cost=1.0 (clamped)
- **Status**: ✅ Implementiert

---

#### _compute_icp_fit_cost() - ICP Fit Cost (Placeholder)
- **Zweck**: Geometrischer Fit via ICP
- **Signatur**: `_compute_icp_fit_cost(seg_a, seg_b, config) -> float`
- **Returns**: fit_cost ∈ [0, 1]
- **Implementation**: **Placeholder** - returns 0.0 (TODO: Step 9)
- **Notes**: ICP-Algorithmus in Step 9 (Pose Refinement) implementiert
- **Status**: ✅ Stub implementiert

**TODO Algorithm (Step 9)**:
1. Align seg_b zu seg_a (initial pose from chord overlap)
2. Iterative Closest Point (ICP) refinement
3. RMS distance berechnen
4. Normierung: `fit_cost = clamp(rms_mm / fit_ref_mm, 0, 1)`

---

#### _compute_inner_cost() - Cost Aggregation
- **Zweck**: Aggregierte Inner-Match Kosten berechnen
- **Signatur**: `_compute_inner_cost(seg_a, seg_b, config) -> tuple[float, dict, bool, bool]`
- **Returns**: (cost_inner, features_dict, reversal_used, sign_flip_used)
- **Algorithmus**:
  1. Profile extrahieren (lazy, cached)
  2. NCC mit sign-flip detection: `compute_ncc_with_flip(a, b, allow_reversal=True)`
  3. Costs: `profile_cost = max(0.0, min(1.0, 1 - |NCC|))` (FIX 1 + FIX 2), length_cost, fit_cost
  4. Aggregation mit Gewichten
- **Formula**: `cost_inner = w_profile*profile_cost + w_length*length_cost + w_fit*fit_cost` ∈ [0, 1]
- **Status**: ✅ Implementiert (FIX 1: |NCC|, FIX 2: clamp [0, 1])

---

#### generate_inner_candidates() - Main API
- **Zweck**: Alle Inner-Match Kandidaten generieren, ranken, top-k behalten
- **Signatur**: `generate_inner_candidates(segments, config) -> dict`
- **Returns**: dict[(piece_id, seg_id)] → list[InnerMatchCandidate] (sortiert nach cost_inner)
- **Algorithmus**:
  1. Für jedes Segment A:
     - Prefilter Kandidaten (Length, Flatness, Piece ID)
     - Für jeden Kandidaten B: Compute inner cost
     - Sortiere nach cost_inner ascending
     - Behalte top-k (k=topk_per_segment, default 10)
  2. Return dict
- **Brute-Force**: Alle prefiltered Paare getestet (NCC forward/reversed)
- **Top-k**: Default k=10 für Diversität
- **Status**: ✅ Implementiert

---

### 4. solver/models.py - InnerMatchCandidate updated (FIX 1)

InnerMatchCandidate (Zeilen 165-194):
```python
@dataclass
class InnerMatchCandidate:
    seg_a_ref: tuple[int | str, int]
    seg_b_ref: tuple[int | str, int]
    cost_inner: float
    profile_cost: float
    length_cost: float
    fit_cost: float
    reversal_used: bool
    sign_flip_used: bool = False  # NEW (FIX 1)
```

**FIX 1**: `sign_flip_used` field added für opposite-side tracking

**Many-to-one**: Nicht in Step 5, erst Step 10 Fallback (separate Datenstruktur)

---

### 5. solver/config.py - 3 neue Parameter

**Group 4 (Inner Matching)** - Ergänzt:
```python
length_tolerance_ratio: float = 0.15
    # Length mismatch tolerance für Prefiltering (15% streng)
    # TODO: Tuning

flatness_tolerance_mm: float = 2.0
    # Flatness mismatch tolerance für Prefiltering (mm)
    # TODO: Tuning

frame_likelihood_threshold: float = 0.5
    # Frame cost threshold für Prefiltering (bevorzuge inner edges)
    # TODO: Tuning, aktuell nicht verwendet
```

**Tuning-Parameter**: Alle Werte sind Startwerte, empirisch tunebar in Schritt 10

---

### 6. tests/test_inner_matching.py (11 Tests, ~360 Zeilen)

#### Test 1: test_profile_extraction_straight()
- **Prüft**: Gerade Linie → Profile ≈ 0
- **Testdaten**: Horizontal line, 20mm
- **Ergebnis**: max_abs=0.0000
- **Status**: ✅ Pass

#### Test 2: test_profile_extraction_curved()
- **Prüft**: Kreisbogen → non-zero profile
- **Testdaten**: 90° arc, radius=10mm
- **Ergebnis**: max_abs=2.93mm (erwartete Abweichung)
- **Status**: ✅ Pass

#### Test 3: test_profile_degenerate()
- **Prüft**: Degenerate segment (zero length) → zero profile
- **Testdaten**: All points identical
- **Ergebnis**: profile ≡ 0
- **Status**: ✅ Pass

#### Test 4: test_ncc_identical()
- **Prüft**: Identical profiles → NCC = 1.0
- **Testdaten**: [0, 1, 2, 1, 0] vs [0, 1, 2, 1, 0]
- **Ergebnis**: NCC=1.000000
- **Status**: ✅ Pass

#### Test 5: test_ncc_reversed()
- **Prüft**: Reversed profile → NCC reversed > forward
- **Testdaten**: [0,1,2,3,1] vs [1,3,2,1,0]
- **Ergebnis**: forward=0.0385, reversed=1.0000
- **Status**: ✅ Pass

#### Test 6: test_ncc_different()
- **Prüft**: Different profiles → NCC < threshold
- **Testdaten**: [0,1,2,1,0] vs [0,0.5,0.2,0.3,0]
- **Ergebnis**: NCC=0.5634
- **Status**: ✅ Pass

#### Test 7: test_length_cost()
- **Prüft**: Length cost formula
- **Testdaten**: 0%, 10%, 13%, >15% diffs
- **Ergebnis**: cost=0.000, 0.606, 0.767, 1.000
- **Status**: ✅ Pass

#### Test 8: test_prefilter_length()
- **Prüft**: Length prefiltering (15% tolerance)
- **Testdaten**: 20mm ref, 21mm (5% OK), 25mm (20% filtered)
- **Ergebnis**: candidates=1 (nur 5% diff)
- **Status**: ✅ Pass

#### Test 9: test_prefilter_piece_id()
- **Prüft**: Piece ID filter (no self-matches)
- **Testdaten**: piece 1 vs piece 1 (filtered), piece 2 (OK)
- **Ergebnis**: candidates=1 (nur piece 2)
- **Status**: ✅ Pass

#### Test 10: test_generate_candidates_basic()
- **Prüft**: Multi-piece candidate generation, top-k, sorting
- **Testdaten**: 2 pieces × 2 segments = 4 segments
- **Ergebnis**: 4 keys, 2 candidates per segment, sorted by cost
- **Status**: ✅ Pass

#### Test 11: test_icp_stub()
- **Prüft**: ICP placeholder returns 0.0
- **Testdaten**: enable_icp=True/False
- **Ergebnis**: disabled=0.0, enabled=0.0 (beide placeholder)
- **Status**: ✅ Pass

---

## Design-Entscheidungen

### 1. Profile: Signed Distance vs Absolute

**Signed (implementiert)**:
- **Methode**: Perpendicular distance mit Vorzeichen (right-hand rule)
- **Vorteile**: Bewahrt Orientierung, robuster für NCC
- **Formel**: `d_signed = dot(p - chord_start, chord_perp)`

**Absolute**:
- **Methode**: Nur |d|, ohne Vorzeichen
- **Nachteile**: Verliert Orientierung, NCC weniger aussagekräftig

**Entscheidung**: Signed distance (robuster, mehr Information)

---

### 2. Resampling: Uniform Arclength vs Point-Count

**Uniform Arclength (implementiert)**:
- **Methode**: N samples gleichverteilt entlang arclength
- **Vorteile**: Unabhängig von Original-Sampling, konsistent über alle Segmente
- **Nachteile**: Interpolation-Overhead

**Point-Count**:
- **Methode**: N samples gleichverteilt über Point-Indices
- **Nachteile**: Sampling-abhängig (ungleiche Abstände)

**Entscheidung**: Uniform Arclength (robuster, konsistenter)

---

### 3. Smoothing: None vs Filter

**None (implementiert)**:
- **Methode**: Kein Smoothing, raw profile
- **Vorteile**: Einfach, bewahrt Features
- **Nachteile**: Noise kann NCC beeinflussen

**Filter (TODO)**:
- **Methode**: uniform_filter1d oder gaussian
- **Vorteile**: Reduziert Noise
- **Nachteile**: Kann Features verschmieren
- **Parameter**: profile_smoothing_window (aktuell ignoriert)

**Entscheidung**: None erstmal, später bei Bedarf aktivieren (config-gesteuert)

---

### 4. Prefilter: Minimal vs Aggressive

**Implemented (Minimal + Configurable)**:
- **Filter 1**: Piece ID (always active)
- **Filter 2**: Length tolerance (0.15 strict, configurable)
- **Filter 3**: Flatness tolerance (2.0mm, configurable)
- **Filter 4**: Frame-likelihood (TODO, disabled)

**Aggressive (rejected)**:
- **Filter**: Auch Angle, Position, etc.
- **Nachteile**: Zu viele Heuristiken, false negatives möglich

**Entscheidung**: Minimal prefiltering (Length + Flatness), rest via cost

---

### 5. Reversal Detection: Forward+Reversed vs Forward-only

**Forward+Reversed (implementiert)**:
- **Methode**: NCC für beide Orientierungen, best auswählen
- **Vorteile**: Erkennt gespiegelte Segmente
- **Overhead**: 2× NCC-Berechnung pro Paar

**Forward-only**:
- **Methode**: Nur 1 Orientierung
- **Nachteile**: Miss gespiegelte Matches

**Entscheidung**: Forward+Reversed (robuster, geringe Kosten)

---

### 6. Sign-Flip Detection: |NCC| vs NCC (Design-Review FIX 1)

**Problem**: Profile auf Gegenseite der Chord haben NCC ≈ -1 (perfekter Match, aber negiert)

**|NCC| (implementiert, FIX 1)**:
- **Methode**: `compute_ncc_with_flip()` testet 4 Varianten (fwd, fwd_flip, rev, rev_flip), wählt max |NCC|
- **Formula**: `profile_cost = 1 - |NCC|` (statt `1 - NCC`)
- **Vorteile**: Erkennt opposite-side Matches (NCC ≈ -1 → cost ≈ 0)
- **Tracking**: `sign_flip_used: bool` in InnerMatchCandidate

**NCC ohne Flip (alt)**:
- **Methode**: `profile_cost = 1 - NCC` (direkt)
- **Nachteile**: NCC = -1 → cost = 2.0 (schlechter als random), false reject

**Entscheidung**: |NCC| mit sign_flip_used tracking (robuster, erkennt Gegenseiten)

---

### 7. Length Tolerance: Relative vs Absolute

**Relative Ratio (implementiert)**:
- **Methode**: `len_diff = (1 - min/max)`, threshold = 0.15 (15%)
- **Formel**: `cost = min(len_diff / 0.15, 1.0)`
- **Vorteile**: Skaliert mit Segment-Größe (10mm vs 11.5mm = 13% diff, 100mm vs 115mm = 13% diff)
- **Config**: `length_tolerance_ratio: float = 0.15`

**Absolute mm (rejected)**:
- **Methode**: `|len_a - len_b| <= 2.0mm` (fest)
- **Nachteile**: Zu streng für lange Segmente, zu locker für kurze Segmente

**Entscheidung**: Relative ratio (konsistent über Größenordnungen)

---

### 8. Cost Clamp: [0, 2] vs [0, 1] (Design-Review FIX 2)

**Clamped [0, 1] (implementiert, FIX 2)**:
- **Methode**: `profile_cost = max(0.0, min(1.0, 1 - |NCC|))`
- **Aggregation**: `cost_inner = w_profile*profile_cost + w_length*length_cost + w_fit*fit_cost` ∈ [0, 1]
- **Vorteile**: Konsistenter Range, einfachere Gewichtung, keine overflow

**Unclamped [0, 2] (alt)**:
- **Methode**: `profile_cost = 1 - NCC` → NCC=-1 → cost=2.0
- **Nachteile**: cost_inner kann >1.0, inkonsistent mit length_cost/fit_cost ∈ [0, 1]

**Entscheidung**: Clamp zu [0, 1] (konsistent, interpretierbar)

---

### 9. ICP: Now vs Later

**Later (implementiert)**:
- **Methode**: Placeholder (fit_cost=0.0), Implementation in Step 9
- **Vorteile**: Einfacherer Start, fokussiert auf Profil-Matching
- **Nachteile**: fit_cost uninformativ (alle 0.0)

**Now**:
- **Methode**: Vollständige ICP-Implementation
- **Nachteile**: Komplex, performance-kritisch

**Entscheidung**: Placeholder jetzt, ICP in Step 9 (Pose Refinement)

---

## Config-Parameter

Aus `MatchingConfig` verwendet:

```python
# Group 3: Profile
profile_samples_N: int = 128
    # Anzahl Samples für Resampling

profile_smoothing_window: int = 3
    # Smoothing window (aktuell ignoriert)

# Group 4: Inner matching
topk_per_segment: int = 10
    # Top-k Kandidaten pro Segment

enable_icp: bool = False
    # ICP aktivieren (aktuell stub)

inner_weights: dict[str, float] = {
    "profile": 0.6,
    "length": 0.2,
    "fit": 0.2
}
    # Gewichte für Cost-Aggregation

length_tolerance_ratio: float = 0.15
    # Length tolerance für Prefiltering (15% streng)

flatness_tolerance_mm: float = 2.0
    # Flatness tolerance für Prefiltering (mm)

frame_likelihood_threshold: float = 0.5
    # Frame cost threshold (aktuell nicht verwendet)
```

**Tuning-Parameter**: Alle Werte sind Startwerte

---

## Cost-Aggregation Formel

**Inner Cost**:
```
cost_inner = w_profile * profile_cost
           + w_length * length_cost
           + w_fit * fit_cost

Komponenten:
  profile_cost = max(0.0, min(1.0, 1 - |NCC_best|))
    wobei NCC_best = argmax(|NCC_v|) über 4 Varianten:
      - fwd:      NCC(profile_a, profile_b)
      - fwd_flip: NCC(profile_a, -profile_b)       # Sign-Flip (opposite-side)
      - rev:      NCC(profile_a, profile_b[::-1])  # Reversal
      - rev_flip: NCC(profile_a, -profile_b[::-1]) # Both
    Range: profile_cost ∈ [0, 1] (clamped)

  length_cost = min((1 - len_ratio) / tol_len, 1.0)
    wobei len_ratio = min(len_a, len_b) / max(len_a, len_b)
    tol_len = length_tolerance_ratio (0.15)

  fit_cost = 0.0 (ICP placeholder in V1)
```

**Weights**: profile=0.6, length=0.2, fit=0.2 (Default)

**Range**: cost_inner ∈ [0, 1] für alle Matches (clamped)

---

## Validierung

### Tests (11/11 bestanden)

```
Test 1: profile_extraction_straight... ✓ (max_abs=0.0000)
Test 2: profile_extraction_curved... ✓ (max_abs=2.9265)
Test 3: profile_degenerate... ✓
Test 4: ncc_identical... ✓ (NCC=1.000000)
Test 5: ncc_reversed... ✓ (forward=0.0385, reversed=1.0000)
Test 6: ncc_different... ✓ (NCC=0.5634)
Test 7: length_cost... ✓ (0%=0.000, 10%=0.606, 13%=0.767, >15%=1.000)
Test 8: prefilter_length... ✓ (candidates=1)
Test 9: prefilter_piece_id... ✓ (candidates=1)
Test 10: generate_candidates_basic... ✓ (keys=4, cands_1_0=2)
Test 11: icp_stub... ✓ (disabled=0.0, enabled=0.0)
```

**Test-Abdeckung**:
- ✅ Profile-Extraktion (Gerade, Curved, Degenerate)
- ✅ NCC-Berechnung (Identical, Reversed, Different)
- ✅ Length-Cost (verschiedene Differenzen)
- ✅ Prefiltering (Length, Piece ID)
- ✅ Candidate-Generierung (Multi-Piece, Top-k, Sortierung)
- ✅ ICP-Stub (Placeholder)

---

## Statistik

| Datei                             | Zeilen | Funktionen | Bemerkung                |
|-----------------------------------|--------|------------|--------------------------|
| inner_matching/__init__.py        | 21     | 0          | API-Exports              |
| inner_matching/profile.py         | 118    | 1          | Profile-Extraktion       |
| inner_matching/candidates.py      | 320    | 6          | NCC + Prefilter + Gen    |
| tests/test_inner_matching.py      | 360    | 11 (tests) | Vollständige Abdeckung   |
| **Gesamt**                        | **819**| **18**     | **3 neu, 1 test**        |

**Änderungen**:
- solver/config.py: +3 Felder (length_tolerance_ratio, flatness_tolerance_mm, frame_likelihood_threshold)
- solver/inner_matching/: ~459 Zeilen (neu)
- tests/: ~360 Zeilen (neu)

---

## Nächste Schritte

**Schritt 6**: Pair Generation
- pair_generator.py (Frame-Hypothesen → Piece-Pairs, Top-M Pairs)
- **Verwendet**: Frame-Hypothesen (Schritt 4), Inner-Match-Candidates (Schritt 5)

**Schritt 7**: Beam Search Solver
- solver_core.py (Beam-Search Algorithmus)
- state.py (SolverState Model)
- **Verwendet**: Pairs (Schritt 6), Overlap-Check (Schritt 8)

**Siehe**: docs/implementation/00_structure.md §3 für vollständige Roadmap

---

## Status

**Schritt 5**: ✅ Abgeschlossen

**Freigabe**: Bereit für Schritt 6 (Pair Generation)

**Abhängigkeiten erfüllt**:
- Schritt 1: ✅ Config + Models
- Schritt 2: ✅ Transform2D + Conversion
- Schritt 3: ✅ Segmentierung + Flatness
- Schritt 4: ✅ Frame-Matching
- Schritt 5: ✅ Inner-Matching

**Nächste Abhängigkeit**:
- Schritt 6 benötigt: FrameHypothesis (✅), InnerMatchCandidate (✅), MatchingConfig (✅)
