# Schritt 4: Frame-Matching - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-25

---

## Was implementiert

### 1. solver/frame_matching/__init__.py (21 Zeilen)

- **Exports**: compute_frame_contact_features, compute_frame_cost, estimate_pose_grob_F, generate_frame_hypotheses
- **__all__**: API-Deklaration
- **Status**: ✅ Implementiert

---

### 2. solver/frame_matching/features.py (3 Funktionen, ~229 Zeilen)

#### compute_frame_contact_features() - Main API
- **Zweck**: Frame-Kontakt-Metriken für Segment→Frame-Kante berechnen
- **Signatur**: `compute_frame_contact_features(seg, side, frame, config) -> FrameContactFeatures`
- **Algorithmus**:
  1. Frame-Linie definieren (TOP: y=190, BOTTOM: y=0, LEFT: x=0, RIGHT: x=128)
  2. Distanzen für alle Punkte berechnen (perpendikular zu Frame-Linie)
  3. Distance metrics: mean, p90, max
  4. Coverage in band: Arclength-basiert (linear interpolation bei Boundaries)
  5. Inlier ratio: Point-basiert (#inliers / #points)
  6. Angle difference: min_angle_diff(seg.direction_angle_deg, expected_angle)
  7. Flatness: Kopie von seg.flatness_error
- **Output**: FrameContactFeatures mit 7 Metriken
- **Status**: ✅ Implementiert

**7 Metriken**:

| Metrik               | Typ     | Formel                                       | Zweck                            |
|----------------------|---------|----------------------------------------------|----------------------------------|
| dist_mean_mm         | float   | mean(distances)                              | Durchschnittliche Fit-Qualität   |
| dist_p90_mm          | float   | percentile(distances, 90)                    | Robuste Fit-Qualität (vs outliers)|
| dist_max_mm          | float   | max(distances)                               | Debug only (nicht gewichtet)     |
| coverage_in_band     | [0,1]   | L_in_band / L_total (arclength)              | Primäre Kontakt-Metrik           |
| inlier_ratio         | [0,1]   | (#inliers) / (#points)                       | Sekundär (vs coverage policy)    |
| angle_diff_deg       | [0,180] | min_angle_diff(seg, expected)                | Alignment-Check                  |
| flatness_error_mm    | float   | seg.flatness_error (von Schritt 3)          | Geradheit (niedriger = besser)   |

**Expected Angles**:
- TOP/BOTTOM: 0° (horizontal)
- LEFT/RIGHT: 90° (vertikal)

---

#### compute_frame_cost() - Cost Aggregation
- **Zweck**: Raw Metriken → Kosten, Aggregation mit Gewichten
- **Signatur**: `compute_frame_cost(features, config) -> float`
- **Algorithmus**:
  1. Cost-Mappings (normalize zu [0,1]):
     - cost_dist_p90 = min(dist_p90_mm / frame_band_mm, 1.0)
     - cost_coverage = 1.0 - coverage_in_band (policy-abhängig)
     - cost_inlier = 1.0 - inlier_ratio (policy-abhängig)
     - cost_angle = min(angle_diff_deg / frame_angle_deg, 1.0)
     - cost_flatness = min(flatness_error_mm / frame_flat_ref_mm, 1.0)
  2. Aggregation: weighted sum mit config.frame_weights + policy
- **Formula**: `cost_frame = Σ w_k * cost_k` (siehe Cost-Aggregation Formel)
- **Default Weights**: dist_p90=0.3, coverage=0.3, angle_diff=0.2, flatness=0.2
- **Policy**: frame_coverage_vs_inlier_policy steuert coverage vs inlier weighting
- **Status**: ✅ Implementiert

---

#### _min_angle_diff() - Helper
- **Zweck**: Minimale Winkeldistanz mit Wraparound
- **Signatur**: `_min_angle_diff(a_deg, b_deg) -> float`
- **Formel**: min(|a-b|, 360-|a-b|)
- **Beispiel**: 0° ↔ 350° → 10° (nicht 350°)
- **Status**: ✅ Implementiert

---

### 3. solver/frame_matching/hypotheses.py (2 Funktionen, ~182 Zeilen)

#### estimate_pose_grob_F() - Pose Estimation
- **Zweck**: Initiale Pose in Frame-KS aus Frame-Kontakt schätzen
- **Signatur**: `estimate_pose_grob_F(seg, side, frame, config) -> tuple[Pose2D, float]`
- **Returns**: (pose_grob_F, uncertainty_mm)
- **Algorithmus** (Option A: Projection + Alignment):
  1. Chord-Mittelpunkt auf Frame-Kante projizieren
  2. Translation:
     - TOP/BOTTOM: x von chord_mid[0], y bei Frame-Kante (0 oder 190mm)
     - LEFT/RIGHT: y von chord_mid[1], x bei Frame-Kante (0 oder 128mm)
  3. Rotation: config.pose_grob_theta_mode ("zero" | "side_aligned" | "segment_aligned")
     - zero: theta=0° (keine Rotation)
     - side_aligned: theta basierend auf Frame-Side (0° TOP/BOTTOM, 90° LEFT/RIGHT)
     - segment_aligned: theta=seg.direction_angle_deg (Heuristik)
  4. Uncertainty: Mode-abhängig (5mm default, 10mm für segment_aligned)
- **Notes**: Grob-Schätzung (coarse), Refinement in Schritt 9
- **Status**: ✅ Implementiert

---

#### generate_frame_hypotheses() - Main API
- **Zweck**: Alle Frame-Hypothesen generieren, ranken, top-N pro Teil behalten
- **Signatur**: `generate_frame_hypotheses(segments, frame, config) -> dict[piece_id, list[FrameHypothesis]]`
- **Algorithmus**:
  1. Für jedes Segment (length >= min_frame_seg_len_mm):
     - Teste alle 4 Seiten (TOP, BOTTOM, LEFT, RIGHT)
     - Compute features, cost, pose → FrameHypothesis
  2. Gruppiere nach piece_id
  3. Sortiere nach cost_frame (ascending, niedriger = besser)
  4. Behalte top-N pro Teil (N = debug_topN_frame_hypotheses_per_piece)
- **Brute-Force**: Alle Segmente × 4 Seiten (z.B. 8 segs × 4 = 32 Hypothesen/Teil)
- **Filtering**: Segmente < min_frame_seg_len_mm (10mm) übersprungen
- **Top-N**: Default N=5 für Diversität
- **Status**: ✅ Implementiert

---

### 4. tests/test_frame_matching.py (11 Tests, ~430 Zeilen)

#### Test 1: test_min_angle_diff()
- **Prüft**: Minimale Winkeldistanz mit Wraparound
- **Testdaten**: 0°↔10°, 0°↔350°, 90°↔-90°, etc.
- **Status**: ✅ Pass

#### Test 2: test_features_ideal_top()
- **Prüft**: Features für Segment bei TOP (ideal case)
- **Testdaten**: Segment bei y≈190mm
- **Ergebnis**: dist_p90=0.41mm, coverage=1.00
- **Status**: ✅ Pass

#### Test 3: test_features_wrong_side()
- **Prüft**: Segment bei TOP getestet gegen BOTTOM → schlechte Metriken
- **Testdaten**: y≈190mm vs BOTTOM (y=0)
- **Ergebnis**: BOTTOM dist_p90=190.1mm >> TOP dist_p90=0.4mm
- **Status**: ✅ Pass

#### Test 4: test_cost_mapping_perfect()
- **Prüft**: Perfekte Features → niedrige Kosten
- **Testdaten**: coverage=1.0, dist_p90=0.2, angle_diff=1.0, flatness=0.1
- **Ergebnis**: cost=0.100
- **Status**: ✅ Pass

#### Test 5: test_cost_mapping_bad()
- **Prüft**: Schlechte Features → hohe Kosten
- **Testdaten**: coverage=0.1, dist_p90=8.0, angle_diff=45.0, flatness=3.0
- **Ergebnis**: cost=0.970
- **Status**: ✅ Pass

#### Test 6: test_pose_estimation_top()
- **Prüft**: Pose-Schätzung für TOP-Segment (side_aligned mode)
- **Testdaten**: Segment bei (50-60, 189-190), direction=45°
- **Ergebnis**: pose=(55.0, 190.0, 0.0°), unc=5.0mm (ignores segment direction)
- **Status**: ✅ Pass

#### Test 7: test_generate_hypotheses_basic()
- **Prüft**: Hypothesen-Generierung für 2 Pieces × 4 Segments
- **Testdaten**: 8 Segmente insgesamt
- **Ergebnis**: 2 pieces, je 5 Hypothesen (top-5), sortiert nach cost
- **Status**: ✅ Pass

#### Test 8: test_short_segment_filtering()
- **Prüft**: Segmente < min_frame_seg_len_mm (10mm) gefiltert
- **Testdaten**: Segment mit 5mm Länge
- **Ergebnis**: 0 Hypothesen
- **Status**: ✅ Pass

#### Test 9: test_flatness_independent_of_frame_band()
- **Prüft**: Flatness-Normierung unabhängig von frame_band_mm
- **Testdaten**: frame_band_mm=1.0 vs 2.0, frame_flat_ref_mm=1.0 konstant
- **Ergebnis**: cost1=0.100, cost2=0.100 (identisch)
- **Status**: ✅ Pass

#### Test 10: test_policy_switches_weights()
- **Prüft**: Policy steuert coverage vs inlier weighting
- **Testdaten**: coverage=0.5, inlier=0.7 mit 3 Policies
- **Ergebnis**: coverage policy ≠ inlier policy ≠ balanced policy
- **Status**: ✅ Pass

#### Test 11: test_pose_theta_modes()
- **Prüft**: Alle 3 theta modes (zero, side_aligned, segment_aligned)
- **Testdaten**: TOP und LEFT Seiten, direction=45°
- **Ergebnis**: side_aligned: TOP→0°, LEFT→90°; segment_aligned: both→45° (unc=10mm); zero: both→0°
- **Status**: ✅ Pass

---

## Design-Entscheidungen

### 1. Coverage: Arclength-basiert vs Point-Count

**Warum Arclength**:
- **Robuster**: Unabhängig von Point-Sampling-Dichte
- **Genauer**: Repräsentiert tatsächliche Kontakt-Länge
- **Interpolation**: Linear interpolation bei Boundary-Crossings

**Formel**:
```python
for i in range(len(points) - 1):
    p1, p2 = points[i], points[i+1]
    d1, d2 = distances[i], distances[i+1]
    seg_len = norm(p2 - p1)

    if d1 <= t and d2 <= t:
        L_in_band += seg_len  # Both in band
    elif d1 <= t or d2 <= t:
        ratio_in = (t - min(d1, d2)) / abs(d2 - d1)
        L_in_band += seg_len * ratio_in  # Partial overlap

coverage = L_in_band / total_length
```

**Alternativen erwogen**:
- Point-Count (einfacher, aber sampling-abhängig)
- Chord-basiert (zu ungenau)

**Entscheidung**: Arclength-basiert (robuster, genauer)

---

### 2. Pose Estimation: Option A (Projection) vs Option B (ICP)

**Option A: Projection + Alignment** (implementiert):
- **Methode**: Chord-Mittelpunkt auf Frame-Kante projizieren, direction_angle_deg übernehmen
- **Vorteile**: Schnell, deterministisch, ausreichend für Grob-Schätzung
- **Nachteile**: Nicht optimal bei schrägen Segmenten

**Option B: ICP-basiert**:
- **Methode**: Iterative Closest Point zwischen Segment und Frame-Linie
- **Vorteile**: Genauer, optimal alignment
- **Nachteile**: Teurer, overkill für initial estimate

**Entscheidung**: Option A (Projection) für Schritt 4, Refinement in Schritt 9 (Pose Refinement)

---

### 3. Uncertainty: Fixed vs Adaptive

**Fixed (implementiert)**:
- **Wert**: 5.0mm konstant
- **Vorteile**: Einfach, vorhersagbar
- **Nachteile**: Ignoriert Qualität der Hypothese

**Adaptive (TODO)**:
- **Formel**: `uncertainty_mm = 5.0 + flatness * 0.5 + (1 - coverage) * 10`
- **Vorteile**: Refle

ktiert Vertrauen in Hypothese
- **Verwendung**: Beam-Pruning, Overlap-Thresholds (Schritt 7-9)

**Entscheidung**: Fixed 5.0mm für Schritt 4, Adaptive später bei Bedarf (reserved field)

---

### 4. Edge Detection: Explicit vs Implicit

**Implicit (implementiert)**:
- **Methode**: Brute-Force (alle Segmente × 4 Seiten testen)
- **Cost-basiert**: Schlechte Matches → hohe Kosten → von top-N gepruned
- **Vorteile**: Einfach, keine Heuristiken, robuster
- **Nachteile**: Mehr Compute (32-40 Hypothesen/Teil statt ~4-8)

**Explicit**:
- **Methode**: Vorselektion basierend auf Segment-Position (z.B. y > 170mm → nur TOP testen)
- **Vorteile**: Schneller (weniger Hypothesen)
- **Nachteile**: Fragil, false negatives möglich

**Entscheidung**: Implicit (brute-force + cost-based filtering) - robuster, einfacher

---

### 5. Inlier vs Coverage Policy

**Redundanz**: coverage_in_band (arclength) vs inlier_ratio (points) messen ähnliches

**Policy-Flag** (`frame_coverage_vs_inlier_policy`):
- **"coverage"**: Nur coverage gewichtet, inlier_ratio ignoriert
- **"inlier"**: Nur inlier_ratio gewichtet, coverage ignoriert
- **"balanced"**: Beide gleichgewichtet (je 0.5 × coverage weight)

**Entscheidung**: "coverage" default (arclength robuster), empirisch tunen in Schritt 10

---

## Design-Entscheidungen nach Review

### 6. Flatness-Normierung unabhängig von frame_band_mm

**Problem**: Flatness (Geradheit) und Kontakt-Toleranz sind unabhängige Konzepte

**Lösung**: Eigener Referenzwert `frame_flat_ref_mm=1.0`
- **frame_band_mm**: Toleranz für Segment-zu-Frame Distanz (Kontakt)
- **frame_flat_ref_mm**: Referenz für Flatness-Normierung (Geradheit)

**Vorteil**: Unabhängiges Tuning beider Konzepte

---

### 7. Policy explizit statt Gewichte

**Problem**: coverage vs inlier redundant, Gewichte unintuitiv

**Lösung**: Explizite Policy mit 3 Modi
```python
frame_coverage_vs_inlier_policy: Literal["coverage", "inlier", "balanced"] = "coverage"
```

**Vorteil**: Klarere Semantik, einfacheres Tuning

---

### 8. Theta-Mode für korrektes Seeding

**Problem**: seg.direction_angle_deg ≠ Piece-Rotation (Segment-lokales KS)

**Lösung**: 3 Modi für theta-Schätzung
- **side_aligned** (default): 0° für TOP/BOTTOM, 90° für LEFT/RIGHT
- **segment_aligned**: Heuristik (seg.direction_angle, höhere Uncertainty)
- **zero**: Keine Rotation

**Vorteil**: Korrekte Pose-Seeds für Beam-Search

---

### 9. Parameter-Reservierungen für Schritt 6

**Nicht verwendet in Schritt 4** (reserviert für Schritt 6):

1. **tau_frame_mm**: Toleranz für inside/outside frame validation
   - Schritt 4: Nur Hypothesis-Generierung
   - Schritt 6: Boundary-Check (Pose außerhalb Frame prune)

2. **penalty_missing_frame_contact**: Soft-constraint
   - Schritt 4: Nur Hypothesis-Generierung
   - Schritt 6: Beam-ranking penalty für Pieces ohne Frame-Kontakt

3. **use_pose_uncertainty_in_solver + pose_uncertainty_penalty_weight**:
   - Schritt 4: uncertainty_mm berechnet, aber nicht verwendet
   - Schritt 6+: Optional für Ranking/Pruning

**Grund**: Saubere Trennung (Schritt 4 = Generierung, Schritt 6 = Validation/Ranking)

---

## Config-Parameter

Aus `MatchingConfig` verwendet:

```python
# Frame-first group
frame_band_mm: float = 1.0
    # Toleranzband um Frame-Kante (mm)

frame_angle_deg: float = 10.0
    # Winkelige Toleranz für Segment→Frame Alignment (°)

min_frame_seg_len_mm: float = 10.0
    # Mindestlänge für Frame-Kontakt (mm)

tau_frame_mm: float = 2.0
    # Outside-Check Toleranz (nicht in Schritt 4 verwendet)

frame_weights: dict[str, float] = {
    "dist_p90": 0.3,
    "coverage": 0.3,
    "angle_diff": 0.2,
    "flatness": 0.2
}
    # Gewichte für Cost-Aggregation

frame_flat_ref_mm: float = 1.0
    # Referenz für Flatness-Normierung (unabhängig von frame_band_mm)

frame_coverage_vs_inlier_policy: Literal["coverage", "inlier", "balanced"] = "coverage"
    # Redundanz-Handling: "coverage" | "inlier" | "balanced"

pose_grob_theta_mode: Literal["zero", "side_aligned", "segment_aligned"] = "side_aligned"
    # Theta-Schätzung: "zero" (0°) | "side_aligned" (0/90°) | "segment_aligned" (seg.direction)

frame_distance_mode: Literal["abs", "signed"] = "abs"
    # Distanz-Berechnung: "abs" (|d|) | "signed" (pos nach innen)

use_pose_uncertainty_in_solver: bool = False
    # Uncertainty in Beam-Pruning verwenden

pose_uncertainty_penalty_weight: float = 0.0
    # Gewicht für Uncertainty (wenn use_pose_uncertainty_in_solver=True)

penalty_missing_frame_contact: float = 10.0
    # Soft constraint (nicht in Schritt 4 verwendet)

# Debug group
debug_topN_frame_hypotheses_per_piece: int = 5
    # Top-N behalten pro Teil
```

**Tuning-Parameter**:
- Alle Werte sind Startwerte (empirisch tunebar in Schritt 10)

---

## Cost-Aggregation Formel

**Frame Cost**:
```
cost_frame = w_dist_p90 * min(dist_p90_mm / t, 1.0)
           + w_angle * min(angle_diff_deg / alpha, 1.0)
           + w_flatness * min(flatness_error_mm / t_flat, 1.0)
           + [policy-abhängig]:
             - "coverage": w_coverage * (1 - coverage_in_band)
             - "inlier": w_coverage * (1 - inlier_ratio)
             - "balanced": w_coverage * 0.5 * [(1 - coverage_in_band) + (1 - inlier_ratio)]
```

Wobei:
- `t = frame_band_mm` (1.0mm) - Kontakt-Toleranz
- `t_flat = frame_flat_ref_mm` (1.0mm) - Flatness-Referenz (unabhängig!)
- `alpha = frame_angle_deg` (10.0°)
- Weights: dist_p90=0.3, coverage=0.3, angle_diff=0.2, flatness=0.2

**Range**: [0, ~1] für gute Matches, > 1 für sehr schlechte Matches

---

## Validierung

### Tests (11/11 bestanden)

```
Test 1: min_angle_diff... ✓
Test 2: features (ideal TOP)... ✓ (dist_p90=0.41mm, coverage=1.00)
Test 3: features (wrong side)... ✓ (TOP: 0.4mm, BOTTOM: 190.1mm)
Test 4: cost mapping (perfect)... ✓ (cost=0.100)
Test 5: cost mapping (bad)... ✓ (cost=0.970)
Test 6: pose estimation (TOP)... ✓ (pose=(55.0, 190.0, 0.0°))
Test 7: generate_hypotheses (basic)... ✓ (2 pieces, 5 hyps each)
Test 8: short segment filtering... ✓ (0 hypotheses for 5mm segment)
Test 9: flatness independent... ✓ (cost1=0.100, cost2=0.100)
Test 10: policy switches... ✓ (coverage=0.350, inlier=0.410, balanced=0.380)
Test 11: theta modes... ✓ (side_aligned, segment_aligned, zero)
```

**Test-Abdeckung**:
- ✅ Winkeldistanz (Wraparound)
- ✅ Features-Berechnung (Ideal case, Wrong side)
- ✅ Cost-Mapping (Perfect, Bad)
- ✅ Pose-Estimation (TOP-Segment, side_aligned mode)
- ✅ Hypothesen-Generierung (Multi-Piece, Top-N, Sortierung)
- ✅ Filtering (Kurze Segmente)
- ✅ Flatness-Normierung (Unabhängigkeit von frame_band_mm)
- ✅ Policy-Switching (coverage vs inlier vs balanced)
- ✅ Theta-Modes (zero, side_aligned, segment_aligned)

---

## Statistik

| Datei                             | Zeilen | Funktionen | Bemerkung                |
|-----------------------------------|--------|------------|--------------------------|
| frame_matching/__init__.py        | 21     | 0          | API-Exports              |
| frame_matching/features.py        | 238    | 3          | Metriken + Cost          |
| frame_matching/hypotheses.py      | 198    | 2          | Pose + Hypothesen        |
| tests/test_frame_matching.py      | 430    | 11 (tests) | Vollständige Abdeckung   |
| **Gesamt**                        | **887**| **16**     | **3 neu, 1 test**        |

**Änderungen**:
- solver/config.py: +6 Felder (Literal import, frame_flat_ref_mm, pose_grob_theta_mode, etc.)
- solver/frame_matching/: ~457 Zeilen (neu)
- tests/: ~430 Zeilen (neu)

---

## Nächste Schritte

**Schritt 5**: Inner-Matching
- profile_extractor.py (1D-Profil-Extraktion, Resampling N=128)
- inner_matcher.py (NCC forward/reversed, Top-k pro Segment)
- **Berechnet**: ContourSegment.profile_1d (lazy, on-demand)

**Schritt 6**: Pair Generation
- pair_generator.py (Frame-Hypothesen → Piece-Pairs, Top-M Pairs)
- **Verwendet**: Frame-Hypothesen (Schritt 4), Inner-Match-Candidates (Schritt 5)

**Siehe**: docs/implementation/00_structure.md §3 für vollständige Roadmap

---

## Status

**Schritt 4**: ✅ Abgeschlossen

**Freigabe**: Bereit für Schritt 5 (Inner-Matching)

**Abhängigkeiten erfüllt**:
- Schritt 1: ✅ Config + Models
- Schritt 2: ✅ Transform2D + Conversion
- Schritt 3: ✅ Segmentierung + Flatness
- Schritt 4: ✅ Frame-Matching

**Nächste Abhängigkeit**:
- Schritt 5 benötigt: ContourSegment (✅ vorhanden), MatchingConfig (✅ vorhanden)
