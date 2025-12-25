# Schritt 3: Segmentierung + Flatness - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-23

---

## Was implementiert

### 1. solver/segmentation/contour_segmenter.py (8 Funktionen, ~350 Zeilen)

#### segment_piece() - Main API
- **Zweck**: Piece-Kontur in stabile Segmente aufteilen
- **Signatur**: `segment_piece(piece, config) -> List[ContourSegment]`
- **Algorithmus**:
  1. Split: Krümmungsmaxima via Winkeländerung (Schwelle 30°)
  2. Merge: Segmente < min_frame_seg_len_mm kombinieren
  3. Target: 4-12 Segmente pro Teil
  4. Compute: chord, direction_angle_deg, flatness_error
- **Output**: ContourSegment-Liste mit allen Feldern (außer profile_1d)
- **Status**: ✅ Implementiert

#### _find_split_candidates()
- **Zweck**: Split-Kandidaten via Krümmungsmaxima
- **Methode**: Lokale Tangentenwinkel (Fenster 5 Punkte)
- **Schwelle**: Winkeländerung > 30° → Split
- **Status**: ✅ Implementiert

#### _create_segments_from_splits()
- **Zweck**: Initial-Segmente aus Split-Indizes erstellen
- **Logik**: [0:split[0]], [split[0]:split[1]], ..., [split[-1]:N]
- **Edge-Case**: Keine Splits → Gesamte Kontur als 1 Segment
- **Status**: ✅ Implementiert

#### _merge_short_segments()
- **Zweck**: Kurze Segmente mergen (Länge-basiert)
- **Strategie**: Greedy shortest-first
  - Kürzestes Segment < min_len_mm finden
  - Mit kürzerem Nachbarn mergen
  - Wiederholen bis target_count_range oder keine kurzen Segmente
- **Target**: 4-12 Segmente
- **Status**: ✅ Implementiert

#### _compute_chord()
- **Zweck**: Chord-Endpunkte (start, end)
- **Implementierung**: Erstes/Letztes Punkt im Segment
- **Format**: `(start_pt.copy(), end_pt.copy())` für Immutability
- **Status**: ✅ Implementiert

#### _compute_direction_angle()
- **Zweck**: Chord-Richtungswinkel in Grad [-180, 180)
- **Formel**: `atan2(dy, dx)` mit dy = end_y - start_y
- **Konvention**: CCW positiv (0°=rechts, 90°=oben)
- **Status**: ✅ Implementiert

#### _compute_flatness_error()
- **Zweck**: RMS perpendikular-Distanz Punkte→Chord
- **Algorithmus**:
  1. Chord-Linie: L(t) = start + t*(end - start)
  2. Für jeden Punkt: Perpendikular-Distanz berechnen
  3. RMS: sqrt(mean(distances²))
- **Degenerate-Case**: chord_len < 1e-6 → return 0.0
- **Status**: ✅ Implementiert (V1: RMS Punkt-zu-Sehne)

#### _compute_arclength()
- **Zweck**: Totale Bogenlänge der Punktsequenz
- **Formel**: sum(||p[i+1] - p[i]||) für i in range(M-1)
- **Implementierung**: Euklidische Distanz zwischen konsekutiven Punkten
- **Status**: ✅ Implementiert

---

### 2. solver/segmentation/__init__.py (10 Zeilen)

- **Exports**: segment_piece
- **__all__**: API-Deklaration
- **Status**: ✅ Implementiert

---

### 3. tests/test_segmentation.py (11 Tests, ~340 Zeilen)

#### Test 1: test_arclength()
- **Prüft**: Arclength-Berechnung (gerade Linie)
- **Testdaten**: (0,0)→(3,0)→(3,4), erwartete Länge 7.0
- **Status**: ✅ Pass

#### Test 2: test_chord_computation()
- **Prüft**: Chord-Endpunkte korrekt
- **Testdaten**: 3-Punkt-Segment, start/end shape (2,)
- **Status**: ✅ Pass

#### Test 3: test_direction_angle()
- **Prüft**: Richtungswinkel (Edge-Cases)
- **Testdaten**: 0°, 90°, ±180°, -90°, 45°
- **Status**: ✅ Pass

#### Test 4: test_flatness_error_straight_line()
- **Prüft**: Flatness ≈ 0 für perfekte Gerade
- **Testdaten**: Punkte exakt auf Chord
- **Status**: ✅ Pass

#### Test 5: test_flatness_error_curved()
- **Prüft**: Flatness > 0 für gekrümmtes Segment
- **Testdaten**: Arc mit Mittelpunkt-Deviation 0.5mm
- **Erwartete RMS**: sqrt(0.25/3) ≈ 0.289
- **Status**: ✅ Pass

#### Test 6: test_flatness_error_degenerate_chord()
- **Prüft**: Flatness = 0 für degenerierte Chord
- **Testdaten**: Alle Punkte an gleicher Position
- **Status**: ✅ Pass

#### Test 7: test_segment_piece_basic()
- **Prüft**: Segment-Count, alle Felder korrekt, wraparound merge
- **Testdaten**: Synthetische Quadrat-Kontur (40×40mm, geschlossen)
- **Ergebnis**: 3 Segmente (4 Splits → wraparound merge)
- **Status**: ✅ Pass

#### Test 8: test_segment_piece_min_length()
- **Prüft**: Segmente respektieren min_frame_seg_len_mm
- **Testdaten**: Viele kleine Zick-Zack-Kanten (2mm pro Kante)
- **Ergebnis**: 1/1 Segment >= 10.0mm (merged)
- **Status**: ✅ Pass

#### Test 9: test_segment_piece_simple_contour()
- **Prüft**: Segmentierung einfacher Konturen (Dreieck, geschlossen)
- **Testdaten**: Dreieck mit 3 Kanten (je 15 Punkte)
- **Ergebnis**: 3 Segmente
- **Status**: ✅ Pass

#### Test 10: test_segment_piece_high_threshold()
- **Prüft**: Hoher Schwellwert → weniger Splits
- **Testdaten**: Quadrat mit threshold=60° (statt default 30°)
- **Ergebnis**: 1 Segment (keine Splits bei 60°)
- **Status**: ✅ Pass

#### Test 11: test_wraparound_merge_closed_contour()
- **Prüft**: Wraparound merge für geschlossene Konturen
- **Testdaten**: Quadrat, dist(first, last) = 0.1mm < 1.0mm
- **Ergebnis**: 4 → 3 Segmente (enable_wraparound_merge)
- **Status**: ✅ Pass

---

## Design-Entscheidungen

### 1. Krümmungs-Detektion: Fixed Window + Threshold

**Warum**:
- **Window**: 5 Punkte (±2 um Zentrum)
  - Einfach zu implementieren
  - Robust gegen lokales Rauschen
  - Ausreichend für glatte Konturen
- **Threshold**: 30° Winkeländerung
  - Startwert (empirisch tunebar)
  - Konservativ (vermeidet Über-Segmentierung)

**Alternativen erwogen**:
- Adaptive Window (komplexer, nicht nötig)
- Shape-based Clustering (O(n²), zu teuer)

**Entscheidung**: Fixed Window 5, Threshold 30° (Start-Parameter, TODO: Tuning)

---

### 2. Merge-Strategie: Greedy Shortest-First

**Warum**:
- **Einfach**: Iterativ kürzestes Segment mit Nachbar mergen
- **Vorhersagbar**: Klare Regel (keine Heuristiken)
- **Effektiv**: Erreicht target_count_range + min_len Constraint

**Algorithmus**:
1. Finde kürzestes Segment < min_len_mm
2. Merge mit kürzerem Nachbarn (links/rechts)
3. Wiederhole bis target_count_range ODER keine kurzen Segmente

**Alternativen erwogen**:
- Bottom-up Clustering (komplexer)
- Adaptive Krümmungs-basiert (schwer zu tunen)

**Entscheidung**: Greedy shortest-first (einfach, robust)

---

### 3. Flatness: RMS Punkt-zu-Sehne (V1)

**Warum**:
- **RMS-Metrik**: Standard-Deviation-Ansatz
  - Robust gegen Segment-Länge
  - Geometrisch intuitive Bedeutung
- **Punkt-zu-Sehne**: Straight-Line-Abweichung
  - Einfach zu berechnen (O(M) pro Segment)
  - Ausreichend für Frame-Contact-Detektion

**Formel**:
```python
# Perpendikular-Distanz von Punkt P zu Chord-Linie L
vec_to_pt = P - start
proj_len = dot(vec_to_pt, chord_dir)
perp = vec_to_pt - proj_len * chord_dir
dist = norm(perp)

# RMS
rms = sqrt(mean(distances²))
```

**Alternativen erwogen**:
- RMS zu gefittetem Arc/Spline (teurer, nicht nötig)
- Max-Distanz statt RMS (weniger robust)

**Entscheidung**: RMS Punkt-zu-Sehne (V1, bewusst einfach)

---

### 4. Edge-Cases

**A) Zu wenige Segmente (< 4)**:
- **Policy**: Akzeptieren, Log-Warnung
- **Grund**: Sehr einfache Teile (z.B. Dreieck) können < 4 Segmente haben
- **Implementierung**: Keine Erzwingung, Merge stoppt bei Constraint-Verletzung

**B) Zu viele Segmente (> 12)**:
- **Policy**: Weiter mergen bis in Range ODER min_len verletzt
- **Implementierung**: Merge-Loop läuft bis target_max ODER keine kurzen Segmente

**C) Geschlossene Konturen**:
- **Annahme**: Erster Punkt ≈ Letzter Punkt (geschlossene Polygon)
- **Implementierung**: Wraparound merge (siehe Design-Entscheidung 4 nach Review)
- **Kriterium**: distance(contour[0], contour[-1]) < wraparound_merge_dist_mm
- **Aktion**: Merge letztes + erstes Segment → neues erstes Segment

**D) Degenerierte Chord (start ≈ end)**:
- **Policy**: flatness_error = 0.0
- **Implementierung**: Check chord_len < 1e-6 → return 0.0

---

### 5. ContourSegment Felder

Alle Felder werden in Schritt 3 berechnet, außer `profile_1d`:

| Feld                  | Berechnung                | Status     |
|-----------------------|---------------------------|------------|
| piece_id              | Von Input PuzzlePiece     | ✅ Set     |
| segment_id            | Sequentiell (0, 1, 2,...) | ✅ Set     |
| points_mm             | Segment-Punkte            | ✅ Set     |
| length_mm             | _compute_arclength()      | ✅ Set     |
| chord                 | _compute_chord()          | ✅ Set     |
| direction_angle_deg   | _compute_direction_angle()| ✅ Set     |
| flatness_error        | _compute_flatness_error() | ✅ Set     |
| profile_1d            | Lazy (Step 5)             | ✗ None     |

---

## Config-Parameter

Aus `MatchingConfig` verwendet:

```python
# Segmentation group
target_seg_count_range: tuple[int, int] = (4, 12)
    # Ziel: 4-12 Segmente pro Teil

curvature_angle_threshold_deg: float = 30.0
    # Split-Kriterium: lokaler Winkelwechsel > threshold

curvature_window_pts: int = 5
    # Fenster für Tangentenwinkel (ungerade Zahl)

enable_wraparound_merge: bool = True
    # Erlaubt Merge zwischen letztem und erstem Segment (zyklisch)

wraparound_merge_dist_mm: float = 1.0
    # Max Distanz zwischen contour[0] und contour[-1] für 'geschlossen'

# Frame-first group
min_frame_seg_len_mm: float = 10.0
    # Mindestlänge für Merge-Threshold
```

**Tuning-Parameter**:
- Alle Werte sind Startwerte (empirisch tunebar in Schritt 10)

---

## Design-Entscheidungen nach Review

### 4. Konfigurierbare Parameter (statt Hardcoded)

**Warum**:
- **Flexibilität**: Parameter empirisch tunebar (threshold, window)
- **Policy-Control**: Wraparound merge aktivierbar/deaktivierbar
- **Testbarkeit**: Verschiedene Schwellwerte in Tests prüfbar

**Implementierung**:
- `curvature_angle_threshold_deg`: Split-Kriterium (default 30°)
- `curvature_window_pts`: Tangentenwinkel-Fenster (default 5)
- `enable_wraparound_merge`: Policy-Flag (default true)
- `wraparound_merge_dist_mm`: Closed-Contour-Kriterium (default 1.0mm)

**Entscheidung**: Alle Segmentation-Parameter in Config (Design Review 2025-12-25)

---

### 5. Wraparound Merge für geschlossene Konturen

**Warum**:
- **Problem**: Geschlossene Konturen (Puzzle-Außenkanten) haben erstes Segment ≈ letztes Segment
- **Folge**: Künstliche Split-Kante bei [0] (willkürlich)
- **Lösung**: Merge letztes + erstes Segment wenn closed

**Algorithmus**:
```python
if distance(contour[0], contour[-1]) < wraparound_merge_dist_mm:
    merged_first = vstack([segments[-1], segments[0]])
    result = [merged_first] + segments[1:-1]
```

**Kriterium**: distance < 1.0mm (default) → closed

**Alternativen erwogen**:
- Kontur-Resampling mit fester Start-Position (zu komplex)
- Manuelle Wrap-Annotation (nicht automatisch)

**Entscheidung**: Automatische Wraparound merge für dist < threshold (Design Review 2025-12-25)

---

## Offene Punkte für spätere Schritte

### 1. Tuning-Parameter (Schritt 10: Integration)
- **curvature_angle_threshold_deg**: 30° Startwert, empirisch tunen
- **curvature_window_pts**: 5 Punkte Startwert, ggf. adaptiv
- **merge_strategy**: Greedy vs. andere (A/B-Test)

### 2. Profile-Berechnung (Schritt 5: Inner Matching)
- **profile_1d**: 1D-Profil (Resampling N=128)
- **Orientierung**: Konsistent mit Chord-Richtung (perpendikular right-hand)
- **Lazy**: Nur berechnen wenn für Matching benötigt

### 4. Debug-Ausgaben (Schritt 9)
- **Segment-Count Statistik**: Min/Mean/Max pro Batch
- **Längen-Distribution**: Histogramm
- **Flatness-Distribution**: Histogramm
- **Visualisierung**: Kontur + Segment-Grenzen + Chord-Linien

---

## Validierung

### Tests (11/11 bestanden)

```
Test 1: Arclength... ✓
Test 2: Chord computation... ✓
Test 3: Direction angle... ✓
Test 4: Flatness (straight line)... ✓
Test 5: Flatness (curved)... ✓
Test 6: Flatness (degenerate chord)... ✓
Test 7: segment_piece (basic)... ✓ (3 segments)
Test 8: segment_piece (min length)... ✓ (1/1 segments >= 10.0mm)
Test 9: segment_piece (triangle)... ✓ (3 segments)
Test 10: segment_piece (high threshold)... ✓ (1 segment with threshold=60.0°)
Test 11: wraparound merge (closed)... ✓ (4 → 3 segments with wraparound)
```

**Test-Abdeckung**:
- ✅ Arclength-Berechnung (Synthese-Daten)
- ✅ Chord-Computation (Start/End)
- ✅ Direction-Angle (Edge-Cases: 0°, 90°, ±180°)
- ✅ Flatness (Gerade, Gekrümmt, Degeneriert)
- ✅ segment_piece (Basis, Min-Length, Einfache Kontur)
- ✅ Konfigurierbare Parameter (high threshold)
- ✅ Wraparound Merge (closed contours)

---

## Statistik

| Datei                             | Zeilen | Funktionen | Bemerkung                |
|-----------------------------------|--------|------------|--------------------------|
| segmentation/__init__.py          | 12     | 0          | API-Exports              |
| segmentation/contour_segmenter.py | 463    | 9          | Core + Wraparound        |
| tests/test_segmentation.py        | 341    | 11 (tests) | Vollständige Abdeckung   |
| **Gesamt**                        | **816**| **20**     | **2 neu, 1 test**        |

**Änderungen (Initial)**:
- solver/segmentation/: +360 Zeilen (neu)
- tests/: +270 Zeilen (neu)

**Änderungen (Design Review)**:
- solver/config.py: +12 Zeilen (4 neue Config-Felder)
- solver/segmentation/contour_segmenter.py: +113 Zeilen (wraparound merge + config params)
- tests/test_segmentation.py: +71 Zeilen (2 neue Tests, 1 modifiziert)

---

## Nächste Schritte

**Schritt 4**: Frame-Matching
- frame_matcher/features.py (7 Metriken: dist_mean/p90/max, coverage, inlier, angle_diff, flatness)
- frame_matcher/cost.py (Cost-Mapping + Aggregation)
- frame_matcher/hypotheses.py (Top-N Frame-Hypothesen pro Teil)
- **Verwendet**: ContourSegment.flatness_error, length_mm, direction_angle_deg

**Schritt 5**: Inner-Matching
- profile_extractor.py (1D-Profil-Extraktion, Resampling N=128)
- inner_matcher.py (NCC forward/reversed, Top-k pro Segment)
- **Berechnet**: ContourSegment.profile_1d (lazy, on-demand)

**Siehe**: docs/implementation/00_structure.md §3 für vollständige Roadmap

---

## Status

**Schritt 3**: ✅ Abgeschlossen (inkl. Design Review 2025-12-25)

**Design Review**: ✅ Implementiert
- Konfigurierbare Parameter (4 neue Config-Felder)
- Wraparound Merge für geschlossene Konturen
- Tests erweitert (11/11 bestanden)

**Freigabe**: Bereit für Schritt 4 (Frame-Matching)

**Abhängigkeiten erfüllt**:
- Schritt 1: ✅ Config + Models
- Schritt 2: ✅ Transform2D + Conversion
- Schritt 3: ✅ Segmentierung + Flatness (inkl. Review)

**Nächste Abhängigkeit**:
- Schritt 4 benötigt: ContourSegment (✅ vorhanden), FrameModel (✅ vorhanden), MatchingConfig (✅ vorhanden)
