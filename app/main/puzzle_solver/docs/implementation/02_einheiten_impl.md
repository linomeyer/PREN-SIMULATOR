# Schritt 2: Einheiten & Koordinatensysteme - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-23

---

## Was implementiert

### 1. Transform2D Methoden (solver/config.py, +35 Zeilen)

#### to_matrix() → np.ndarray
- **Zweck**: Konvertiert Transform2D zu 3×3 homogener Matrix
- **Implementierung**: Standard 2D-Transformationsmatrix
  ```python
  [[cos(θ) -sin(θ) tx]
   [sin(θ)  cos(θ) ty]
   [0       0       1 ]]
  ```
- **Status**: ✅ Implementiert

#### from_matrix(mat) → Transform2D
- **Zweck**: Erstellt Transform2D aus Matrix
- **Implementierung**:
  - Translation aus mat[:2, 2]
  - Rotation via arctan2(mat[1,0], mat[0,0])
  - Normalisierung zu [-180, 180) Grad
- **Status**: ✅ Implementiert

#### compose(other) → Transform2D
- **Zweck**: Kombiniert zwei Transformationen
- **Implementierung**: Matrix-Multiplikation self @ other
- **Status**: ✅ Implementiert

#### inverse() → Transform2D
- **Zweck**: Berechnet inverse Transformation
- **Implementierung**: np.linalg.inv auf Matrix, dann from_matrix
- **Status**: ✅ Implementiert

#### apply(points) → np.ndarray
- **Zweck**: Wendet Transform auf Punkte an
- **Implementierung**:
  - Konvertierung zu homogenen Koordinaten (N, 3)
  - Matrix-Multiplikation
  - Rückkonvertierung zu (N, 2)
- **Validierung**: Shape-Check für (N, 2) Input
- **Status**: ✅ Implementiert

---

### 2. solver/utils/conversion.py (8 Funktionen, ~380 Zeilen)

**⚠ Module-Assumptions**:
- Isotrope Skalierung (sx = sy)
- Achsen-paralleles Pixel-KS (keine Rotation)
- Anisotrope/rotierte Pixel-KS nicht unterstützt in V1

#### convert_points_px_to_mm()
- **Zweck**: Punkte von Pixel zu mm
- **Signatur**: `(points_px, scale_px_to_mm, origin_offset_mm?) -> points_mm`
- **Implementierung**: `points_mm = points_px * scale + offset`
- **Validierung**: Shape-Check (N, 2)
- **Status**: ✅ Implementiert

#### convert_contour_px_to_mm()
- **Zweck**: Konturen von Pixel zu mm (Wrapper)
- **Verwendung**: PuzzlePiece.contour_mm Konvertierung
- **Status**: ✅ Implementiert

#### convert_bbox_px_to_mm()
- **Zweck**: Bounding Box von Pixel zu mm
- **Signatur**: `(x_min, y_min, x_max, y_max)_px -> (x_min, y_min, x_max, y_max)_mm`
- **Verwendung**: PuzzlePiece.bbox_mm Konvertierung
- **Status**: ✅ Implementiert

#### convert_pieces_px_to_mm() **[Design-Review]**
- **Zweck**: Batch-Konvertierung von PuzzlePiece-Liste
- **Signatur**: `(pieces, scale_px_to_mm, origin_offset_mm?) -> List[PuzzlePiece]`
- **Implementierung**: Konvertiert contour_mm, bbox_mm, center_mm pro Piece
- **Besonderheit**: Immutable (neue Instanzen), preserviert mask/image
- **Status**: ✅ Implementiert (nach Design-Review)

#### validate_pieces_format() **[Design-Review Runde 1+2]**
- **Zweck**: Input-Validierung (fail-fast)
- **Signatur**: `(pieces, max_mm_extent=1000.0) -> None` **[Runde 2]**
- **Prüfungen**:
  - contour_mm shape (N, 2), N >= 3
  - bbox_mm tuple(4), plausible bounds
  - center_mm shape (2,) wenn vorhanden
  - mm-Plausibilität (max < max_mm_extent) **[Runde 2: konfigurierbar]**
- **Raises**: ValueError bei invaliden Inputs
- **Status**: ✅ Implementiert (nach Design-Review Runde 2)

#### get_default_T_MF() **[Design-Review Runde 1]**
- **Zweck**: Default T_MF für Simulator-Visualisierung
- **Rückgabe**: Transform2D(200, 200, 0)
- **Semantik**: Rahmen-untere-linke-Ecke bei (200, 200) mm
- **Status**: ✅ Implementiert (nach Design-Review)

#### PlaceholderScaleError (Exception) **[Design-Review Runde 2]**
- **Zweck**: Runtime-Guard gegen versehentliche Produktion-Verwendung
- **Base**: RuntimeError
- **Verwendung**: get_default_scale_simulator() raises by default
- **Status**: ✅ Implementiert (nach Design-Review Runde 2)

#### Platzhalter-Funktionen
- **get_default_scale_simulator(allow_in_production=False)**: **[Runde 2: Parameter]**
  - Startwert 0.1mm/px (TODO: Kalibrierung)
  - **⚠ WARNING**: Nur für visuelle Integration, NICHT für Scoring/Pruning
  - **Raises**: PlaceholderScaleError wenn allow_in_production=False (default)
  - Verhindert versehentliche Produktion-Verwendung
- **get_machine_origin_offset_placeholder()**: (0.0, 0.0) (TODO: Physikalisches Setup)
- **extract_scale_from_metadata()**: NotImplementedError (TODO: Schritt 10)
- **Status**: ✅ Dokumentiert als bewusst offen, Runtime-Guard implementiert

---

### 3. FrameModel Docstring Update (solver/config.py, +12 Zeilen)

#### Erweiterte Dokumentation
- **Coordinate Systems**: Frame (F) vs Machine (M) präzisiert
- **Transform T_MF**: Richtung und Verwendung dokumentiert (p_M = T_MF.apply(p_F))
- **Placeholder Values**: 4 Use-Cases dokumentiert:
  1. T_MF = None für reine Frame-Koordinaten
  2. T_MF für Simulator-Visualisierung
  3. T_MF wird bei physikalischer Maschine kalibriert
  4. Verweis auf utils.conversion Platzhalter

---

### 4. solver/utils/__init__.py (17 Zeilen)

- **Exports**: convert_contour_px_to_mm, convert_bbox_px_to_mm, convert_points_px_to_mm
- **__all__**: Explizite API-Deklaration
- **Status**: ✅ Implementiert

---

## Design-Entscheidungen

### 1. Homogene Koordinaten für Transformationen
**Warum**:
- Standard-Ansatz in Computer Vision / Robotik
- Ermöglicht Komposition via Matrixmultiplikation
- Rotation + Translation in einer Operation

**Implementierung**:
```python
# Homogene Koordinaten (N, 3)
points_h = np.hstack([points, np.ones((N, 1))])
# Matrix @ Punkte
transformed_h = (mat @ points_h.T).T
# Zurück zu Kartesisch
return transformed_h[:, :2]
```

**Alternativen erwogen**:
- Separate Rotation/Translation: Komplexer, fehleranfälliger
- Quaternions: Overkill für 2D

**Entscheidung**: Homogene Koordinaten (bewährter Standard)

---

### 2. np.linalg.inv für inverse()
**Warum**:
- Einfach, korrekt, robust
- NumPy-Implementierung optimiert
- Kein manuelles Invertieren (Fehlerquelle)

**Implementierung**:
```python
mat = self.to_matrix()
mat_inv = np.linalg.inv(mat)
return Transform2D.from_matrix(mat_inv)
```

**Alternativen erwogen**:
- Manuelle Inversion für 2D (R^T, -R^T * t):
  - Schneller, aber fehleranfällig
  - Kein nennenswerter Performance-Gewinn (<1µs Unterschied)

**Entscheidung**: np.linalg.inv (Klarheit > Mikrooptimierung)

---

### 3. Separate Konvertierungsfunktionen
**Warum**:
- Klarheit: convert_contour vs convert_bbox vs convert_points
- Flexibilität: Verschiedene Datentypen (ndarray vs tuple)
- Dokumentation: Spezifische Use-Cases erkennbar

**Struktur**:
```
conversion.py:
  - convert_points_px_to_mm()    # Generisch (N, 2)
  - convert_contour_px_to_mm()   # Wrapper für Konturen
  - convert_bbox_px_to_mm()      # Spezifisch für (x_min, y_min, x_max, y_max)
```

**Alternative erwogen**: Eine generische Funktion mit Type-Overloads
- Komplexer, schlechtere IDE-Unterstützung

**Entscheidung**: Separate Funktionen (Einfachheit, Typsicherheit)

---

### 4. Platzhalter-Funktionen für unbekannte Parameter
**Warum**:
- Maschine noch nicht gebaut → scale/offset unbekannt
- Platzhalter explizit als TODO dokumentiert
- Erlaubt Simulator-Integration ohne physikalische Daten

**Implementierung**:
```python
def get_default_scale_simulator() -> float:
    # Placeholder: Assume 1px = 0.1mm (to be calibrated)
    return 0.1

def get_machine_origin_offset_placeholder() -> tuple[float, float]:
    # Placeholder: No offset (Machine origin = Camera (0, 0) scaled)
    return (0.0, 0.0)
```

**Policy**: In Code + Docstrings als bewusst offen markiert

---

### 7. KS-Mix Risiken & Guards **[Design-Review]**

**Problem**: Solver arbeitet intern in Frame (F), aber Inputs/Outputs in Machine (M)

**Risiko**: Spätere Module könnten versehentlich F/M mixen

**Guards implementiert**:
1. **Klare Konvertierungs-Policy**:
   - Input: M→F Konvertierung via convert_pieces_px_to_mm()
   - Verarbeitung: Strikt in F (alle Berechnungen)
   - Output: F→M Konvertierung via T_MF nur wenn gesetzt
2. **Dokumentation**:
   - models.py: Alle Pose2D mit KS-Tag im Kontext
   - FrameHypothesis: pose_grob_F explizit benannt
   - PuzzleSolution: poses_F vs poses_M getrennt
3. **Type-Hints** (für Code-Reviews):
   - Pose2D mit expliziter KS-Dokumentation
   - Keine generische "Pose" ohne Kontext
4. **Validierung**: validate_pieces_format() prüft mm-Plausibilität

**Empfehlung**: Code-Reviews in Schritten 3-9 auf KS-Konsistenz prüfen

---

## Abweichungen vom Design

**Keine**: 100% gemäss docs/implementation/00_structure.md §3 Schritt 2

**Ergänzungen nach Design-Review** (2025-12-23):
1. **Shape-Validierung** in apply() und convert_points_px_to_mm()
   - ValueError bei falscher Input-Shape
   - Verhindert Silent Failures
2. **Platzhalter-Funktionen** in conversion.py (initial)
   - Nicht in initial spec, aber hilfreich für Integration
   - Als TODO dokumentiert
3. **convert_pieces_px_to_mm()** [Design-Review Runde 1]
   - Batch-Konvertierung für PuzzlePiece-Liste
   - Immutable Transformation (neue Instanzen)
4. **validate_pieces_format()** [Design-Review Runde 1]
   - Input-Validierung mit ValueError (fail-fast)
   - mm-Plausibilitätsprüfung (< 1000mm)
5. **get_default_T_MF()** [Design-Review Runde 1]
   - Default T_MF = (200, 200, 0) für Simulator
   - Dokumentiert in FrameModel docstring
6. **WARNING zu get_default_scale_simulator()** [Design-Review Runde 1]
   - Explizite Warnung: Nur visuelle Integration
   - NICHT für Scoring/Pruning verwenden
7. **Module-Level Assumptions** [Design-Review Runde 1]
   - Isotrope Skalierung dokumentiert
   - Anisotrope/rotierte Pixel-KS nicht unterstützt
8. **PlaceholderScaleError Exception** [Design-Review Runde 2]
   - Runtime-Guard für get_default_scale_simulator()
   - Verhindert versehentliche Produktion-Verwendung
   - allow_in_production=True nur für Visual-Tests
9. **max_mm_extent Parameter** [Design-Review Runde 2]
   - validate_pieces_format() konfigurierbar
   - Default 1000mm, anpassbar für andere KS
   - Bessere Fehler-Messages mit Limit-Anzeige

---

## Offene Punkte für spätere Schritte

### 1. Kalibrierung (Schritt 10: Integration)
- **scale_px_to_mm**: Von Kamera-Kalibrierung bestimmen
- **origin_offset_mm**: Von physikalischem Setup bestimmen
- **T_MF**: Von mechanischem Aufbau bestimmen
- **Methode**: Kalibrier-Target (Schachbrett, ArUco-Marker)

### 2. Koordinatensystem-Tagging in Debug
- pose_F vs pose_M explizit in Debug-Ausgaben markieren
- Implementierung in Schritt 9 (Debug-Bundle)

### 3. Corner Radius
- **corner_radius_mm**: Physikalisch messen nach Rahmenbau
- Verwendung in Schritt 4 (Frame-Matching) für präzisere Ecken-Behandlung

---

## Validierung

### Manuelle Tests (empfohlen)

```python
from app.main.puzzle_solver.solver.config import Transform2D
from app.main.puzzle_solver.solver.utils.conversion import (
    convert_points_px_to_mm, convert_contour_px_to_mm, convert_bbox_px_to_mm
)
import numpy as np

# Test 1: Transform2D to_matrix / from_matrix
t = Transform2D(x_mm=10, y_mm=20, theta_deg=90)
mat = t.to_matrix()
t_back = Transform2D.from_matrix(mat)
assert np.allclose([t_back.x_mm, t_back.y_mm, t_back.theta_deg],
                   [10, 20, 90], atol=1e-10)

# Test 2: compose
t1 = Transform2D(x_mm=5, y_mm=0, theta_deg=0)
t2 = Transform2D(x_mm=0, y_mm=10, theta_deg=90)
t_comp = t1.compose(t2)
# t1 then t2: translate (5,0), then translate (0,10) and rotate 90°

# Test 3: inverse
t = Transform2D(x_mm=10, y_mm=20, theta_deg=45)
t_inv = t.inverse()
t_identity = t.compose(t_inv)
assert np.allclose([t_identity.x_mm, t_identity.y_mm, t_identity.theta_deg],
                   [0, 0, 0], atol=1e-10)

# Test 4: apply
t = Transform2D(x_mm=10, y_mm=0, theta_deg=0)
points = np.array([[0, 0], [1, 0]])
points_t = t.apply(points)
expected = np.array([[10, 0], [11, 0]])
assert np.allclose(points_t, expected, atol=1e-10)

# Test 5: Pixel to mm conversion
points_px = np.array([[100, 200], [150, 250]])
scale = 0.5  # 0.5mm per pixel
points_mm = convert_points_px_to_mm(points_px, scale)
expected = np.array([[50, 100], [75, 125]])
assert np.allclose(points_mm, expected, atol=1e-10)

# Test 6: Bbox conversion
bbox_px = (100, 200, 300, 400)
bbox_mm = convert_bbox_px_to_mm(bbox_px, 0.5)
expected = (50.0, 100.0, 150.0, 200.0)
assert bbox_mm == expected

print("✅ Alle Tests bestanden")
```

### Import-Test
```bash
.venv/bin/python -c "
from app.main.puzzle_solver.solver import Transform2D
from app.main.puzzle_solver.solver.utils import (
    convert_contour_px_to_mm, convert_bbox_px_to_mm
)
print('✅ Imports erfolgreich')
"
```

---

## Statistik

**Initial (vor Design-Review)**:

| Datei                  | Zeilen | Funktionen | Änderungen               |
|------------------------|--------|------------|--------------------------|
| config.py (Transform2D)| +35    | 5          | Methoden implementiert   |
| config.py (FrameModel) | +12    | 0          | Docstring erweitert      |
| utils/__init__.py      | 17     | 0          | Neu erstellt             |
| utils/conversion.py    | ~160   | 5 (3+2)    | Neu erstellt             |
| **Initial Gesamt**     | **~224**| **10**    | **1 geändert, 2 neu**   |

**Nach Design-Review Runde 1** (2025-12-23):

| Datei                  | Zeilen | Funktionen | Änderungen               |
|------------------------|--------|------------|--------------------------|
| config.py (Transform2D)| +35    | 5          | Methoden implementiert   |
| config.py (FrameModel) | +18    | 0          | Docstring erweitert (T_MF default) |
| utils/__init__.py      | 23     | 0          | +3 exports (review funcs)|
| utils/conversion.py    | ~380   | 8 (6+2)    | +3 funcs (batch/validate/T_MF) |
| tests/test_step2.py    | ~250   | 9          | +3 tests (review coverage) |
| **Gesamt Runde 1**     | **~706**| **22**    | **1 geändert, 3 neu**   |

**Nach Design-Review Runde 2 (FINAL)** (2025-12-23):

| Datei                  | Zeilen | Funktionen/Classes | Änderungen               |
|------------------------|--------|-------------------|--------------------------|
| config.py (Transform2D)| +35    | 5 funcs           | Methoden implementiert   |
| config.py (FrameModel) | +18    | 0                 | Docstring erweitert (T_MF default) |
| utils/__init__.py      | 28     | 0                 | +1 export (PlaceholderScaleError) |
| utils/conversion.py    | ~420   | 8 funcs + 1 class | +PlaceholderScaleError, +params |
| tests/test_step2.py    | ~300   | 10 tests          | +1 test (PlaceholderScaleError) |
| **FINAL Gesamt**       | **~801**| **23 + 1 exc**   | **1 geändert, 3 neu**   |

**Änderungen Gesamt**:
- config.py: +53 Zeilen (5 Methoden + erweiterte Docstrings)
- utils/conversion.py: ~420 Zeilen (+40 nach Review 2: Exception + param updates)
- utils/__init__.py: 28 Zeilen (+5 nach Review 2)
- tests/: ~300 Zeilen (10 Tests, +1 nach Review 2)

---

## Nächste Schritte

**Schritt 3**: Segmentierung + Flatness
- segmentation/contour_segmenter.py
- Split an Krümmungsmaxima
- Merge bis Mindestlänge (min_frame_seg_len_mm)
- Flatness V1: RMS Punkt-zu-Sehne

**Schritt 4**: Frame-Matching
- frame_matcher/features.py (7 Metriken)
- frame_matcher/cost.py (Aggregation)
- frame_matcher/hypotheses.py (Top-N)

**Siehe**: docs/implementation/00_structure.md §3 für vollständige Roadmap

---

## Status

**Schritt 2**: ✅ Abgeschlossen

**Freigabe**: Bereit für Schritt 3

**Abhängigkeiten erfüllt**:
- Schritt 1: ✅ Config + Models
- Schritt 2: ✅ Transform2D + Conversion

**Nächste Abhängigkeit**:
- Schritt 3 benötigt: ContourSegment, MatchingConfig.target_seg_count_range (bereits vorhanden)
