# Schritt 1: Fundament - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-23

---

## Was implementiert

### 1. solver/config.py (3 Klassen, 245 Zeilen)

#### Transform2D
- **Zweck**: 2D-Transformationen (Translation + Rotation)
- **Felder**: x_mm, y_mm, theta_deg (alle float, Defaults 0.0)
- **Methoden** (Stubs für Schritt 2):
  - `to_matrix() -> np.ndarray`: 3×3 homogene Matrix
  - `from_matrix(mat) -> Transform2D`: Aus Matrix erstellen
  - `compose(other) -> Transform2D`: Transformationen kombinieren
  - `inverse() -> Transform2D`: Inverse Transformation
  - `apply(points) -> np.ndarray`: Auf Punkte anwenden
- **Status**: Grundstruktur implementiert, Methoden-Implementierung in Schritt 2

#### FrameModel
- **Zweck**: A5-Rahmen-Geometrie (128×190mm)
- **Felder**:
  - `inner_width_mm: 128.0` (fix)
  - `inner_height_mm: 190.0` (fix)
  - `corner_radius_mm: None` (bewusst offen)
  - `T_MF: None` (bewusst offen, Rahmen→Maschine Transform)
- **Dokumentation**: Koordinatensystem Frame (F) mit Ursprung untere-linke Ecke

#### MatchingConfig
- **Zweck**: Alle Solver-Parameter (7 Gruppen, 20+ Parameter)
- **Gruppen**:
  1. **Frame-first** (6 Parameter): frame_band_mm, frame_angle_deg, min_frame_seg_len_mm, tau_frame_mm, frame_weights, penalty_missing_frame_contact
  2. **Segmentation** (1 Parameter): target_seg_count_range
  3. **Profile** (2 Parameter): profile_samples_N, profile_smoothing_window
  4. **Inner matching** (3 Parameter): topk_per_segment, enable_icp, inner_weights
  5. **Solver** (2 Parameter): beam_width, max_expansions
  6. **Overlap** (4 Parameter): overlap_depth_max_mm_prune/final, polygon_nonconvex_strategy, nonconvex_aggregation
  7. **Confidence/Fallback** (4 Parameter): k_conf, fallback_conf_threshold, enable_many_to_one_fallback, many_to_one_max_chain_len
  8. **Debug** (3 Parameter): debug_topN_frame_hypotheses_per_piece, debug_topN_inner_candidates_per_segment, export_debug_json
- **Defaults**: Alle aus docs/implementation/00_structure.md §6.1 übernommen
- **Dokumentation**:
  - Alle Parameter mit Docstrings
  - TODO-Markers für Tuning-Parameter
  - Empfehlungen in Kommentaren (z.B. nonconvex_aggregation: "max" empfohlen)

---

### 2. solver/models.py (8 Datenmodelle, ~260 Zeilen)

#### Pose2D
- **Zweck**: 2D-Pose (Position + Orientierung)
- **Felder**: x_mm, y_mm, theta_deg
- **Dokumentation**: Koordinatensystem muss im Kontext getaggt sein

#### PuzzlePiece
- **Zweck**: Input-Modell für extrahierte Puzzle-Teile
- **Felder**: piece_id, contour_mm, mask, bbox_mm, image (optional), center_mm (optional)
- **Dokumentation**: Schnittstelle zu piece_extraction Modul, Koordinatensystem M

#### ContourSegment
- **Zweck**: Segmentiertes Konturstück
- **Felder**: piece_id, segment_id, points_mm (N×2), length_mm, chord, direction_angle_deg, flatness_error, profile_1d (lazy)
- **Dokumentation**: Alle Koordinaten in mm, Koordinatensystem aus Kontext

#### FrameContactFeatures
- **Zweck**: Rohmetriken für Rahmenkontakt
- **Felder**: 7 Metriken (dist_mean/p90/max, coverage_in_band, inlier_ratio, angle_diff_deg, flatness_error_mm)
- **Dokumentation**: **KEINE Costs**, nur Rohwerte (Costs in Schritt 4)

#### FrameHypothesis
- **Zweck**: Frame-first Platzierungs-Hypothese
- **Felder**: piece_id, segment_id, side (TOP/BOTTOM/LEFT/RIGHT), pose_grob_F, features, cost_frame, is_committed
- **Dokumentation**: pose_grob_F im Rahmen-KS (F)

#### InnerMatchCandidate
- **Zweck**: Innenkanten-Match-Kandidat
- **Felder**: seg_a_ref, seg_b_ref (beide tuple), cost_inner, profile_cost, length_cost, fit_cost, reversal_used
- **Dokumentation**: Tuple-Refs (piece_id, segment_id) für Einfachheit und Hashability

#### SolutionStatus (Enum)
- **Zweck**: Lösungs-Status
- **Werte**: OK, OK_WITH_FALLBACK, LOW_CONFIDENCE, NO_SOLUTION, REFINEMENT_FAILED, INVALID_INPUT
- **Dokumentation**: Referenz zu docs/design/09_edgecases.md §C

#### PuzzleSolution
- **Zweck**: Finale Puzzle-Lösung
- **Felder**: poses_F, poses_M (optional), matches (list, Typ TBD), total_cost, confidence, status, overlap_violations, debug (optional)
- **Dokumentation**:
  - poses_F primär (Frame-KS)
  - poses_M nur wenn T_MF gesetzt
  - overlap_violations für A/B-Test (robustere Metrik als confidence)

**NOTE-Kommentar**: SolverState NICHT hier definiert (Single Source of Truth in beam_solver/state.py)

---

### 3. solver/__init__.py (API-Gerüst, 105 Zeilen)

#### solve_puzzle() API
- **Signatur**: `solve_puzzle(pieces, frame, config) -> PuzzleSolution`
- **Implementierung**: NotImplementedError mit Status-Info
- **Dokumentation**:
  - Google-Style Docstring
  - Args, Returns, Raises, Notes
  - Implementierungs-Status (Checkboxen für Schritte 1-10)
  - Referenzen zu Design-Dokumenten

#### __all__ Export
- Alle öffentlichen Klassen/Funktionen exportiert
- Gruppiert: Main API, Config, Models

---

## Design-Entscheidungen

### 1. dataclass für alle Modelle
**Warum**:
- Clean, deklarativ
- Auto `__init__`, `__repr__`, `__eq__`
- Optional `frozen=True` für Immutability (später)
- Type Hints first-class

**Alternativen erwogen**:
- Plain class: Mehr Boilerplate
- NamedTuple: Immutable, aber weniger Features
- Pydantic: Overkill für interne Modelle

**Entscheidung**: dataclass (Standard-Library, perfekt für diese Use-Case)

---

### 2. Type Hints: Python 3.10+ Syntax
**Warum**:
- `dict[K, V]` statt `Dict[K, V]` (kürzer, modern)
- `X | Y` statt `Union[X, Y]` (lesbarer)
- `from __future__ import annotations` für Forward References

**Anforderung**: Python 3.10+

**Projekt-Kompatibilität**: Zu prüfen in Schritt 10 (Integration)

---

### 3. Enum für SolutionStatus
**Warum**:
- Type Safety (keine Magic Strings)
- IDE-Unterstützung (Autocomplete)
- Explizite Werte-Menge

**Alternative erwogen**: String-Literals mit Literal["OK", ...]

**Entscheidung**: Enum (bewährte Praxis)

---

### 4. tuple für seg_ref statt eigene Klasse
**Warum**:
- Einfacher (kein Extra-Typ)
- Hashable (für Sets/Dicts)
- Kompatibel mit EdgeID = tuple[PieceID, int] (siehe 00_structure.md §2)

**Alternative erwogen**: @dataclass EdgeRef

**Entscheidung**: tuple (einfacher, ausreichend)

---

### 5. Transform2D Methoden als Stubs
**Warum**:
- API-Definition jetzt, Implementierung Schritt 2
- Klare Abhängigkeiten (Schritt 2: Einheiten & KS)
- NotImplementedError mit klarem Hinweis

**Implementierung Schritt 2**:
- `to_matrix()`: 3×3 homogene Matrix (Rotation + Translation)
- `from_matrix()`: Rückkonversion
- `compose()`: T1 ∘ T2
- `inverse()`: T⁻¹
- `apply()`: T(points)

---

### 6. Docstring-Stil: Google-Style
**Warum**:
- Strukturiert (Args, Returns, Raises, Notes)
- Lesbar (auch ohne Rendering)
- Standard in Python-Community

**Beispiel**:
```python
def solve_puzzle(pieces, frame, config):
    """
    Solve puzzle using multi-hypothesis beam search.

    Args:
        pieces: List of PuzzlePiece
        frame: FrameModel
        config: MatchingConfig

    Returns:
        PuzzleSolution

    Raises:
        NotImplementedError

    Notes:
        - Input in mm
        - Output in Frame (F)
    """
```

---

## Abweichungen vom Design

**Ergänzungen nach Design-Review (2025-12-23)**:
1. **PuzzlePiece Klasse hinzugefügt** (fehlte in initial spec)
   - Input-Modell für Pieces mit contour_mm, mask, bbox_mm
   - Definiert Schnittstelle zu piece_extraction Modul
2. **ContourSegment.chord spezifiziert** (tuple structure präzisiert)
   - `(start_pt, end_pt)` statt `(a_mm, b_mm)` für Klarheit
3. **Segment-Orientierung dokumentiert**
   - Chord direction (end - start)
   - Profile orientation (perpendicular right-hand)
   - reversal_used Semantik erklärt
4. **Config Policy-Flags ergänzt**:
   - `frame_coverage_vs_inlier_policy: "balanced"` (coverage vs inlier preference)
   - `penalty_composite_used: 5.0` (many-to-one penalty)
   - `polygon_nonconvex_strategy`: Triangle limit (max 50) dokumentiert

**Ansonsten**: 100% gemäss docs/implementation/00_structure.md §6.1, §2

---

## Offene Punkte für Schritt 2

### 1. Transform2D Methoden implementieren
- `to_matrix()`: 3×3 homogene Matrix
- `from_matrix()`: Aus Matrix erstellen
- `compose()`, `inverse()`, `apply()`

### 2. Konverter Pixel→mm
- Neue Datei: `utils/conversion.py`
- Funktion: `convert_contour_px_to_mm(contour, scale_px_to_mm)`
- Integration in solve_puzzle() API

### 3. KS-Tagging
- Koordinatensystem-Tags in Debug-Ausgaben
- (x_F, y_F) vs (x_M, y_M) explizit markieren

### 4. T_MF Platzhalter
- Default-Wert für Simulator (außerhalb A4)
- Dokumentiert als "bewusst offen"

---

## Validierung

### Type Checking (optional, empfohlen)
```bash
cd app/main/puzzle_solver/solver
mypy config.py models.py __init__.py
```

### Import-Test
```python
from app.main.puzzle_solver.solver import (
    solve_puzzle, MatchingConfig, FrameModel, PuzzleSolution
)

# Config erstellen
config = MatchingConfig()
frame = FrameModel()

# API aufrufen (wirft NotImplementedError)
try:
    solution = solve_puzzle([], frame, config)
except NotImplementedError as e:
    print(f"Erwartet: {e}")
```

---

## Statistik

| Datei         | Zeilen | Klassen     | Funktionen | Felder (gesamt) |
|---------------|--------|-------------|------------|-----------------|
| config.py     | ~260   | 3           | 5 (stubs)  | 26              |
| models.py     | ~260   | 8 (+1 Enum) | 0          | 44              |
| __init__.py   | ~105   | 0           | 1 (stub)   | -               |
| **Gesamt**    | **~625**| **11**     | **6**      | **70**          |

**Nach Design-Review**: +1 Klasse (PuzzlePiece), +3 Config-Felder, +6 Model-Felder

---

## Nächste Schritte

**Schritt 2**: Einheiten & KS
- Transform2D Methoden implementieren
- utils/conversion.py erstellen
- KS-Tagging in Debug
- T_MF Platzhalter definieren

**Schritt 3**: Segmentierung + Flatness
- segmentation/contour_segmenter.py
- Split an Krümmungsmaxima
- Merge bis Mindestlänge
- Flatness V1 (RMS Punkt-zu-Sehne)

**Siehe**: docs/implementation/00_structure.md §3 für vollständige Roadmap

---

## Status

**Schritt 1**: ✅ Abgeschlossen

**Freigabe**: Bereit für Schritt 2
