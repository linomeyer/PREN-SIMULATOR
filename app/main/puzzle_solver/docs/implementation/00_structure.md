# Solver V2: Struktur & Implementierungs-Roadmap

## 1. Abhängigkeiten zu bestehenden Modulen

### 1.1 Input (von piece_extraction/)

**Aktuell verfügbar:**
```python
class PuzzlePiece:
    contour: np.ndarray[(N,2)]  # Konturpunkte
    mask: np.ndarray
    bbox: (x, y, w, h)
    image: np.ndarray  # RGBA
    center: (cx, cy)
```

**Benötigt für Solver V2:**
```python
# Erweiterung um Metadaten (optional als separate Wrapper-Klasse)
class PuzzlePieceInput:
    piece: PuzzlePiece
    # Metadaten
    unit: "px" | "mm"
    coord_system: "P" | "M" | "F"  # Pixel/Maschine/Frame
    scale_px_to_mm: float | None
```

**Policy**: Wenn möglich direkt PuzzlePiece verwenden + Konvertierung zu mm im Solver-Einstieg.

---

### 1.2 Output (zu Simulator/Visualisierung)

```python
class PuzzleSolution:
    # Primäre Ausgabe
    poses_F: dict[piece_id -> Pose2D]  # im Rahmen-KS
    poses_M: dict[piece_id -> Pose2D] | None  # wenn T_MF gesetzt

    # Constraints
    matches: list[MatchConstraint]

    # Qualität
    total_cost: float
    confidence: float
    status: SolutionStatus  # OK, OK_WITH_FALLBACK, LOW_CONFIDENCE, etc.

    # Debug
    debug: DebugBundle
```

**Visualisierung braucht:**
- Posen (x_mm, y_mm, theta_deg) pro Teil
- KS-Information (F oder M)
- Optional: Debug-Overlay (Hypothesen, Matches, Costs)

---

### 1.3 Kalibrierung/Skalierung

**Annahme**: Bestehende Pipeline liefert bereits metrische Koordinaten ODER Skalierungsfaktor ist bekannt.

**Fallback**: Wenn Konturen in Pixel → manuelle Skalierung über Config (`scale_px_to_mm`).

---

## 2. Interne solver/ Modul-Struktur

```
app/main/puzzle_solver/solver/
│
├── __init__.py
│   └── API: solve_puzzle(pieces, frame, config) -> PuzzleSolution
│
├── config.py
│   ├── MatchingConfig         # Alle Parameter (siehe 03_matching.md)
│   ├── FrameModel              # Rahmengeometrie, T_MF
│   └── Transform2D             # 2D-Transformation (x, y, theta)
│
├── models.py
│   ├── ContourSegment          # Segment mit Geometrie + Features
│   ├── FrameHypothesis         # Rahmenkontakt-Hypothese
│   ├── InnerMatchCandidate     # Innenmatch-Kandidat
│   ├── PuzzleSolution          # Finale Lösung
│   ├── DebugBundle             # Debug-Daten
│   ├── Pose2D, MatchConstraint, etc.
│   └── NOTE: SolverState wird NUR in beam_solver/state.py definiert (Single Source of Truth)
│
├── segmentation/
│   ├── __init__.py
│   └── contour_segmenter.py
│       ├── segment_piece(piece) -> list[ContourSegment]
│       ├── Krümmungs-basierter Split
│       ├── Längen-basierter Merge
│       └── Flatness V1 (RMS Punkt-zu-Sehne)
│
├── frame_matching/
│   ├── __init__.py
│   ├── features.py
│   │   └── compute_frame_features(seg, side, frame) -> FrameContactFeatures
│   │       ├── dist_mean, dist_p90, dist_max
│   │       ├── coverage_in_band
│   │       ├── inlier_ratio
│   │       ├── angle_diff
│   │       └── flatness_error
│   └── hypotheses.py
│       ├── estimate_pose_grob_F(segment, side, frame) -> (Pose2D, float)
│       │   └── Returns: (pose, uncertainty_mm)
│       │       ├── Projektion Segment → Rahmenlinie
│       │       ├── Alignment Chord mit Rahmenkante
│       │       ├── uncertainty_mm: Reserved for future adaptive pruning
│       │       │   └── TODO: Overlap-Schwellen, Beam-Gewichtung
│       │       └── TODO: Definition in 04_scoring.md §3.3
│       └── generate_frame_hypotheses(segments, frame, config) -> list[FrameHypothesis]
│           ├── Features pro Segment×Seite
│           ├── Cost-Mapping + Aggregation
│           ├── pose_grob_F via estimate_pose_grob_F()
│           └── Top-N pro Teil
│
├── inner_matching/
│   ├── __init__.py
│   ├── profile.py
│   │   └── extract_1d_profile(segment, N, smoothing) -> np.ndarray
│   │       ├── Resampling auf N Samples (Start: 128)
│   │       ├── Signed chord distance
│   │       └── Optional: Glättung (window offen)
│   └── candidates.py
│       └── compute_inner_candidates(segments, config) -> dict[seg_ref -> list[InnerMatchCandidate]]
│           ├── Prefilter: Länge, Flatness, Frame-Likelihood
│           ├── NCC forward/reversed
│           ├── Optional: ICP fit_cost
│           ├── Cost-Aggregation
│           └── Top-k pro Segment
│
├── beam_solver/
│   ├── __init__.py
│   ├── state.py
│   │   └── class SolverState  # Single Source of Truth
│   │       ├── poses_F: dict[PieceID -> Pose2D]
│   │       ├── placed: set[PieceID]
│   │       ├── committed_frame_constraints: list[FrameHypothesis]
│   │       ├── active_constraints: list[MatchConstraint]
│   │       ├── cost_total: float, cost_breakdown: dict[str, float]
│   │       ├── unplaced_pieces: set[PieceID]
│   │       ├── open_edges: set[EdgeID]
│   │       │   └── EdgeID = tuple[PieceID, int]  # (piece_id, segment_id)
│   │       ├── debug_trace: list[TraceEvent]
│   │       └── is_complete() -> bool
│   │           └── all(pieces placed) AND len(open_edges) == 0
│   ├── expansion.py
│   │   └── expand_state(state, ...) -> list[SolverState]
│   │       ├── Move 1: Place via FrameHypothesis
│   │       ├── Move 2: Place via InnerMatchCandidate
│   │       └── Pose-Update, Constraint-Commit
│   └── solver.py
│       └── run_beam_solver(frame_hyps, inner_cands, frame, config) -> list[SolverState]
│           ├── Seeding (Frame seeds + leerer State)
│           ├── Beam-Loop (Expansion + Ranking)
│           ├── Pruning (outside, overlap, conflicts)
│           └── Completion-Check
│
├── collision/
│   ├── __init__.py
│   └── overlap.py
│       ├── compute_penetration_depth(poly_a, poly_b) -> float
│       │   ├── SAT/MTV für konvexe Polygone
│       │   └── Nonkonvex-Strategie (A: decomposition / B: triangulation / C: library)
│       └── check_overlap(state, threshold) -> bool
│
├── refinement/
│   ├── __init__.py
│   └── pose_refiner.py
│       └── refine_solution(solution, constraints, frame, config) -> PuzzleSolution
│           ├── Zielfunktion: J_frame + J_inner + J_overlap + J_reg
│           ├── Optimierer (Nelder-Mead / GN / Zweistufig)
│           ├── Final Check: penetration_depth ≤ 0.1mm
│           └── Debug: cost trajectory, overlap trajectory
│
├── fallback/
│   ├── __init__.py
│   └── many_to_one.py
│       └── apply_fallback(segments, config) -> list[ContourSegment]
│           ├── Trigger: confidence < threshold
│           ├── Erzeuge composite Segmente (chain_len=2)
│           ├── Erweitere Kandidatenraum
│           └── Rerun Solver
│
└── debug/
    ├── __init__.py
    └── export.py
        └── export_debug(debug_bundle, path)
            ├── Config-Dump (JSON)
            ├── Metriken (Frame-Hypothesen, Inner-Kandidaten)
            ├── Solver-Trace (Beam stats, Prune counts)
            └── Final Checks (Coverage, Overlap, Konsistenz)
```

---

## 3. Implementierungs-Reihenfolge (Bottom-Up)

### Phase 1: Fundament (Schritte 1-2)

#### Schritt 1: Config + Models
**Dateien**: `config.py`, `models.py`

**Tasks**:
- MatchingConfig mit allen Parametern:
  - Frame: frame_band_mm, frame_angle_deg, min_frame_seg_len_mm, tau_frame_mm, frame_weights, penalty_missing_frame_contact
  - Segmentation: target_seg_count_range
  - Profile: profile_samples_N, profile_smoothing_window
  - Inner: topk_per_segment, enable_icp, inner_weights
  - Solver: beam_width, max_expansions
  - Overlap: overlap_depth_max_mm_prune/final, polygon_nonconvex_strategy, nonconvex_aggregation
  - Confidence/Fallback: k_conf, fallback_conf_threshold, enable_many_to_one_fallback, many_to_one_max_chain_len
  - Debug: debug_topN_frame_hypotheses_per_piece, debug_topN_inner_candidates_per_segment, export_debug_json
- FrameModel (128×190mm, corner_radius bewusst offen, T_MF bewusst offen)
- Transform2D (x, y, theta + Matrix-Konversion)
- Datenmodelle (in models.py):
  - ContourSegment (points_mm, length, chord, direction, flatness, profile_1d)
  - FrameHypothesis (piece_id, seg_id, side, pose_grob_F, features, cost_frame)
  - InnerMatchCandidate (seg_a, seg_b, cost_inner, komponenten)
  - PuzzleSolution (poses_F, poses_M, matches, cost, conf, debug)
  - DebugBundle (config_dump, frame_hypotheses, inner_candidates, solver_summary, final_checks)
  - NOTE: SolverState wird NICHT hier definiert, sondern in beam_solver/state.py (Single Source of Truth)

**Output**: API-Gerüst `solve_puzzle()` existiert (stub), Serialisierung funktioniert (JSON).

---

#### Schritt 2: Einheiten & KS
**Dateien**: Erweiterung `models.py`, neue `utils/conversion.py`

**Tasks**:
- Konverter: Pixel→mm (über scale_px_to_mm)
- KS-Tagging: Koordinaten werden als (x_F, y_F) oder (x_M, y_M) klar markiert
- T_MF als Platzhalter (z.B. außerhalb A4, aber sichtbar in Config)
- Debug: KS-Information in allen Ausgaben

**Output**: Reproduzierbare Koordinatenflüsse, mm-basiertes Arbeiten garantiert.

---

### Phase 2: Kontursegmentierung (Schritt 3)

#### Schritt 3: Segmentierung + Flatness V1
**Dateien**: `segmentation/contour_segmenter.py`

**Tasks**:
- Split-Kandidaten: Krümmungsmaxima (Winkeländerung entlang Kontur)
- Merge: Segmente < min_frame_seg_len_mm zusammenführen
- Ziel: Segmentanzahl in target_seg_count_range (z.B. 4-12)
- Flatness V1: RMS(point_to_chord_distance) in mm
- Output: ContourSegment mit stabilen IDs, Längen, Richtungen, Flatness

**Debug**:
- Segmentanzahl pro Teil
- Segmentlängen (min/mean/max)
- Flatness-Werte

**Output**: Stabile Segment-Inputs für alle folgenden Schritte.

---

### Phase 3: Matching-Grundlagen (Schritte 4-5)

#### Schritt 4: Frame-Matching
**Dateien**: `frame_matching/features.py`, `frame_matching/hypotheses.py`

**Tasks**:
- Features-Berechnung pro Segment×Seite:
  - Distanzmetriken: dist_mean, dist_p90, dist_max (Punkte → Rahmenlinie)
  - Coverage: coverage_in_band (±t), inlier_ratio
  - Richtung: angle_diff (Segment vs Seite)
  - Form: flatness_error
- Cost-Mapping (konfigurierbare Weights)
- Aggregation: cost_frame = Σ w_k * cost_k
- pose_grob_F Schätzung (Option A: Projektion auf Rahmenlinie, bewusst offen)
- Top-N Hypothesen pro Teil speichern

**Debug**:
- Top-N Hypothesen pro Teil inkl. alle Rohmetriken
- Cost-Breakdown

**Output**: Frame-first Kandidaten existieren, Qualität messbar.

---

#### Schritt 5: Inner-Matching
**Dateien**: `inner_matching/profile.py`, `inner_matching/candidates.py`

**Tasks**:
- 1D-Profil-Extraktion:
  - Resampling auf N=128 Samples (konfigurierbar)
  - Signed chord distance
  - Optional: Glättung (window=3/5, bewusst offen)
- Similarity:
  - NCC forward: ncc(profile_a, profile_b)
  - NCC reversed: ncc(profile_a, reverse(profile_b))
  - profile_cost = 1 - max(corr_forward, corr_reversed)
- Optional: ICP fit_cost (enable_icp in Config)
- Prefilter: Längenfenster, optional Frame-Likelihood
- Cost-Aggregation: cost_inner = w_profile*profile_cost + w_len*len_cost + w_fit*fit_cost
- Top-k pro Segment

**Debug**:
- Top-k Kandidaten pro Segment inkl. Komponenten, reversal_used
- Kandidatenanzahl nach Prefilter

**Output**: Innenkanten-Kandidatenraum verfügbar.

---

### Phase 4: Globaler Solver (Schritte 6-7)

#### Schritt 6: Beam-Solver V1
**Dateien**: `beam_solver/state.py`, `beam_solver/expansion.py`, `beam_solver/solver.py`

**Tasks**:
- SolverState-Klasse (in beam_solver/state.py, Single Source of Truth):
  - Felder: poses_F, placed, committed_frame_constraints, active_constraints
  - Kosten: cost_total, cost_breakdown
  - Frontier (zwei separate Sets, keine Kollision):
    - unplaced_pieces: set[PieceID]
    - open_edges: set[EdgeID] mit EdgeID = tuple[PieceID, int]  # (piece_id, segment_id)
  - Debug: debug_trace
  - Methode: is_complete() -> bool (all placed AND len(open_edges) == 0)
- Seeding:
  - Option A: Best-Frame Seeds (Top-1 Rahmenhypothese pro Teil)
  - Option B: Leerer State (Fallback)
  - Empfehlung: Hybrid (A+B)
- Expansion:
  - Move 1: Place via FrameHypothesis → commit Frame-Constraint
  - Move 2: Place via InnerMatchCandidate → Place relativ zu platziertem Teil
- Pruning (Stub zunächst):
  - Outside Frame: tau_frame_mm Toleranz
  - Overlap: stub (0-check oder simplified)
  - Committed conflicts: harte Pruning
- Beam-Ranking: cost_total (min)
- Frontier-Mode: Hybrid
  - Phase 1: Expansion basierend auf unplaced_pieces
  - Phase 2 (ab ≥2 platziert): Expansion basierend auf open_edges (Kanten ohne Match)
- Completion-Check: is_complete() = len(placed) == n AND len(open_edges) == 0

**Debug**:
- Beam stats pro Iteration (size, expansions, prunes)
- Prune reason counts
- Best-cost progression
- Trace: commits, expansions, pruning

**Output**: Erste vollständige Lösungen (grob), Solver-Trace vorhanden.

---

#### Schritt 7: Overlap-Modul
**Dateien**: `collision/overlap.py`

**Tasks**:
- SAT/MTV für konvexe Polygone
- Nonkonvex-Strategie:
  - **Default: Triangulation** (Option B)
  - **PERFORMANCE KRITISCH**: n*(n-1)/2 Paarprüfungen × Anzahl Dreiecke pro Teil
  - Aggregation: "max" (konservativ) | "mean" | "p90" (robust)
  - Config: `polygon_nonconvex_strategy`, `nonconvex_aggregation`
- penetration_depth(poly_a, poly_b) → float (mm)
  - Bei Triangulation: max over all triangle pairs
- Startwerte:
  - overlap_depth_max_mm_prune = 1.0mm
  - overlap_depth_max_mm_final = 0.1mm
- Integration in Solver-Pruning

**Risiko-Dokumentation**:
- Triangulation einfach implementierbar (z.B. via shapely/scipy)
- Performance: O(n² × t_a × t_b) wobei t_i = Anzahl Dreiecke Teil i
- Typisch: 6 Teile × ~10-20 Dreiecke/Teil → ~400-1600 SAT-Prüfungen pro State
- Beam-Pruning limitiert States → akzeptabel für Simulator
- TODO: Falls zu langsam → Bounding-Box Pre-Check oder konvexe Zerlegung

**Debug**:
- max penetration depth pro State
- Pair causing max depth
- Anzahl overlap pairs > 0
- Anzahl SAT-Prüfungen pro State (Performance-Monitoring)

**Output**: Overlap sauber beschnitten, falsche Hypothesen fliegen früh raus.

---

### Phase 5: Robustheit (Schritte 8-9)

#### Schritt 8: Confidence + Fallback
**Dateien**: `fallback/many_to_one.py`, Erweiterung `beam_solver/solver.py`

**Tasks**:
- Confidence-Mapping: conf = exp(-k_conf * cost_total)
- k_conf als bewusst offener Parameter (Start: 1.0)
- Fallback-Trigger: conf < 0.5 (konfigurierbar)
- Many-to-one:
  - Erzeuge composite Segmente (adjazente Ketten, chain_len=2)
  - Erweitere Inner-Kandidatenraum
  - Rerun Solver mit erweiterten Kandidaten
- Debug:
  - before/after cost/conf
  - Anzahl composite Segmente
  - Welche Matches composite verwendet haben

**Output**: Robustheit steigt, besonders bei Segmentierungs-Splits.

---

#### Schritt 9: Pose-Refinement
**Dateien**: `refinement/pose_refiner.py`

**Tasks**:
- Zielfunktion:
  - J_frame: Rahmenkontakt-Kosten (committed Hypothesen)
  - J_inner: Innenkanten-Kosten (pose-abhängig, Punkt-zu-Punkt oder ICP)
  - J_overlap: Barrier/Penalty (λ * max(0, d - ε)²)
  - J_reg: optional Regularisierung (kleine Winkeländerungen)
- Optimierer wählen (bewusst offen, dokumentieren):
  - Option A: Nelder-Mead (gradient-frei, robust)
  - Option B: Gauss-Newton / LM (effizient, braucht glatte J)
  - Option C: Zweistufig (J_frame+J_inner → Overlap-Barrier)
- Abbruch:
  - max_refine_iters
  - delta_cost < threshold
  - penetration_depth_max ≤ overlap_depth_max_mm_final
- Final Check: penetration_depth ≤ 0.1mm

**Debug**:
- cost_total init/final
- Breakdown: frame/inner/overlap
- penetration_depth init/final
- Iterationslog (optional)
- Status: success/failed (reason)

**Output**: "Praktisch kein Overlap" zuverlässig erreicht.

---

### Phase 6: Integration (Schritt 10)

#### Schritt 10: Integration + Tests
**Dateien**: Erweiterung `__init__.py`, neue `tests/`, Integration in Simulator

**Tasks**:
- Integration in bestehende Simulator-Visualisierung
- A/B Vergleich:
  - Gleiche Inputs für alt/neu
  - Compare outputs, logs
- Regression-Suite:
  - Gespeicherte Szenen
  - Seed-Läufe
  - Debug-Exports archivieren
- Debug-Export pro Run:
  - JSON (config + metrics + trace)
  - Optional: Visualisierungen (Overlay)

**Output**: Stabiler Entwicklungsprozess, nachvollziehbare Verbesserungen.

---

## 4. Mapping: design/*.md → solver/ Code-Module

| Design-Dok           | Solver-Module                                      | Schwerpunkt                                    |
|----------------------|----------------------------------------------------|------------------------------------------------|
| 00_overview.md       | `__init__.py`, `config.py`                         | API, Gesamtkonzept, Anforderungen              |
| 01_coordinates.md    | `config.py` (Transform2D, FrameModel)              | KS-Definitionen, T_MF, Einheiten               |
| 02_datamodels.md     | `models.py`                                        | Alle Datenstrukturen, Schnittstellen           |
| 03_matching.md       | `config.py` (MatchingConfig), `segmentation/`, `frame_matching/`, `inner_matching/` | Config, Algorithmus (8 Phasen), Segmentierung  |
| 04_scoring.md        | `frame_matching/features.py`, `inner_matching/candidates.py`, `debug/export.py` | Metriken, Cost-Mappings, Debug-Ausgaben        |
| 05_solver.md         | `beam_solver/state.py`, `beam_solver/expansion.py`, `beam_solver/solver.py` | Solver-State, Expansion, Pruning, Beam-Search  |
| 06_collision.md      | `collision/overlap.py`                             | SAT/MTV, Overlap-Berechnung, Nonkonvex         |
| 07_refinement.md     | `refinement/pose_refiner.py`                       | Pose-Optimierung, Zielfunktion, Optimierer     |
| 08_fallback.md       | `fallback/many_to_one.py`                          | Many-to-one Matching, Composite Segmente       |
| 09_edgecases.md      | Alle Module (Error Handling, Logging)              | Edge Cases, Failure Modes, Status-Handling     |

---

## 5. Offene Punkte (bewusst offen, als Config/TODO führen)

### 5.1 Bewusst offen (Felder vorhanden, Startwerte/Platzhalter)
- `T_MF`: Platzhalter (außerhalb A4), später reale Kalibrierung
- `corner_radius_mm`: Default None oder 0, später festlegen
- `profile_smoothing_window`: Default 3, als Tuning markieren
- `k_conf`: Default 1.0, als Tuning markieren
- `polygon_nonconvex_strategy`: Default "triangulation", Performance kritisch
- `nonconvex_aggregation`: Default "max", als Tuning markieren (max/mean/p90)
- `pose_grob_F` Definition: Projektion + Alignment, API in frame_matching/hypotheses.py
- `overlap_depth_max_mm_*`: Startwerte 1.0/0.1mm, als Annahme markieren
- `frame_weights`, `inner_weights`: Startwerte gesetzt, als Tuning markieren
- `penalty_missing_frame_contact`: Default 10.0, als Tuning markieren
- `many_to_one_max_chain_len`: Default 2, Option 3 möglich

### 5.2 Policy
- **Kein Hardcoding**: alle Werte in Config
- **Dokumentation**: bewusst offene Parameter in Code + Config kommentieren
- **Debug**: offene Parameter stets in Debug-Export ausgeben
- **Tuning**: als "TODO: Tuning" markieren

---

## 6. Erste Implementierungs-Schritte (Konkret)

### 6.1 Schritt 1a: Config anlegen
**Datei**: `app/main/puzzle_solver/solver/config.py`

```python
from dataclasses import dataclass, field
import numpy as np

@dataclass
class Transform2D:
    x_mm: float = 0.0
    y_mm: float = 0.0
    theta_deg: float = 0.0

    def to_matrix(self) -> np.ndarray:
        """Convert to 3x3 homogeneous transformation matrix."""
        # TODO: Implement
        pass

@dataclass
class FrameModel:
    inner_width_mm: float = 128.0
    inner_height_mm: float = 190.0
    corner_radius_mm: float | None = None  # bewusst offen
    T_MF: Transform2D | None = None  # bewusst offen

@dataclass
class MatchingConfig:
    # Frame-first
    frame_band_mm: float = 1.0
    frame_angle_deg: float = 10.0
    min_frame_seg_len_mm: float = 10.0
    tau_frame_mm: float = 2.0
    frame_weights: dict[str, float] = field(default_factory=lambda: {
        "dist_p90": 0.3,
        "coverage": 0.3,
        "angle_diff": 0.2,
        "flatness": 0.2
    })  # TODO: Tuning
    penalty_missing_frame_contact: float = 10.0  # TODO: Tuning

    # Segmentation
    target_seg_count_range: tuple[int, int] = (4, 12)

    # Profile
    profile_samples_N: int = 128
    profile_smoothing_window: int = 3  # TODO: Tuning

    # Inner matching
    topk_per_segment: int = 10
    enable_icp: bool = False
    inner_weights: dict[str, float] = field(default_factory=lambda: {
        "profile": 0.6,
        "length": 0.2,
        "fit": 0.2
    })  # TODO: Tuning

    # Solver
    beam_width: int = 20
    max_expansions: int = 1000

    # Overlap
    overlap_depth_max_mm_prune: float = 1.0  # Annahme, Tuning
    overlap_depth_max_mm_final: float = 0.1  # Annahme, Tuning
    polygon_nonconvex_strategy: str = "triangulation"  # Option B (Default, Performance kritisch!)
    nonconvex_aggregation: str = "max"
    # "max": empfohlen für Pruning (konservativ)
    # "mean"/"p90": nur für Diagnose/Ranking-Experimente
    # TODO: Tuning

    # Confidence/Fallback
    k_conf: float = 1.0  # TODO: Tuning
    fallback_conf_threshold: float = 0.5
    enable_many_to_one_fallback: bool = True
    many_to_one_max_chain_len: int = 2  # Start: 2, Option: 3

    # Debug
    debug_topN_frame_hypotheses_per_piece: int = 5
    debug_topN_inner_candidates_per_segment: int = 5
    export_debug_json: bool = True
```

---

### 6.2 Schritt 1b: Models anlegen
**Datei**: `app/main/puzzle_solver/solver/models.py`

```python
from dataclasses import dataclass, field
import numpy as np
from typing import Optional

@dataclass
class Pose2D:
    x_mm: float
    y_mm: float
    theta_deg: float

@dataclass
class ContourSegment:
    piece_id: int | str
    segment_id: int
    points_mm: np.ndarray  # (M, 2)
    length_mm: float
    chord: tuple[np.ndarray, np.ndarray]  # (a_mm, b_mm)
    direction_angle_deg: float
    flatness_error: float
    profile_1d: Optional[np.ndarray] = None  # lazy

@dataclass
class FrameContactFeatures:
    dist_mean_mm: float
    dist_p90_mm: float
    dist_max_mm: float
    coverage_in_band: float
    inlier_ratio: float
    angle_diff_deg: float
    flatness_error_mm: float

@dataclass
class FrameHypothesis:
    piece_id: int | str
    segment_id: int
    side: str  # "TOP" | "BOTTOM" | "LEFT" | "RIGHT"
    pose_grob_F: Pose2D
    features: FrameContactFeatures
    cost_frame: float
    is_committed: bool = False

# ... weitere Modelle (InnerMatchCandidate, SolverState, etc.)
```

---

### 6.3 Schritt 1c: API-Gerüst
**Datei**: `app/main/puzzle_solver/solver/__init__.py`

```python
from typing import List
from .config import MatchingConfig, FrameModel
from .models import PuzzleSolution
# TODO: Import PuzzlePiece from existing module
# from app.main.puzzle_solver.piece_extraction.extractor import PuzzlePiece

def solve_puzzle(pieces, frame: FrameModel, config: MatchingConfig) -> PuzzleSolution:
    """
    Solve puzzle using multi-hypothesis beam search.

    Args:
        pieces: List of PuzzlePiece
        frame: FrameModel (128x190mm A5)
        config: MatchingConfig

    Returns:
        PuzzleSolution with poses, matches, confidence, debug
    """
    # TODO: Implement nach Schritten 2-9
    raise NotImplementedError("Solver V2 wird schrittweise implementiert")
```

---

## 7. Test-Prioritäten & Strategie

### 7.1 Early Tests (nach Schritt 3-5)

**Ziel**: Fundament validieren vor Solver-Integration

#### Test 1: Segmentierung (nach Schritt 3)
```python
def test_segmentation():
    # Input: Beispiel-Teil (n=1)
    piece = load_test_piece("corner_piece.pkl")

    # Segmentierung
    segments = segment_piece(piece)

    # Assertions
    assert 4 <= len(segments) <= 12  # target_seg_count_range
    for seg in segments:
        assert seg.length_mm >= 10.0  # min_frame_seg_len_mm
        assert seg.flatness_error >= 0

    # Debug-Visualisierung
    plot_segments(piece, segments)
```

#### Test 2: Frame-Features (nach Schritt 4)
```python
def test_frame_features():
    # Input: Segment bekanntermaßen an TOP
    seg = create_test_segment_top()
    frame = FrameModel()

    # Features
    features = compute_frame_features(seg, "TOP", frame)

    # Assertions (Rohmetriken prüfen)
    assert features.dist_p90_mm < 2.0  # Sollte nah an Rahmenkante sein
    assert features.coverage_in_band > 0.8
    assert abs(features.angle_diff_deg) < 10.0

    # Negative Test (falsche Seite → schlechtere Metriken)
    features_wrong = compute_frame_features(seg, "LEFT", frame)
    # FrameContactFeatures hat kein cost-Feld, prüfe Rohmetriken
    assert features_wrong.dist_p90_mm > features.dist_p90_mm
    assert features_wrong.coverage_in_band < features.coverage_in_band
```

#### Test 3: Overlap (konvex, nach Schritt 7)
```python
def test_overlap_convex():
    # Einfache konvexe Polygone
    poly_a = np.array([[0,0], [10,0], [10,10], [0,10]])
    poly_b = np.array([[5,5], [15,5], [15,15], [5,15]])

    # Overlap-Berechnung
    depth = compute_penetration_depth(poly_a, poly_b)

    # Assertions
    assert 4.5 <= depth <= 5.5  # Theoretisch 5mm Eindringung

    # Kein Overlap
    poly_c = np.array([[20,0], [30,0], [30,10], [20,10]])
    depth_none = compute_penetration_depth(poly_a, poly_c)
    assert depth_none == 0.0
```

---

### 7.2 Integration Tests (nach Schritt 6)

#### Test 4: Beam-Solver ohne Overlap
```python
def test_beam_solver_no_overlap():
    # Input: 4 Teile (2×2), ideale Konturen (kein Overlap möglich)
    pieces = load_test_case("simple_2x2.pkl")
    frame_hyps = generate_frame_hypotheses(...)
    inner_cands = compute_inner_candidates(...)

    # Solver (ohne Overlap-Pruning)
    config = MatchingConfig(overlap_depth_max_mm_prune=999)
    states = run_beam_solver(frame_hyps, inner_cands, frame, config)

    # Assertions
    assert len(states) > 0  # Mindestens eine Lösung
    best = states[0]
    assert len(best.placed) == 4  # Alle Teile platziert
    assert best.cost_total < 20.0  # Plausible Kosten
```

#### Test 5: Beam-Solver mit Overlap-Pruning
```python
def test_beam_solver_with_overlap_pruning():
    # Input: gleiche Teile wie Test 4
    pieces = load_test_case("simple_2x2.pkl")

    # Solver (mit Overlap-Pruning)
    config = MatchingConfig(overlap_depth_max_mm_prune=1.0)
    states = run_beam_solver(...)

    # Assertions
    assert len(states) > 0
    best = states[0]

    # Final Overlap-Check
    depth = compute_state_overlap(best)
    assert depth <= config.overlap_depth_max_mm_prune
```

---

### 7.3 Regression Tests (nach Schritt 10)

#### Test 6: A/B Vergleich
```python
def test_ab_comparison():
    # Gleiche Inputs für alt + neu
    pieces = load_test_case("real_world_6pieces.pkl")

    # Solver V1 (alt)
    solution_v1 = solve_puzzle_v1(pieces)

    # Solver V2 (neu)
    solution_v2 = solve_puzzle(pieces, frame, config)

    # Vergleich
    print(f"V1: conf={solution_v1.confidence:.3f}, overlap_violations={solution_v1.overlap_violations}")
    print(f"V2: conf={solution_v2.confidence:.3f}, overlap_violations={solution_v2.overlap_violations}")

    # Robustere Metrik: Overlap-Verletzungen statt Confidence
    # (Confidence ist abstrakt, Overlap ist messbar)
    assert solution_v2.overlap_violations <= solution_v1.overlap_violations
    # Optional: auch Kosten vergleichen
    # assert solution_v2.total_cost <= solution_v1.total_cost * 1.1
```

---

### 7.4 Test-Daten Management

**Struktur**:
```
tests/
├── test_data/
│   ├── simple_2x2.pkl          # Einfacher Fall (n=4)
│   ├── real_world_6pieces.pkl  # Realistisch (n=6)
│   └── edge_case_n5.pkl        # n=5 (kein Grid)
├── test_segmentation.py
├── test_frame_matching.py
├── test_inner_matching.py
├── test_overlap.py
├── test_beam_solver.py
└── test_integration.py
```

**Policy**:
- Test-Daten versioniert in Git (falls klein) oder via DVC/LFS
- Debug-Exports aus Tests archivieren für spätere Regression

---

## 8. Nächste Schritte nach 00_structure.md

1. **Schritt 1a-c umsetzen** (Config + Models + API-Gerüst)
2. **Schritt 2**: Einheiten/KS-Handling
3. **Schritt 3**: Segmentierung implementieren + **Test 1**
4. **Schritt 4**: Frame-Matching + **Test 2**
5. **Schritt 5**: Inner-Matching
6. **Schritt 6**: Beam-Solver + **Test 4**
7. **Schritt 7**: Overlap + **Test 3, Test 5**
8. **Schritt 8-9**: Confidence/Fallback + Refinement
9. **Schritt 10**: Integration + **Test 6**
10. **Debug-Export**: Nach jedem Schritt testen

---

## 9. Status

**Aktuell**: Struktur finalisiert nach Design-Review
**Nächster Schritt**: Schritt 1a - Config anlegen
