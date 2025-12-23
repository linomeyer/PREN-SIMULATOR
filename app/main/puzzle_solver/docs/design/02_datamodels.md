## Datenmodelle und Schnittstellen

### Ziel
Klare, versionierbare Datenstrukturen zwischen bestehenden Pipeline-Teilen (Extraktion) und neuem Matching/Solver. Fokus:
- eindeutige Einheiten (mm vs Pixel),
- Koordinatensystem-Tagging,
- Debug- und Reproduzierbarkeit.

---

### Eingangs-Schnittstelle (vom bestehenden Simulator)

#### `PuzzlePiece` (Input)
Minimal benötigte Felder für Matching:
- `id: str | int`
- `contour: np.ndarray[(N,2)]`
  - Punkte als (x,y)
  - Einheit: **Pixel oder mm**, muss explizit angegeben/erkennbar sein.
- Optional (nicht zwingend, aber nützlich):
  - `mask: np.ndarray`
  - `bbox: (x, y, w, h)`
  - `center: (cx, cy)`
  - `image: np.ndarray` (RGBA)

**Zusatzanforderung**:
- `contour_metadata` (oder äquivalent):
  - `unit: "px" | "mm"`
  - `coord_system: "P" | "M" | "F" | "A4" | ...`
  - `scale_px_to_mm` falls unit=px und scale bekannt
  - Hinweis: Wenn im Projekt schon implizit mm verwendet werden, trotzdem explizit taggen (Debug!).

---

### Rahmen- und Transformationsmodelle

#### `FrameModel`
- `inner_width_mm: float = 128.0`
- `inner_height_mm: float = 190.0`
- `corner_radius_mm: float | None` (**bewusst offen**)
- `T_MF: Transform2D | None` (**bewusst offen**)
  - Falls None: Ausgabe kann im Rahmen-KS erfolgen; Maschinen-KS Ausgabe erst, wenn gesetzt.

#### `Transform2D`
- Repräsentation:
  - `x_mm: float`, `y_mm: float`, `theta_deg: float`
  - plus optional `mat3: np.ndarray[(3,3)]`
- Methoden:
  - `to_matrix()`, `from_matrix()`
  - `compose(a,b)`, `inverse()`, `apply(points)`

---

### Interne Matching-Modelle

#### `ContourSegment`
- Identität:
  - `piece_id`, `segment_id`
- Geometrie:
  - `points_mm: np.ndarray[(M,2)]` (immer mm)
  - `length_mm: float`
  - `chord: (a_mm, b_mm)` (Endpunkte)
  - `direction_angle_deg: float`
- Form:
  - `flatness_error: float`
  - `profile_1d: np.ndarray[(N,)] | None` (lazy)
- Tags:
  - `is_frame_candidate: bool | float` (optional; als Score besser)

#### `FrameContactFeatures`
(als dict oder dataclass)
- `dist_mean_mm`
- `dist_p90_mm`
- `dist_max_mm`
- `coverage_in_band` (0..1)
- `inlier_ratio` (0..1)
- `angle_diff_deg`
- `flatness_error_mm`
- Optional:
  - `support_points_count`
  - `corner_consistency_cost` (erst später)

#### `FrameHypothesis`
- `piece_id`, `segment_id`, `side: {TOP,BOTTOM,LEFT,RIGHT}`
- `pose_grob_F: Pose2D` (im Rahmen-KS; Definition bewusst offen, aber als Feld vorhanden)
- `features: FrameContactFeatures`
- `cost_frame: float`
- `is_committed: bool` (nur im Solver-State)

#### `InnerMatchCandidate`
- `seg_a_ref: (piece_id, segment_id)`
- `seg_b_ref: (piece_id, segment_id)`
- `cost_inner: float`
- Komponenten (für Debug):
  - `profile_cost`
  - `length_cost`
  - `fit_cost` (ICP optional)
  - `reversal_used: bool`
- Optional:
  - `relative_transform: Pose2D` (A→B oder B→A)

---

### Solver- und Lösungsmodelle

#### `SolverState`
- `poses_F: dict[piece_id -> Pose2D]` (Posen im Rahmen-KS)
- `placed: set[piece_id]`
- `committed_frame_constraints: list[FrameHypothesis]`
- `open_frontier`: repräsentiert offene Matching-Interfaces (Segmente/Constraints)
- `cost_total: float`
- `cost_breakdown: dict[str->float]` (z.B. frame_cost, inner_cost, penalties)
- `debug_trace: list[TraceEvent]`
  - inkl. Pruning-Gründe, Auswahlentscheidungen

#### `PuzzleSolution`
- `poses_F: dict[piece_id -> Pose2D]`
- `poses_M: dict[piece_id -> Pose2D] | None` (wenn `T_MF` gesetzt)
- `matches: list[MatchConstraint]`
- `total_cost: float`
- `confidence: float`
- `debug: DebugBundle`

#### `DebugBundle`
- `config_dump: dict`
- `area_score`
- `frame_hypotheses_by_piece`: Top-N Hypothesen inkl. Features und cost_frame
- `inner_candidates_by_segment`: Top-N Kandidaten inkl. Komponenten
- `solver_summary`: beam stats, expansions, prune counts by reason
- `final_checks`: coverage, open_edges, overlap_depth, etc.
- `artifacts`: optional Pfade zu Visualisierungen/JSON

---

### Schnittstellen / API-Vorschlag

#### Matching-Einstieg
- `solve_puzzle(pieces: list[PuzzlePiece], frame: FrameModel, config: MatchingConfig) -> PuzzleSolution`

#### Teil-Module (für Testbarkeit)
- `segment_piece(piece) -> list[ContourSegment]`
- `compute_frame_hypotheses(segments, frame, config) -> list[FrameHypothesis]`
- `compute_inner_candidates(segments, config) -> dict[seg_ref -> list[InnerMatchCandidate]]`
- `run_solver(frame_hypotheses, inner_candidates, frame, config) -> list[SolverState]`
- `refine_solution(solution, constraints, frame, config) -> PuzzleSolution`
- `export_debug(solution.debug, path)`

---

### Offene/variable Aspekte (bewusst nicht finalisiert, aber als Felder vorhanden)
- Definition von `pose_grob_F` (wie initial aus Segment+Seite geschätzt wird).
- Ausgabe-KS: primär Rahmen-KS; Maschinen-KS erst wenn `T_MF` gesetzt.
- Overlap-Strategie für nicht-konvexe Polygone (Konfigurationsfeld + dokumentierte Wahl).