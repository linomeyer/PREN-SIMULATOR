# Schritt 6: Beam-Solver - Implementierung

**Status**: ✅ Abgeschlossen

**Datum**: 2025-12-26

---

## Was implementiert

### 1. solver/beam_solver/state.py (310 Zeilen)

#### SolverState (Dataclass)
- **Zweck**: Core state representation für Beam Search - partial/complete puzzle solution
- **Felder** (10):
  - `n_pieces`: Total pieces in puzzle
  - `placed_pieces`: Set[PieceID] (platzierte Teile)
  - `unplaced_pieces`: Set[PieceID] (Frontier Teil 1)
  - `poses_F`: Dict[PieceID → Pose2D] (Frame-Koordinaten, mm)
  - `committed_edges`: Set[EdgeID] (geschlossene Kanten)
  - `open_edges`: Set[EdgeID] (Frontier Teil 2, verfügbar für Matching)
  - `cost_total`: float (Gesamtkosten, lower = better)
  - `cost_breakdown`: Dict[str, float] (Kostenaufschlüsselung)
  - `committed_frame_constraints`: Dict[PieceID → FrameHypothesis]
  - `matches`: List (Match-Historie)
- **EdgeID Format**: `tuple[PieceID, int]` = (piece_id, segment_id) - hashable für Sets
- **Koordinatensystem**: Alle Poses in Frame-Koordinaten (F), mm-Einheiten
- **Status**: ✅ Vollständig

#### Methoden

**`__init__(all_piece_ids, **kwargs)`** (Zeile 100-127)
- Erstellt leeren oder vordefinierten State
- Initialisiert alle Felder mit Defaults
- Validiert n_pieces vs placed/unplaced

**`copy() -> SolverState`** (Zeile 128-155)
- Deep copy aller Sets/Dicts (Mutationen isoliert)
- Immutable values (Pose2D, int, float) shared (safe)
- **Test**: S4 (copy mutation isolation)

**`is_complete() -> bool`** (Zeile 157-177)
- **D1 Implementation**: `all_placed AND open_edges==0` (strikt)
- Completion = wasserdicht (alle Kanten matched)
- **Tests**: S8, B3

**`get_frontier() -> tuple[set, set]`** (Zeile 179-196)
- Returns `(unplaced_pieces, open_edges)` - zwei separate Sets
- **Nicht** Union-Type (D2 Frontier-Kollision Fix)
- **Test**: S5

**`validate_invariants()`** (Zeile 198-260)
- Validiert 6 Invarianten:
  - **I1**: placed_pieces ⊆ all_piece_ids
  - **I2**: unplaced_pieces = all - placed
  - **I3**: poses_F.keys() == placed_pieces
  - **I4**: open_edges nur placed pieces
  - **I5**: committed ∩ open = ∅
  - **I6**: cost_total ≥ 0.0
- **Tests**: S6 (valid), S7 (invalid open_edge)

**`seed_with_frame_hypothesis(all_piece_ids, hyp) -> SolverState`** (Zeile 262-310, classmethod)
- Erstellt Seed-State mit platzierten Frame-Hypothese
- Setzt pose = hyp.pose_grob_F
- Committed: (piece_id, segment_id) zu committed_edges
- Cost = hyp.cost_frame
- open_edges leer (Segmente noch unbekannt, wird von expansion.py gefüllt)
- **Test**: S3

---

### 2. solver/beam_solver/expansion.py (532 Zeilen)

#### expand_state (Zeile 34-214)
- **Zweck**: State-Transition - erzeugt Nachfolger-States via Frame/Inner Placement
- **Signatur**: `expand_state(state, pieces, segments, frame_hyps, inner_cands, config, frame) -> list[SolverState]`
- **Algorithmus**:
  1. Get frontier (unplaced_pieces, open_edges)
  2. **Move 1**: Place via Frame (if unplaced_pieces available)
     - Top-N frame hypotheses per piece (branching_cap)
     - Commit hypothesis, open all non-committed segments
  3. **Move 2**: Place via InnerMatch (if open_edges available)
     - Find candidates matching open edges
     - Compute pose via D3 (180° flip + chord-midpoint)
     - Close matched edge, open new edges from piece_b
  4. **Pruning**: _check_valid_state (inside frame, overlap stub, dedup)
  5. Return: Pruned list of successor states
- **Branching**: Top-N frame hyps (D4), all valid inner matches
- **Status**: ✅ Vollständig

#### _compute_pose_from_inner_match (Zeile 217-300)
- **Zweck**: D3 Implementation - Pose-Berechnung für InnerMatch
- **Algorithmus**:
  ```
  theta_B = pose_A.theta_deg + 180.0  # Flip

  # Chord-Midpoint Alignment
  chord_A_F = transform(chord_A_local, pose_A)
  mid_A = (chord_A_F[0] + chord_A_F[1]) / 2

  chord_B_rotated = rotate(chord_B_local, theta_B)
  mid_B_local = (chord_B_rotated[0] + chord_B_rotated[1]) / 2

  pos_B = mid_A - mid_B_local  # Translation
  pose_B = Pose2D(pos_B.x, pos_B.y, theta_B)
  ```
- **Test E5**: (pose_A=(0,0,0), seg_a (0,0)→(20,0)) → pose_B=(20,0,180) exakt
- **Limitations**: reversal_used/sign_flip_used nicht implementiert (V1)
- **Status**: ✅ Funktioniert für einfache Cases

#### _check_valid_state (Zeile 303-343)
- **Zweck**: Pruning-Filter - validiert State-Transition
- **Checks**:
  1. `_check_inside_frame()` → False = prune
  2. `_overlap_stub()` > threshold → prune (Schritt 7: echtes SAT/MTV)
  3. `_check_committed_frame_constraints()` → **DISABLED V1** (Zeile 336-341)
- **Return**: True = valid, False = prune
- **Tests**: E8 (inside), E9 (overlap), E10 (committed disabled)

#### _check_inside_frame (Zeile 347-428)
- **Zweck**: BBox-basierte Frame-Boundarycheck (mit tau_frame_mm Toleranz)
- **Algorithmus**:
  ```
  # Step 1: Compute bbox center (rotation pivot)
  bbox_cx = (x_min + x_max) / 2.0
  bbox_cy = (y_min + y_max) / 2.0

  # Step 2: Translate corners to origin (center)
  corners_centered = corners - bbox_center

  # Step 3: Rotate around origin (= bbox center)
  corners_rotated = R @ corners_centered.T

  # Step 4: Translate back + apply pose
  corners_F = corners_rotated + bbox_center + pose.translation

  # Check inside frame (with tau tolerance)
  inside = (bbox_F.min >= -tau AND bbox_F.max <= frame_bounds + tau)
  ```
- **Hard Prune vs Soft Penalty**:
  - **_check_inside_frame** (Zeile 347-428): **HARD PRUNE** (boolean return, false = reject state)
  - **penalty_missing_frame_contact** (expansion.py:149-153): **SOFT PENALTY** (cost addition)
  - **Unterschied**: Hard prune eliminiert States sofort, Penalty erlaubt exploration mit erhöhten Kosten
  - **Use Case**: Prune wenn komplett außerhalb, Penalty wenn ohne Frame-Hypothese platziert (inner-only)
- **Rotation Fix** (2025-12-26):
  - **Problem**: Rotation um Origin (0,0) → offset für non-centered bboxes
  - **Fix**: Rotation um bbox center (Zeile 388-410)
  - **Impact**: Origin-aligned bboxes (0,0,20,10) jetzt korrekt
- **Tests**: E8 (inside mit tau=2.0 tolerance), E3 (penalty für inner-only piece, inside frame), E2b/E2c (rotation 90°/180°)

#### _overlap_stub (Zeile 422-434)
- **Zweck**: Placeholder für Schritt 7 (SAT/MTV Overlap-Detection)
- **Aktuell**: Returns 0.0 (kein echtes Overlap-Check)
- **Schritt 7**: Ersetzen durch Polygon-Overlap (design/06_collision.md)
- **Test**: E9 (stub behavior)

#### _check_committed_frame_constraints (Zeile 437-486)
- **Zweck**: Prüft ob neues Piece committed frame constraints verletzt
- **Status**: **DISABLED V1** (nicht anwendbar während Expansion)
- **Begründung**: Expansion platziert neue Pieces, re-posed keine committed Pieces → keine Konflikte möglich
- **Code Comment**: Zeile 336-341 in _check_valid_state
- **Test**: E10 (erwartet kein Pruning, function disabled)

#### Cost-Update & Validation

**Cost Integrity**: state.py:261-283 (`validate_cost_consistency`)
```python
def validate_cost_consistency(self):
    expected = sum(self.cost_breakdown.values())
    if abs(self.cost_total - expected) > 1e-6:
        raise ValueError(f"Cost inconsistency: {self.cost_total} != {expected}")
```

**Breakdown Components**:
- `'frame'`: Summe aller Frame-Hypothesen-Kosten (expansion.py:192-194)
- `'inner'`: Summe aller InnerMatchCandidate-Kosten (expansion.py:144-146)
- `'penalty'`: Summe aller Penalties (missing frame contact, expansion.py:149-153)

**Hinweis V1**: 'overlap' Key wird NICHT in cost_breakdown geführt.
- overlap_stub() returns 0.0 (hard prune wenn > threshold, kein cost component)
- Schritt 7: Könnte als Key hinzugefügt werden wenn echtes Overlap-Modul integriert (TBD)

**Update Pattern** (expansion.py:143-153):
```python
# Add cost component
new_state.cost_total += candidate.cost_inner
new_state.cost_breakdown['inner'] = \
    new_state.cost_breakdown.get('inner', 0.0) + candidate.cost_inner

# Penalty if no frame constraint
if piece_b not in new_state.committed_frame_constraints:
    penalty = config.penalty_missing_frame_contact
    new_state.cost_total += penalty
    new_state.cost_breakdown['penalty'] = \
        new_state.cost_breakdown.get('penalty', 0.0) + penalty
```

**Validation Call Sites**:
- state.py:328 (seed_with_frame_hypothesis after cost initialization)
- Implicitly enforced via I6 invariant (cost_total >= 0.0)

**Tests**: S9b (validates cost_total == sum(breakdown), detects inconsistency)

**Status**: ✅ Implemented 2025-12-26

---

#### Frontier-Mode (Sequential Switching)

**Algorithm** (expansion.py:86-169):
```python
n_placed = len(state.placed_pieces)
threshold = 1  # V1: Switch after first piece

# Phase 2: Inner matching (priority, n_placed >= threshold)
if n_placed >= threshold and len(open_edges) > 0 and len(inner_candidates) > 0:
    # Place via InnerMatchCandidate (Move 2)
    ...

# Phase 1: Frame placement (bootstrap OR fallback)
if (n_placed < threshold or len(open_edges) == 0 or len(inner_candidates) == 0) and len(unplaced_pieces) > 0:
    # Place via FrameHypothesis (Move 1)
    ...
```

**Phases**:
- **Phase 1** (n_placed < 1): Frame-only (bootstrap empty state)
- **Phase 2** (n_placed >= 1): Inner-priority (fallback to Frame if no open_edges/candidates)

**Threshold Choice**:
- V1: threshold=1 (pragmatic)
- Prevents empty-state inner matching (no placed pieces → no open_edges)
- Allows inner matching after first piece placed

**Conditions**:
- Sequential (not mutually exclusive): Phase 1 runs wenn Phase 2 conditions NICHT erfüllt
- Fallback robust: Wenn open_edges leer oder keine candidates → Frame placement

**Branching Cap**: D4 - `config.debug_topN_frame_hypotheses_per_piece = 5` (expansion.py:84)

**Tests**: E3 (n_placed=1, inner match), E4 (n_placed=1, no candidates → fallback frame), E5-E6 (inner matching)

**Status**: ✅ Implemented 2025-12-26 (Fixed from earlier mutually exclusive version)

---

#### _deduplicate_states (Zeile 498-541)
- **Zweck**: Entfernt duplicate States (gleiche placed_pieces + poses_F)
- **Key**: `(frozenset(placed_pieces), tuple(sorted((id, pose) for id, pose in poses_F.items())))`
- **Return**: First occurrence kept (niedrigste cost bevorzugt durch pre-sort)
- **Test**: E11

---

### 3. solver/beam_solver/solver.py (168 Zeilen)

#### beam_search (Zeile 28-112)
- **Zweck**: Multi-Hypothesen Beam Search Main Loop
- **Signatur**: `beam_search(pieces, segments, frame_hyps, inner_cands, config, frame) -> list[SolverState]`
- **Algorithmus**:
  ```
  1. Seeding (Hybrid):
     beam = _create_initial_beam(...)
     # Empty state (cost 0.0) + top-1 frame hyp per piece

  2. Expansion Loop:
     all_complete = []
     all_partial = beam.copy()  # D5: Track all for fallback

     while beam AND expansions < max_expansions:
       a. Expand: successors = [expand_state(s) for s in beam]
       b. Extract complete: all_complete += [s for s in succ if s.is_complete()]
       c. Prune incomplete: beam = top-N by cost_total
       d. Track partial: all_partial += beam  # D5
       expansions += 1

  3. Return (D5):
     if all_complete:
       return sorted(all_complete, key=cost)  # Best complete first
     else:
       return sorted(all_partial, key=cost)[:beam_width]  # Best partial (D5)
  ```
- **D1 Check**: `s.is_complete()` (all_placed AND open_edges==0)
- **D5 Fallback**: Return best partial states statt [] (debugfreundlich)
- **Tests**: B1-B9 (seeding, pruning, completion, ranking, collapse, limits)
- **Status**: ✅ Vollständig

#### _create_initial_beam (Zeile 115-168)
- **Zweck**: Hybrid Seeding (D1 Test-Spec §B1)
- **Seeding Strategy**:
  - **Seed A**: Empty state (cost 0.0) → robuste inner-only solutions
  - **Seed B**: Top-1 frame hypothesis per piece → fast convergence
- **Algorithmus**:
  ```
  beam = [SolverState(all_piece_ids)]  # Empty

  for piece_id in all_piece_ids:
    if frame_hyps[piece_id]:
      best_hyp = frame_hyps[piece_id][0]  # Top-1
      seed = SolverState.seed_with_frame_hypothesis(all_piece_ids, best_hyp)
      beam.append(seed)

  return sorted(beam, key=cost)[:beam_width]
  ```
- **Return**: Sorted by cost_total (empty first bei cost=0.0)
- **Test**: B1 (expected: 1 empty + n_pieces seeds, first cost=0.0)
- **Status**: ✅ Vollständig

---

## Design-Entscheidungen

### D1: Completion-Definition
**Problem**: Wann ist Puzzle vollständig?

**Alternativen**:
- Option A: Strikt (all_placed AND open_edges==0)
- Option B: Relaxed (all_placed nur)

**Entscheidung**: **Option A (strikt)**
- `is_complete() = (len(placed_pieces) == n_pieces) AND (len(open_edges) == 0)`
- **Begründung**: "Wasserdicht" completion - alle Kanten matched, keine offenen Edges
- **Implementation**: state.py Zeile 157-177 (is_complete Methode)
- **Tests**: S8 (True/False cases), B3 (completion detection stoppt loop)

---

### D2: open_edges Policy
**Problem**: Welche Segmente werden open_edges?

**Alternativen**:
- Option A: Alle nicht-committed Segmente öffnen
- Option B: Nur "wahrscheinlich matchbare" (heuristic)

**Entscheidung**: **Option A (alle öffnen) + Debug-Kandidaten**
- **Regel**:
  - Frame placement: Alle Segmente außer committed → open_edges
  - Inner match: Matched edge schließen, neue Segmente von piece_b öffnen
- **Begründung**: Robustheit (keine Heuristik-Fehler), Beam-Pruning filtert ineffiziente
- **Implementation**:
  - expansion.py Zeile 136-140 (Frame placement)
  - expansion.py Zeile 196-200 (Inner match)
- **Tests**: S3 (seed öffnet keine edges), E6 (open/close Logik)

---

### D3: Pose-Berechnung InnerMatch
**Problem**: Wie wird pose_B aus InnerMatch berechnet?

**Alternativen**:
- Option A: 180° flip + Chord-Midpoint alignment
- Option B: Feature-based registration (ICP-Style)

**Entscheidung**: **Option A (180° flip)**
- **Algorithmus**:
  ```
  theta_B = pose_A.theta_deg + 180.0  # Gegenteilseite

  # Chord-Midpoint in Frame-Koordinaten
  chord_A_F = transform(chord_A, pose_A)
  mid_A_F = (chord_A_F[0] + chord_A_F[1]) / 2

  # Chord-Midpoint in piece_B local (rotiert)
  chord_B_rotated = rotate(chord_B, theta_B)
  mid_B_local = (chord_B_rotated[0] + chord_B_rotated[1]) / 2

  # Translation: align midpoints
  pos_B = mid_A_F - mid_B_local
  pose_B = Pose2D(pos_B.x, pos_B.y, theta_B)
  ```
- **Begründung**: Einfach, deterministisch, grobe Pose reicht (Refinement Schritt 9)
- **Implementation**: expansion.py Zeile 217-300 (_compute_pose_from_inner_match)
- **Test E5 Spec**: (pose_A=(0,0,0), chord_a (0,0)→(20,0)) → pose_B=(20,0,180) exakt
- **Tests**: E2, E3, E5, E6
- **Limitation**: reversal_used/sign_flip_used nicht implementiert (V1)

---

### D4: Branching Cap
**Problem**: Wie viele Frame-Hypothesen pro Piece in expand_state?

**Alternativen**:
- Option A: Alle (exponentielles Wachstum)
- Option B: Reuse debug_topN_frame_hypotheses_per_piece
- Option C: Neuer Parameter branching_cap

**Entscheidung**: **Option B (Reuse debug_topN)**
- **Parameter**: `config.debug_topN_frame_hypotheses_per_piece = 5` (default)
- **Code**: `frame_hyps[:branching_cap]` (expansion.py Zeile 84-85)
- **Begründung**:
  - Verhindert exponentielles Wachstum (Beam-Collapse)
  - Dual-Purpose Parameter (Debug + Solver)
  - Tuning-freundlich (1 Parameter)
- **Implementation**: expansion.py Zeile 84-85
- **Tests**: E1 (frame placement), E12 (branching limit)

---

### D5: NO_SOLUTION Verhalten
**Problem**: Was returnen wenn keine complete solution?

**Alternativen**:
- Option A: Leere Liste []
- Option B: Best partial state (debugfreundlich)

**Entscheidung**: **Option B (best partial)**
- **Regel**: Return `sorted(all_partial_states, key=cost)[:beam_width]`
- **Cases**:
  - Beam collapse (alle States gepruned)
  - max_expansions Limit erreicht
- **Return**: Liste mit mind. 1 SolverState (`is_complete() == False`)
- **Begründung**:
  - Debug-Analyse möglich (welches Teil fehlt?)
  - Partial solution besser als nichts
  - Visualisierung zeigt Fortschritt
- **Implementation**: solver.py Zeile 106-112 (return logic)
- **Tests**: B5 (beam collapse), B6 (max_expansions)

---

## Abweichungen vom Design

### 1. Completion-Definition (Strikt vs Design-Variante)

**Original Design**: design/05_solver.md erwähnt completion ohne explizite open_edges-Bedingung
**Implementation**: D1 strikt - `is_complete() = all_placed AND open_edges==0`

**Begründung**:
- "Wasserdicht" completion (alle Kanten matched)
- Verhindert false positives (alle platziert aber Lücken vorhanden)
- Konsistent mit Frontier-Semantik (open_edges = verfügbare Matches)

**Code**: state.py:176-177
```python
return (len(self.placed_pieces) == self.n_pieces and
        len(self.open_edges) == 0)
```

**Tests**: S8 (True/False cases), B3 (completion detection)

**Status**: ✅ Finalized (alle Tests passing)

---

### 3. Committed Frame Conflict Check (Disabled)
- **Original Design**: expansion.py sollte prüfen ob neue Pieces committed frame constraints verletzen
- **Implementation**: Function existiert (_check_committed_frame_constraints, Zeile 446-495) aber disabled in _check_valid_state (Zeile 336-342)
- **Begründung**: Nicht anwendbar während Expansion
  - Expansion platziert neue Pieces, re-posed keine committed Pieces
  - Konflikte können nur entstehen wenn committed Pieces neu platziert werden → Move 3 (hypothesis switch, V2)
  - Aktuell: Constraint-Konsistenz durch Konstruktion garantiert
- **Code Comment**: "V1: Disabled (not applicable during expansion)"
- **Test E10**: Erwartet kein Pruning, function disabled
- **TODO**: Schritt 9 (Refinement) oder V2 Move 3 könnte das reaktivieren

---

### 4. reversal_used / sign_flip_used (NotImplementedError)
- **Design**: InnerMatchCandidate hat Felder `reversal_used`, `sign_flip_used` (aus Schritt 5)
- **Implementation**: D3 Pose-Berechnung implementiert nur einfache Cases
  - Keine bedingte Logik für reversal/sign_flip
  - 180° flip immer angewendet (funktioniert für "normal" orientation)
- **Status**: Test-Spec §6.1 dokumentiert: "Noch nicht implementiert"
- **Impact**:
  - Einfache Puzzles funktionieren (32/32 tests)
  - Edge cases (gespiegelte Profile) könnten fehlschlagen
- **TODO**: Erweiterte Pose-Berechnung in V2
  - Conditional logic: if reversal_used → theta += extra_flip
  - Dokumentation in design/05_solver.md §Move 2 erweitern

---

### Ansonsten
✅ 100% gemäß design/05_solver.md und implementation/00_structure.md §3

---

## Offene Punkte für Schritt 7+

### 1. overlap_stub() durch SAT/MTV ersetzen
- **Aktuell**: _overlap_stub() returns 0.0 (kein echtes Overlap-Check)
- **Schritt 7**: Implementiere design/06_collision.md
  - SAT (Separating Axis Theorem) für konvexe Polygone
  - MTV (Minimum Translation Vector) für overlap_depth
  - Nonconvex-Strategie wählen (Triangulation / Convex Hull / Minkowski)
- **Impact**:
  - Pruning wird genauer (false positives eliminiert)
  - Weniger invalide Hypothesen in Beam
  - Overlap-basierte Cost-Komponente aktiviert
- **Integration**: _overlap_stub() Signatur bleibt (transparent für expansion.py)

---

### 2. Pose-Refinement Integration
- **Aktuell**: Grobe Poses via Frame/Inner (D3 Algorithmus)
- **Schritt 9**: Implementiere design/07_refinement.md
  - Optimierer wählen (Nelder-Mead / Gauss-Newton / Zweistufig)
  - Cost-Funktion: J = J_frame + J_inner + J_overlap
  - Final Check: penetration_depth ≤ 0.1mm
- **Impact**:
  - BBox-Offset korrigiert (~1-2mm → <0.1mm)
  - Overlap → 0 (praktisch keine Penetration)
  - Kosten optimiert (bessere Frame/Inner Alignment)
- **Integration**: Refinement als Post-Processing nach beam_search()

---

### 3. Confidence + Fallback (Schritt 8)
- **Aktuell**: Keine Confidence-Berechnung
- **Schritt 8**: Implementiere design/08_fallback.md
  - Confidence: `conf = exp(-k_conf * cost_total)`
  - Fallback-Trigger: `conf < 0.5`
  - Many-to-one Matching (chain_len=2)
- **Impact**:
  - Robustheit bei Segmentierungs-Splits
  - Weniger NO_SOLUTION Cases (D5)
- **Integration**: Fallback-Layer vor/nach beam_search()

---

## Validierung

### Test Coverage
```bash
pytest tests/test_beam_solver.py -v
# 33/33 passing (0.23s)
```

**Test-Gruppen**:
- **S1-S9b**: SolverState (10 tests)
  - Invarianten, Copy, Frontier, Completion
  - **S9b**: Cost consistency validation (NEW - validates cost_total == sum(cost_breakdown))
- **E1-E14**: expand_state (14 tests)
  - Frame/Inner Placement, Pruning, Edge cases
  - **E2b, E2c**: Rotation correctness (90°, 180° with origin-aligned bboxes - NEW)
- **B1-B9**: beam_search (9 tests)
  - Seeding, Loop, Completion, Ranking, Fallback

**Kritische Invarianten** (alle validiert):
- ✅ EdgeID = tuple[PieceID, int] (keine Hash-Kollisionen)
- ✅ Frontier = zwei separate Sets (unplaced_pieces, open_edges)
- ✅ SolverState Single Source of Truth (nur in state.py)
- ✅ D1-D5 Design Decisions implementiert
- ✅ Koordinaten: Frame (F), mm-Einheiten
- ✅ I1-I6 State Invariants (validated via validate_invariants())

**Numerische Erwartungen** (exakt):
- Test E5: pose_B = Pose2D(20, 0, 180) (exakt)
- Test B1: first state cost = 0.0 (empty state)
- Test E3: penalty = 10.0 (config.penalty_missing_frame_contact)
- Test B5/B6: len(results) >= 1 (D5 best partial)

---

### Config-Parameter (verwendet)
Alle aus `MatchingConfig` (Schritt 1):
- `beam_width: int = 3` (MAIN parameter, B2)
- `max_expansions: int = 20` (B6 termination)
- `overlap_depth_max_mm_prune: float = 1.0` (E9 threshold)
- `penalty_missing_frame_contact: float = 10.0` (E3/E4/B7)
- `tau_frame_mm: float = 2.0` (E8 boundary tolerance)
- `debug_topN_frame_hypotheses_per_piece: int = 5` (D4 branching cap)

---

### Abhängigkeiten (erfüllt)
- ✅ ContourSegment (Schritt 3: Segmentierung)
- ✅ FrameHypothesis (Schritt 4: Frame-Matching)
- ✅ InnerMatchCandidate (Schritt 5: Inner-Matching)
- ✅ MatchingConfig (Schritt 1: Fundament)
- ✅ Pose2D, Transform2D (Schritt 2: Koordinaten)

---

### Design-Docs Cross-Reference
- ✅ `design/05_solver.md`: Solver-Algorithmus (Moves, Pruning, Termination)
- ✅ `design/09_edgecases.md`: Edge Cases (Beam collapse, max_expansions)
- ✅ `implementation/00_structure.md` §3: Schritt 6 Struktur
- ✅ `implementation/06_beam_solver_test_spec.md` §3-6: Test-Spec + Design Decisions

---

## Statistik

| Modul | Zeilen | Klassen | Funktionen | Tests |
|-------|--------|---------|------------|-------|
| state.py | 335 | 1 | 7 methods (inkl. validate_cost_consistency) | 10 (S1-S9b) |
| expansion.py | 541 | 0 | 1 main + 5 helpers | 14 (E1-E14) |
| solver.py | 168 | 0 | 2 (main + helper) | 9 (B1-B9) |
| __init__.py | 22 | 0 | 0 (exports) | - |
| **Gesamt** | **1066** | **1** | **15** | **33/33 ✅** |

**Implementierungs-Details**:
- 10 Fields in SolverState (placed, unplaced, poses, edges, cost, ...)
- 6 Invarianten (I1-I6, validated)
- 7 Methoden (inkl. validate_cost_consistency für cost integrity)
- 2 Placement-Moves (Frame, Inner with sequential switching)
- 3 Pruning-Checks (inside frame with rotation fix, overlap stub, dedup)
- 5 Design Decisions (D1-D5, finalized)

**Test-Runtime**: 0.23s (alle 33 tests)

---

## Nächste Schritte

### Schritt 7: Overlap-Modul (SAT/MTV)
**Ziel**: Overlap sauber messen + prunen

**Module**:
- `solver/overlap/polygon_overlap.py`
- SAT für konvexe Polygone
- Nonconvex-Strategie (Triangulation / Convex Hull / Minkowski)

**Integration**:
- Ersetze _overlap_stub() in expansion.py
- Pruning-Schwellen: prune 1.0mm, final 0.1mm (aus MatchingConfig)

**Geschätzt**: ~2-3h (komplex, Polygon-Operationen)

**Basis**: design/06_collision.md

---

### Schritt 8: Confidence + Fallback
**Ziel**: Robustheit bei Segmentierungs-Splits

**Algorithmus**:
- confidence = exp(-k_conf * cost_total)
- Fallback-Trigger: conf < 0.5
- Many-to-one Matching (chain_len=2)

**Basis**: design/08_fallback.md

---

### Schritt 9: Pose-Refinement
**Ziel**: Overlap→0, Sub-mm Genauigkeit

**Optimierer**: Nelder-Mead / GN / Zweistufig
**Cost**: J = J_frame + J_inner + J_overlap

**Basis**: design/07_refinement.md

---

## Status

**Schritt 6**: ✅ Abgeschlossen (2025-12-26)

**Freigabe**: Bereit für Schritt 7 (Overlap-Modul)

**Review**: Design-LLM validation ausstehend
