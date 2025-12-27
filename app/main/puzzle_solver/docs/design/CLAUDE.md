# Solver V2 Implementation Orchestrator

**Zweck**: Workflow-Koordination für schrittweise Implementierung des neuen Solvers gemäss Design.

---

## Design-Grundlage

Alle Implementierungsschritte folgen strikt den Design-Dokumenten:

- `00_overview.md` - Gesamtkonzept, Anforderungen, Rahmenbedingungen
- `01_coordinates.md` - Koordinatensysteme, Transformationen
- `02_datamodels.md` - Datenstrukturen, Schnittstellen
- `03_matching.md` - Config, Algorithmus (8 Phasen)
- `04_scoring.md` - Metriken, Costs, Debug-Ausgaben
- `05_solver.md` - Beam-Solver, States, Expansion
- `06_collision.md` - Overlap (SAT/MTV)
- `07_refinement.md` - Pose-Optimierung
- `08_fallback.md` - Many-to-one Matching
- `09_edgecases.md` - Edge Cases, Failure Modes

---

## Implementierungsstrategie

### Prinzipien
- **Bottom-Up**: Fundament zuerst, dann darauf aufbauen
- **Testbar**: Nach jedem Schritt lauffähiger Zwischenstand
- **Messbar**: Debug-Daten nach jedem Schritt verfügbar
- **Konfigurierbar**: Alle Parameter zentral in Config

### Reihenfolge (10 Schritte)

#### Schritt 1: Fundament (Config + Models)
**Ziel**: Datenstrukturen und Konfiguration
- `config.py`: MatchingConfig, FrameModel, Transform2D
- `models.py`: ContourSegment, FrameHypothesis, InnerMatchCandidate, SolverState, PuzzleSolution, DebugBundle
- **Basis**: 01_coordinates.md, 02_datamodels.md
- **Output**: API-Gerüst vorhanden, Serialisierung funktioniert

#### Schritt 2: Einheiten & KS
**Ziel**: mm-basiertes Arbeiten, KS-Tagging
- Konverter Pixel→mm
- KS-Tagging in Debug
- T_MF als bewusst offener Parameter
- **Basis**: 01_coordinates.md
- **Output**: Reproduzierbare Koordinatenflüsse

#### Schritt 3: Segmentierung + Flatness V1
**Ziel**: Stabile Kontursegmente
- Split an Krümmungsmaxima
- Merge bis Mindestlänge
- Flatness RMS (Punkt-zu-Sehne)
- **Basis**: 03_matching.md Phase 1
- **Output**: ContourSegment mit stabilen IDs

#### Schritt 4: Frame-Matching
**Ziel**: Frame-first Hypothesen
- Features: dist_mean/p90/max, coverage, inlier, angle_diff, flatness
- Cost-Mapping + Aggregation
- Top-N Hypothesen pro Teil
- **Basis**: 03_matching.md Phase 2, 04_scoring.md
- **Output**: FrameHypothesis + Debug-Metriken

#### Schritt 5: Inner-Matching (1D-Profil)
**Ziel**: Innenmatch-Kandidaten
- 1D-Profil (Resampling N=128)
- NCC forward/reversed
- Top-k pro Segment
- **Basis**: 03_matching.md Phase 3, 04_scoring.md
- **Output**: InnerMatchCandidate + Debug

#### Schritt 6: Beam-Solver V1
**Ziel**: Erste vollständige Lösungen
- Seeding (Frame + leer)
- Expansion (Place via Frame/Inner)
- Soft→Hard Commit
- Pruning (outside, overlap stub, conflicts)
- **Basis**: 05_solver.md
- **Output**: Lösungen (grob), Solver-Trace

#### Schritt 7: Overlap-Modul
**Ziel**: Overlap sauber messen + prunen
- SAT/MTV für konvexe Polygone
- Nonkonvex-Strategie (Option A/B/C wählen)
- Startwerte: prune 1.0mm, final 0.1mm
- **Basis**: 06_collision.md
- **Output**: Overlap beschneidet falsche Hypothesen

#### Schritt 8: Confidence + Fallback
**Ziel**: Robustheit bei Segmentierungs-Splits
- confidence = exp(-k_conf * cost)
- Fallback-Trigger: conf < 0.5
- Many-to-one (chain_len=2)
- **Basis**: 08_fallback.md
- **Output**: Robustheit steigt

#### Schritt 9: Pose-Refinement
**Ziel**: Overlap→0, Kosten optimieren
- Optimierer wählen (Nelder-Mead / GN / Zweistufig)
- J = J_frame + J_inner + J_overlap
- Final Check: penetration_depth ≤ 0.1mm
- **Basis**: 07_refinement.md
- **Output**: "Praktisch kein Overlap"

#### Schritt 10: Integration + Tests
**Ziel**: Stabiler Entwicklungsprozess
- Integration in Simulator-Visualisierung
- A/B Vergleich alt/neu
- Regression-Suite
- **Basis**: 00_overview.md
- **Output**: Reproduzierbare Verbesserungen

---

## Bewusst offene Parameter

Folgende Parameter sind **nicht final festgelegt**, müssen aber als Felder existieren:

- `T_MF` (Rahmen→Maschine Transform)
- `corner_radius_mm`
- `profile_smoothing_window`
- `k_conf` (Confidence-Mapping)
- `polygon_nonconvex_strategy`
- `pose_grob_F` (Definition der Initialpose aus FrameHypothesis)
- `overlap_depth_max_mm_*` (Startwerte, Tuning nötig)

**Policy**: Alle als Config-Felder führen, Startwerte/Platzhalter setzen, im Debug ausgeben.

---

## Aktuelle Aufgabe

Siehe `implementation/00_structure.md` für detaillierte Struktur und Abhängigkeiten.

**Status**: ✅ Struktur finalisiert, freigegeben nach Design-Review

**Review-Iterationen**:

**Iteration 1** (Struktur-Fixes):
1. Config-Felder ergänzt: frame_weights, inner_weights, penalty_missing_frame_contact, many_to_one_max_chain_len, nonconvex_aggregation
2. SolverState Single Source of Truth: NUR in beam_solver/state.py
3. pose_grob_F API festgelegt: estimate_pose_grob_F(segment, side, frame) -> (Pose2D, float)
4. Frontier/Completion formalisiert: open_frontier = set[PieceID] ∪ set[EdgeID], Completion = all placed AND all edges matched
5. Nonconvex-Risiko dokumentiert: Triangulation Default, Performance O(n²×t_a×t_b), Aggregation als Tuning-Parameter
6. Test-Prioritäten hinzugefügt: Early Tests (Segmentierung, Frame-Features, Overlap), Integration Tests (Beam-Solver), Regression (A/B)

**Iteration 2** (Blocker-Fixes + Dokumentation):
1. Frontier-Typen Kollision behoben: Zwei separate Sets (unplaced_pieces, open_edges) statt Union
2. EdgeID Hash→Tuple: EdgeID = tuple[PieceID, int] (keine Hash-Kollisionen)
3. Test 2 fix: Rohmetriken prüfen (FrameContactFeatures hat kein cost-Feld)
4. uncertainty_mm dokumentiert: Reserved for future adaptive pruning (TODO: Overlap-Schwellen, Beam-Gewichtung)
5. Aggregation-Kommentar: "max" empfohlen für Pruning, "mean/p90" nur für Diagnose
6. A/B Test Assertion robuster: overlap_violations statt confidence

**Nächster Schritt**: ~~Schritt 1-5~~ ✅ → **Schritt 6 - Beam-Solver V1** 🔄

---

## Implementierungs-Fortschritt

### ✅ Schritt 1: Fundament (Config + Models) - ABGESCHLOSSEN

**Datum**: 2025-12-23

**Fixes nach Review**: FrameHypothesis.uncertainty_mm nachgetragen

**Statistik**: 584 Zeilen, 10 Klassen, 61 Felder

**Tests**: Models structure validation

**Dokumentation**: `implementation/01_fundament_impl.md`

---

### ✅ Schritt 2: Einheiten & Koordinatensysteme - ABGESCHLOSSEN

**Datum**: 2025-12-23

**Fixes nach Review**:
- PuzzlePiece px+mm dual fields (extraction→solver workflow)
- Guards: validate_pieces_format(require_mm_fields), segment_piece(contour_mm None-check)

**Statistik**: +224 Zeilen, 10 Funktionen (Transform2D + Conversion)

**Tests**: 11/11 (test_step2.py)

**Dokumentation**: `implementation/02_einheiten_impl.md`

---

### ✅ Schritt 3: Segmentierung + Flatness - ABGESCHLOSSEN

**Datum**: 2025-12-24

**Implementation**: solver/segmentation/contour_segmenter.py (~473 Zeilen)

**Fixes nach Review**: Keine (Design-Review akzeptiert ohne Änderungen)

**Tests**: 12/12 (test_segmentation.py)

**Dokumentation**: `implementation/03_segmentierung_impl.md`

**Design-Entscheidungen**: Curvature-based splitting, arclength coverage, wraparound merge

---

### ✅ Schritt 4: Frame-Matching - ABGESCHLOSSEN

**Datum**: 2025-12-25

**Implementation**: solver/frame_matching/ (~457 Zeilen: features.py, hypotheses.py)

**Fixes nach Review**: Doku-Klarstellungen (CCW rotation, TopN dual-purpose, reserved params)

**Tests**: 11/11 (test_frame_matching.py)

**Dokumentation**: `implementation/04_frame_matching_impl.md`

**Design-Entscheidungen**:
- Arclength-basierte coverage (robuster vs point-count)
- Projection-based pose (Option A, schneller)
- Policy-Switch (coverage vs inlier vs balanced)
- Theta-Modes (zero/side_aligned/segment_aligned)

---

### ✅ Schritt 5: Inner-Matching - ABGESCHLOSSEN

**Datum**: 2025-12-25/26

**Implementation**: solver/inner_matching/ (~470 Zeilen: profile.py, candidates.py)

**Fixes nach Review**:
- Weight normalization policy (sum=1.0 assumption + runtime warning)
- Debug fields: ncc_best, best_variant (trace/analysis)
- Unused V1 params marked: profile_smoothing_window, frame_likelihood_threshold

**Tests**: 16/16 (test_inner_matching.py)

**Dokumentation**: `implementation/05_inner_matching_impl.md`

**Design-Entscheidungen**:
- Sign-flip detection (4 NCC variants: fwd/fwd_flip/rev/rev_flip)
- Cost clamping [0,1] (consistent aggregation)
- Profile: Signed distance (orientation-aware)

---

### ✅ Schritt 6: Beam-Solver - ABGESCHLOSSEN

**Datum**: 2025-12-26

**Implementation**: beam_solver/state.py, expansion.py, solver.py (~911 Zeilen)

**Tests**: 32/32 (test_beam_solver.py)

**Statistik**:
- state.py: 311 Zeilen, 9 tests (S1-S9)
- expansion.py: 532 Zeilen, 14 tests (E1-E14 inkl. rotation validation)
- solver.py: 168 Zeilen, 9 tests (B1-B9)

**Design Decisions Finalisiert**:
- D1: Completion = all_placed AND open_edges==0 (strikt)
- D2: open_edges = alle non-committed Segmente + Debug-Kandidaten
- D3: InnerMatch Pose = 180deg flip + chord-midpoint
- D4: Branching cap = debug_topN_frame_hypotheses_per_piece (N=5)
- D5: NO_SOLUTION = return best partial state

**Known Limitations**:
- reversal/sign_flip: NotImplementedError (V1)
- Committed conflict check: Disabled (V1)

**Dokumentation**:
- docs/implementation/06_beam_solver_test_spec.md (finalisiert)

---

### 🔄 Schritt 7: Overlap-Modul (SAT/MTV) - NÄCHSTER

---

## Lessons Learned (Schritte 1-5)

### Workflow-Erfolge ✅

1. **Test-Spec vor Code**: Design-LLM erstellt Test-Spezifikation → verhindert Test-Fitting
   - Claude Code implementiert gegen fixe Spec
   - Design-LLM validiert Tests NACH Implementation

2. **3-Phasen Validation**:
   - Phase 1: Test-Spec gegen Design-Docs
   - Phase 2: Implementation gegen Test-Spec
   - Phase 3: Code-Review gegen Design-Docs

3. **Schrittweise Reviews**: Jeder Schritt einzeln validiert vor nächstem
   - Frühe Bug-Erkennung (FIX 1/2 in Schritt 5 vor Schritt 6)
   - Keine akkumulierten Tech-Debts

4. **Code-Doku Synchronisation**: Nach jedem Fix beide prüfen
   - Implementation → Doku update → Validation

5. **Design-First Approach**: Alle Algorithmen aus design/*.md Docs
   - Keine "guess & check" Implementation
   - Klare Referenz für Reviews

### Häufige Probleme ⚠️

1. **Test-Fitting**: Claude Code passt Tests an Code statt umgekehrt
   - Symptom: Tests passen immer, aber Design verletzt
   - Lösung: Design-LLM validiert Test-Quality nach Implementation

2. **Fehlende Config-Parameter**: In späteren Schritten nachgetragen
   - Beispiel: FrameHypothesis.uncertainty_mm fehlte (Schritt 1 → nachgetragen in Schritt 4)
   - Lösung: ALLE Parameter in Schritt 1, auch wenn "reserved for later"

3. **Optional-Felder ohne Guards**: None-Zugriff Risiko
   - Beispiel: PuzzlePiece.contour_mm=None → segment_piece crash
   - Lösung: Guards am Moduleingang (validate_pieces_format, segment_piece)

4. **Doku-Code Drift**: Formeln/Algorithmen unterschiedlich
   - Beispiel: NCC formula (1-NCC vs 1-|NCC|) in Schritt 5
   - Lösung: Nach Code-Änderung IMMER Doku aktualisieren + cross-validate

5. **Unnamed Assumptions**: "Obvious" defaults nicht dokumentiert
   - Beispiel: inner_weights sum=1.0 (nicht explizit bis Review)
   - Lösung: ASSUMPTION-Tags in Docstrings + runtime validation

### Best Practices für Schritt 6+

**Config-Vollständigkeit**:
- ✅ ALLE Parameter vorhanden? (beam_width, max_expansions, penalties)
- ✅ Reserved params als "UNUSED V1" markiert?
- ✅ Default values plausibel dokumentiert?

**Guards & Validation**:
- ✅ SolverState invariants? (placed_pieces ⊆ all pieces, etc.)
- ✅ None-checks am Moduleingang?
- ✅ validate_*() functions für komplexe inputs?

**Test-Quality**:
- ✅ Konkrete numerische Erwartungen? (nicht nur "should work")
- ✅ Edge cases aus design/09_edgecases.md abgedeckt?
- ✅ Test-Spec BEVOR Implementation?

**Design-Alignment**:
- ✅ Code folgt design/*.md Algorithmen exakt?
- ✅ Doku update nach jeder Code-Änderung?
- ✅ Cross-validation (Code ↔ Doku ↔ Tests)?

---

## Nächste Schritte: Schritt 6 Implementation Plan

### Phase 1: Test-Spezifikation (Design-LLM)

**Module 1: beam_solver/state.py** (SolverState)
- Test: State initialization (empty, seeded)
- Test: Frontier computation (unplaced_pieces, open_edges)
- Test: is_complete() logic (all placed AND all edges matched)
- Test: State invariants (placed ⊆ all, no conflicts)

**Module 2: beam_solver/expansion.py** (expand_state)
- Test: Place piece via frame hypothesis
- Test: Place piece via inner match (2-piece connection)
- Test: Edge opening/closing after placement
- Test: Invalid expansions filtered (overlap stub, outside frame)

**Module 3: beam_solver/solver.py** (beam_search)
- Test: Seeding (frame hypotheses + empty state)
- Test: Beam expansion loop (breadth-first)
- Test: Pruning (beam_width limit, overlap stub)
- Test: Solution extraction (complete states ranked)

### Phase 2: Implementation (Claude Code)

**Reihenfolge**: state.py → expansion.py → solver.py

**Nach jedem Modul**:
1. Run tests
2. Design-LLM: Test-Quality validation
3. Fix issues before next module

### Phase 3: Integration & Validation

**Design-LLM Review**:
- Implementation gegen design/05_solver.md
- Edge cases aus design/09_edgecases.md
- Config-Vollständigkeit
- Guards & None-checks

**Kritische Checks**:
- ✅ SolverState NUR in state.py (Single Source of Truth)
- ✅ EdgeID = tuple[PieceID, int] (no hash collision)
- ✅ Frontier: separate Sets (not Union)
- ✅ Overlap: Stub returns 0.0 (implementation Schritt 7)
- ✅ No hardcoded magic numbers (all in MatchingConfig)

### Abhängigkeiten

**Benötigt aus Schritt 1-5**:
- ✅ ContourSegment (Schritt 3)
- ✅ FrameHypothesis (Schritt 4)
- ✅ InnerMatchCandidate (Schritt 5)
- ✅ MatchingConfig.beam_width, max_expansions (Schritt 1)

**Liefert für Schritt 7+**:
- SolverState (beam search states)
- expand_state() (state transitions)
- beam_search() (main solver loop)

---

## Workflow-Regeln

1. **Keine Implementierung ohne Design-Referenz**
2. **Nach jedem Schritt: Debug-Export testen**
3. **Startwerte explizit als "Annahme" markieren**
4. **Offene Parameter als TODO in Code + Config dokumentieren**
5. **Keine Hardcoding-Magie, alles konfigurierbar**
