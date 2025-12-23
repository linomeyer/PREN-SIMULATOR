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

**Nächster Schritt**: Schritt 1 - Fundament (Config + Models)

---

## Workflow-Regeln

1. **Keine Implementierung ohne Design-Referenz**
2. **Nach jedem Schritt: Debug-Export testen**
3. **Startwerte explizit als "Annahme" markieren**
4. **Offene Parameter als TODO in Code + Config dokumentieren**
5. **Keine Hardcoding-Magie, alles konfigurierbar**
