# Test-Spec: Edge Cases & Failure Modes (Schritt 9)

> Quelle: `docs/design/09_edgecases.md` (ca. Zeilen 8–237)
> System-Kontext: `docs/design/00_structure.md`
> Format-Referenz: `docs/test_spec/08_fallback_test_spec.md`

## 1. Übersicht

**Scope:** Tests fuer definierte Edge Cases (A1–A6) sowie Failure Modes (F1–F5 + INVALID_INPUT) inkl. Debug-Pflichtdaten und kontrollierter Status-Codes.  
**Ziel:** Kein Crash; deterministische, diagnostizierbare Ergebnisse (auch bei Failure).

**Status-Enum (muss existieren):** `OK`, `OK_WITH_FALLBACK`, `LOW_CONFIDENCE_SOLUTION`, `NO_SOLUTION`, `REFINEMENT_FAILED`, `INVALID_INPUT`

**Test-Kategorien:**
- Geometrie/Segmentation (Konturform, Nicht-Konvexitaet, kleine Kanten)
- Input-Validierung (n, fehlende Konturen, Einheiten/KS)
- Matching-Robustheit (Rauschen, Ausreisser, Symmetrie)
- Solver/Resource (Beam leer, max_expansions, Pruning zu strikt)
- Refinement/Overlap-Numerik (final overlap threshold, Flapping)
- Debug/Observability (Pflichtfelder, Completeness)

**Abhaengige Module (aus Struktur):**
- `segmentation/contour_segmenter.py` (A1, A2, A4)
- `frame_matching/features.py`, `frame_matching/hypotheses.py` (A1, A5, F5)
- `inner_matching/profile.py`, `inner_matching/candidates.py` (A2, A4, F1)
- `beam_solver/solver.py` + `beam_solver/state.py` (A5, A6, F1, F5)
- `collision/overlap.py` (A3, F4)
- `fallback/many_to_one.py` (A4, F2)
- `refinement/pose_refiner.py` (F3, F5)
- `debug/export.py` + `models.DebugBundle` (alle Failures)

**Total Test-Count (Schaetzung):**
- Edge Cases: 15
- Failure Modes: 14
- Debug Info: 6
- Integration: 6  
**Total:** 41 Tests

**Globale Toleranzen (Default):**
- `TOL_MM = 1e-3`
- `TOL_DEG = 1e-2`
- `TOL_COST = 1e-6`
- `TOL_CONF = 1e-3`

---

## 1.1 Extrahierte Requirements (aus 09_edgecases.md)

### MUST (kritisch)
1. **Kein Crash/keine ungefangenen Exceptions**; Failure ist ein kontrollierter Zustand mit Status-Code. (Ziel, ca. Zeilen 1–6)  
2. **Bei jedem non-OK Status**: Debug-Minimum muss vorhanden sein (Config-Dump, n, area_score, Top-Hypothesen/Kandidaten, prune stats + last best state; bei refinement Iter-Info). (D, ca. Zeilen 129–140)  
3. **A1**: Fehlende/zu kleine Rahmenkontakte duerfen keinen harten Abbruch ausloesen; `penalty_missing_frame_contact` greift, Platzierung via Innenmatches muss moeglich bleiben. (A1, ca. Zeilen 10–18)  
4. **F2**: Bei `confidence < fallback_conf_threshold` muss Many-to-one-Fallback neu laufen; wenn weiterhin niedrig: `LOW_CONFIDENCE_SOLUTION` (mit Loesung). (F2, ca. Zeilen 85–92)  
5. **F3**: Wenn Refinement scheitert: Status `REFINEMENT_FAILED`, Output ist beste pre-refinement Loesung + Diagnose. (F3, ca. Zeilen 93–100)

### SHOULD (wichtig)
1. **A5**: Symmetrien behandeln via Top-K und Tie-break Reihenfolge (cost_total, penetration, frame coverage, area_score). (A5, ca. Zeilen 47–58)  
2. **F5**: Inside/Outside Checks: Solver toleranter (`tau_frame_mm`), Refinement strenger. (F5, ca. Zeilen 109–118)  
3. **F4**: Near-zero/Flapping faelle muessen geloggt werden (Achsen-Overlaps, Toleranznutzung). (F4, ca. Zeilen 101–108)

### MAY (optional)
1. **F1**: Auto-Variantenlauf (Konfig vorbereitet), aber nicht Bestandteil V1. (F1, ca. Zeilen 71–84)  
2. **A2**: Optional zusaetzliche Statistik der Punktabstaende als Debug. (A2, ca. Zeilen 20–29)  
3. **A4**: Design-Aenderung: Many-to-one im Normalmodus zulassen (falls Fallback zu oft). (A4, ca. Zeilen 38–46)

---

## 2. Edge Cases Tests (15 Tests)

### Test EC-01: A1 Kleine Rahmenberuehrung wird nicht hart geblockt
**Kategorie:** Geometrie / Constraint (soft)  
**Setup:**
- `min_frame_seg_len_mm = 10.0`
- Ein Piece mit einem echten Aussenkanten-Segment `length_mm = 9.0` (knapp darunter), das geometrisch an einer Rahmenkante liegt.
- Inner-Matching bietet mindestens 1 gueltige Platzierung zu bereits platziertem Nachbarn.
**Expected:**
- Kein `INVALID_INPUT`, kein Crash.
- Final `status in {OK, OK_WITH_FALLBACK, LOW_CONFIDENCE_SOLUTION}` (je nach Gesamtqualitaet), aber **nicht** `NO_SOLUTION` allein wegen fehlender Frame-Hypothese.
- `solution.cost_breakdown["penalty_missing_frame_contact"] == penalty_missing_frame_contact ± TOL_COST`.
**Rationale:** A1 fordert Soft-Constraint Verhalten statt Abbruch.  
**Modul:** segmentation, frame_matching, beam_solver

### Test EC-02: A1 Debug markiert Pieces ohne ausreichende Rahmenhypothese
**Kategorie:** Debug / Observability  
**Setup:** Wie EC-01, aber zusaetzlich FrameHypothesen werden generiert.  
**Expected:**
- `debug.frame_hypotheses[piece_id]` existiert und enthaelt mindestens 1 Eintrag (auch wenn unter Schwellwert).  
- `debug.flags["missing_frame_contact_pieces"]` enthaelt `piece_id` (Liste oder Set).  
**Rationale:** Diagnosefaehigkeit bei A1 ist explizit gefordert.  
**Modul:** frame_matching, debug/models

### Test EC-03: A2 Rauschen/Ausreisser: Profil-Resampling ist laengenstabil
**Kategorie:** Input/Robustheit  
**Setup:**
- Segment mit Arclength-basiertem Resampling.
- Zwei Konturen: (a) glatt, (b) gleiche Kurve + 5% Ausreisserpunkte (spikes) + jitter (sigma=0.2mm).
- `profile_samples_N = 128`, `profile_smoothing_window = 3`.
**Expected:**
- Resampled Profil-Array hat exakt `N=128` Werte (int).
- `std(profile_b - profile_a) <= 0.5 # mm` (numerischer Stabilitaetsindikator; ggf. anpassen, aber fix in Test).  
**Rationale:** A2 fordert Resampling und minimale Glättung zur Stabilisierung.  
**Modul:** inner_matching/profile

### Test EC-04: A2 Quantilmetriken: dist_p90 reagiert robuster als dist_mean
**Kategorie:** Robustheit / Features  
**Setup:**
- Frame-Kontakt-Features auf Segment nahe Rahmenkante.
- Fuege 5% Ausreisserpunkte mit Abstand 8mm zur Kante ein.
**Expected:**
- `features.dist_p90_mm <= 2.0`
- `features.dist_mean_mm >= 2.5` (mean verschlechtert sich messbar)
**Rationale:** A2 verlangt Quantile (p90) statt nur Mittelwert.  
**Modul:** frame_matching/features

### Test EC-05: A2 Profilglättung minimal und konfigurierbar
**Kategorie:** Config/Robustheit  
**Setup:**
- Gleiches Segmentprofil, laufe Extraktion mit `profile_smoothing_window=3` und `=5`.  
**Expected:**
- Beide liefern gleiche Laenge `N`.
- `corr(window=5) >= corr(window=3) - 0.05` fuer ein noisy fixture (Korrelation darf nicht schlechter werden als 0.05).  
**Rationale:** A2 verlangt „so wenig wie noetig“ aber konfigurierbar.  
**Modul:** inner_matching/profile, config

### Test EC-06: A3 Nicht-konvexe Kontur wird via Strategie verarbeitet (triangulation default)
**Kategorie:** Geometrie / Collision  
**Setup:**
- Nicht-konvexes Polygon (z.B. konkave L-Form) als Piece-Outline in mm.
- `polygon_nonconvex_strategy="triangulation"`.  
**Expected:**
- `compute_penetration_depth(nonconvex_a, nonconvex_b)` liefert deterministisch `>= 0.0`.
- `debug.collision["nonconvex_strategy"] == "triangulation"`.
- `debug.collision["components_per_piece"][piece_id] >= 2` (Dreiecke/Komponenten).  
**Rationale:** A3 fordert nonkonvexe Behandlung und Logging.  
**Modul:** collision/overlap, debug

### Test EC-07: A3 Strategie-Wechsel wird im Debug reflektiert
**Kategorie:** Config/Debug  
**Setup:** Wie EC-06, aber `polygon_nonconvex_strategy` iteriert ueber `"triangulation"`, `"convex_decomposition"`, `"library"`, sofern implementiert.  
**Expected:**
- Debug-Feld `nonconvex_strategy` entspricht Config.
- Debug enthaelt pro Piece `components_per_piece` (int) auch wenn Strategie „library“ ist (dann ggf. `== 1`).  
**Rationale:** A3 verlangt klare Diagnose ueber Strategie.  
**Modul:** collision/overlap, debug

### Test EC-08: A3 Komponentenanzahl ist plausibel und >0
**Kategorie:** Geometrie / Sanity  
**Setup:** 3 nicht-konvexe Polygone mit unterschiedlicher Konkavitaet (leicht/mittel/stark).  
**Expected:**
- `components(light) <= components(medium) <= components(strong)` (monoton, wenn Triangulation/Decomposition genutzt).  
**Rationale:** Basic Sanity fuer „Anzahl Dreiecke pro Teil“ Debug aus A3.  
**Modul:** collision/overlap

### Test EC-09: A4 Fallback-Haeufigkeit wird gezaehlt
**Kategorie:** Robustheit / Telemetrie  
**Setup:**
- 10 Solver-Laeufe auf Edge-Fixture, bei dem Segmentierung regelmaessig oversplittet (gezielt).  
**Expected:**
- `expected_count = fixture.expected_fallback_triggered_runs` (deterministisch aus Fixture-Metadaten)
- `debug.stats["fallback_triggered_runs"] == expected_count`
- `debug.stats["total_runs"] == 10` 
**Rationale:** A4 fordert Zaehlung, um Designziel „Fallback selten“ pruefbar zu machen.  
**Modul:** fallback, debug/export

### Test EC-10: A4 Composite Match Nutzung in finaler Loesung wird gezaehlt
**Kategorie:** Fallback / Debug  
**Setup:** Ein Run mit Fallback, in dem die beste Loesung mindestens 1 composite Match nutzt.  
**Expected:**
- `solution.status == OK_WITH_FALLBACK`
- `debug.fallback["composite_matches_used_count"] == 1` (oder groesser, aber exakt im Fixture)
- `debug.fallback["composite_matches_used"]` ist Liste nicht-leer.  
**Rationale:** A4 fordert explizit Composite Usage Tracking.  
**Modul:** fallback/many_to_one, models.DebugBundle

### Test EC-11: A5 Symmetrie: Top-K Loesungen werden ausgegeben
**Kategorie:** Solver / Ambiguitaet  
**Setup:**
- Symmetrisches Puzzle (z.B. 4 identische Rechtecke, mehrere gueltige Rotationen).
- Config `topk_solutions = 3` (muss existieren, falls A5 implementiert wird).  
**Expected:**
- `len(solution.debug["top_solutions"]) == 3`
- Jede Top-Loesung enthaelt: `cost_total` (float), `max_penetration_mm` (float), `frame_coverage` (float), `area_score` (float).  
**Rationale:** A5 verlangt Top-K + Rationale.  
**Modul:** beam_solver, refinement/collision (penetration), debug

### Test EC-12: A5 Tie-break Reihenfolge ist deterministisch
**Kategorie:** Solver / Ranking  
**Setup:**
- Konstruiere 2 Loesungen mit:
  - gleicher `cost_total`
  - unterschiedliche `max_penetration_mm` (A kleiner als B)
- Beide in Top-K Pool.  
**Expected:**
- Rang(A) < Rang(B) (A gewinnt wegen penetration).  
**Rationale:** A5 nennt klare Tie-break Prioritaet.  
**Modul:** beam_solver (ranking), debug

### Test EC-13: A5 Tie-break Stufe 3/4 greift korrekt
**Kategorie:** Solver / Ranking  
**Setup:**
- 3 Loesungen mit gleicher `cost_total` und gleicher `max_penetration_mm`.
- Unterschiedliche `frame_coverage` und `area_score`.  
**Expected:**
- Hoehere `frame_coverage` gewinnt; bei Gleichstand hoeherer `area_score` gewinnt.
- Ranking ist stabil und deterministisch (gleicher Input -> gleiche Reihenfolge).  
**Rationale:** Minimiert „beliebige Wahl“ bei Symmetrien.  
**Modul:** beam_solver, debug

### Test EC-14: A6 n=5 ohne Grid-Annahmen (Completion nach State-Check)
**Kategorie:** Input/Algorithmik  
**Setup:**
- n=5 Pieces, keine Rasterstruktur.
- SolverState completion: `all placed AND len(open_edges)==0` (aus Struktur)  
**Expected:**
- Kein Zugriff auf Grid-spezifische Heuristiken (Test: `debug.solver_summary["grid_assumptions_used"] == False` oder Feld existiert und ist false).  
- Debug: `debug.solver_summary["n_pieces"] == 5`.  
**Rationale:** A6 fordert „keine Grid-Annahmen“.  
**Modul:** beam_solver/state, beam_solver/solver, debug

### Test EC-15: A6 Frontier-Mode wird geloggt
**Kategorie:** Debug/Observability  
**Setup:** n=5 fixture; frontier_mode Hybrid wie in Struktur beschrieben.  
**Expected:**
- `debug.solver_summary["frontier_mode"] in {"unplaced_pieces", "open_edges", "hybrid"}` und entspricht realer Nutzung.
**Rationale:** A6 fordert logging von n + frontier_mode.  
**Modul:** beam_solver/solver, debug

---

## 3. Failure Modes Tests (14 Tests)

### Test FM-01: F1 Beam laeuft leer -> NO_SOLUTION
**Trigger:** `topk_per_segment = 0` und `debug_topN_*` weiterhin >0, so dass Kandidatenraum leer bleibt.  
**Expected Status:** `NO_SOLUTION`  
**Expected Debug Info:**
```python
{
  "status": "NO_SOLUTION",
  "config_dump": "dict",
  "solver_summary": {
    "beam_exhausted": True,
    "max_expansions_reached": False,
    "prune_counts": "dict[str,int]"
  },
  "frame_hypotheses": "dict[piece_id, list]",
  "inner_candidates": "dict[seg_id, list]",
  "last_best_state": "SolverState|dict"
}
```
**Rationale:** F1 fordert Diagnose bei leerem Beam (Ursachenanalyse).  
**Modul:** beam_solver, debug

### Test FM-02: F1 max_expansions erreicht -> NO_SOLUTION
**Trigger:** `max_expansions = 5` bei Fixture, das mind. 20 Expansionen braeuchte.  
**Expected Status:** `NO_SOLUTION`  
**Expected Debug Info:**
- `solver_summary.max_expansions_reached == True`
- `solver_summary.expansions_done == 5`
- `solver_summary.best_cost_progression` Laenge == Iterationenanzahl (int)  
**Rationale:** F1 unterscheidet Beam leer vs Limit erreicht.  
**Modul:** beam_solver, debug

### Test FM-03: F2 confidence < threshold -> Fallback Rerun -> OK_WITH_FALLBACK
**Trigger:** Run1: `confidence = 0.49`, `fallback_conf_threshold = 0.5`, many-to-one enabled. Run2 verbessert conf auf >= 0.5.  
**Expected Status:** `OK_WITH_FALLBACK`  
**Expected Debug Info:**
```python
{
  "fallback": {
    "fallback_triggered": True,
    "confidence_before": 0.49,
    "confidence_after": ">=0.5",
    "rerun_count": 1
  }
}
```
**Rationale:** Direkte Umsetzung von F2.  
**Modul:** fallback, beam_solver, debug  
**Hinweis:** Details zur Fallback-Mechanik sind in `08_fallback_test_spec.md` (ca. Zeilen 158–187) spezifiziert.

### Test FM-04: F2 Fallback bleibt niedrig -> LOW_CONFIDENCE_SOLUTION
**Trigger:** Run1 conf=0.2; Run2 conf=0.3; threshold=0.5.  
**Expected Status:** `LOW_CONFIDENCE_SOLUTION`  
**Expected:**
- `solution` enthaelt eine vollstaendige oder bestmoegliche Pose-Zuweisung (kein None/empty bei vorhandenem best state).
- `debug.fallback["rerun_count"] == 1` (keine Schleife).  
**Rationale:** F2 verlangt Warnstatus statt Crash.  
**Modul:** fallback, beam_solver

### Test FM-05: F2 Kein Trigger wenn conf == threshold
**Trigger:** conf=0.5, threshold=0.5 (strikt `<`).  
**Expected Status:** Kein Fallback; `status != OK_WITH_FALLBACK`; `debug.fallback["fallback_triggered"] == False`.  
**Rationale:** Boundary Condition fuer Trigger.  
**Modul:** fallback

### Test FM-06: F3 Refinement scheitert wegen final overlap > threshold
**Trigger:** Pre-refinement Loesung hat `max_penetration_mm = 0.8`; Optimierer kann unter `overlap_depth_max_mm_final = 0.1` nicht kommen (z.B. constraints inkonsistent).  
**Expected Status:** `REFINEMENT_FAILED`  
**Expected:**
- Output-Pose entspricht **pre-refinement best** (identische `poses_F` innerhalb `TOL_MM/TOL_DEG`).
- `debug.refinement["stop_reason"] in {"max_iters", "diverged", "stalled", "constraint_infeasible"}`
- `debug.refinement["penetration_depth_final_mm"] > 0.1`  
**Rationale:** F3 fordert Rueckfall auf beste pre-refinement Loesung.  
**Modul:** refinement, collision, debug

### Test FM-07: F3 cost trajectory und penetration trajectory sind vorhanden
**Trigger:** Wie FM-06.  
**Expected Debug Info:**
- `len(debug.refinement["cost_trajectory"]) >= 1`
- `len(debug.refinement["penetration_trajectory_mm"]) == len(cost_trajectory)`  
**Rationale:** F3 fordert Trajectories im Debug.  
**Modul:** refinement, debug

### Test FM-08: F4 Overlap-Flapping near-zero wird geloggt
**Trigger:** Zwei Polygone mit fast-touching Kontakt: theoretische penetration 0.0–0.02mm; wiederhole mit minimaler Pose-Perturbation (±0.01mm).  
**Expected Status:** `OK` oder `REFINEMENT_FAILED` je nach Resultat, aber deterministisch pro Input.  
**Expected Debug Info:**
- `debug.collision["near_zero_cases"]` Laenge >= 1
- Eintrag enthaelt: `min_axis_overlap_mm` (float), `epsilon_mm` (float), `classified_as_overlap` (bool)  
**Rationale:** F4 verlangt robuste Diagnose fuer numerische Instabilitaet.  
**Modul:** collision/overlap, refinement

### Test FM-09: F4 epsilon/tolerance Usage ist explizit im Debug
**Trigger:** Wie FM-08, mit `epsilon_mm` (Konfig/Hardcap) > 0.0.  
**Expected:**
- `debug.collision["tolerance_used"] == True`
- `debug.collision["epsilon_mm"] == epsilon_mm ± TOL_MM`  
**Rationale:** Nachvollziehbarkeit, warum Klassifikation so ausfaellt.  
**Modul:** collision/overlap, debug

### Test FM-10: F5 Frame-inside pruning im Solver ist nicht zu strikt
**Trigger:** Ein Piece ist bei Seed-Pose um `1.5mm` ausserhalb; `tau_frame_mm = 2.0mm`.  
**Expected Status:** Kein prune wegen „outside frame“ im Solver (state bleibt im Beam).  
**Expected Debug Info:**
- `prune_counts["outside_frame"]` erhoeht sich **nicht** in diesem Fall.
- Optional: `debug.solver_trace` enthaelt outside-distance `<= 2.0`.  
**Rationale:** F5 fordert robusten Start (Solver tolerant).  
**Modul:** beam_solver, frame_matching

### Test FM-11: F5 Refinement inside check ist strenger als Solver
**Trigger:** Gleiches Fixture wie FM-10, aber Refinement final check nutzt strengere Toleranz (z.B. 0.5mm).  
**Expected:**
- Solver akzeptiert Seed, Refinement markiert violation oder korrigiert.
- Wenn nicht korrigierbar: `REFINEMENT_FAILED` mit `debug.refinement["stop_reason"]` passend.  
**Rationale:** F5 explizit.  
**Modul:** refinement, beam_solver

### Test FM-12: INVALID_INPUT wenn n nicht in {4,5,6}
**Trigger:** n=7 pieces.  
**Expected Status:** `INVALID_INPUT`  
**Expected Debug Info:**
- `debug.failure_reason == "invalid_n_pieces"`
- `debug.n_pieces == 7`  
**Rationale:** Status-Enum nennt diesen Fall explizit.  
**Modul:** API/solve_puzzle, models, debug

### Test FM-13: INVALID_INPUT bei fehlender Kontur
**Trigger:** Ein Piece ohne `contour` oder leeres Array.  
**Expected Status:** `INVALID_INPUT`  
**Expected:**
- `debug.failure_reason == "missing_contour"`
- `debug.affected_pieces == [piece_id]`  
**Rationale:** Input-Validierung muss frueh und kontrolliert sein.  
**Modul:** API/solve_puzzle, segmentation

### Test FM-14: INVALID_INPUT bei ungueltigem Einheitensetup
**Trigger:** Pieces in Pixel ohne `scale_px_to_mm` (oder unit mismatch), obwohl Solver mm erwartet.  
**Expected Status:** `INVALID_INPUT`  
**Expected:**
- `debug.failure_reason in {"missing_scale_px_to_mm", "unit_mismatch"}`
- `debug.required_fields_missing` enthaelt `"scale_px_to_mm"` (oder aequivalent).  
**Rationale:** 00_structure betont mm-basiertes Arbeiten; ohne Skalierung nicht definierbar. (00_structure.md ca. Zeilen 40–65, 112–140)  
**Modul:** API/solve_puzzle, utils/conversion

---

## 4. Debug Information Tests (6 Tests)

### Test DBG-01: Debug-Minimum bei jedem non-OK Status ist vollstaendig
**Kategorie:** Debug/Schema  
**Setup:** Erzeuge je 1 Run mit Status `NO_SOLUTION`, `LOW_CONFIDENCE_SOLUTION`, `REFINEMENT_FAILED`, `INVALID_INPUT`.  
**Expected (alle muessen existieren):**
- `debug.config_dump` (dict)
- `debug.n_pieces` (int)
- `debug.area_score` (float; falls nicht berechenbar -> `None` + `debug.area_score_reason`)
- `debug.frame_hypotheses` (dict)
- `debug.inner_candidates` (dict)
- `debug.solver_summary.prune_counts` (dict[str,int])
- `debug.last_best_state` (dict oder serialisierbar)
- Wenn refinement relevant: `debug.refinement` inkl. last iter stats  
**Rationale:** Direkt aus Debug-Minimum (D).  
**Modul:** models.DebugBundle, debug/export

### Test DBG-02: Datentypen sind stabil (serialisierbar)
**Kategorie:** Debug/Format  
**Setup:** Serialisiere `DebugBundle` zu JSON (export_debug_json true).  
**Expected:**
- JSON-Export ohne Exception.
- Keine `NaN`/`Infinity` Werte (oder: explizit als Strings gemappt, aber konsistent).  
**Rationale:** Debug muss reproduzierbar exportierbar sein (00_structure debug/export).  
**Modul:** debug/export, models

### Test DBG-03: Top-N Listen respektieren Config
**Kategorie:** Debug/Completeness  
**Setup:** `debug_topN_frame_hypotheses_per_piece = 5`, `debug_topN_inner_candidates_per_segment = 5`.  
**Expected:**
- Pro piece max 5 Hypothesen in Debug (oder weniger, wenn nicht vorhanden).
- Pro segment max 5 Kandidaten in Debug.  
**Rationale:** Debug-Volumen kontrolliert halten, aber konsistent.  
**Modul:** frame_matching, inner_matching, debug

### Test DBG-04: prune_counts Keys sind stabil und nicht leer
**Kategorie:** Solver/Debug  
**Setup:** Fixture, das mindestens 2 verschiedene Prune-Gründe triggert (`outside_frame`, `overlap`, `conflict`).  
**Expected:**
- `len(prune_counts.keys()) >= 2`
- Alle Werte int >= 0
- Keys sind aus definierter Menge (Schema): `{"outside_frame","overlap","conflict","beam_width","other"}` (ggf. anpassen, aber fixieren).  
**Rationale:** F1 verlangt prune counts nach reason.  
**Modul:** beam_solver, debug

### Test DBG-05: last_best_state ist konsistent zum Solution-Output
**Kategorie:** Debug/Integrity  
**Setup:** NO_SOLUTION Run mit gespeichertem last_best_state.  
**Expected:**
- `debug.last_best_state.cost_total == min(debug.solver_summary.best_cost_progression)`
- `len(last_best_state.placed) == debug.solver_summary.max_placed_seen`  
**Rationale:** Reproduzierbarkeit und Plausibilitaet.  
**Modul:** beam_solver/state, debug

### Test DBG-06: Failure reason ist immer gesetzt bei INVALID_INPUT
**Kategorie:** Input/Debug  
**Setup:** INVALID_INPUT (FM-12/13/14).  
**Expected:**
- `debug.failure_reason` ist nicht-leer (string).
- `debug.affected_pieces` existiert (Liste; kann leer sein bei n invalid).  
**Rationale:** Nutzer muss Ursache verstehen (Ziel der Doc).  
**Modul:** API/solve_puzzle, debug

---

## 5. Integration Tests (6 Tests)

### Test INT-01: E2E A1+A2 kombiniert (kleine Frame-Kante + Rauschen)
**Setup:** 6-piece fixture: ein corner piece mit kurzer Aussenkante + noisy contours.
**Expected:**
- Kein Crash; Status in Enum.
- **Wenn non-OK: Debug-Minimum (DBG-01):**
  - `debug.config_dump`, `debug.n_pieces`, `debug.area_score`
  - `debug.frame_hypotheses`, `debug.inner_candidates`
  - `debug.solver_summary.prune_counts`, `debug.last_best_state`
- **Zusätzlich bei INVALID_INPUT (DBG-06):**
  - `debug.failure_reason` (non-empty), `debug.affected_pieces` (list)
**Rationale:** Kombinierte Grenzfälle sind realistisch.
**Modul:** end-to-end

### Test INT-02: E2E A3 Nicht-konvex + Overlap-Pruning
**Setup:** 5-piece fixture, mind. 2 nicht-konvexe pieces, overlap_depth_max_mm_prune=1.0.  
**Expected:**
- Solver liefert entweder OK/LOW_CONFIDENCE oder NO_SOLUTION, aber immer kontrolliert.
- Debug loggt Strategie + Komponentenanzahl (EC-06).  
**Rationale:** Cross-module Robustheit (collision + solver).  
**Modul:** end-to-end

### Test INT-03: E2E A5 Symmetrie: Top-K + Tie-break
**Setup:** Symmetrisches 4-piece fixture, topk_solutions=3.  
**Expected:**
- `debug.top_solutions` vorhanden und sortiert nach Tie-break (EC-12/13).  
**Rationale:** Validiert Gesamtverhalten statt Unit-only.  
**Modul:** end-to-end

### Test INT-04: E2E F2 Fallback Aktivierung und Debug-Vergleich
**Setup:** Fixture, bei dem Run1 low-conf ist, Run2 (mit composites) besser.  
**Expected:**
- Status OK_WITH_FALLBACK
- Debug enthaelt before/after cost+conf + composite usage.  
**Rationale:** F2 + A4 uebergreifend.  
**Modul:** fallback + solver

### Test INT-05: E2E F3 Refinement Failed liefert pre-refinement Loesung
**Setup:** Fixture mit inkonsistenten Constraints, Refinement kann overlap nicht unter 0.1mm bringen.  
**Expected:**
- REFINEMENT_FAILED
- poses entsprechen pre-refinement (FM-06).  
**Rationale:** Unerlaesslich fuer Simulator/Visualisierung.  
**Modul:** refinement

### Test INT-06: E2E INVALID_INPUT wird frueh abgefangen
**Setup:** n=7 oder missing contour.  
**Expected:**
- INVALID_INPUT innerhalb definierter kurzer Laufzeit (z.B. < 50ms in Unit-Test Umgebung; wenn messbar).
- Keine Nebenwirkungen (kein partial debug export ohne Status).  
**Rationale:** Fail-fast ohne Crash.  
**Modul:** API layer

---

## 6. Test-Fixtures

**Fixture-Typen (minimal, deterministisch):**
1. **Polygon Fixtures (mm):** konvexe Rechtecke, konkave L-Formen, near-touching Paare (F4).  
2. **Segment Fixtures:** `ContourSegment(points_mm, length_mm, chord, direction_angle_deg, flatness_error, profile_1d)` aus 00_structure/models (ca. Zeilen 248–330).  
3. **Noisy Contours Generator:** baseline Kontur + jitter + spikes (A2).  
4. **Symmetry Fixture:** identische Pieces mit zwei gueltigen globalen Orientierungen (A5).  
5. **Fallback Fixture:** oversplit Segmente, nur composite macht Match moeglich (A4/F2).  
6. **Invalid Input Templates:** n invalid, contour missing, unit mismatch (INVALID_INPUT).

**Mock/Helper Anforderungen:**
- deterministische RNG seeds (z.B. seed=0) fuer noisy fixtures
- eindeutige piece_id/segment_id Zuweisung
- simple frame model 128x190mm (default)

---

## 7. Strategy Decisions (aus „Offene Fragen“)

### D1: `T_MF` Handling (KS Rahmen→Maschine)
**Optionen:** A) Platzhalter ohne phys. Bedeutung; B) reale Kalibrierung; C) Maschinen-KS output deaktivieren bis bekannt  
**Empfehlung:** A (Platzhalter) + immer F-KS ausgeben (gem. Doc).  
**Test-Impact:** INVALID_INPUT vermeiden; Test DBG-01 fordert beide KS-Felder oder klare `None`/Reason.

### D2: Eckenradius Rahmen
**Optionen:** A) 0; B) definierter Radius  
**Empfehlung:** A als Default, B als Config; Debug muss Radius ausgeben.  
**Test-Impact:** Frame-inside Checks muessen fixture-abhaengig parametrisiert werden.

### D3: `pose_grob_F` Definition
**Optionen:** A) Projektion; B) chord alignment; C) mini-optimierung pro Hypothese  
**Empfehlung:** B fuer deterministische Seeds (weniger drift als A) + optional C spaeter.  
**Test-Impact:** F1/F5 Tests muessen die Seed-Qualitaet ueber Debug messbar machen (best-cost progression, prune reasons).

### D4: Nonkonvex Overlap Strategie
**Optionen:** A) convex decomposition; B) triangulation; C) library  
**Empfehlung:** B als Default (Design nennt Performance-Risiko, aber implementierbar).  
**Test-Impact:** EC-06..08 (Komponentenanzahl + Logging) sind Pflicht.

### D5: Profil-Glättung window
**Optionen:** 3 / 5 / adaptiv  
**Empfehlung:** Default 3, Test EC-05 sichert, dass 5 nicht destabilisiert.  
**Test-Impact:** noisy fixtures brauchen fixe numerische Schwellen.

### D6: ICP/Fit-Kosten Innenmatching
**Optionen:** off / on  
**Empfehlung:** off als Default; on fuer disambiguierung spaeter.  
**Test-Impact:** Wenn on: neue Failure/Timeout Tests (nicht Teil dieses Specs, aber erweiterbar).

### D7: `k_conf` Kalibrierung
**Optionen:** A) 1.0; B) datengetrieben  
**Empfehlung:** A als Start; B als Regression-Kalibrierung.  
**Test-Impact:** F2 Tests nutzen fixture-fixierte cost_total, damit conf stabil reproduzierbar ist.

### D8: Overlap-Grenzwerte
**Optionen:** prune 1.0mm / final 0.1mm vs andere  
**Empfehlung:** Defaults wie Doc; Schwellen als Config und in Debug ausgeben.  
**Test-Impact:** FM-06/INT-05 fixieren 0.1mm als harte Erwartung.

### D9: Completion Definition
**Optionen:** A) alle Teile platziert + Checks; B) zusaetzlich keine offenen Interfaces  
**Empfehlung:** A als V1 (laut Struktur `is_complete()`), B spaeter falls Interfaces formalisiert werden.  
**Test-Impact:** EC-14/15 muessen Completion-Policy und Frontier-Mode im Debug validieren.
