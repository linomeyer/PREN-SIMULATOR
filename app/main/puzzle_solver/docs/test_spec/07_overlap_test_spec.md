# 07_overlap_test_spec.md
Test-Spezifikation Schritt 7: Overlap/Collision Detection (SAT/MTV)

Basis (intern):
- `implementation/00_structure.md` ca. Zeilen 355-385 (Schritt 7: Default Triangulation, Aggregation, Startwerte 1.0mm/0.1mm, Debug).  [oai_citation:0‡00_structure.md](sediment://file_00000000805871f4b632ecebc35ec4ea)

## 1. Scope und Begriffe

### Ziel
Implementierung und Absicherung von:
- `solver/overlap/collision.py`
  1) `penetration_depth(poly_a, poly_b) -> float`
  2) `penetration_depth_max(state, pieces) -> float`
  3) SAT/MTV Helper (konvex)

### Definition: penetration_depth
- Output ist die **MTV-Laenge** (minimal translation vector magnitude), um `poly_a` und `poly_b` **disjunkt** zu machen.
- Keine Ueberlappung (inkl. nur Beruehrung) => `0.0`.
- Symmetrisch: `penetration_depth(a,b) == penetration_depth(b,a)`.

### Polygon-Konventionen (Tests erwarten diese Invarianten)
- Punkte in mm, `np.ndarray` shape `(N,2)`.
- Simple polygons (keine Selbstschnitte). CCW bevorzugt, aber Implementation soll robust gegen CW sein.
- Numerik:
  - `EPS = 1e-9` fuer interne Stabilitaet
  - `TOL_MM = 1e-3` fuer Assertions (0.001mm), ausser explizit anders.

### Non-konvex Strategie (D6)
Aus `implementation/00_structure.md` Schritt 7:
- Default: **Triangulation** (Option B), Aggregation per Config: `"max" | "mean" | "p90"`; Performance-Risiko dokumentiert.  [oai_citation:1‡00_structure.md](sediment://file_00000000805871f4b632ecebc35ec4ea)
Test-Spec deckt:
- Konvex: harte numerische Erwartungen.
- Non-konvex: strategie-agnostische Erwartungen + (optional) V1-spezifische Checks fuer Triangulation.

## 2. Testdaten: Hilfs-Polygone (mm)

### Konvex
- `SQ10 = [(0,0),(10,0),(10,10),(0,10)]`
- `SQ10_shift(dx,dy) = SQ10 + (dx,dy)`
- `RECT20x10 = [(0,0),(20,0),(20,10),(0,10)]`
- `THIN = [(0,0),(0.1,0),(0.1,10),(0,10)]`  # sehr schmal

### Konkav (Non-konvex)
- `L_SHAPE = [(0,0),(10,0),(10,3),(3,3),(3,10),(0,10)]`  # konkav, simple

## 3. Unit-Tests: penetration_depth(poly_a, poly_b)

### Test 1: No Overlap (separated squares)
**Setup:**
- a = SQ10
- b = SQ10_shift(20, 0)
**Expected:**
- penetration_depth(a,b) == 0.0 (abs <= TOL_MM)

### Test 2: Tangential edge contact (touching, no overlap)
**Setup:**
- a = SQ10
- b = SQ10_shift(10, 0)  # beruehrt bei x=10
**Expected:**
- penetration_depth(a,b) == 0.0 (abs <= TOL_MM)

### Test 3: Tangential corner contact (touching at one corner)
**Setup:**
- a = SQ10
- b = SQ10_shift(10, 10)  # beruehrt bei (10,10)
**Expected:**
- penetration_depth(a,b) == 0.0 (abs <= TOL_MM)

### Test 4: Small Overlap (0.5mm)
**Setup:**
- a = SQ10
- b = SQ10_shift(9.5, 0)  # Ueberlappung in x: 0.5
**Expected:**
- 0.49 <= penetration_depth(a,b) <= 0.51

### Test 5: Larger Overlap (3.0mm)
**Setup:**
- a = SQ10
- b = SQ10_shift(7.0, 0)  # Ueberlappung in x: 3.0
**Expected:**
- 2.99 <= penetration_depth(a,b) <= 3.01

### Test 6: Full Containment (inner square inside outer square)
**Setup:**
- outer = [(0,0),(20,0),(20,20),(0,20)]
- inner = [(5,5),(15,5),(15,15),(5,15)]
**Expected:**
- MTV minimal entlang x oder y bis disjunkt: 5.0mm
- 4.99 <= penetration_depth(outer, inner) <= 5.01
- 4.99 <= penetration_depth(inner, outer) <= 5.01  # Symmetrie

### Test 7: Identical polygons (maximaler Overlap-Fall)
**Setup:**
- a = SQ10
- b = SQ10  # identisch
**Expected:**
- MTV minimal ist 10.0mm (entlang x oder y)
- 9.99 <= penetration_depth(a,b) <= 10.01

### Test 8: Symmetry property (random shift overlap)
**Setup:**
- a = RECT20x10
- b = [(15,2),(35,2),(35,12),(15,12)]  # Overlap in x: 5, in y: 8
**Expected:**
- depth_ab = penetration_depth(a,b)
- depth_ba = penetration_depth(b,a)
- abs(depth_ab - depth_ba) <= TOL_MM
- Und: 4.99 <= depth_ab <= 5.01 (min overlap ist 5.0)

### Test 9: Translation invariance
**Setup:**
- a1 = SQ10
- b1 = SQ10_shift(9.5, 0)  # depth ~0.5
- a2 = SQ10_shift(100, -50)
- b2 = SQ10_shift(109.5, -50)
**Expected:**
- abs(penetration_depth(a1,b1) - penetration_depth(a2,b2)) <= TOL_MM
- und beide ca. 0.5mm (siehe Test 4 Range)

### Test 10: Robustness to polygon orientation (CW vs CCW)
**Setup:**
- a_ccw = SQ10
- a_cw = reversed(SQ10)  # gleiche Geometrie, andere Reihenfolge
- b = SQ10_shift(9.5, 0)
**Expected:**
- penetration_depth(a_ccw, b) ~ 0.5mm (Range wie Test 4)
- penetration_depth(a_cw, b)  ~ 0.5mm (Range wie Test 4)

### Test 11: Very thin polygon stability (no overlap)
**Setup:**
- a = THIN
- b = THIN shifted by dx=1.0 (separated, da width=0.1)
**Expected:**
- penetration_depth(a,b) == 0.0 (abs <= TOL_MM)
**Notes:** Test ist primär Stabilitaet (keine NaNs/Exceptions).

## 4. Unit-Tests: Non-konvex Polygone (strategie-agnostisch + V1)

> Diese Tests muessen fuer alle D6-Optionen bestehen. Numerische Erwartungen sind bewusst als Range/Property definiert, weil die exakte MTV je nach Approximation/Decomposition variieren kann.

### Test 12: Non-convex no overlap
**Setup:**
- a = L_SHAPE
- b = SQ10_shift(30, 0)
**Expected:**
- penetration_depth(a,b) == 0.0 (abs <= TOL_MM)

### Test 13: Non-convex clear overlap (must be > 0)
**Setup:**
- a = L_SHAPE
- b = [(2,2),(6,2),(6,6),(2,6)]  # liegt im Innenbereich und schneidet die L-Geometrie sicher
**Expected (strategie-agnostisch):**
- penetration_depth(a,b) > 0.1  # mindestens klar > 0
- penetration_depth(a,b) < 4.0  # obere Schranke, damit kein absurdes Ergebnis
**Optional V1 (Triangulation + max aggregation):**
- depth sollte stabil in einem engeren Band liegen, nach erster Implementationsmessung nachziehen.

## 5. Unit-Tests: penetration_depth_max(state, pieces)

### Annahme fuer Schnittstelle (Step7)
- `state.poses_F[piece_id]` existiert fuer platzierte Teile.
- `pieces[piece_id].contour_mm` ist **piece-lokal**; Overlap-Modul erzeugt Weltpolygon via Pose (Transform2D).
Falls eure reale Datenhaltung anders ist, muessen diese Tests an die effektive Konvention angepasst werden (aber genau eine Konvention muss gelten).

### Test 14: Max depth over 3 pieces (1 overlap pair)
**Setup:**
- pieces:
  - p1.contour_mm = SQ10 (lokal)
  - p2.contour_mm = SQ10 (lokal)
  - p3.contour_mm = SQ10 (lokal)
- state.poses_F:
  - p1 at (0,0,0)
  - p2 at (9.5,0,0)  # overlap 0.5
  - p3 at (30,0,0)   # kein overlap
**Expected:**
- penetration_depth_max(state,pieces) in [0.49, 0.51]

### Test 15: Max depth returns the maximum (two overlapping pairs)
**Setup:**
- p1 at (0,0,0)
- p2 at (9.5,0,0)  # overlap 0.5
- p3 at (7.0,0,0)  # overlap zwischen p1 und p3: 3.0
**Expected:**
- penetration_depth_max(...) in [2.99, 3.01]

### Test 16: Ignores unplaced pieces (pose missing)
**Setup:**
- pieces p1,p2,p3 wie oben
- state.poses_F enthaelt nur p1,p2; p3 unplaced
- p1/p2 wie Test 4: overlap 0.5
**Expected:**
- penetration_depth_max(...) in [0.49, 0.51]
- keine Exception wegen p3

## 6. Config-Schwellenwerte: Prune vs Final (Step7 Integration)

Startwerte aus `implementation/00_structure.md` ca. Zeilen 367-369: prune=1.0mm, final=0.1mm.  [oai_citation:2‡00_structure.md](sediment://file_00000000805871f4b632ecebc35ec4ea)

Da `collision.py` laut Moduldefinition nur Depth liefert, werden Threshold-Checks als Mini-Helper empfohlen (oder als Solver-Tests in Schritt 6/9). Falls ihr in `collision.py` Helpers zulasst:

### Optional Helper A: should_prune_overlap(depth, config) -> bool
**Test 17: Prune threshold**
- depth = 1.01, config.overlap_depth_max_mm_prune = 1.0 => True
- depth = 1.00 => False (oder True, aber muss festgelegt sein; Empfehlung: strict `>`)

### Optional Helper B: is_final_overlap_ok(depth, config) -> bool
**Test 18: Final threshold**
- depth = 0.09, config.overlap_depth_max_mm_final = 0.1 => True
- depth = 0.11 => False

Wenn ihr keine Helpers wollt: diese beiden Tests werden als Integrations-Tests im Solver/Refinement platziert (Schritt 7/9).

## 7. Strategie-Empfehlung (D6) fuer V1

### Option A: Convex Decomposition
- Pro: potenziell weniger Primitive als Triangulation, gute Genauigkeit
- Contra: Implementationsaufwand hoch (robuste Zerlegung bei beliebigen Konturen), viele Edgecases

### Option B: Triangulation (Default in `00_structure.md`)
- Pro: einfach implementierbar, etabliert; in V1 gut testbar; passt zu Risiko-Doku und Performance-Abschaetzung in `00_structure.md` ca. 361-377  [oai_citation:3‡00_structure.md](sediment://file_00000000805871f4b632ecebc35ec4ea)
- Contra: Performance kann steigen (t_a * t_b), Aggregation beeinflusst Konservativitaet

### Option C: Polygon-Library Approximation (z.B. shapely)
- Pro: schnell zu einem robusten Ergebnis (wenn Dependency ok)
- Contra: Abhaengigkeit/Deployment (Raspi, Wheels), Numerik/Robustheit je nach Backend, schwerer "white-box" zu debuggen

**Empfehlung V1:** Option B (Triangulation) + Aggregation `"max"` als konservativer Default, weil bereits im Projekt-Plan so vorgesehen und gut mit Debug-Metriken (SAT-Pruefungen/State) ueberwachbar (`00_structure.md` ca. 379-383).  [oai_citation:4‡00_structure.md](sediment://file_00000000805871f4b632ecebc35ec4ea)  
(Alternative: `"p90"` als spaetere Performance/Robustness-Option, aber erst nach Messung.)

## 8. Akzeptanzkriterien (Step7)
- Alle Tests 1-16 muessen deterministisch bestehen.
- Keine NaNs/Exceptions bei CW/CCW, duennen Polygonen, containment.
- Symmetrie und Translation-Invarianz muessen eingehalten sein (Tests 8-9).
- Non-konvex: mindestens korrektes 0 vs >0 Verhalten (Tests 12-13), exakte MTV kann in V1 als Range abgesichert werden.