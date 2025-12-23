## Overlap und Kollisionsmodell

### Ziel
- Overlap (Überschneidung) darf am Ende praktisch nicht existieren.
- Während des Solvers dürfen kleine Overlaps toleriert werden, dürfen aber nicht eskalieren (Pruning).
- Overlap wird über **maximale Eindringtiefe** gemessen (nicht Fläche).

---

## Definition: Eindringtiefe (Penetration Depth)
- Eindringtiefe `d >= 0` zwischen zwei Polygonen ist die minimale Translation, die nötig ist, um die Polygone gerade zu trennen.
- Operationalisierung:
  - **SAT (Separating Axis Theorem)**: prüft, ob eine Trennachse existiert.
  - **MTV (Minimum Translation Vector)**: liefert die Richtung und Länge der minimalen Translation, falls Overlap besteht.
- Metrik im System:
  - `penetration_depth(P, Q) = ||MTV(P, Q)||` wenn Overlap
  - sonst `0`

**State-Level Metrik**:
- `penetration_depth_max(state) = max_{i<j} penetration_depth(P_i, P_j)`

---

## Konvex vs. nicht-konvex (bewusst offen, mit Optionen)
SAT/MTV gilt direkt für **konvexe** Polygone. Puzzle-Konturen sind potenziell nicht-konvex.

### Option A: Konvexe Zerlegung (Convex Decomposition)
- Zerlege jede Teilkontur in konvexe Komponenten `{C_i_k}`.
- Penetration zwischen zwei Teilen:
  - `max_k,l penetration_depth(C_a_k, C_b_l)`
- **Pro**: saubere MTV-Definition, geometrisch korrekt.
- **Contra**: Implementationsaufwand, Zerlegung muss robust sein.

### Option B: Triangulation
- Trianguliere jede Teilkontur zu Dreiecken `{T_i_k}` (Dreiecke sind konvex).
- Penetration zwischen zwei Teilen:
  - `max_k,l penetration_depth(T_a_k, T_b_l)`
- **Pro**: Triangulation oft verfügbar; robust.
- **Contra**: viele Dreiecke → mehr Paarprüfungen; MTV über max von Dreiecken ist konservativ.

### Option C: Polygon-Library + Näherung
- Nutze Polygon-Intersection für „Overlap ja/nein“ und approximative Tiefe (z.B. über Distanz zur separierenden Verschiebung, Sampling, etc.).
- **Pro**: schnell implementierbar.
- **Contra**: Tiefe ggf. nicht exakt/definiert; schwerer reproduzierbar.

**Festlegung für V1**:
- Overlap-Metrik ist MTV-Länge via SAT/MTV.
- Strategie für nicht-konvex ist **bewusst offen**; Wahl muss im Code dokumentiert werden (Config-Feld `polygon_nonconvex_strategy`).

---

## Schwellenwerte (konfigurierbar, Startwerte als Annahme)
- `overlap_depth_max_mm_prune`:
  - Zweck: frühes Verwerfen klar falscher Hypothesen.
  - **Startannahme**: 1.0 mm (tuning)
- `overlap_depth_max_mm_final`:
  - Zweck: finaler Akzeptanztest nach Refinement.
  - **Startannahme**: 0.1 mm (tuning)

Hinweis: Diese Werte sind bewusst als Startwerte gewählt; sie müssen über reale Konturqualität / Kalibrierfehler evaluiert werden. Debug-Ausgaben müssen die Werte pro Run loggen.

---

## Kontur-Offsets / Dilatation (Option, bewusst offen)
Bei vision-basierten Konturen kann systematischer Fehler auftreten (Kontur liegt innen/ausserhalb der echten Geometrie).

Option: Offset (Minkowski Sum) um `dilate_mm`:
- `P' = offset_polygon(P, dilate_mm)`
- Overlap wird dann auf `P'` geprüft.

- **Pro**: kann Robustheit erhöhen, wenn Konturen systematisch „zu klein“ sind.
- **Contra**: beeinflusst Ergebnisse stark; muss sehr sauber parametrisiert werden.

**Status**: bewusst offen (Default `dilate_mm = 0`).

---

## Solver-Integration

### Pruning
Ein `SolverState` wird verworfen, wenn:
- `penetration_depth_max(state) > overlap_depth_max_mm_prune`

### Refinement (Barrier)
Im Pose-Refinement wird Overlap nicht nur geprüft, sondern aktiv minimiert:
- Penalty-Term z.B. `λ * max(0, penetration_depth_max - ε)^2`
- Ziel: `penetration_depth_max <= overlap_depth_max_mm_final`

---

## Debug-Ausgaben (Pflicht)
- Pro Solver-State (mindestens für best paths oder aggregate):
  - `penetration_depth_max_mm`
  - Teilpaar mit max Depth
  - Anzahl Paare mit Depth > 0
- Final:
  - `penetration_depth_max_mm_final`
  - falls > final threshold: Fail reason + Werte

---

## Offene Punkte (bewusst offen, aber markiert)
- Wahl der nicht-konvex Strategie (A/B/C).
- Ob und wie `dilate_mm` eingesetzt wird.
- Numerische Stabilität bei nahezu tangentialen Kontakten (Epsilon-Behandlung).