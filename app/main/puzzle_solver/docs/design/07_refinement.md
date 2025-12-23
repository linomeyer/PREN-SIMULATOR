## Pose-Refinement

### Ziel
Nach dem globalen Solver liegt eine konsistente, aber grobe Lösung vor. Pose-Refinement soll:
- kleine Fehler in Translation/Rotation korrigieren,
- Overlap gegen 0 drücken (praktisch kein Eindringen),
- Rahmenkontakt und Innenkanten-Konsistenz verbessern,
- dabei deterministisch und debugbar bleiben.

---

## Eingaben
- Initiale Lösung aus Solver:
  - `poses_F_init: dict[piece_id -> Pose2D]` (Rahmen-KS)
  - `committed_frame_constraints: list[FrameHypothesis]`
  - `active_constraints: list[InnerMatchConstraint]` (aus gewählten Kandidaten)
- `FrameModel`, `MatchingConfig`

---

## Variablen
Für jedes Teil i:
- `x_i, y_i, theta_i` (im Rahmen-KS)
- Optional: falls ein Teil als „stabiler Anchor“ gilt, kann es fixiert werden (reduziert Freiheitsgrade). Das ist eine Option, keine Pflicht.

---

## Zielfunktion (Cost-Minimization)
Gesamtziel:
- `min  J = J_frame + J_inner + J_overlap + J_regularize`

### 1) Rahmenkontakt-Term `J_frame`
Basierend auf committed frame constraints (und optional soft frame hints):
- Für jedes committed `FrameHypothesis h`:
  - verwende die gleichen Rahmenmetriken wie in Phase 2 (dist, coverage, angle, flatness),
  - aber jetzt als kontinuierliche Kostenfunktion der Pose.
- Aggregation:
  - `J_frame = Σ w_frame * cost_frame(h, pose_i)`

Hinweis: Wenn frame contact global soft ist, kann `J_frame` nur committed Hypothesen stark gewichten; nicht-committed können höchstens schwach einfliessen.

### 2) Innenkanten-Term `J_inner`
Für jede gewählte Innenmatch-Constraint (Segment A ↔ Segment B):
- Ziel: Segmentpunkte sollen nach Transformation gut übereinander liegen.
- Optionen für Kosten:
  - **Option A (schnell)**: Profilbasierte Kosten (1D) als konstant (nicht pose-abhängig) → wenig Nutzen im Refinement.
  - **Option B (präzise)**: Punkt-zu-Punkt oder Punkt-zu-Linie Distanz nach Transformation (ähnlich ICP).
    - Beispiel: `Σ || T(p_a_k) - T(p_b_k)||^2` nach korrektem Alignment (inkl. Umkehrung).
- Für Refinement sinnvoll ist Option B (pose-abhängig).

Aggregation:
- `J_inner = Σ w_inner * cost_inner_constraint(c, poses)`

### 3) Overlap-Term `J_overlap` (Barrier / Penalty)
- Verwende penetration depth `d = penetration_depth_max(poses)`.
- Barrier-ähnlich:
  - `J_overlap = λ * max(0, d - ε)^2`
- ε kann 0 oder ein kleines numerisches Epsilon sein.

Ziel: `d` soll <= `overlap_depth_max_mm_final` werden.

### 4) Regularisierung `J_regularize` (optional)
- Kleine Regularisierung gegen extreme Winkeländerungen:
  - `Σ μ * wrap(theta_i - theta_i_init)^2`
- Nutzt man nur, wenn Solver initial instabil ist.

---

## Optimierungsverfahren (Optionen, bewusst nicht final festgelegt)

### Option A: Gradient-freies Verfahren (Nelder-Mead / Powell)
- **Pro**: wenig Annahmen, robust bei nicht-smooth (Overlap).
- **Contra**: kann langsam sein; Skalierung/Parameter wichtig.

### Option B: Gauss-Newton / Levenberg-Marquardt (Least Squares)
- **Pro**: sehr effizient, wenn J als Sum-of-squares formuliert.
- **Contra**: Overlap-Barrier ist nicht glatt; braucht sorgfältige Handhabung.

### Option C: Zweistufig
1) Minimierung von `J_frame + J_inner` (glatt)
2) Overlap-Barriere schrittweise erhöhen (Continuation), bis `d` klein ist
- **Pro**: sehr stabil in der Praxis.
- **Contra**: Implementationsaufwand höher.

**Empfehlung für V1**: Option C, falls Aufwand akzeptabel; sonst Option A mit klaren Iterationslimits.

---

## Abbruchkriterien
- Max Iterationen `max_refine_iters`
- Verbesserung < `delta_cost_min`
- Overlap erfüllt: `penetration_depth_max <= overlap_depth_max_mm_final` (zusätzliches Kriterium)

---

## Debug-Ausgaben (Pflicht)
- Start vs. Ende:
  - `cost_total_init`, `cost_total_final`
  - Breakdown: frame/inner/overlap
  - `penetration_depth_max_init`, `penetration_depth_max_final`
- Iterationslog (optional, aber hilfreich):
  - cost pro Iteration
  - max overlap depth pro Iteration
- Finaler Status:
  - `refinement_success: bool`
  - falls false: Grund (Iterationslimit / Overlap bleibt / Divergenz)

---

## Output
- Verfeinerte `poses_F`
- Falls `T_MF` gesetzt: transformiere nach Maschinen-KS (M) für Ausgabe.
- Übernimm Debug-Daten in `DebugBundle`.

---

## Offene Punkte (bewusst offen)
- Konkrete Wahl des Optimierers (A/B/C).
- Exakte Definition der pose-abhängigen Innenkanten-Kosten (Punktzuordnung, Umkehrung, Sampling).
- Umgang mit nicht-konvexer Overlap-Berechnung im Refinement (gleiche Strategie wie im Solver).