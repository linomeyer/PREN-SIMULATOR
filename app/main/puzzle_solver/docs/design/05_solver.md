## Solver-Design

### Ziel
Ein globaler Solver, der:
- mehrere Hypothesen parallel verfolgt (Multi-Hypothesen),
- frühe Fehlentscheidungen vermeidet,
- ohne Rasterannahmen funktioniert (n=5 möglich),
- Soft→Hard Regeln korrekt umsetzt,
- Overlap/Frame-Constraints als frühe Pruning-Kriterien nutzt,
- reproduzierbar und debugbar ist.

---

## Zustandsrepräsentation (`SolverState`)

Pflichtfelder:
- `poses_F: dict[piece_id -> Pose2D]`  
  Teilposen im Rahmen-KS (F).
- `placed: set[piece_id]`  
  Welche Teile sind bereits platziert.
- `committed_frame_constraints: list[FrameHypothesis]`  
  Rahmenhypothesen, die in diesem Branch als „fix“ gelten.
- `active_constraints: list[MatchConstraint]`  
  Innenmatch-Constraints, die für Platzierung genutzt wurden (für Debug/Refinement).
- `cost_total: float` und `cost_breakdown: dict[str->float]`
- `open_frontier`: Repräsentation „was ist noch offen“  
  (siehe Optionen unten).
- `debug_trace: list[TraceEvent]`  
  Events: commit, expansion, prune reason, cost update.

---

## Soft→Hard Regel (commit policy)

- Rahmenkontakt ist global **soft** (Penalty möglich).
- Innerhalb eines Branch gilt:
  - Sobald ein Teil mittels einer Rahmenhypothese platziert wird, wird diese Hypothese **committed**.
  - Committed bedeutet:
    - zukünftige Platzierungen dürfen diesen Rahmenkontakt nicht widersprechen (harte Konsistenz in diesem Branch).
- Nicht-committed Rahmenhypothesen bleiben Vorschläge und dürfen ersetzt/ignoriert werden.

**Wichtig**: „widerspricht committed“ ist ein harter Prune-Grund. „hat keine committed Rahmenhypothese“ ist nur Penalty.

---

## Initialisierung (Seeding)

Ziel: Beam nicht leer starten, aber auch ohne gute Rahmenhypothesen funktionieren.

Optionen:

### Seed-Option A: Best-Frame Seeds (Standard)
- Pro Teil Top-N Rahmenhypothesen berechnet.
- Erzeuge initiale States durch Auswahl der besten Hypothese für 1..m Teile (m klein, z.B. 1 oder 2).
- Vorteil: schnell guter Anchor.
- Nachteil: wenn Rahmenhypothesen schlecht, können Seeds fehlen.

### Seed-Option B: Mixed Seeds (robuster)
- Erzeuge zusätzlich einen „leeren“ State ohne Platzierung.
- Solver kann dann über Innenmatches starten.
- Vorteil: funktioniert auch wenn Rahmenhypothesen fehlen.
- Nachteil: Suche kann breiter werden.

**Empfehlung**: A + B (leerer State als Backup).

---

## Frontier-Definition (was wird als nächstes erweitert?)

Das ist zentral. Mehrere Varianten:

### Frontier-Ansatz 1: „Place next unplaced piece“ (einfach)
- Wähle ein unplatziertes Teil und versuche:
  - Platzierung via Rahmenhypothese
  - Platzierung via Innenmatch zu bereits platziertem Teil
- Pro: einfach, kontrollierbar.
- Contra: ohne klare Priorisierung kann Suche ineffizient werden.

### Frontier-Ansatz 2: „Open interfaces“ (segment-/constraint-getrieben)
- Frontier enthält „offene“ Segmente an der Grenze des bereits platzierten Clusters.
- Expansion versucht, diese offenen Segmente mit Kandidaten zu matchen.
- Pro: natürliche Constraint-Propagation.
- Contra: braucht saubere Definition, welche Segmente „offen“ sind.

### Frontier-Ansatz 3: Hybrid
- Starte mit Ansatz 1 bis mehrere Teile platziert sind,
- wechsle dann zu Ansatz 2.

**Entscheidung**: Hybrid (1→2), weil initial ohne Cluster „open interfaces“ unklar sind.

---

## Expansionen (Move-Generator)

Ein `SolverState` erzeugt neue States durch kontrollierte Aktionen:

### Move 1: Place via FrameHypothesis
Preconditions:
- Teil i ist unplatziert.
- Rahmenhypothese `h` existiert für i (Top-N).

Action:
- setze `poses_F[i] = h.pose_grob_F` (oder refine local)
- `placed.add(i)`
- commit `h` (in state)
- update cost_total (frame cost + penalties update)

### Move 2: Place via InnerMatchCandidate
Preconditions:
- Teil A platziert, Teil B unplatziert.
- Kandidat `m` existiert: seg_a(A) ↔ seg_b(B).

Action:
- berechne Pose von B relativ zu A:
  - `pose_B = pose_A ∘ T_match` (T_match aus Kandidat oder aus Segmentalignment)
- add constraint `m` to active_constraints
- update cost_total (inner cost + penalties update)
- optional: wenn B gleichzeitig gute Rahmenhypothese hat, kann sie als weiche Zusatzinformation einfliessen (noch nicht commit)

### Move 3: Re-commit / switch frame hypothesis (optional, später)
- Nur falls explizit zugelassen; erhöht Komplexität.
- Für V1: nicht nötig; Beam-Search übernimmt „alternative branch“.

---

## Kosten-Update und Ranking

- Pro Move werden Costs additiv akkumuliert:
  - `cost_total = Σ(cost_frame_committed) + Σ(cost_inner_constraints) + penalties + overlap_cost(optional)`
- Overlap wird primär für Pruning verwendet; kann optional als Cost-Term addiert werden (Barrier erst im Refinement).

Ranking innerhalb Beam:
- Primär: `cost_total` (min)
- Tie-break (optional, debug):
  - geringere Overlap (falls >0)
  - höhere Rahmen-Coverage
  - bessere Flächenbilanz-Score

---

## Pruning (harte Ausschlusskriterien)

Prune ein `new_state`, wenn:

1) **Outside Frame**
- Teilkontur liegt ausserhalb Rahmen-Innenkontur über `tau_frame_mm`.
- `tau_frame_mm` wird als Sicherheitsband verwendet, um Rauschen zu tolerieren.

2) **Overlap**
- `penetration_depth_max(new_state) > overlap_depth_max_mm_prune`

3) **Committed Frame Conflict**
- neue Pose widerspricht einer committed Rahmenhypothese:
  - z.B. ein committed Segment soll an TOP liegen, Pose legt es klar nicht an TOP (Definition über Feature/Geometrie).
- Wichtig: nur gegen committed, nicht gegen alle Hypothesen.

4) (Optional) **Unmögliche Reststruktur**
- z.B. zu viele unmatchable Segmente, oder Kandidatenraum leer.
- Für V1 optional, um Komplexität gering zu halten.

---

## Termination / Completion

Ein State ist „complete“, wenn:
- `len(placed) == n`
- alle nötigen Constraints vorhanden sind (Definition abhängig vom Frontier-Modell):
  - Option A: genügt, wenn alle Teile platziert und Costs akzeptabel sind
  - Option B: zusätzlich „keine offenen Interfaces“ (wenn Frontier-Ansatz 2 verwendet)

Für V1:
- Completion = alle Teile platziert, plus final checks (Overlap, Coverage) bestehen oder Refinement kann sie herstellen.

---

## Output des Solvers

- Liste vollständiger States (Top-K) oder mindestens der beste State.
- Für Debug: beste K States + Pruning-Statistiken.
- Danach Phase 3 (Refinement) auf best state anwenden.

---

## Konfigurationshebel (Solver)
- `beam_width`
- `topk_per_segment`
- `max_iterations` / `max_expansions`
- `seed_mode` (A/B/hybrid)
- `frontier_mode` (hybrid 1→2)
- `pruning_strictness` (z.B. overlap prune threshold)

---

## Offene Punkte (bewusst offen, aber dokumentiert)
- Exakte Definition, wann ein Teil „outside frame“ ist bei nicht-0 corner radius.
- Exakte Definition, wie `pose_grob_F` aus FrameHypothesis geschätzt wird (wirkt stark auf Solver).
- Ob Overlap als reiner Prune oder zusätzlich als Cost-Term behandelt wird (V1: primär prune, Refinement: Barrier).