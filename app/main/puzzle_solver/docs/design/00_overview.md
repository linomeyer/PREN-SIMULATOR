## Projekt-Übersicht

- **Zweck**: Entwurf und Implementationsplan für ein robustes Matching-Verfahren, das aus bereits segmentierten/extrahierten Puzzle-Teilen (Konturen) eine vollständige, konsistente Puzzle-Lösung im bekannten A5-Zielrahmen bestimmt.

- **Problemstellung**: Das bestehende Matching (Kanten-Pairing + deterministischer Build, teilweise diskrete Rotationen) ist in der Praxis unzuverlässig. Die neue Lösung soll unter realistischen Bedingungen (Rauschen, unregelmässige Kanten, variable Teileanzahl) reproduzierbar eine Lösung finden oder sauber diagnostizieren, warum keine Lösung möglich ist.

- **Ziel-Output**:
  - Pro Teil: absolute Soll-Pose `(x_soll_mm, y_soll_mm, winkel_soll_deg)` in einem eindeutig definierten Koordinatensystem (Ausgabe-KS wird über Transformationsparameter bestimmt).
  - Zusätzlich: strukturierte Debug-Daten (Metriken, Costs, Hypothesen, Solver-Trails) für Parametertuning und Fehleranalyse.

- **Rahmenbedingungen**:
  - Teileanzahl variabel: **4, 5 oder 6**.
  - Zielrahmen: Innenmass **128 × 190 mm**, Orientierung konzeptionell bekannt; Pose `T_MF` (Rahmen→Maschine) und Eckenradius sind **bewusst offen** (Platzhalter im Simulator, später fixieren).
  - Keine Spiegelung; 2D Translation + kontinuierliche Rotation.
  - Invarianten: Puzzle ist rechteckig im Rahmen, **keine Löcher**, Overlap am Ende praktisch 0 (max Eindringtiefe als Kriterium).

- **Leitprinzipien**:
  - **Frame-first**: Rahmenconstraints liefern starke Anker, jedoch als **Soft-Constraint** modelliert (robust gegen Segmentierungsschwächen).
  - **Multi-Hypothesen**: Globaler Solver hält mehrere Kandidaten (Beam/Top-K) statt früh zu committen.
  - **Messbarkeit**: Jede Metrik (Rahmenkontakt, Innenmatch, Overlap, Coverage) wird einzeln ausgegeben; Aggregation ist konfigurierbar.
  - **Iterierbarkeit**: Parameter sind zentral gebündelt; späterer Auto-Tuning-Loop ist möglich, ohne Kernlogik umzuschreiben.


## Anforderungen

### Funktional

- **F1 — Vollständige Lösung**: Aus einer Menge extrahierter Teile wird eine Puzzle-Lösung erzeugt, die alle Teile platziert und orientiert.
- **F2 — Absolute Soll-Posen**: Ausgabe pro Teil einer Soll-Pose `(x_soll_mm, y_soll_mm, winkel_soll_deg)` im definierten Ausgabe-KS (siehe später: KS/Rahmenmodell).
- **F3 — Variable Teileanzahl**: Funktioniert für `n ∈ {4,5,6}` ohne fixe Raster-/Gridannahmen.
- **F4 — Frame-first**: Rahmenkontakt wird als primäre Informationsquelle genutzt (Rahmengeometrie bekannt), aber als **Soft-Constraint** (kein hartes „jedes Teil muss“).
- **F5 — Innenkanten-Matching**: Matching funktioniert auch bei „wilden“ Kantenformen (nicht Tab/Slot-typisch).
- **F6 — Many-to-one (Fallback)**: Wenn Direktmatching (1:1 Segment ↔ Segment) keine ausreichend gute Gesamtlösung liefert, wird ein Fallback aktiviert, das begrenzt many-to-one Matches zulässt (adjazente Segmente A1+A2 ↔ B).
- **F7 — Overlap-Minimierung**: Overlap wird nicht akzeptiert; im Solver wird früh beschnitten (prune), und nach Refinement muss Overlap praktisch 0 sein. Metrik: **maximale Eindringtiefe**.
- **F8 — Robustheit gegen Rauschen**: Kontur-/Kalibrierrauschen wird über Toleranzen, Quantile, Glättung und Multi-Hypothesen abgefedert.
- **F9 — Debug-Ausgaben**: Das System liefert pro Lauf eine strukturierte Debug-Ausgabe mit:
  - Rahmenhypothesen pro Teil/Segment inkl. Einzelmetriken
  - Innenmatch-Kandidaten (Top-k) inkl. Score-Komponenten
  - Solver-Trace (States, Pruning-Gründe, Costs)
  - finale Checks (Coverage, Overlap, Konsistenz, Flächenbilanz-Score)

### Nicht-Funktional

- **NF1 — Performance (Simulator)**:
  - Für `n<=6` sollen typische Läufe interaktiv bleiben.
  - Kandidatenbildung begrenzt durch Top-k pro Segment und Beam-Width.
- **NF2 — Determinismus & Reproduzierbarkeit**:
  - Bei gleichen Inputs und gleicher Konfiguration identische Ergebnisse (keine unkontrollierte Randomness).
  - Falls Randomness für Sampling genutzt wird: Seed ist Teil der Konfiguration und wird geloggt.
- **NF3 — Konfigurierbarkeit**:
  - Alle Schwellen/Toleranzen/Weights sind in einer zentralen Config definiert und werden im Debug mit ausgegeben.
  - „Bewusst offene“ Parameter (z.B. `T_MF`, Eckenradius, k_conf) sind explizit als offen dokumentiert, aber technisch als Felder vorhanden.
- **NF4 — Erweiterbarkeit**:
  - Neue Metriken können ergänzt werden, ohne den Solver zu brechen (Feature-Dump + Weighting).
  - Alternativer Solver (ILP/MIP) kann später als Benchmark ergänzt werden.
- **NF5 — Plattform**:
  - Zielsystem perspektivisch Raspi/Embedded; daher bevorzugt einfache numerische Operationen, kontrollierte Komplexität und optionale Heavy-Parts (z.B. ICP) abschaltbar.

### Constraints (fix / bewusst offen)

- **Fix**:
  - Rahmeninnenmass 128×190 mm.
  - Keine Spiegelung; Rotation kontinuierlich.
  - Keine Löcher im finalen Puzzle.
- **Bewusst offen** (muss als Parameter geführt werden, nicht hartkodiert):
  - Rahmenpose `T_MF` (Rahmen→Maschine).
  - Eckenradius der Innenkontur.
  - Profilglättung und k_conf Startwert (Tuning).
  - Overlap-Grenzwerte (Startwerte vorhanden, aber als Tuning markieren).



## Nächste Schritte

> Ziel: Umsetzung mit einem Code-Agent. Die Reihenfolge ist so gewählt, dass nach jedem Schritt ein lauffähiger Zwischenstand existiert und Debug-Daten die nächsten Entscheidungen stützen.

### 1) Projekt-Gerüst für neues Matching schaffen
- Neues Paket/Modul anlegen (parallel zum bestehenden Matching, um A/B Vergleich zu ermöglichen).
- Einführen von:
  - `MatchingConfig`
  - `FrameModel`
  - `PuzzleSolution`
  - `DebugBundle`
- Serialisierung: Config + Debug als JSON exportierbar (Pfad/Run-ID).

**Ergebnis**: Minimales API `solve_puzzle(pieces, frame, config)` existiert (noch stub), Debug-Export funktioniert.

---

### 2) Einheiten und KS-Handling stabilisieren
- Sicherstellen, dass Matching intern in **mm** arbeitet:
  - Konverter: Pixel→mm über vorhandene Kalibrierung/Skalierung.
- KS-Tagging im Debug:
  - Welche Koordinaten im Rahmen-KS (F) vs Maschinen-KS (M) vorliegen.
- `T_MF` als bewusst offener Parameter:
  - Wenn nicht gesetzt: Ausgabe nur in F (und im Debug als „T_MF undefined“ markieren).
  - Wenn gesetzt: zusätzlich Ausgabe in M.

**Ergebnis**: Reproduzierbare, eindeutig interpretierbare Koordinatenflüsse.

---

### 3) Coarse Segmentierung + Flatness V1 implementieren
- Kontursegmentierung:
  - Split an Krümmungsmaxima / Richtungswechsel.
  - Merge bis Mindestlänge und Zielsegmentbereich erreicht ist.
- Flatness V1:
  - RMS Punkt-zu-Sehne Distanz (mm).
- Output:
  - `ContourSegment` pro Teil mit stabilen IDs.
- Debug:
  - Segmentanzahl pro Teil
  - Segmentlängen, flatness_error, direction_angle

**Ergebnis**: Stabiler Segment-Input für alle folgenden Schritte.

---

### 4) Rahmenkontakt-Features (alle Metriken) + Hypothesenbildung
- Für jedes Segment und jede Rahmen-Seite:
  - berechne `dist_mean/p90/max`, `coverage_in_band`, `inlier_ratio`, `angle_diff`, `flatness_error`.
- Cost-Mapping + Aggregation:
  - `cost_frame = Σ w_k * cost_k`
- Hypothesen:
  - `FrameHypothesis(piece, seg, side, pose_grob_F, cost_frame, features)`
- Debug:
  - Top-N Hypothesen pro Teil mit allen Einzelmetriken.

**Ergebnis**: Frame-first Kandidaten existieren, Qualität ist messbar.

---

### 5) 1D-Profil-Extraktion + Direkt-Innenmatching (Top-k)
- Profil:
  - Resampling auf `profile_samples_N=128` (Startannahme, konfigurierbar).
  - minimaler Smoothing (window offen, aber implementierbar als Parameter).
- Similarity:
  - NCC forward/reversed, `profile_cost = 1 - max_corr`.
- Kandidaten:
  - Prefilter (Länge, optional frame-likelihood).
  - Top-k pro Segment speichern.
- Debug:
  - Top-k Kandidaten pro Segment inkl. Komponenten und reversal_used.

**Ergebnis**: Innenkanten-Kandidatenraum für Solver verfügbar.

---

### 6) Beam-Solver V1 (ohne Refinement, aber mit Pruning)
- Implementiere:
  - Seeding (Frame seeds + leerer state)
  - Expansion (Place via FrameHypothesis, Place via InnerMatchCandidate)
  - Soft→Hard commit pro Branch
  - Pruning: outside frame (`tau_frame_mm`), overlap (stub oder simplified zunächst), committed conflicts
  - Beam-Ranking via cost_total
- Debug:
  - Beam stats, prune reason counts, best-cost progression

**Ergebnis**: Erste vollständige Lösungen möglich (noch grob), Solver-Trace vorhanden.

---

### 7) Overlap-Modul (SAT/MTV) integrieren
- Implementiere SAT/MTV für konvexe Polygone.
- Ergänze nonkonvex-Strategie (Option A/B/C):
  - Wahl dokumentieren im Code + Debug.
- Setze Startwerte:
  - prune: 1.0 mm
  - final: 0.1 mm
- Pruning im Solver aktivieren.
- Debug:
  - max penetration depth + verursachendes Paar

**Ergebnis**: Overlap wird sauber beschnitten, falsche Hypothesen fliegen früh raus.

---

### 8) Confidence-Mapping + Fallback many-to-one
- Implementiere:
  - `confidence = exp(-k_conf * cost_total)` (k_conf als Parameter)
  - Fallback-Trigger: conf < 0.5 (Start)
- Many-to-one:
  - Erzeuge composite segments (chain_len=2 Start)
  - erweitere Innenmatching, rerun solver
- Debug:
  - before/after cost/conf
  - composite usage

**Ergebnis**: Robustheit steigt, besonders bei Segmentierungs-Splits.

---

### 9) Pose-Refinement implementieren
- Wähle Optimierer (Optionen im Design; V1 pragmatisch).
- Ziel:
  - Overlap depth gegen final threshold
  - Rahmen/Innenkosten verbessern
- Final checks:
  - penetration_depth_max <= overlap_depth_max_mm_final
- Debug:
  - cost trajectory, overlap trajectory, stop reason

**Ergebnis**: „praktisch kein Overlap“ wird zuverlässig erreicht.

---

### 10) Integration, Regression, Vergleich
- Integration in bestehende Simulator-Visualisierung.
- A/B Vergleich mit altem Solver:
  - gleiche Inputs, compare outputs, logs.
- Regression-Suite:
  - mehrere gespeicherte Szenen / Seed-Läufe
  - Debug-Exports archivieren

**Ergebnis**: Stabiler Entwicklungsprozess und nachvollziehbare Verbesserungen.