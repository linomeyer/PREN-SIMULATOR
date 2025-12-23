## Fallback many-to-one

### Ziel
Abdecken von Fällen, in denen eine „logische“ Kontaktkante eines Teils in der Segmentierung als mehrere benachbarte Segmente vorliegt (oder umgekehrt). Many-to-one Matching wird **nicht** standardmässig verwendet, sondern nur dann aktiviert, wenn das Direktmatching (1:1) keine ausreichend gute Gesamtlösung liefert.

Trigger (vereinbart):
- Aktivierung, wenn `confidence < fallback_conf_threshold` (Start: 0.5, konfigurierbar).

---

## Begriffe
- **Direktmatching (1:1)**: ein Segment `A` wird einem Segment `B` zugeordnet.
- **Many-to-one**:
  - `(A1 + A2 + ... + Ak) ↔ B`
  - wobei `A1..Ak` **adjazent** entlang derselben Teilkontur sind (kontinuierlicher Abschnitt).

---

## Voraussetzungen / Scope
- Teileanzahl klein (4..6), dennoch soll Kandidatenraum nicht explodieren.
- Many-to-one wird daher strikt begrenzt:
  - nur **adjazente** Segmente,
  - nur kleine Kettenlängen (z.B. k=2, optional k=3),
  - nur wenn Länge/Profil-Kompatibilität plausibel ist.

Diese Grenzen werden als Config geführt:
- `many_to_one_max_chain_len` (Option: 2 oder 3)
- `many_to_one_enable_only_if_triggered` (true)

---

## Ablauf (Fallback-Pipeline)

### Schritt 1: Diagnose (aus Debug/Run)
Beim Trigger wird im Debug explizit festgehalten:
- `fallback_triggered = true`
- `trigger_reason = confidence_below_threshold`
- welche Cost-Komponenten dominant waren (frame vs inner vs penalties vs overlap)

### Schritt 2: Erzeuge zusammengesetzte Segmente (nur lokal)
Für jedes Teil:
- betrachte alle Segmente in Konturreihenfolge
- bilde Ketten:
  - `k=2`: (seg_i + seg_{i+1})
  - optional `k=3`: (seg_i + seg_{i+1} + seg_{i+2})
- für jede Kette:
  - `points_mm` = konkatenierte Punktliste (mit Duplikat-Endpunkten bereinigen)
  - `length_mm`, `flatness_error`, `direction_angle` neu berechnen
  - 1D-Profil neu berechnen (Resampling auf N)

**Wichtig**: Keine globalen Re-Splits; nur Kettenbildung über existierende Segmente, damit Debug nachvollziehbar bleibt.

### Schritt 3: Kandidatenbildung erweitern
- Innenmatching wird erneut ausgeführt, aber Kandidaten dürfen jetzt:
  - `composite_seg ↔ atomic_seg`
  - `atomic_seg ↔ composite_seg`
  - optional `composite ↔ composite` (typisch vermeiden, sonst combinatorics)
- Prefilter:
  - Längenfenster: composite_len ~ other_len
  - optional Flatness/Frame-Likelihood Gate (Innen bevorzugt)

### Schritt 4: Solver rerun
- Beam-Search wird erneut gestartet, gleiche Pruning-Regeln.
- Konfigurierbar:
  - `beam_width_fallback` kann grösser sein als normaler beam.
  - `topk_per_segment_fallback` kann leicht erhöht werden.

### Schritt 5: Auswahl
- Wenn Fallback eine Lösung mit besserer `cost_total` bzw. höherer Confidence liefert, wird sie gewählt.
- Debug hält beide Varianten (vor Fallback / nach Fallback) als Vergleich.

---

## Scoring im Many-to-one Kontext
- Profilmatching wie bei 1:1, nur auf zusammengesetztem Profil.
- Zusätzliche Penalty möglich:
  - `penalty_composite_used` (klein) um many-to-one nicht unnötig zu bevorzugen.
- Ziel: many-to-one ist „Rettungsanker“, nicht Standardpfad.

---

## Debug-Ausgaben (Pflicht)
- `fallback_triggered` (bool)
- `confidence_before`, `cost_before`
- `confidence_after`, `cost_after`
- Anzahl erzeugter composite Segmente pro Teil
- Welche Matches composite verwendet haben (Liste)
- Penalty-Anteile (falls `penalty_composite_used` aktiv)

---

## Risiken / Edge Cases
- **Segmentierung generell zu fein**: many-to-one würde ständig aktiv werden.
  - Gegenmassnahme: Segmentierung in Phase 1 stabilisieren (target_seg_count_range).
- **Composite-Segmente werden fälschlich als Rahmenkanten interpretiert**:
  - Gegenmassnahme: many-to-one nur für Innenmatching verwenden (nicht für Frame-first).
- **Kettenbildung über Kontur-„Wrap-around“** (Ende→Anfang):
  - Entscheidung: entweder erlauben (wenn Kontur zyklisch) oder verbieten; als Config flag führen.
- **Explodierender Kandidatenraum**:
  - Gegenmassnahme: chain_len begrenzen und Prefilter strikt halten.

---

## Offene Punkte (bewusst offen, aber dokumentiert)
- Ob `k=3` initial erlaubt wird oder erst später.
- Ob composite↔composite Matching überhaupt zugelassen wird (V1 eher nein).
- Ob many-to-one auch für Frame-first zulässig wäre (V1 nein).