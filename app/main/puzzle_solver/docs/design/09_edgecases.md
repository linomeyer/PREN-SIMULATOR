## Edge Cases und Failure Modes

### Ziel
Systematisch definieren, wie der Algorithmus mit Grenzfällen umgeht, und wie Failures diagnostiziert/kommuniziert werden. Ein Failure ist kein „Crash“, sondern ein kontrollierter Zustand mit klaren Debug-Informationen.

---

## A) Edge Cases (Geometrie / Input)

### 1) Extrem kleine Rahmenberührung pro Teil
- **Problem**: Segmentierung kann die kleine Aussenkante unter `min_frame_seg_len_mm` „wegmergen“ oder als Innenstück klassifizieren.
- **Behandlung**:
  - Rahmenkontakt bleibt Soft-Constraint: kein harter Abbruch.
  - `penalty_missing_frame_contact` greift.
  - Solver kann Teil über Innenmatches platzieren.
- **Debug**:
  - Markiere Teile ohne ausreichende Rahmenhypothese.
  - Logge beste Rahmenhypothese trotzdem (auch wenn unter Schwellwert).

### 2) Starke Konturrauschanteile / Ausreisserpunkte
- **Problem**: Flatness und Profil werden instabil.
- **Behandlung**:
  - Resampling auf Arclength.
  - Quantilmetriken (p90) statt nur Mittelwert.
  - Profilglättung minimal (Config).
- **Debug**:
  - Profilkorrelationen und FlatnessError ausgeben.
  - Optional: Statistik der Punktabstände.

### 3) Nicht-konvexe Konturen
- **Problem**: SAT/MTV benötigt konvexe Polygone.
- **Behandlung**:
  - `polygon_nonconvex_strategy` wählen (convex decomposition / triangulation / library).
- **Debug**:
  - Logge gewählte Strategie.
  - Logge Anzahl konvexer Komponenten/Dreiecke pro Teil.

### 4) Many-to-one Matching ist tatsächlich häufig
- **Problem**: Fallback wäre ständig aktiv → Designziel verfehlt.
- **Behandlung**:
  - Segmentierung in Phase 1 anpassen (weniger Split, mehr Merge; target_seg_count_range enger).
  - Alternativ: many-to-one auch im Normalmodus zulassen (Designänderung).
- **Debug**:
  - Zähle, wie viele Runs Fallback aktivieren.
  - Zähle, wie oft composite Matches in finaler Lösung genutzt werden.

### 5) Symmetrische oder nahezu symmetrische Lösungen
- **Problem**: mehrere globale Minima, Solver kann beliebige wählen.
- **Behandlung**:
  - Top-K Lösungen behalten (K konfigurierbar).
  - Tie-break:
    1) geringster cost_total
    2) kleinste penetration depth
    3) beste Rahmen-Coverage
    4) beste Flächenbilanz-Score
- **Debug**:
  - Ausgabe der Top-K Lösungen inkl. tie-break rationale.

### 6) n=5 ohne Rasterstruktur
- **Problem**: heuristische Rasterannahmen würden brechen.
- **Behandlung**:
  - Solver macht keine Grid-Annahmen.
  - Completion basiert auf „alle Teile platziert + Checks“.
- **Debug**:
  - Logge n und den verwendeten frontier_mode.

---

## B) Failure Modes (kontrollierte Abbrüche)

### F1) Keine vollständige Lösung im Solver
- **Symptom**: Beam läuft leer oder erreicht `max_expansions`, ohne completion.
- **Mögliche Ursachen**:
  - Kandidatenraum zu stark beschnitten (topk zu klein, strikte Prefilter).
  - Rahmenhypothesen/pose_grob schlecht.
  - Overlap pruning zu streng.
- **Reaktion**:
  - Ergebnisstatus: `NO_SOLUTION`
  - Optional: Auto-Variantenlauf (nicht Bestandteil V1, aber vorbereitet durch Config).
- **Debug-Pflicht**:
  - Beam-Statistik, prune counts nach reason.
  - Top frame hypotheses und top inner candidates.
  - Parameterdump.

### F2) Lösung existiert, aber Confidence zu niedrig
- **Symptom**: `confidence < fallback_conf_threshold`
- **Reaktion**:
  - Fallback many-to-one aktivieren und Solver neu laufen lassen.
  - Falls weiterhin niedrig: Status `LOW_CONFIDENCE_SOLUTION` (Lösung + Warnung).
- **Debug**:
  - Vergleich before/after fallback: cost/conf und dominante Kostenkomponenten.

### F3) Refinement scheitert
- **Symptom**: Overlap bleibt > final threshold oder Optimierer divergiere/stoppt.
- **Reaktion**:
  - Status `REFINEMENT_FAILED`
  - Output: beste pre-refinement Lösung + Diagnose
- **Debug**:
  - cost trajectory, max penetration depth trajectory, stop reason.

### F4) Overlap-Check unzuverlässig (numerisch)
- **Symptom**: Flapping: mal overlap, mal nicht bei minimalen Änderungen.
- **Reaktion**:
  - Epsilon/robuste Geometriebehandlung (konfigurierbar).
  - Debug: logge near-zero cases.
- **Debug**:
  - minimal axis overlap values, tolerance usage.

### F5) Frame-inside Check zu strikt
- **Symptom**: viele prunes wegen outside frame, obwohl Lösung visuell möglich.
- **Reaktion**:
  - `tau_frame_mm` erhöhen (Start 2mm ist bewusst robust).
  - Inside check im Refinement strenger als im Solver.
- **Debug**:
  - outside distances (max outside) pro prune.

---

## C) Ergebnisstatus (empfohlenes Enum)
- `OK`
- `OK_WITH_FALLBACK`
- `LOW_CONFIDENCE_SOLUTION`
- `NO_SOLUTION`
- `REFINEMENT_FAILED`
- `INVALID_INPUT` (z.B. n nicht in {4,5,6} oder fehlende Konturen)

---

## D) Debug-Outputs für Failures (Minimum)
Bei jedem non-OK Status:
- vollständige Config
- n, area_score
- Top frame hypotheses pro piece
- Top inner candidates pro segment (zumindest für relevante)
- solver prune stats + last best state
- falls refinement: last iter stats

Damit ist ein Failure reproduzierbar und analysierbar.



## Offene Fragen

Die folgenden Punkte bleiben **bewusst offen** (wie vereinbart) und werden als Parameter/Optionen geführt. Für die Implementierung muss jeweils:
- ein Feld/Interface existieren,
- ein Default/Platzhalter gesetzt sein (wo nötig),
- der Status und die Annahme im Debug ausgegeben werden.

### 1) Rahmenpose `T_MF` (Rahmen-KS → Maschinen-KS)
- **Status**: Maschine noch nicht gebaut; numerische Pose unbekannt.
- **Erwartung**:
  - Im Simulator: Platzhalter wählen (garantiert ausserhalb A4) und in Config/FrameModel sichtbar machen.
  - Später: reale Definition/Kalibrierung ersetzt Platzhalter.
- **Risiko**: Ausgabe in Maschinen-KS ist ohne `T_MF` nicht physikalisch interpretierbar; deshalb immer auch Rahmen-KS ausgeben.

### 2) Rahmen-Innenkontur: Eckenradius
- **Status**: nicht festgelegt.
- **Optionen**:
  - A) `corner_radius_mm = 0` (perfektes Rechteck)
  - B) definierter Radius (z.B. mechanisch bedingt)
- **Impact**:
  - Inside/Outside Checks und Rahmenkontakt-Features könnten angepasst werden (Rundung).

### 3) Definition von `pose_grob_F` (Initialpose aus Rahmenhypothese)
- **Status**: noch nicht spezifiziert.
- **Optionen**:
  - A) Projektion des Segmentes auf Rahmenlinie + Translation so, dass Segmentpunkte minimalen Abstand haben
  - B) Endpunkt-/Chord-Alignment + Snap an Linie (mit Winkelrestriktion über alpha)
  - C) lokale Mini-Optimierung pro Hypothese (1–5 Iterationen) bevor Solver startet
- **Impact**:
  - Stark auf Solver-Stabilität (Seeds) und Overlap-Pruning.

### 4) Nicht-konvexe Overlap-Strategie (SAT/MTV Basis)
- **Status**: Wahl offen, aber SAT/MTV als Definition gesetzt.
- **Optionen**:
  - A) Convex decomposition
  - B) Triangulation
  - C) Library-basierte Näherung
- **Impact**:
  - Performance (Anzahl Teilprüfungen)
  - Genauigkeit der Eindringtiefe und Pruning-Verhalten

### 5) Profil-Glättung (`profile_smoothing_window`)
- **Status**: offen, „so wenig wie nötig“.
- **Optionen**:
  - A) window=3 (detailtreu)
  - B) window=5 (robuster)
  - C) adaptiv (komplexer)
- **Impact**:
  - Matching-Robustheit vs. Detailverlust

### 6) ICP / Fit-Kosten im Innenmatching
- **Status**: optional.
- **Optionen**:
  - A) ICP deaktiviert (nur Profil + Länge) → schneller, weniger komplex
  - B) ICP aktiviert als sekundäres Kriterium → robustere Disambiguierung
- **Impact**:
  - Performance und Implementationsaufwand
  - Robustheit bei ähnlichen Profilen

### 7) Confidence-Parameter `k_conf`
- **Status**: offen, weil absolute Skala von `cost_total` erst nach Implementierung/Tuning stabil ist.
- **Optionen**:
  - A) Startwert 1.0 und empirisch kalibrieren
  - B) k_conf so wählen, dass typische „gute“ Lösung conf≈0.8 und „schlechte“ conf≈0.2 ergibt (datengetrieben)
- **Impact**:
  - Fallback-Trigger Empfindlichkeit

### 8) Overlap-Grenzwerte (Startwerte sind Annahmen)
- **Status**: als Startwerte gesetzt, müssen empirisch validiert werden:
  - `overlap_depth_max_mm_prune = 1.0 mm` (Annahme)
  - `overlap_depth_max_mm_final = 0.1 mm` (Annahme)
- **Impact**:
  - Pruning aggressiv vs. tolerant
  - Refinement Erreichbarkeit

### 9) Definition „Completion“ (bei segmentbasiertem Frontier)
- **Status**: offen, abhängig vom finalen Frontier-Modell.
- **Optionen**:
  - A) „alle Teile platziert + Checks ok“
  - B) zusätzlich „keine offenen Interfaces“ (wenn Interfaces formal geführt werden)
- **Impact**:
  - Fehlalarme vs. strengere Konsistenz

---

## Nächste Schritte

1. **Konfigurations- und Datenmodelle anlegen**: `MatchingConfig`, `FrameModel`, Segment- und Hypothesen-Modelle, DebugBundle.
2. **Segmentierung + Flatness V1 implementieren**: deterministisch, wenige Segmente, mm-Einheit garantiert.
3. **Rahmenkontakt-Features implementieren**: alle Metriken + Costs + Top-N Hypothesen pro Teil, Debug-Export.
4. **Profil-Extraktion implementieren**: Resampling (N=128 Start), minimaler Smoothing (offen), NCC forward/reversed, Top-k Kandidaten.
5. **Beam-Solver implementieren**: Seeding, Expansion, Soft→Hard Commit, Pruning, Trace/Stats.
6. **Overlap-Modul implementieren**: SAT/MTV für konvexe Polygone, plus gewählte nonkonvex-Strategie (Option A/B/C dokumentieren).
7. **Fallback many-to-one implementieren**: Trigger über Confidence, chain_len=2 Start, Debug-Vergleich.
8. **Pose-Refinement implementieren**: Optimierer wählen (Optionen), Overlap-Barriere, Final-Checks.
9. **Integration in Simulator**: Ausgabe-Schnittstelle, Visualisierung, Debug-Export pro Run, Vergleichsläufe/Regression.