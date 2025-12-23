## MatchingConfig und Parameter

### Ziel
Alle Toleranzen, Gewichte, Limits und „bewusst offenen“ Werte werden zentral gebündelt, so dass:
- Variantenläufe (manuell oder später automatisiert) möglich sind,
- Debug-Ausgaben stets die verwendete Konfiguration enthalten,
- keine Schwellenwerte im Code verstreut sind.

### Struktur (Vorschlag)
`MatchingConfig` enthält mindestens folgende Gruppen. Feldnamen sind Vorschläge; entscheidend ist Konsistenz.

#### 1) Rahmenkontakt / Frame-first
- `frame_band_mm` (t): Bandbreite für „Kontaktband“ um eine Rahmenkante.
  - **Start**: 1.0 mm (konfigurierbar)
- `frame_angle_deg` (alpha): Richtungsfenster für Segment-Orientierung vs. Rahmenkante.
  - **Start**: 10° (konfigurierbar)
- `min_frame_seg_len_mm`: Mindestlänge eines Segmentes, das als Rahmenkontakt-Kandidat gelten darf.
  - **Start**: 10 mm (konfigurierbar)
- `tau_frame_mm`: Toleranzband für Inside/Outside (Pruning-Stabilität).
  - **Start**: 2.0 mm (fixer Startwert, konfigurierbar)
- `frame_weights`: Gewichte für Aggregation der Rahmenmetriken (Costs).
  - Metriken siehe Abschnitt „Scoring und Debug-Ausgaben“.
- `penalty_missing_frame_contact`: Penalty, wenn ein Teil keine ausreichende Rahmenhypothese hat (Soft-Constraint).
  - **Start**: offen (tuning), aber Feld muss existieren.

#### 2) Segmentierung
- `target_seg_count_range`: gewünschter Segmentbereich pro Teil (grob, nicht zu fein).
  - **Start**: z.B. (4, 12) (konfigurierbar)
- `curvature_split_params`: Parameter für Split-Kriterien (z.B. Schwellen für Krümmungsmaxima).
  - **Start**: offen (tuning)
- `merge_min_len_mm`: Mindestlänge für Merge; kann an `min_frame_seg_len_mm` gekoppelt oder separat sein.

#### 3) 1D-Profil / Feature-Extraktion
- `profile_samples_N`:
  - **Start**: 128 (konfigurierbar; in Doku explizit als Startannahme vermerken)
- `profile_smoothing_window`:
  - **Status**: bewusst offen (Optionen: 3 / 5 / adaptiv)
- `profile_cost_method`:
  - Optionen: normalized cross-correlation / DTW (falls nötig)
  - Start: Cross-correlation (einfacher, schneller), DTW optional als Erweiterung.

#### 4) Innenmatching / Kandidaten
- `topk_per_segment`: Anzahl der besten Kandidaten pro Segment, die in den Solver gehen.
  - **Start**: z.B. 10 (konfigurierbar)
- `inner_weights`: Gewichte für Aggregation der Innenmatch-Kosten:
  - z.B. `w_profile`, `w_length`, `w_fit`
- `enable_icp`: bool, ob ICP-Fit-Kosten genutzt werden.
  - **Start**: optional (abhängig von Performance/Qualität)

#### 5) Solver (Multi-Hypothesen)
- `beam_width`: maximale Anzahl Hypothesen pro Iterationsstufe.
  - **Start**: z.B. 20..50 (konfigurierbar)
- `max_expansions` / `max_iterations`:
  - Schutz gegen Laufzeitexplosion (konfigurierbar)
- `commit_policy` (Soft→Hard Regel):
  - Frame-Hypothesen werden pro Branch committed, wenn verwendet.

#### 6) Overlap / Kollision
- `overlap_method`: "SAT_MTV" (gesetzt)
- `overlap_depth_max_mm_prune`:
  - **Start**: 1.0 mm (konfigurierbar; als plausibler Startwert markiert)
- `overlap_depth_max_mm_final`:
  - **Start**: 0.1 mm (konfigurierbar; als plausibler Startwert markiert)
- `polygon_nonconvex_strategy`:
  - Optionen: convex decomposition / triangulation / library-approx
  - **Status**: bewusst offen; Wahl muss dokumentiert werden.

#### 7) Confidence / Fallback
- `confidence_mapping`: `exp(-k_conf * cost_total)`
- `k_conf`:
  - **Status**: bewusst offen (Start z.B. 1.0; tuning)
- `fallback_conf_threshold`:
  - **Start**: 0.5 (konfigurierbar; als Startannahme vermerken)
- `enable_many_to_one_fallback`: bool (Start: true)

#### 8) Debug / Logging
- `debug_topN_frame_hypotheses_per_piece`: z.B. 5
- `debug_topN_inner_candidates_per_segment`: z.B. 5
- `export_debug_json`: bool
- `debug_run_id` / `random_seed` (falls relevant)

### Parameter-Policy (wichtig)
- **Kein** implizites Hardcoding ausser in Default-Initialisierung.
- Jeder Lauf speichert:
  - vollständige Config (serialisiert),
  - Input-Meta (n, Flächenbilanz-Score),
  - finale Auswahl + Confidence.

### Offene Parameter (bewusst offen lassen, aber im Config-Feld vorhanden)
- `T_MF` (im FrameModel oder Config referenziert)
- `corner_radius_mm` (FrameModel)
- `profile_smoothing_window`
- `k_conf`
- `polygon_nonconvex_strategy`
- `penalty_missing_frame_contact` (Startwert offen, aber Pflichtfeld)



## Algorithmus-Konzept

### Überblick (Phasen)
1. **Sanity/Meta**: Teileanzahl, Flächenbilanz-Score (Debug).
2. **Coarse Segmentierung**: wenige, stabile Kontursegmente pro Teil; Flatness-V1.
3. **Frame-first Hypothesen**: pro Segment und Rahmen-Seite Metriken berechnen, zu Costs aggregieren, Top-N pro Teil behalten.
4. **Innenmatching (direkt 1:1)**: Top-k Segment↔Segment Kandidaten über 1D-Profil (ggf. + ICP).
5. **Globaler Solver (Multi-Hypothesen)**: Beam-Search über Platzierungen/Constraints; Soft→Hard Commit pro Branch; früh prunen via Rahmen/Overlap.
6. **Fallback many-to-one**: nur wenn globale Confidence < Threshold.
7. **Pose-Refinement**: kontinuierliche Optimierung der Posen (Kosten minimieren, Overlap→0).
8. **Final Checks + Output**: Konsistenz, Overlap, Coverage; Ausgabe in F und optional M.

---

### Phase 0: Sanity/Meta (Debug)
- `n = len(pieces)` muss in `{4,5,6}` liegen; sonst als Debug-Warnung (kein harter Abbruch).
- Flächenbilanz-Score:
  - `area_frame = 128 * 190`
  - `area_sum = Σ area(piece_i)`
  - `area_ratio = area_sum / area_frame`
  - **Nutzung**: Debug/Ranking, nicht als Filter (vorerst).

---

### Phase 1: Coarse Segmentierung + Flatness V1
**Ziel**: Segmentierung nicht zu fein, damit Matching-Raum kontrollierbar bleibt, aber genug Struktur zur Erkennung von Rahmenkontakt/Innenmatches.

- Split-Kandidaten:
  - Krümmungsmaxima / grosse Richtungswechsel entlang Kontur.
- Merge:
  - Segmente < `min_frame_seg_len_mm` werden gemerged.
  - Ziel: Segmentanzahl in `target_seg_count_range` (z.B. 4..12).
- Flatness V1 pro Segment:
  - `flatness_error = RMS(point_to_chord_distance)`
  - wird als Feature sowohl für Rahmenkontakt als auch für „Segment ist eher innen“ genutzt.

**Output**: `segments_all: list[ContourSegment]` (mm, stabile IDs)

---

### Phase 2: Frame-first Hypothesen (Rahmenkontakt als Soft-Constraint)
**Ziel**: Für jedes Teil mehrere plausible Rahmenplatzierungen erzeugen; idealerweise hat jedes Teil mindestens eine gute Hypothese (aber Soft).

Für jedes Segment und jede Rahmen-Seite (TOP/BOTTOM/LEFT/RIGHT):
1. Berechne Features (Metriken) gegen die entsprechende Rahmenlinie:
   - `dist_mean`, `dist_p90`, `dist_max`
   - `coverage_in_band (±t)`
   - `inlier_ratio`
   - `angle_diff`
   - `flatness_error`
   - (optional) corner/global consistency later
2. Aggregiere zu `cost_frame = Σ w_k * cost_k(features_k)`
3. Schätze `pose_grob_F` (initiale Platzierung im Rahmen-KS) aus Segment+Seite (Definition bewusst offen; als Modul kapseln).
4. Speichere Hypothese inkl. vollständigem Feature-Dump.

**Selektion**:
- Behalte Top-N Hypothesen pro Teil (z.B. N=5..10).
- Fehlende gute Hypothesen → später Penalty (Soft).

**Output**: `frame_hypotheses_by_piece`

---

### Phase 3: Innenmatching (direkt 1:1 Segment↔Segment)
**Ziel**: Kandidatenmenge reduzieren, aber robuste Similarity.

Für jedes Segment `a`:
- Kandidaten `b` aus anderen Teilen:
  - Prefilter: Längenfenster (konfigurierbar) und evtl. Flatness/Frame-Likelihood (Innen bevorzugt).
- Ähnlichkeit:
  - Erzeuge 1D-Profil:
    - Resampling auf `N=profile_samples_N` entlang Arclength.
    - Profilwert pro Sample: signed chord distance (Vorzeichen konsistent über Segmentrichtung).
    - Optional milde Glättung (`profile_smoothing_window`, offen).
  - Similarity:
    - Normalisierte Kreuzkorrelation (NCC) von `profile_a` mit `reverse(profile_b)` (und ggf. auch nicht-reversed testen).
    - `profile_cost = 1 - max_corr`
- Optional: `fit_cost` via ICP (abschaltbar).
- Aggregation: `cost_inner = w_profile*profile_cost + w_length*length_cost + w_fit*fit_cost`
- Speichere Top-k Kandidaten pro `a`.

**Output**: `inner_candidates[a_ref] = list[InnerMatchCandidate]`

---

### Phase 4: Globaler Solver (Beam / Multi-Hypothesen)
**Ziel**: Konsistente globale Lösung ohne Rasterannahmen, robust gegen lokale Fehlmatches.

**State** enthält:
- Teilposen (im Rahmen-KS)
- committed Rahmenhypothesen (Soft→Hard pro Branch)
- offene Frontier (welche Inneninterfaces noch zu matchen sind)
- akkumulierte Kosten + Penalties

**Soft→Hard Regel**:
- Rahmenkontakt ist global soft.
- In einem Branch wird eine Rahmenhypothese **committed**, sobald sie zur Platzierung/Initialisierung eines Teils genutzt wird.
- Widerspruchspruning gilt nur gegen committed Constraints innerhalb desselben Branch.

**Expansionen** (Beispiele):
- Platziere ein noch unplatziertes Teil anhand einer Rahmenhypothese (seed/anchor).
- Platziere ein Teil anhand eines Innenmatch-Kandidaten zu einem bereits platzierten Teil (relativer Transform, pose update).

**Pruning**:
- Outside-Check: Teilkontur muss innerhalb Rahmen (mit `tau_frame_mm`).
- Overlap: `penetration_depth_max(state) > overlap_depth_max_mm_prune` → prune.
- Widerspruch zu committed Rahmenhypothesen → prune.

**Ranking**:
- `cost_total = frame_cost + inner_cost + penalties`
- Beam behält die `beam_width` besten States pro Iteration.

**Output**:
- Liste vollständiger Kandidatenlösungen (oder beste Teil-Lösungen + Diagnose)

---

### Phase 5: Confidence-Berechnung und Fallback
- Wähle beste Lösung: `argmin(cost_total)`
- Mappe zu `confidence`:
  - `conf = exp(-k_conf * cost_total)` (k_conf offen, konfigurierbar)
- Wenn `conf < fallback_conf_threshold`:
  - aktiviere many-to-one Matching (Phase 6) und rerun (oder erweitere Kandidatenraum).

---

### Phase 6: Fallback many-to-one (adjacent concatenation)
**Ziel**: Fälle abdecken, in denen Segmentierung eine Kante in mehrere Segmente splittet, die zusammen zu einer Gegenseite passen.

- Erzeuge zusammengesetzte Segmente:
  - `(A1+A2)` nur wenn A1 und A2 benachbart auf derselben Teilkontur sind.
  - Optional Länge/Flatness Gate.
- Wiederhole Kandidatenbildung und Solverlauf.
- Debug: explizit loggen, dass Fallback aktiv war.

---

### Phase 7: Pose-Refinement (kontinuierlich)
**Ziel**: Endoverlap praktisch 0 und Kantenfit maximieren.

- Variablen: Pose jedes Teils (x,y,theta) im Rahmen-KS.
- Ziel: Minimierung
  - Rahmenkontaktkosten (für committed/gewichtete Kontakte)
  - Innenmatchkosten (Profil/ICP-basiert)
  - Overlap-Barrier (penalty bei penetration_depth > 0)
- Abbruch: Konvergenz oder Iterationslimit.
- Final Check:
  - `penetration_depth_max <= overlap_depth_max_mm_final`

---

### Phase 8: Final Checks und Output
- Rahmenabdeckung / Konsistenz:
  - Coverage pro Seite (diagnostisch)
- Innenkanten:
  - keine offenen Interfaces (oder definierter Rest, wenn segmentbasiert)
- Overlap:
  - penetration depth unter finalem Grenzwert
- Ausgabe:
  - Posen in Rahmen-KS (F)
  - wenn `T_MF` gesetzt: transformiere nach Maschinen-KS (M)
- Debug Export:
  - Config + Metriken + Solver-Trace + finale Checks