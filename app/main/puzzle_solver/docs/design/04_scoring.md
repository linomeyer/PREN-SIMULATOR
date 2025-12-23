## Scoring und Debug-Ausgaben

### Ziel
Alle verwendeten Metriken werden:
1) **einzeln berechnet**,  
2) **als numerische Werte geloggt**,  
3) **in Costs überführt**,  
4) **aggregiert** (gewichtbar, konfigurierbar).  

Damit ist später per Praxisläufen evaluierbar, welche Kombinationen/Weights robust sind.

---

## Scoring-Konvention (einheitlich)
- Intern wird ausschliesslich **Cost minimiert**.
- Jede Metrik liefert entweder direkt einen Cost oder wird über eine definierte Mapping-Funktion in einen Cost transformiert.
- Gesamt-Cost ist Summe von Teil-Costs plus Penalties:
  - `cost_total = cost_frame + cost_inner + cost_overlap + cost_penalties`
- Debug-Konfidenz:
  - `confidence = exp(-k_conf * cost_total)`  
  - `k_conf` ist konfigurierbar und bewusst offen (Startwert später).

---

## A) Rahmenkontakt-Metriken (pro Segment, pro Rahmen-Seite)

### Rohmetriken (immer im Debug)
Für ein Segment `seg` und eine Rahmen-Seite `side`:

1. **Distanzmetriken** (Segmentpunkte zu Rahmenlinie projiziert)
   - `dist_mean_mm`
   - `dist_p90_mm` (90%-Quantil, robust gegen Ausreisser)
   - `dist_max_mm` (optional fürs Debug; nicht zwingend gewichten)

2. **Coverage im Band**
   - Bandbreite `t = config.frame_band_mm`
   - `coverage_in_band = (L_in_band / L_total)` in [0..1]
   - Längen- oder Arclength-basiert (nicht Punktanzahl), um Sampling-Effekte zu reduzieren.

3. **Inlier Ratio**
   - `inlier_ratio = (#points with |dist|<=t) / #points` in [0..1]
   - Punktbasiert; primär diagnostisch, optional sekundär gewichten.

4. **Richtungskohaerenz**
   - `angle_diff_deg = |wrap(seg.direction_angle - expected_side_angle)|`
   - expected_side_angle: TOP/BOTTOM ~ 0°, LEFT/RIGHT ~ 90° (nur als Orientierung, keine harte 90°-Rotation).

5. **Flatness**
   - `flatness_error_mm = RMS(point_to_chord_distance(seg.points))`
   - Niedrig => segment ist „gerade“.

6. **Support Count (optional)**
   - `support_points_count`: Anzahl Punkte mit gutem Abstand und guter Orientierung (diagnostisch).

---

### Cost-Mappings (definiert, konfigurierbar)
Jede Rohmetrik wird in einen Cost überführt, z.B.:

- `cost_dist_p90 = clamp(dist_p90_mm / t, 0, Cmax)`  
- `cost_coverage = 1 - coverage_in_band`
- `cost_inlier = 1 - inlier_ratio`
- `cost_angle = clamp(angle_diff_deg / alpha, 0, Cmax)` (alpha = frame_angle_deg)
- `cost_flat = clamp(flatness_error_mm / flat_ref, 0, Cmax)` (flat_ref konfigurierbar oder datengetrieben)

**Aggregation**:
- `cost_frame = Σ w_k * cost_k`  
- `frame_weights` in Config.

**Wichtig (Redundanz vermeiden)**:
- `coverage_in_band` und `inlier_ratio` sind verwandt.
- Policy:
  - Option A: nur Coverage gewichten, Inlier nur Debug
  - Option B: Inlier mit sehr kleinem Gewicht
  - Option C: nur eines aktivieren; das andere bleibt Debug-only

Diese Policy wird als Konfigurationsflag geführt, damit Varianten vergleichbar sind.

---

## B) Innenmatching-Metriken (pro Segmentpaar-Kandidat)

### Rohmetriken (immer im Debug)
1. **Längen-Kompatibilität**
   - `len_a_mm`, `len_b_mm`, `len_ratio = min/ max`
2. **Profil-Similarity**
   - 1D-Profil je Segment (Sampling N=profile_samples_N)
   - Similarity über normalisierte Kreuzkorrelation (NCC)
   - Teste beide Orientierungen:
     - `corr_forward = ncc(profile_a, profile_b)`
     - `corr_reversed = ncc(profile_a, reverse(profile_b))`
   - `corr = max(corr_forward, corr_reversed)`
   - `reversal_used = (corr_reversed > corr_forward)`
3. **Profilkosten**
   - `profile_cost = 1 - corr` (in [0..2] je nach NCC-Definition; clamp auf [0..1] falls NCC∈[0,1])
4. **Optional: ICP/Geometrie-Fit**
   - `fit_cost` z.B. RMS nach ICP (mm)
   - nur wenn `enable_icp=True` (Config)

### Cost-Mappings
- `cost_len = clamp(|1 - len_ratio| / len_tol, 0, Cmax)` (len_tol konfigurierbar)
- `cost_profile = profile_cost` (nach definierter Normalisierung)
- `cost_fit = clamp(fit_rms_mm / fit_ref_mm, 0, Cmax)` (fit_ref_mm konfigurierbar)

**Aggregation**:
- `cost_inner = w_profile*cost_profile + w_len*cost_len + w_fit*cost_fit`

---

## C) Overlap-Metrik (global, state-level)
- Overlap wird als **maximale Eindringtiefe** definiert:
  - `penetration_depth_max_mm(state)` über alle Teilpaare.
- Solver-Pruning:
  - `> overlap_depth_max_mm_prune` ⇒ prune
- Final Check:
  - `<= overlap_depth_max_mm_final`

**Debug**:
- max penetration depth
- pair causing max depth
- Anzahl overlap pairs > 0
- optional Histogramm über depths (für Tuning)

---

## D) Penalties (Soft Constraints)
- **Missing Frame Contact Penalty**:
  - Wenn ein Teil keine brauchbare Rahmenhypothese hat:
    - `penalty_missing_frame_contact` addieren
  - Debug: list of affected pieces

- Optional weitere Soft-Penalties:
  - „zu geringe Seiten-Coverage“
  - „zu viele offene Interfaces“
  - „Flächenbilanz-Abweichung“ (vorerst nur Debug/Ranking)

---

## Debug-Ausgaben (Pflichtumfang)

### 1) Pro Run (Header)
- Run-ID, Timestamp
- `n_pieces`
- vollständige `MatchingConfig` serialisiert (inkl. Defaults)
- Rahmenmodell (128×190, corner_radius, tau_frame_mm, T_MF Status)

### 2) Rahmenhypothesen (pro Teil)
Für jedes Teil:
- Top-N Hypothesen (N konfigurierbar)
- pro Hypothese:
  - side, segment_id
  - `cost_frame`
  - alle Rohmetriken (dist_*, coverage, inlier, angle_diff, flatness)
  - `pose_grob_F` (mit KS-Tag)
- Summary:
  - best cost, #hypothesen unter einem Schwellwert (diagnostisch)

### 3) Innenkandidaten (pro Segment)
- pro Segment Top-k Kandidaten:
  - target segment ref
  - `cost_inner`
  - `profile_cost`, `len_ratio`, `fit_cost`
  - `reversal_used`
- Summary:
  - Kandidatenanzahl nach Prefilter
  - Verteilung der Costs

### 4) Solver-Trace / Statistik
- Beam stats pro Iteration:
  - beam size, expansions, prunes
- Pruning reasons count:
  - outside frame
  - overlap prune
  - committed constraint conflict
  - 기타 (falls vorhanden)
- Best state cost progress
- Falls Fallback aktiv:
  - Flag + Grund (confidence < threshold)

### 5) Final Checks
- `cost_total` und Breakdown (frame/inner/penalties/overlap)
- `confidence`
- `penetration_depth_max_mm` (final)
- Seiten-Coverage (diagnostisch)
- offene Interfaces (diagnostisch)
- Flächenbilanz-Score

### 6) Export-Format
- JSON (stabil, maschinenlesbar)
- Optional: zusätzlich CSV-Auszüge für schnelle Analyse
- Optional: Visualisierung-artefacts (Plots/Overlay), referenziert im DebugBundle