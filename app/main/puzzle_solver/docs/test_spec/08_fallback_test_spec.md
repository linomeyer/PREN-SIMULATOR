# 08_fallback_test_spec.md
Test-Spezifikation Schritt 8: Confidence + Fallback (Many-to-one)

## 1. Übersicht
**Ziel:** Robustheit fuer Low-Confidence-Loesungen via Many-to-one Fallback (adjazente Segment-Ketten) mit genau 1 Rerun.

**Definitionen:**
- Confidence: `conf = exp(-k_conf * cost_total)`
- Trigger: `conf < fallback_conf_threshold`
- Many-to-one: `(A1+...+Ak) ↔ B` mit adjazenten Segmenten entlang derselben Kontur (k=2/3)

**Toleranzen:**
- `TOL_CONF = 1e-3`
- `TOL_COST = 1e-6`
- `TOL_LEN_MM = 1e-3` (Segmentlaengen)

**Config (Step8, muss existieren und ist zu testen):**
- `k_conf: float` (Default 1.0)
- `fallback_conf_threshold: float` (Default 0.5)
- `many_to_one_max_chain_len: int` (Default 2; Option 3)
- `many_to_one_enable: bool` (Default true)
- `many_to_one_enable_only_if_triggered: bool` (Default true)

---

## 2. Confidence-Berechnung (Unit-Tests)

### Test 1: cost_total=0 -> conf=1.0
**Setup:** cost_total=0.0, k_conf=1.0  
**Expected:** abs(conf-1.0) <= TOL_CONF

### Test 2: cost_total=ln(2) -> conf~0.5
**Setup:** cost_total=0.693147, k_conf=1.0  
**Expected:** 0.499 <= conf <= 0.501

### Test 3: k_conf Variation wirkt korrekt
**Setup:** cost_total=0.693147, k_conf=0.5  
**Expected:** 0.706 <= conf <= 0.708

### Test 4: cost_total=10 -> conf≈0
**Setup:** cost_total=10.0, k_conf=1.0  
**Expected:** 0.0 <= conf <= 5e-4

### Test 5: cost_total=+inf -> conf=0
**Setup:** cost_total=+inf  
**Expected:** conf == 0.0 (oder <= TOL_CONF)

### Test 6: cost_total<0 Policy ist definiert (kein Silent)
**Setup:** cost_total=-1.0  
**Expected (eine muss gelten):**
- (A) Exception (ValueError/AssertionError), oder
- (B) clamp(conf)=1.0 und Debug/Status markiert „invalid/negative cost“

---

## 3. Composite-Segmente (Unit-Tests)

**Annahme:** `create_composite_segments(segments_in_contour_order, config)` erzeugt Composites mit:
- `points_mm` (Join-Duplikate entfernt)
- neu berechneten Features (`length_mm`, `flatness_error`, `direction_angle`, `profile_1d`)
- Referenz auf Quelle: `source_segment_ids`

### Test 7: chain_len=2 erzeugt Composite-Paare

**Setup:**
- 2 Pieces mit unterschiedlichen Längen
- Piece 1: 4 Segmente (lengths [10, 12, 8, 15])
- Piece 2: 3 Segmente (lengths [20, 22, 24])
- max_composites_per_piece_k2 = 6

**Expected:**
- 5 k=2 Composites total (3 von Piece 1, 2 von Piece 2)
- Adjacent pairs only: Piece 1: (s0+s1, s1+s2, s2+s3), Piece 2: (s0+s1, s1+s2)
- Segment IDs: 2000, 2001, 2002, 2003, 2004
- Längen kompatibel für ±20% Filter zwischen Pieces

**Rationale:** Realistic 2-piece fixture enables ±20% length filter validation (same-piece filter blocks intra-piece matches).

### Test 8: chain_len=3 erzeugt Triples

**Setup:**
- Same fixture as Test 7 (2 pieces, 4+3 segments)
- max_composites_per_piece_k3 = 3

**Expected:**
- 8 Composites total (5 k=2 + 3 k=3)
- k=2 Composites: 5 total (as Test 7)
- k=3 Composites: 3 total (Piece 1: s0+s1+s2, s1+s2+s3; Piece 2: s0+s1+s2)
- Segment IDs: 2000-2004 (k=2), 3000-3002 (k=3)
- Length filter applies to all composite↔atomic candidate generation

**Rationale:** Validates both chain lengths simultaneously with realistic multi-piece constraints.

### Test 9: Nicht-adjazent wird nie kombiniert
**Setup:** Segmente s0,s1,s2  
**Expected:** kein Composite aus (s0+s2)

### Test 10: Punkte-Konkatenation entfernt Join-Duplikat
**Setup:** s0 endet mit Punkt P, s1 startet mit P  
**Expected:** composite.points_mm enthaelt P nur einmal; len = len(s0)+len(s1)-1

### Test 11: Laenge wird additiv neu berechnet
**Setup:** s0.length=10, s1.length=12  
**Expected:** 21.999 <= composite.length <= 22.001

### Test 12: Wrap-around Policy ist getestet
**Setup:** Kontur zyklisch (s_last neben s0); Config-Flag `many_to_one_allow_wraparound` (falls implementiert)  
**Expected:** wenn false -> kein (s_last+s0); wenn true -> existiert (s_last+s0)

---

## 4. Fallback-Trigger (Unit-Tests)

### Test 13: conf > threshold -> kein Trigger
**Setup:** conf=0.6, threshold=0.5, many_to_one_enable=true  
**Expected:** fallback_triggered=false

### Test 14: conf < threshold -> Trigger + Reason
**Setup:** conf=0.49, threshold=0.5, many_to_one_enable=true  
**Expected:** fallback_triggered=true; debug.trigger_reason == "confidence_below_threshold"

### Test 15: conf == threshold -> kein Trigger
**Setup:** conf=0.5, threshold=0.5  
**Expected:** fallback_triggered=false (strikt `<`)

### Test 16: many_to_one_enable=false -> nie Trigger
**Setup:** conf=0.1, threshold=0.5, many_to_one_enable=false  
**Expected:** fallback_triggered=false; kein Rerun

### Test 17: multiple solutions -> Trigger basiert auf selektierter Best-Policy
**Setup:** best_solution (nach cost_total) hat conf=0.49, andere conf=0.8  
**Expected:** Trigger erfolgt (weil selektierte Loesung low-conf)

---

## 5. Kandidaten-Erweiterung (Unit-Tests)

**Annahme:** `extend_inner_candidates(original_candidates, atomic_segs, composite_segs, config)`.

### Test 18: Original-Kandidaten bleiben erhalten
**Setup:** candidates={(a0,b0),(a1,b1)}  
**Expected:** candidates_after ⊇ candidates_before

### Test 19: Composite↔Atomic ist bidirektional
**Setup:** Composite A01, Atomic B0  
**Expected:** Kandidaten enthalten (A01,B0) und (B0,A01) (oder aequivalent)

### Test 20: Composite↔Composite ist V1 default: aus
**Setup:** Composite A01 und B01  
**Expected:** kein (A01,B01), sofern nicht explizit konfiguriert

### Test 21: Prefilter (Laenge) blockiert unplausible Paare (Start: ±20%)
**Setup:** atomic_len=10, composite_len=30  
**Expected:** kein Kandidat, da 30 nicht in [8,12]

---

## 6. Rerun-Mechanismus (End-to-End / Integration-nahe Tests)

### Test 22: Rerun verbessert -> Fallback-Loesung wird gewaehlt
**Setup:**
- Run1: cost=1.0, k_conf=1.0 -> conf≈0.367 (Trigger bei threshold=0.5)
- Run2 (mit composites): cost=0.2 -> conf≈0.819
**Expected:**
- finale Loesung == Run2
- status == OK_WITH_FALLBACK
- debug enthaelt before/after cost+conf und composite usage

### Test 23: Rerun schlechter -> Original bleibt
**Setup:** Run1 cost=1.0; Run2 cost=1.2  
**Expected:** finale Loesung == Run1; debug dokumentiert Vergleich

### Test 24: Max 1 Rerun (keine Schleife)
**Setup:** Run1 low-conf; Run2 weiterhin low-conf  
**Expected:** fallback_rerun_count == 1

### Test 25: Debug-Pflichtfelder vorhanden (bei Trigger)
**Expected keys:**
- `fallback_triggered`
- `confidence_before`, `cost_before`
- `confidence_after`, `cost_after`
- `composites_created_per_piece` (oder aequivalent)
- `composite_matches_used` (Liste)
- optional: penalty/Cost-Komponenten

---

## 7. Strategie-Decisions (Dokumentationspflicht)
- **D7 k_conf:** Default 1.0 (Tuning spaeter)
- **D8 threshold:** Default 0.5 (konfigurierbar)
- **D9 chain_len:** V1=2, optional 3
- **D10 rerun:** genau 1x in V1
- **D11 prefilter:** Start ±20% Laengenfenster (dokumentiert)

---

## 8. Implementation-Hinweise
- Neu: `solver/fallback/many_to_one.py`
  - `create_composite_segments(...)`
  - `extend_inner_candidates(...)`
- Anpassung: `solver/beam_solver/solver.py` (confidence check + 1x rerun)
- Anpassung: `solver/config.py` (Config-Felder oben)
