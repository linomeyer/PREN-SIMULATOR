# Kantenabgleich

Schlägt komplementäre Kantenpaare zwischen verschiedenen Teilen anhand eines Mehrfach-Scores und einfacher Regeln vor.

Dateien im Paket:
- edge_matching/edge_matcher.py
- edge_matching/edge_matcher_visualizer.py

## Hauptklasse: EdgeMatcher
Eingaben
- `EdgeDetector`: liefert Kanten pro Teil mit Merkmalen (`length`, `angle`, `classification`, `shape_signature`).

Ausgaben
- Eine rangierte Liste von `EdgeMatch`-Objekten sowie Hilfsstatistiken (z. B. abgeleitete Randteile aus `flat`-Kanten).

## Matching-Strategie
1. Kandidatenfilter über Klassenkompatibilität
   - Bevorzugt `flat` ↔ `flat` für Ränder, `tab` ↔ `slot` für innere Verbindungen.
   - Reduziert Vergleichszahl stark und erhöht die Präzision.

2. Zusammengesetzter Score pro Kandidatenpaar
   - Formähnlichkeit: Korrelation der krümmungsbasierten Formsignaturen; Vergleich ggf. gespiegelt, um Richtung zu berücksichtigen.
   - Längenähnlichkeit: bestraft grosse Längendifferenzen (z. B. Verhältnis oder exponentielle Abnahme).
   - Klassenkompatibilität: belohnt erwartete Komplementarität (tab/slot) bzw. Gleichheit bei `flat`.

3. Eindeutige beste Treffer
   - Pro Kante wird nur der jeweils beste Partner oberhalb eines Mindest-Scores (`min_score`) behalten, um Mehrfachverwendungen zu vermeiden.

Warum dieser Ansatz
- Leichtgewichtig und erklärbar; harmoniert mit PCA-/Krümmungsfeatures und verhindert früh viele-zu-eins-Konflikte.

## Tuning und Parameter
- `min_score` (Route-Query): filtert schwache/zweifelhafte Matches. Höher setzen für mehr Präzision, niedriger für mehr Recall.
- Relative Gewichte: Formähnlichkeit dominiert typischerweise; Länge und Klassenzwang dienen als Plausibilitätscheck.

## Visualisierung
- `MatchVisualizer.visualize_best_matches(...)` erzeugt nebeneinander/überlagert dargestellte Hinweise:
  - Markierte Kanten mit Typen und Scores.
  - Nützlich zum Debugging von False Positives und zur Schwellwertanpassung.

## Grenzen
- Bei unzuverlässiger Kantenklassifikation kann die Klassenkompatibilität vorübergehend gelockert werden – jedoch mit Strafterm, um offensichtliche Fehlpaarungen zu vermeiden.
- Sehr ähnliche Formen über mehrere Kanten erzeugen Bindungen; die Eindeutigkeitsregel hilft, ggf. `min_score` erhöhen.
