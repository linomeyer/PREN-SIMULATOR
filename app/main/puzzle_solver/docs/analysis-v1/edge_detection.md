# Kantenerkennung

Findet pro Teil die Ecken, schneidet die Kontur in vier Kanten und berechnet Merkmale für Klassifikation und Matching.

Dateien im Paket:
- edge_detection/edge_detector.py
- edge_detection/edge_detector_visualizer.py

## Wichtige Typen
- `PieceEdge` (eine Instanz pro Seite):
  - `piece_id`, `edge_type` (`top`, `right`, `bottom`, `left`)
  - `points` (Polyline aus der Teil-Kontur)
  - `start_point`, `end_point` (Eckkoordinaten)
  - Abgeleitete Merkmale: `length`, `angle`, `shape_signature`

- `EdgeDetector`:
  - Input: `List[PuzzlePiece]`
  - Output: `List[PieceEdge]` und `piece_edges[piece_id][edge_type]`

## Schritte
1. Eckenermittlung (Minimale-Umgebungs-Rechteck → „gesnappte“ Ecken)
   - `cv2.minAreaRect(contour)` berechnen; 4 Rechteckpunkte mit `cv2.boxPoints` ermitteln.
   - Konsistente Reihenfolge: oben-links → oben-rechts → unten-rechts → unten-links (Sortierung nach y, dann x).
   - Für jede ideale Rechteck-Ecke den nächstgelegenen Konturpunkt suchen (Euklidisch). Dadurch liegen die Ecken genau auf der echten Kontur, die rechteckige Struktur bleibt erhalten.
   - Warum: Das rotierte Bounding-Box-Modell ist robust gegenüber Tabs/Slots und liefert eine stabile Orientierung; Snapping sorgt für realitätsnahe Kanten.

2. Kantenextraktion (kürzerer Weg entlang der Kontur)
   - Für jedes benachbarte Eckpaar die Indizes auf der Kontur finden und entlang des kürzeren Bogens (vorwärts/rückwärts) laufen. So entstehen vier Polylinien.
   - Warum: Stellt sicher, dass jede Kante dem nächstliegenden Kontursegment entspricht und nicht dem langen Weg.

3. Merkmalsberechnung auf `PieceEdge`
   - Länge: `cv2.arcLength(points, False)`; dient Grössen-/Kompatibilitätsvergleich.
   - Winkel: PCA auf Kantenpunkten für robuste Orientierungsbaseline; Normalisierung auf [0°, 360°]. Richtung entspricht `start_point → end_point`.
     - Warum PCA: stabiler als reine Endpunkt-Steilheit bei gekrümmten Verläufen; weniger rauschanfällig.
   - Formsignatur (krümmungsbasiert): gleichmässige Abtastung (Standard 50), Winkeldifferenzen zwischen Segmenten berechnen, auf [-π, π] normieren.
     - Warum: erfasst das Muster aus Auswölbung/Einbuchtung unabhängig von Position/Rotation.

4. Kantenklassifikation: flat vs tab vs slot
   - Senkrechte Abstände der Kantenpunkte zur Geraden (`start`→`end`) berechnen.
   - Kennzahlen: maximale Abweichung, Standardabweichung, Anteil signifikanter Abweichungen (>3% der Basislänge), sowie vorzeichenbehaftete Fläche unter der Abstandskurve.
   - Entscheidung:
     - Globale Abweichungen klein → `flat`.
     - Sonst: Vorzeichen von (0.7×Fläche + 0.3×signierter Maximalabweichung) → `tab` bei positiv, `slot` bei negativ.
   - Warum: kombiniert globale und lokale Evidenz für robuste Klassifikation trotz Rauschen.

## Ausgaben
- Pro Teil: Mapping `piece_edges[piece_id] = {top,right,bottom,left}`.
- Global: Liste `edges` aller `PieceEdge`.
- Statistiken: Zählungen, Mittelwerte (Längen, Winkeldistribution), Klassenverteilung.

## Visualisierung
- `EdgeVisualizer.visualize_piece_edges(segmenter, edge_detector, filename)` zeichnet:
  - Teilkonturen, gesnappte Ecken, Kanten-Polylinien mit Richtung/Labels und Klassenfarbe.
  - Hilfreich zur Prüfung von Eckreihenfolge und Kanten-Schnitt.

## Grenzen
- Sehr gezackte Konturen können den PCA-Winkel beeinträchtigen; `ImageCleaner` hilft.
- Sehr tiefe Tabs/Slots können gesnappte Ecken leicht verschieben; insgesamt robust für typische Jigsaw-Schnitte.
- Funktioniert nur für Rechteckige Teile konsistent
