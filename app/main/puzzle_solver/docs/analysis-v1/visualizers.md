# Visualisierer

Beschreibt die Hilfsklassen, die Zwischenstände und Ergebnisse als Bilder ausgeben.

Dateien im Paket:
- piece_extraction/extractor_visualizer.py
- edge_detection/edge_detector_visualizer.py
- edge_matching/edge_matcher_visualizer.py
- solver/solver_visualizer.py

## PieceVisualizer
- Visualisiert jede erkannte Teilmaske als RGBA-Bild (transparenter Hintergrund).
- Typische Dateinamen enthalten den Upload-Dateinamen und einen Suffix (z. B. `pieces_...png`).
- Nutzen: schnelle Qualitätskontrolle der Segmentierung (Anzahl, Grösse, Konturglätte).

## EdgeVisualizer
- Zeichnet pro Teil: Kontur, gesnappte Ecken, vier Kanten mit Richtung und Klassenfarbe (`flat/tab/slot`).
- Nutzen: Validierung der Eckreihenfolge, der Kanten-Schnitte und der Klassifikation.

## MatchVisualizer
- Stellt gefundene Kantenpaare dar, inklusive Teil- und Kanten-IDs sowie Scores der Teilmetriken.
- Nutzen: Fehleinschätzungen erkennen, Schwellwerte (`min_score`) feinjustieren.

## SolutionVisualizer
- Zeigt das zusammengesetzte Raster mit platzierten Teilen und Rotationen.
- Erzeugt zusätzlich Renderings, die die Teil-RGBA-Zuschnitte an den Grid-Positionen zusammensetzen.
- Nutzen: Beurteilung der Gesamtqualität und Plausibilität der Lösung.

## Ausgabepfad
- Standardmässig werden Bilder unter `app/static/output` abgelegt und können über die Route `/output/<path>` abgerufen werden.
