# Solver

Setzt das Puzzle-Layout aus den Kanten-Matches zusammen, berechnet Rotationen und ordnet Teile auf einem Raster an.

Dateien im Paket:
- solver/solver.py
- solver/solver_visualizer.py

## Hauptklasse: PuzzleSolver
Eingaben
- `EdgeMatcher` mit gefundenen Paaren zwischen Kanten.
- `List[PuzzlePiece]` aus der Teile-Extraktion.

Ausgaben
- `PuzzleSolution` mit platzierten Teilen, Grid-Größe, Konfidenz und verwendeten Matches.
- Visualisierungen über `SolutionVisualizer`.

## Vorgehen (vereinfacht)
1. Rand- und Eckteile bestimmen
   - Über die Anzahl `flat`-Kanten pro Teil (2 → Ecke, 1 → Rand, 0 → Innen).
   - Warum: ermöglicht Start an einer stabilen Ecke bzw. am Rand, reduziert Freiheitsgrade.

2. Orientierung ableiten
   - Aus den von der Kantenerkennung gelieferten Kantenwinkeln (PCA-Winkel) die Grundrotation des Teils schätzen.
   - Warum: bringt Kanten näher an die vier Hauptpositionen (oben/rechts/unten/links) für das Grid.

3. Layout schrittweise aufbauen
   - Von einer Ecke ausgehend passende Nachbarn über die besten Matches auswählen.
   - Positionierung im Grid: die übereinstimmenden Kanten bestimmen die Zielseite und damit die Rotation des Nachbarteils.
   - Konsistenzprüfungen (z. B. bereits belegte Zellen, widersprüchliche Rotationen) verhindern Konflikte.

4. Abschätzen der Lösungsgüte
   - Konfidenz aus verwendeten Match-Scores und Konsistenz des resultierenden Rasters ableiten.

## Rotationen und Kantenpositionen
- Hilfsmethoden ordnen die ursprünglichen Kanten eines Teils nach einer Rotation den Grid-Seiten (oben/rechts/unten/links) zu.
- So lässt sich aus „Kante A des Teils passt zu Kante B des Nachbarn“ die benötigte Drehung und die Zielzelle ableiten.

## Visualisierung
- `SolutionVisualizer.visualize_solution(solution, pieces, filename, edge_detector)` erzeugt:
  - Übersicht des Grids mit platzierten IDs und Rotationen.
  - Renderings mit zusammengesetzten Teilebildern an den Grid-Positionen.

## Grenzen
- Fehlklassifizierte `flat`-Kanten oder schwache Matches können zu Fehllagen führen; ein höheres `min_score` aus dem Matching hilft.
- Ohne ausreichend eindeutige Matches kann keine vollständige Lösung gefunden werden.
