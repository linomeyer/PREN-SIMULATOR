# Architektur: puzzle_solver

Diese Dokumentation beschreibt die Architektur des Pakets `puzzle_solver`. Sie deckt die Verarbeitungspipeline, zentrale Module und Algorithmen ab sowie die Anwendungsrouten, die die Schritte orchestrieren.

Hinweis: Alle hier beschriebenen Implementierungen liegen im Paket `puzzle_solver`. Die Routen befinden sich in `app/main/routes.py` und sind als Überblick enthalten, da sie die Pipeline starten.

## Inhaltsverzeichnis
- Überblick und Datenfluss
- Kapitel
  - Vorverarbeitung (Was und Warum) — siehe `preprocessing.md`
  - Teile-Extraktion (Segmentierung) — siehe `piece_extraction.md`
  - Kantenerkennung (Ecken, Kanten, Features, Klassifikation) — siehe `edge_detection.md`
  - Kantenabgleich (Bewertung, Kompatibilität, Eindeutigkeit) — siehe `edge_matching.md`
  - Solver (Layout-Aufbau und Rotationen) — siehe `solver.md`
  - Visualisierer (Bilder und Interpretation) — siehe `visualizers.md`
- Anwendungsrouten — siehe `routes.md`

## Überblick und Datenfluss
1. Vorverarbeitung (optional): Globale Bildkorrekturen (Tonnenverzerrung, Perspektive, Vignettierung, chromatische Aberration).
2. Teile-Extraktion: Segmentierung der Puzzleteile und Berechnung grundlegender Eigenschaften pro Teil.
3. Kantenerkennung: Pro Teil Ecken finden, Kantenpolylinien extrahieren und Kantenformen klassifizieren.
4. Kantenabgleich: Kanten-Signaturen vergleichen und komplementäre Paare vorschlagen.
5. Lösen: Endlayout durch Platzieren/Rotieren anhand Match-Graph und Klassifikationen aufbauen.

Artefakte jedes Schritts werden von dedizierten Visualizer-Klassen erzeugt und unter `app/static/output` gespeichert (bei Aufruf über die Routen).

## Umfang
Der Fokus liegt ausschließlich auf dem Paket `puzzle_solver`. Die Routen werden als Überblick dargestellt, Details verbleiben im Paket.

## Grobes Klassendiagramm (Pakete und Dateien)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                            puzzle_solver (Kern)                              │
│                                                                              │
│                                                                              │
│  ┌─────────────────────┐   ┌────────────────────────┐   ┌──────────────────┐ │
│  │ preprocessing/      │   │ piece_extraction/      │   │ edge_detection/  │ │
│  │ - GlobalCleaner     │   │ - PuzzlePiece          │──▶│ - PieceEdge      │ │
│  │   (global_cleaner)  │   │ - PieceSegmenter       │   │ - EdgeDetector   │ │
│  │ - ImageCleaner      │   │   (extractor)          │   │   (edge_detector)│ │
│  │   (image_cleaner)   │   └────────────────────────┘   └──────────────────┘ │
│  └─────────────────────┘               │                       │             │
│           ▲                            │                       │             │
│           │                            ▼                       │             │
│           │                 ┌────────────────────────┐         │             │
│           │                 │ edge_matching/         │         │             │
│           └────────────────▶│ - EdgeMatcher          │◀────────┘             │
│                             │   (edge_matcher)       │                       │
│                             └────────────────────────┘                       │
│                                            │                                 │
│                                            ▼                                 │
│                             ┌────────────────────────┐                       │
│                             │ solver/                │                       │
│                             │ - PuzzleSolver         │                       │
│                             │ - PuzzleSolution       │                       │
│                             │   (solver)             │                       │
│                             └────────────────────────┘                       │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                        Visualizer                                            │
│                                                                              │
│  piece_extraction/extractor_visualizer.py   →  PieceVisualizer               │
│  edge_detection/edge_detector_visualizer.py →  EdgeVisualizer                │
│  edge_matching/edge_matcher_visualizer.py   →  MatchVisualizer               │
│  solver/solver_visualizer.py                →  SolutionVisualizer            │
│                                                                              │
│  Nutzen: Beobachten/Visualisieren der Schritte; schreiben PNGs nach          │
│          app/static/output                                                   │
└──────────────────────────────────────────────────────────────────────────────┘

Datenfluss (vereinfacht):
preprocessing.global_cleaner → (optional) korrigiertes Bild → piece_extraction.extractor →
edge_detection.edge_detector → edge_matching.edge_matcher → solver.solver

Hinweis: Visualizer hängen sich an die jeweiligen Schritte an und erzeugen Artefakte in
`app/static/output`.
```
