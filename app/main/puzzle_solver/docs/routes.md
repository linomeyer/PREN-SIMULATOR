# Anwendungsrouten (Überblick)

Die folgenden Endpunkte orchestrieren die Schritte der Puzzle-Lösung und erzeugen Visualisierungen unter `app/static/output`.
Die Implementierung liegt in `app/main/routes.py`.

Hinweis: Die Dokumentation bleibt inhaltlich auf das Paket `puzzle_solver` fokussiert; die Routen sind hier zusammengefasst, da sie den Ablauf steuern.

## GET `/`
- Rendert die Hauptoberfläche (`index.html`).
- Antwort ist HTML, kein JSON.

## POST `/upload`
Beschreibung
- Lädt ein Quellbild hoch und speichert es unter `app/main/img`.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "filename": "puzzle.jpg",
  "message": "File uploaded successfully"
}
```

## POST `/calibrate`
Beschreibung
- Führt eine Kalibrierung für den `GlobalCleaner` mit hochgeladenen Testbildern durch.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "message": "Calibration complete with 3 test image(s)",
  "calibration_info": {
    "barrel_distortion": {"enabled": true, "k1": -0.28, "k2": 0.04, "k3": 0.0, "fx": 4608, "fy": 4608, "cx": 2304.0, "cy": 1296.0, "image_size": [4608, 2592]},
    "perspective": {"enabled": false, "transform_matrix": null},
    "vignette": {"enabled": true, "strength": 0.2, "center": [2304.0, 1296.0], "radius": 1296.0},
    "chromatic_aberration": {"enabled": false, "r_offset": [0, 0], "b_offset": [0, 0]}
  },
  "parameters": {
    "k1": -0.28,
    "k2": 0.04,
    "k3": 0.0,
    "fx": 4608,
    "fy": 4608,
    "cx": 2304.0,
    "cy": 1296.0,
    "image_size": [4608, 2592]
  }
}
```

## GET `/clean-global/<filename>`
Beschreibung
- Wendet globale Korrekturen mittels `GlobalCleaner` an. Das bereinigte Bild wird für spätere Schritte gecacht.
- Zusätzlich wird ein Vorher/Nachher-Bild geschrieben.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "filename": "puzzle.jpg",
  "calibration_info": {
    "barrel_distortion": {"enabled": true, "k1": -0.28, "k2": 0.04, "k3": 0.0, "fx": 4608, "fy": 4608, "cx": 2304.0, "cy": 1296.0, "image_size": [4608, 2592]},
    "perspective": {"enabled": false, "transform_matrix": null},
    "vignette": {"enabled": true, "strength": 0.2, "center": [2304.0, 1296.0], "radius": 1296.0},
    "chromatic_aberration": {"enabled": false, "r_offset": [0, 0], "b_offset": [0, 0]}
  },
  "images": {
    "before_after": "global_clean_puzzle.jpg"
  }
}
```

## GET `/extract/<filename>`
Beschreibung
- Führt die Teile-Extraktion mit `PieceSegmenter` aus (auf bereinigtem Bild, falls vorhanden).

Optionale Query-Parameter
- `blur_kernel`, `morph_kernel`, `morph_close_iter`, `morph_open_iter` (siehe `piece_extraction.md`).

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "num_pieces": 6,
  "statistics": {
    "num_pieces": 6,
    "avg_area": 152345.7,
    "std_area": 12345.9,
    "min_area": 139876.0,
    "max_area": 168234.0,
    "avg_perimeter": 1689.3
  },
  "pieces": [
    {
      "id": 0,
      "center": [512, 384],
      "bbox": {"x": 430, "y": 320, "width": 180, "height": 210},
      "area": 150234.0,
      "perimeter": 1654.8
    },
    {
      "id": 1,
      "center": [1280, 420],
      "bbox": {"x": 1205, "y": 330, "width": 185, "height": 205},
      "area": 152890.0,
      "perimeter": 1702.1
    }
  ],
  "images": {
    "pieces": [
      "pieces_puzzle_0.png",
      "pieces_puzzle_1.png",
      "pieces_puzzle_2.png"
    ]
  }
}
```

## GET `/clean-pieces/<filename>`
Beschreibung
- Führt lokale Nachbearbeitung der extrahierten Teile mittels `ImageCleaner` aus.
- Erstellt außerdem einen Konturvergleich (rot: original, grün: bereinigt).

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "num_pieces": 6,
  "statistics": {
    "num_pieces": 6,
    "avg_area": 151998.2,
    "std_area": 12100.4,
    "min_area": 139500.0,
    "max_area": 167900.0,
    "avg_perimeter": 1681.6
  },
  "images": {
    "cleaned_pieces": [
      "cleaned_puzzle_0.png",
      "cleaned_puzzle_1.png"
    ],
    "contour_comparison": "contour_comparison_puzzle.jpg"
  }
}
```

## GET `/detect-edges/<filename>`
Beschreibung
- Führt Kantenerkennung mit `EdgeDetector` aus.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "num_edges": 24,
  "statistics": {
    "avg_length": 420.5,
    "class_distribution": {"flat": 8, "tab": 8, "slot": 8}
  },
  "edge_info": [
    {
      "piece_id": 0,
      "edges": [
        {"edge_type": "top", "length": 420.2, "angle": 5.7, "classification": "flat"},
        {"edge_type": "right", "length": 401.1, "angle": 93.4, "classification": "tab"},
        {"edge_type": "bottom", "length": 419.8, "angle": 181.2, "classification": "slot"},
        {"edge_type": "left", "length": 402.7, "angle": 271.0, "classification": "tab"}
      ]
    }
  ],
  "images": {
    "edge_pieces": [
      "edges_puzzle_0.png",
      "edges_puzzle_1.png"
    ]
  }
}
```

## GET `/match-edges/<filename>`
Beschreibung
- Sucht eindeutige beste Kantenpaare mit `EdgeMatcher`.

Optionale Query-Parameter
- `min_score` (float): Mindestscore zum Filtern schwächerer Matches.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "num_matches": 10,
  "statistics": {
    "border_pieces": 4,
    "corner_pieces": 2
  },
  "matches": [
    {
      "edge1": {
        "piece_id": 0,
        "edge_type": "right",
        "classification": "tab",
        "length": 401.1
      },
      "edge2": {
        "piece_id": 1,
        "edge_type": "left",
        "classification": "slot",
        "length": 399.9
      },
      "scores": {
        "compatibility": 0.92,
        "length_similarity": 0.98,
        "shape_similarity": 0.90,
        "classification_match": true
      },
      "rotation": {"degrees": 0.0}
    }
  ],
  "images": {
    "match_visualizations": [
      "matches_puzzle_0.png",
      "matches_puzzle_1.png"
    ]
  }
}
```

## GET `/solve-puzzle/<filename>`
Beschreibung
- Führt den kompletten Lösungsschritt aus: Kanten erkennen, Kanten abgleichen (unter Beachtung von `min_score`) und Puzzle via `PuzzleSolver` zusammensetzen.

Antwort (vollständiges JSON‑Beispiel)
```
{
  "success": true,
  "solution": {
    "grid_rows": 2,
    "grid_cols": 3,
    "confidence": 0.86,
    "pieces_placed": 6,
    "total_pieces": 6,
    "placed_pieces": [
      {"piece_id": 0, "grid_position": {"row": 0, "col": 0}, "rotation": 0.0},
      {"piece_id": 1, "grid_position": {"row": 0, "col": 1}, "rotation": 90.0}
    ],
    "grid_layout": [[0,1,2],[3,4,5]],
    "matches_used": 5
  },
  "images": {
    "solution_visualizations": [
      "solution_grid_puzzle.png",
      "solution_render_puzzle.png"
    ]
  }
}
```

## GET `/output/<path>`
Beschreibung
- Liefert generierte Visualisierungen aus dem Output-Ordner aus (PNG/JPG-Dateien). Antwort ist Binärbild, kein JSON.

## Fehlerfälle (Beispiele)
```
{"error": "File not found"}
{"error": "Please extract pieces first"}
{"error": "No file provided"}
```
