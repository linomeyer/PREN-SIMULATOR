# Teile-Extraktion

Extrahiert einzelne Puzzleteile aus einem Gesamtbild und erzeugt `PuzzlePiece`-Objekte mit Geometrie und RGBA-Zuschnitt.

Dateien im Paket:
- piece_extraction/extractor.py
- piece_extraction/extractor_visualizer.py

## Hauptklassen
- `PuzzlePiece`: enthält `contour`, `mask`, `bbox (x,y,w,h)`, `image (RGBA)`, `center`.
- `PieceSegmenter`: führt die Segmentierungspipeline aus und liefert eine Liste von `PuzzlePiece`.

## Pipeline
1. Grauwertkonvertierung: `cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)`
   - Warum: Segmentierung arbeitet auf Helligkeit; reduziert Störungen durch Farbvariationen.
2. Entrauschung (Gaussian Blur): Kernel = `blur_kernel` (Standard 7)
   - Warum: glättet kleine Schwankungen für stabileres Otsu-Thresholding.
3. Thresholding (Otsu, invertiert): `cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`
   - Warum: wählt den Schwellwert automatisch; Invertierung macht (typisch dunklere) Teile zum Vordergrund (255).
4. Morphologie: quadratischer Kernel `morph_kernel` (Standard 5)
   - Closing-Iterationen `morph_close_iter` (Standard 3): schliesst Lücken in Teilregionen.
   - Opening-Iterationen `morph_open_iter` (Standard 1): entfernt kleine Störpixel.
   - Warum: erzeugt sauberere, zusammenhängende Masken für Konturen.
5. Konturerkennung: `cv2.findContours(..., cv2.RETR_EXTERNAL, ...)`
   - Es werden nur äussere Konturen behalten; Flächen werden per `[min_area, max_area]` gefiltert.
6. RGBA-Zuschnitt: Bounding Box berechnen; Bildausschnitt in RGBA; Alphakanal aus Masken-ROI.
   - Warum: Originaloptik bleibt erhalten, Hintergrund ist transparent – hilfreich für Visualisierung/Platzierung.

## Einstellbare Parameter (via /extract Query)
- `blur_kernel`: ungerade Zahl ≥ 3 (Standard 7)
- `morph_kernel`: ungerade Zahl ≥ 3 (Standard 5)
- `morph_close_iter`: Ganzzahl ≥ 0 (Standard 3)
- `morph_open_iter`: Ganzzahl ≥ 0 (Standard 1)

Hinweise zur Wahl
- `blur_kernel` erhöhen bei texturiertem Hintergrund; verringern bei feinen Details nahe der Kante.
- `morph_close_iter` erhöhen, um kleine Lücken zu schliessen; `open`-Iterationen erhöhen gegen Salz-und-Pfeffer-Rauschen.

## Ausgaben und Statistiken
- `segment_pieces()` liefert eine Liste von `PuzzlePiece`.
- `get_piece_statistics()` liefert `num_pieces`, `avg_area`, `std_area`, `min_area`, `max_area`, `avg_perimeter`.
- Visuals: `PieceVisualizer.visualize_pieces()` schreibt Teilbilder nach `app/static/output`.

## Warum dieser Ansatz
- Otsu + Morphologie ist robust und schnell bei klaren Hintergründen.
- Nur äussere Konturen verhindern, dass Innenmuster der Teile die Maske beeinflussen.
- Flächenfilter schützen vor Kleinstartefakten und zufälligen Verschmelzungen.

## Grenzen
- Ungleichmässige Beleuchtung kann Otsu irritieren; Vorverarbeitung → GlobalCleaner nutzen oder Parameter anpassen.
- Sehr nahe beieinander liegende Teile können verschmelzen; grössere Closing-Kernel vorsichtig testen und Ergebnis prüfen.
