# Vorverarbeitung

Dieses Kapitel beschreibt die optionalen globalen und lokalen Reinigungsschritte, die die Segmentierung und die anschließende Kantenauswertung verbessern.

Dateien im Paket:
- preprocessing/global_cleaner.py
- preprocessing/image_cleaner.py

## GlobalCleaner (Kamera-/Optik-Korrekturen)
Zweck
- Wendet vor der Segmentierung kamerabezogene Korrekturen an, damit Form und Lage der Puzzleteile möglichst unverfälscht sind.

Funktionen
- Tonnenverzerrung: nutzt Koeffizienten k1, k2, k3 sowie Intrinsics fx, fy, cx, cy zum Entzerren.
- Perspektivische Entzerrung (optional): kompensiert schräge Aufnahmen.
- Vignettierung (optional): reduziert Helligkeitsabfall zum Rand für stabileres Thresholding.
- Chromatische Aberration (optional): korrigiert Kanalversatz, um Kanten zu schärfen.

Warum hier eingesetzt
- Gleichmäßigere Geometrie und Beleuchtung verbessern Thresholding, Konturqualität sowie Eck-/Kantenerkennung. Verzerrungen können gerade Ränder krümmen und die Klassifikation „flat/tab/slot“ verfälschen.

Wichtige API
- calibrate(test_images): bestimmt Parameter aus Testbildern und speichert den Zustand.
- clean(image): wendet die aktivierten Korrekturen auf ein BGR-Bild an.
- get_calibration_info(): liefert aktive Parameter/Flags (wird z. B. in der Antwort von /clean-global zurückgegeben).

Typischer Ablauf
1. Kalibrierungsbilder hochladen und Route /calibrate aufrufen.
2. /clean-global/<filename> aufrufen, um ein korrigiertes Bild für die nächsten Schritte zu erzeugen.

Hinweise
- Ohne Kalibrierungsdatei werden Defaultwerte genutzt; die Route gibt dann eine Warnung zurück.
- Besonders wirksam für Puzzleränder und lange, gerade Kanten: Tonnen- und Perspektivkorrektur.

## ImageCleaner (lokale Nachbearbeitung pro Teil)
Zweck
- Verfeinert nach der Extraktion Maske und RGBA-Zuschnitt je Teil, um kleine Artefakte und Zacken zu reduzieren.

Funktionen
- Leichte Entrauschung des Teil-Bildausschnitts.
- Morphologische Verfeinerungen an der Maske (z. B. Schließen kleiner Löcher, Entfernen von Sprenkeln).

Warum hier eingesetzt
- Sauberere Konturen verbessern Eckenerkennung und die Krümmungssignatur der Kanten. Außerdem werden Visualisierungen klarer.

Ein-/Ausgaben
- Input: Liste von PuzzlePiece-Objekten.
- Output: dieselbe Liste mit verbesserten Kontur-/Maskenfeldern (optional bleibt contour_original für Vergleiche erhalten).

Wann einsetzen
- Empfohlen für Realfotos, bei denen Thresholding/Morphologie das Bildrauschen nicht vollständig entfernt.

Grenzen
- Zu aggressive Morphologie kann feine Merkmale von Tabs/Slots abtragen und damit das Matching erschweren. Kleine Kernel und wenige Iterationen bevorzugen.
