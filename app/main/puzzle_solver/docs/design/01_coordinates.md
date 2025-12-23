## Koordinatensysteme und Rahmenmodell

### Koordinatensysteme

- **Maschinen-KS (M)**:
  - Zweck: Referenzkoordinatensystem der Maschine/Mechanik.
  - Achsen: `+x` nach rechts, `+y` nach oben.
  - Winkel: Grad, `0°` entlang `+x`, positiv **ccw**, Bereich `[-180°, +180°)`.
  - Ursprung: mechanisch definierter Referenzpunkt (z.B. Ecke der Grundplatte). **Numerische Festlegung ist im Simulator vorläufig/placeholder**.

- **Rahmen-KS (F)**:
  - Zweck: Lokales Koordinatensystem des A5-Zielrahmens (Innenkontur).
  - Ursprung: **untere linke Ecke** der Innenkontur.
  - Achsen: `+x` entlang der 128-mm-Kante, `+y` entlang der 190-mm-Kante.
  - Winkelkonvention wie M.

- **Kamera-/Bild-KS (P)** (falls relevant im bestehenden Projekt):
  - Pixelkoordinaten aus Vision-Pipeline.
  - Wird über vorhandene Kalibrierung/Skalierung nach mm in ein metrisches KS überführt.
  - Für das Matching wird **mm** bevorzugt; Pixel nur als Input.

### Transformationen

- **`T_MF` (Rahmen → Maschine)**:
  - 2D-Transform (Translation + Rotation), intern als 3×3 homogene Matrix.
  - `pose_M = T_MF ∘ pose_F`
  - **Status**: bewusst offen (Maschine noch nicht gebaut). Im Simulator wird ein **Platzhalter** verwendet, der garantiert ausserhalb der A4-Startfläche liegt. Dieser Platzhalter muss in Config/Model sichtbar sein.

- **Vision → Metrik**:
  - Bestehende Pipeline liefert bereits korrigierte/extrahierte Teile.
  - Matching erwartet idealerweise Konturen in mm in einem konsistenten metrischen KS (M oder F oder ein Zwischen-KS).
  - Falls Konturen in einem „Startflächen-KS“ (A4) vorliegen, wird dieses als M oder als eigenes KS dokumentiert und eindeutig transformiert.

### Rahmenmodell

- **Innenkontur**:
  - Rechteck mit Innenmass:
    - `inner_width_mm = 128`
    - `inner_height_mm = 190`

- **Eckenradius**:
  - Parameter: `corner_radius_mm`
  - **Status**: bewusst offen (noch nicht festgelegt). Default nicht hartkodieren.
  - Implementationshinweis: Rahmenkontakt-/Inside-Checks müssen sowohl 0-Radius als auch Radius >0 unterstützen (oder Radius zunächst ignorieren und als offene Einschränkung deklarieren).

- **Inside/Outside Toleranzband**:
  - Parameter: `tau_frame_mm`
  - Zweck: Stabilität gegen Rauschen/Quantisierung (Solver-Pruning).
  - Startwert: `tau_frame_mm = 2.0 mm` (konfigurierbar, im Debug ausgeben).

### Ausgabe-Konvention

- **Interne Repräsentation**:
  - Solver kann im Rahmen-KS (F) arbeiten, da Frame-Constraints dort am einfachsten sind.
- **Finale Ausgabe**:
  - Primär: Ausgabe in Maschinen-KS (M) **sofern** `T_MF` definiert ist.
  - Alternativ/zusätzlich: Ausgabe in Rahmen-KS (F) für Debug und frühe Simulatorphasen.
  - In Debug stets angeben, in welchem KS Werte liegen.

### Offene Punkte (bewusst offen lassen, aber formal geführt)
- Numerische Definition von `T_MF` (x0, y0, theta).
- `corner_radius_mm` (und ob in Checks berücksichtigt oder zunächst ignoriert).