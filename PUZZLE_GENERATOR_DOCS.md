# Puzzle Generator - Technische Dokumentation

## 1. High-Level Architektur

### Klassendiagramm (ASCII)

```
┌─────────────────────────────────────────────────────────────────┐
│                         Flask API Layer                          │
│  ┌──────────────┐              ┌────────────────────────────┐   │
│  │  routes.py   │──────────────│  api.py                    │   │
│  │              │              │  - generate_puzzle_images()│   │
│  │ /generate    │              │  - generate_geometry()     │   │
│  │ /calibrate   │              │  - render_clean/solution() │   │
│  │ /health      │              │  - apply_camera_sim()      │   │
│  └──────────────┘              └────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │   config.py  │  │  geometry/   │  │  rendering/  │
         │              │  │              │  │              │
         │ Generator    │  │ Puzzle       │  │ Puzzle       │
         │ Config       │  │ Generator    │  │ Renderer     │
         │              │  │              │  │              │
         │ - Puzzle     │  │ - Piece      │  │ - Shadow     │
         │ - Render     │  │ - Edge       │  │ - Watermark  │
         │ - Camera     │  │              │  │              │
         │ - Output     │  │ cuts.py      │  │              │
         └──────────────┘  │ - Cut (ABC)  │  └──────────────┘
                           │ - 13 Types   │
                           └──────────────┘
                                   │
                    ┌──────────────┼──────────────┐
                    ▼              ▼              ▼
         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐
         │ simulation/  │  │ calibration/ │  │ performance  │
         │              │  │              │  │              │
         │ Camera       │  │ Grid         │  │ @timed       │
         │ Simulator    │  │ Calibrator   │  │ decorator    │
         │              │  │              │  │              │
         │ effects/     │  │ - Hough      │  │ Metrics      │
         │ - 11 Effects │  │ - Optimize   │  │ Tracking     │
         └──────────────┘  └──────────────┘  └──────────────┘
```

### Kurzbeschreibung

Das System ist in 5 Hauptschichten organisiert:

1. **API Layer**: Flask-Endpunkte für HTTP-Requests (Puzzle-Generierung, Kalibrierung)
2. **Business Logic**: Orchestrierung der Puzzle-Generierung in `api.py`
3. **Core Modules**:
   - `geometry/`: Puzzle-Geometrie und Schnitt-Algorithmen
   - `rendering/`: Bilderzeugung mit Schatten und Wasserzeichen
   - `simulation/`: Kamera-Effekt-Pipeline
   - `calibration/`: Kamera-Kalibrierung via Grid-Detection
4. **Configuration**: Typsichere Dataclasses für alle Parameter
5. **Performance**: Dekoratoren für Zeitmessung und Optimierung

---

## 2. API-Aufbau

### Endpoint: `POST /generate`

**Beschreibung**: Generiert ein neues Puzzle mit clean, noisy und solution Images.

**Parameter** (JSON Body, alle optional):

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `layout` | string | `"2x3"` | Puzzle-Layout: `"2x3"` (2 Zeilen, 3 Spalten) oder `"3x2"` |
| `seed` | int \| null | `null` | Random Seed für Reproduzierbarkeit (null = zufällig) |
| `cut_types` | string[] | random 3-5 | Liste der Schnitt-Typen (z.B. `["wavy", "knob_square"]`) |
| `apply_watermark` | boolean | `true` | Wasserzeichen hinzufügen |
| `camera_intensity` | float | `1.0` | Kamera-Effekt-Intensität (0.0-1.0) |
| `enable_performance_logging` | boolean | `false` | Performance-Metriken ausgeben |

**Verfügbare Cut Types**:
```
knob_square, knob_triangular, knob_wavy, knob_square_new, knob_trapezoid,
knob_round, chevron, double_chevron, single_wave, partial_wave
```

**Rückgabewert** (JSON):

```json
{
  "success": true,
  "noisy_image": "base64_encoded_png_data...",
  "solution_image": "base64_encoded_png_data...",
  "metadata": {
    "seed": 12345,
    "layout": "2x3",
    "cut_types": ["wavy", "knob_square", "chevron"],
    "piece_count": 6,
    "resolution": [4608, 2592]
  }
}
```

---

### Endpoint: `POST /calibrate-from-grid`

**Beschreibung**: Kalibriert Barrel Distortion aus einem Puzzle-Bild mit Kalibrierungsgrid.

**Input**:
- JSON Body: `{"image": "base64_encoded_image"}`
- ODER File Upload (multipart/form-data)

**Query Parameter**:

| Parameter | Typ | Default | Beschreibung |
|-----------|-----|---------|--------------|
| `grid_spacing` | float | `236.0` | Erwarteter Grid-Abstand in Pixeln (20mm @ 300 DPI) |
| `visualize` | boolean | `true` | Visualisierung mit detektierten Linien zurückgeben |

**Rückgabewert** (JSON):

```json
{
  "success": true,
  "calibration": {
    "k1": -0.3,
    "k2": 0.05,
    "k3": 0.0,
    "fx": 4608,
    "fy": 4608,
    "cx": 2304.0,
    "cy": 1296.0,
    "image_size": [4608, 2592],
    "num_lines_detected": 35,
    "rms_error": 1.2
  },
  "calibration_saved": "/path/to/calibration_params.json",
  "visualization": "base64_encoded_image..."
}
```

---

### Endpoint: `GET /health`

**Beschreibung**: Health Check für Service-Monitoring.

**Rückgabewert**: `{"status": "ok", "service": "puzzle-generator"}`

---

## 3. Datenfluss

### Von Anfrage bis Bildausgabe

```
1. HTTP Request (POST /generate)
   └─> routes.py: parse JSON parameters
         │
2. Config Creation
   └─> GeneratorConfig(puzzle, render, camera, output)
         │
3. Puzzle Geometry Generation
   └─> PuzzleGenerator.generate()
         ├─> _assign_edge_cuts(): Male/Female Cut Pairing
         ├─> _create_piece(): 4 Edges pro Piece (TRBL clockwise)
         └─> _randomize_placement(): Grid-basierte Verteilung
               ├─> Rotation (0-360°) pro Piece
               ├─> Bounding Box Berechnung
               └─> Safe Radius für Spacing
         │
4. Clean Image Rendering
   └─> PuzzleRenderer.render()
         ├─> _create_background(): White + Noise + Grid
         ├─> _render_shadow(): Optimized Bbox Shadow (optional)
         └─> _transform_piece_points(): Rotation + Translation + Scaling
               │
5. Solution Image Rendering
   └─> PuzzleGenerator.generate_solution()
         ├─> Pieces in Original Grid Position
         └─> Small Gap (15px) zwischen Pieces
               │
6. Camera Simulation (optional)
   └─> CameraSimulator.simulate()
         ├─> BarrelDistortion → Perspective → ChromaticAberration
         ├─> PurpleFringing → Vignette → Oversharpening
         └─> Noise → NoiseReduction → LensSoftness
               │
7. Watermark Application
   └─> Watermark.apply_to_array(): Seed + Layout Metadata
         │
8. Image Encoding
   └─> PIL.Image → Base64 PNG
         │
9. HTTP Response (JSON)
   └─> {"success": true, "noisy_image": "...", "solution_image": "..."}
```

**Key Points**:
- Geometrie wird einmal berechnet, dann für clean/solution wiederverwendet
- Shadow Rendering nutzt scaled Bounding Boxes (0.5x) für Performance
- Camera Effects sind in realistischer Pipeline-Reihenfolge angeordnet
- Scale Factor wird automatisch berechnet wenn Pieces nicht in Canvas passen

---

## 4. Kern-Algorithmen

### 4.1 Schnitt-Generierung (cuts.py)

**Zweck**: Erzeugt parametrische Kurven für Puzzle-Kanten (male/female pairing).

**Pseudo-Code** (Beispiel: KnobSquareCut):

```python
def generate_knob_square(start, end, is_male, depth_ratio):
    edge_length = distance(start, end)
    knob_depth = edge_length * depth_ratio
    knob_region = [0.35, 0.65]  # Middle 30% of edge

    points = []
    for t in [0.0 ... 1.0]:  # 50 steps
        base = start + t * (end - start)
        perp = perpendicular_vector(start, end)

        if knob_region[0] <= t <= knob_region[1]:
            offset = knob_depth if is_male else -knob_depth
        else:
            offset = 0

        point = base + offset * perp
        points.append(point)

    return points
```

**Beispiel: WavyCut**:

```python
def generate_wavy(start, end, is_male, depth_ratio, frequency):
    edge_length = distance(start, end)
    max_amplitude = edge_length * depth_ratio

    points = []
    for t in [0.0 ... 1.0]:
        base = start + t * (end - start)
        perp = perpendicular_vector(start, end)

        amplitude = max_amplitude * sin(2π * frequency * t)
        if not is_male:
            amplitude = -amplitude

        point = base + amplitude * perp
        points.append(point)

    return points
```

**Verfügbare Cut-Typen**: 13 Varianten (siehe `cuts.py:794-814`)

---

### 4.2 Teile-Verteilung (puzzle.py)

**Zweck**: Platziert Puzzle-Pieces randomisiert mit garantiertem Spacing.

**Pseudo-Code**:

```python
def randomize_placement(pieces, margin, spacing, canvas_w, canvas_h):
    # 1. Calculate grid layout
    num_pieces = len(pieces)
    grid_cols = 3 if num_pieces == 6 else 2
    grid_rows = ceil(num_pieces / grid_cols)

    # 2. Assign rotations FIRST (affects bounding box!)
    for piece in pieces:
        piece.rotation = random(0, 360)

    # 3. Calculate max piece size (after rotation)
    max_size = 0
    for piece in pieces:
        bbox_w, bbox_h = get_rotated_bbox(piece, piece.rotation)
        max_size = max(max_size, max(bbox_w, bbox_h))

    # 4. Calculate cell dimensions
    cell_width = max_size + 2*spacing + inter_cell_gap
    cell_height = max_size + 2*spacing + inter_cell_gap

    # 5. Scale down if doesn't fit
    if grid_width > canvas_w or grid_height > canvas_h:
        scale = min(canvas_w/grid_width, canvas_h/grid_height)
        cell_width *= scale
        cell_height *= scale

    # 6. Place pieces in grid cells with random offset
    shuffle(pieces)
    for idx, piece in enumerate(pieces):
        grid_row = idx // grid_cols
        grid_col = idx % grid_cols

        cell_center_x = grid_start_x + (grid_col + 0.5) * cell_width
        cell_center_y = grid_start_y + (grid_row + 0.5) * cell_height

        safe_radius = (min(cell_width, cell_height) - max_size) / 2 * 0.4
        offset_x = random(-safe_radius, safe_radius)
        offset_y = random(-safe_radius, safe_radius)

        piece.center = (cell_center_x + offset_x, cell_center_y + offset_y)

    return {scale_factor, grid_info...}
```

---

### 4.3 Shadow Rendering (renderer.py)

**Zweck**: Optimierte Shadow-Generierung mit Bounding Box Rendering.

**Pseudo-Code**:

```python
def render_shadow(piece, shadow_offset, blur_radius, scale=0.5):
    # 1. Get piece outline
    outline = transform_piece_points(piece)

    # 2. Apply shadow offset
    offset_outline = [(x+dx, y+dy) for x,y in outline]

    # 3. Calculate bounding box
    bbox = {
        left: min(x) - 2*blur_radius,
        right: max(x) + 2*blur_radius,
        top: min(y) - 2*blur_radius,
        bottom: max(y) + 2*blur_radius
    }

    # 4. Render to small canvas (scaled down)
    small_w = bbox.width * scale
    small_h = bbox.height * scale
    small_canvas = create_image(small_w, small_h)
    draw_polygon(small_canvas, offset_outline, color=black)

    # 5. Apply blur on small canvas (fast!)
    blurred = gaussian_blur(small_canvas, blur_radius * scale)

    # 6. Upscale back to full size
    full_shadow = resize(blurred, (bbox.width, bbox.height), LANCZOS)

    # 7. Paste into full canvas at bbox position
    full_canvas.paste(full_shadow, (bbox.left, bbox.top))

    return full_canvas
```

**Performance**: Shadow-Rendering bei 0.5x scale ist ~4x schneller bei minimaler Qualitätsverlust.

---

### 4.4 Camera Effect Pipeline (simulation/camera.py)

**Zweck**: Simuliert cheap Smartphone-Kamera in realistischer Reihenfolge.

**Pseudo-Code**:

```python
def camera_simulate(image):
    # 1. Optical distortions (lens)
    image = barrel_distortion(image, k1, k2, k3)
    image = perspective_distortion(image, angle, strength)

    # 2. Chromatic effects (lens artifacts)
    image = chromatic_aberration(image, offset)
    image = purple_fringing(image, intensity)

    # 3. Lighting falloff
    image = vignette(image, center, strength)

    # 4. Software processing (sharpening)
    image = oversharpening(image, amount)

    # 5. Sensor noise
    image = gaussian_noise(image, sigma)
    image = color_noise(image, intensity)
    image = salt_pepper_noise(image, probability)

    # 6. Software processing (denoising)
    image = noise_reduction(image, strength)

    # 7. Optical blur (cheap lens)
    image = lens_softness(image, blur_radius)

    return image
```

**Optimierung**: `CombinedNoiseEffect` kombiniert Gaussian + Color Noise in einem RNG-Call.

---

### 4.5 Edge Matching (geometry/puzzle.py)

**Zweck**: Garantiert, dass benachbarte Pieces perfekt zusammenpassen.

**Pseudo-Code**:

```python
def assign_edge_cuts(rows, cols, cut_types):
    edge_cuts = {}

    # Horizontal edges (between rows)
    for row in range(rows - 1):
        for col in range(cols):
            cut_type = random.choice(cut_types)
            is_male = random.boolean()

            # Top piece (bottom edge): left-to-right
            cut = create_cut(cut_type, is_male, depth_ratio)
            edge_cuts[(row, col, 'bottom')] = cut

            # Bottom piece (top edge): right-to-left (reversed!)
            edge_cuts[(row+1, col, 'top')] = ReversedCut(cut)

    # Vertical edges (between columns)
    for row in range(rows):
        for col in range(cols - 1):
            cut_type = random.choice(cut_types)
            is_male = random.boolean()

            # Left piece (right edge): top-to-bottom
            cut = create_cut(cut_type, is_male, depth_ratio)
            edge_cuts[(row, col, 'right')] = cut

            # Right piece (left edge): bottom-to-top (reversed!)
            edge_cuts[(row, col+1, 'left')] = ReversedCut(cut)

    # Border edges (straight)
    for row, col in all_pieces:
        if is_border(row, col):
            edge_cuts[(row, col, direction)] = StraightCut()

    return edge_cuts
```

**Key**: `ReversedCut` wrapper invertiert die Punktreihenfolge UND male/female.

---

### 4.6 Grid Calibration (calibration/grid_calibrator.py)

**Zweck**: Berechnet Barrel Distortion Koeffizienten aus gekrümmten Grid-Linien.

**Pseudo-Code**:

```python
def calibrate_from_grid(image):
    # 1. Detect grid lines
    edges = canny_edge_detection(image)
    lines = hough_line_transform(edges)
    h_lines, v_lines = filter_by_angle(lines)

    # 2. Extract points along lines
    all_points = []
    for line in (h_lines + v_lines):
        points = extract_pixels_along_line(image, line)
        all_points.append(points)

    # 3. Optimize distortion coefficients
    def error_function(k1, k2, k3):
        total_error = 0
        for points in all_points:
            undistorted = apply_inverse_distortion(points, k1, k2, k3)
            straightness = measure_line_fit(undistorted)
            total_error += straightness
        return total_error

    # Minimize error using scipy.optimize
    result = minimize(error_function, x0=[0, 0, 0])
    k1, k2, k3 = result.x

    # 4. Calculate camera matrix
    h, w = image.shape
    fx = fy = w  # Assume square pixels
    cx, cy = w/2, h/2

    return {k1, k2, k3, fx, fy, cx, cy, image_size}
```

---

## 5. Technische Details

### Dependencies

#### Web-Framework & Server

| Library | Version | Verwendungszweck |
|---------|---------|------------------|
| **Flask** | 3.1.2 | REST API Framework, HTTP Routing |
| Werkzeug | 3.1.3 | WSGI Utility (Flask Dependency) |
| Jinja2 | 3.1.6 | Template Engine (Flask Dependency) |
| click | 8.3.0 | CLI Support (Flask Dependency) |
| itsdangerous | 2.2.0 | Session Security (Flask Dependency) |
| blinker | 1.9.0 | Signal/Event System (Flask Dependency) |
| MarkupSafe | 3.0.3 | String Escaping (Jinja2 Dependency) |

#### Bildverarbeitung & Numerik

| Library | Version | Verwendungszweck |
|---------|---------|------------------|
| **Pillow** | 12.0.0 | Image Rendering, Drawing, Filters, Shadow Effects |
| **NumPy** | 2.2.6 | Array Operations, Noise Generation, Image Manipulation |
| **OpenCV** | 4.12.0.88 | Grid Detection (Hough), Barrel Distortion, Calibration |
| **SciPy** | 1.15.3 | Optimization (minimize) für Distortion-Koeffizienten |

### Wichtige Datenstrukturen

**PuzzlePiece** (geometry/puzzle.py:21-46):
```python
@dataclass
class PuzzlePiece:
    row: int              # Grid position
    col: int
    edges: List[PieceEdge]  # [Top, Right, Bottom, Left] clockwise
    rotation: float       # Degrees (0-360)
    center: (float, float)  # Placement position
```

**PieceEdge** (geometry/puzzle.py:12-17):
```python
@dataclass
class PieceEdge:
    start: (float, float)
    end: (float, float)
    cut: Cut              # Parametric cut function
    is_border: bool       # True if outer edge
```

**PuzzleGenerationResult** (api.py:18-36):
```python
@dataclass
class PuzzleGenerationResult:
    seed: int
    piece_count: int
    layout: str
    cut_types: list[str]
    resolution: (int, int)
    clean_image: np.ndarray    # RGB uint8
    solution_image: np.ndarray
    noisy_image: np.ndarray    # After camera sim
    debug_image: np.ndarray    # Optional
    debug_info: dict
```

### Besonderheiten

1. **Performance Tracking**: `@timed` Decorator (performance.py) misst Laufzeit aller kritischen Funktionen
2. **Male/Female Cut Pairing**: `ReversedCut` wrapper garantiert perfekte Passung durch Punktinvertierung
3. **Shadow Optimization**: Rendering bei 0.5x scale + LANCZOS upscaling = 4x Speedup
4. **Grid-based Placement**: Verhindert Piece-Overlap durch cell-based safe zones
5. **Scale Factor**: Automatische Skalierung wenn Pieces grösser als Canvas
6. **Calibration Grid**: Wird VOR Barrel Distortion gezeichnet, damit Verzerrung messbar ist
7. **Effect Pipeline**: Realistische Reihenfolge simuliert echte Kamera-Verarbeitung
8. **Watermark**: Enthält Seed + Layout für Reproduzierbarkeit

### Configuration Defaults (config.py)

```python
PuzzleConfig:
  - A5 Format: 210mm x 148mm @ 300 DPI
  - Layout: 2x3 (6 pieces)

RenderConfig:
  - Margin: 200px
  - Spacing: 100px
  - Shadow: Enabled (30px offset, 35px blur, 60% opacity)
  - Grid: 20mm spacing (236px @ 300 DPI)

CameraConfig (EXTREME MODE):
  - Fisheye: 0.65 (strong barrel distortion)
  - Perspective: 0.45
  - Chromatic Aberration: 0.25
  - Purple Fringing: 0.75
  - Noise: 0.25
  - Vignette: 0.5
  - Oversharpening: 1.8
  - Noise Reduction: 0.7

OutputConfig:
  - Resolution: 4608 x 2592 (RasPi Camera v3)
  - Format: PNG (optimized, 256 colors)
```

### Code-Struktur

```
app/puzzle_gen/
├── __init__.py
├── api.py                    # Business Logic
├── routes.py                 # Flask Endpoints
├── config.py                 # Configuration Dataclasses
├── performance.py            # Performance Tracking
├── geometry/
│   ├── puzzle.py             # PuzzleGenerator, PuzzlePiece
│   └── cuts.py               # 13 Cut Types (Cut ABC)
├── rendering/
│   ├── renderer.py           # PuzzleRenderer
│   └── watermark.py          # Watermark Application
├── simulation/
│   ├── camera.py             # CameraSimulator
│   └── effects/
│       ├── base.py           # CameraEffect ABC
│       ├── geometric.py      # Barrel, Perspective
│       ├── chromatic.py      # Aberration, Fringing
│       ├── optical.py        # Vignette, Softness
│       ├── noise.py          # Gaussian, Color, S&P
│       └── processing.py     # Sharpen, Denoise
└── calibration/
    └── grid_calibrator.py    # GridCalibrator (Hough + Optimize)
```

---

## Performance-Metriken

Typische Ausführungszeiten (4608x2592, 6 Pieces):

| Operation | Zeit |
|-----------|------|
| Geometry Generation | ~50ms |
| Clean Image Rendering | ~300ms |
| Shadow Rendering (6 pieces) | ~400ms |
| Solution Image Rendering | ~250ms |
| Camera Simulation | ~800ms |
| **Total** | **~1.8s** |

Optimierungen:
- Shadow Render Scale: 0.5x → ~4x faster
- Combined Noise Effect → ~30% faster
- Bounding Box Shadow → ~10x faster vs. full-canvas

---

## Verwendungsbeispiel (Python)

```python
from app.puzzle_gen.api import generate_puzzle_images
from app.puzzle_gen.config import GeneratorConfig, PuzzleConfig, Layout

# Configure
config = GeneratorConfig(
    puzzle=PuzzleConfig(
        layout=Layout.LAYOUT_2X3,
        seed=42
    )
)

# Generate
result = generate_puzzle_images(
    config=config,
    cut_types=['knob_square', 'wavy', 'chevron'],
    layout='2x3',
    apply_watermark=True,
    apply_camera=True
)

# Access results
print(f"Seed: {result.seed}")
clean_img = result.clean_image  # np.ndarray (H, W, 3)
noisy_img = result.noisy_image
solution_img = result.solution_image

# Save to disk
result.save_all(output_dir='./output', image_format='png')
```

---

**Dokumentation erstellt**: 2025-12-03
**Version**: 1.0
**Code Basis**: `/Users/ciril/Documents/coding/PREN-SIMULATOR/app/puzzle_gen`
