# Visual Time Perception and Symbolic Reasoning Clock

An interactive research and learning environment for symbolic clock interpretation.

The project combines a styled canvas clock generator, realtime capture pipeline, symbolic inference backend, metrics telemetry, grounding-preset scripting, and dataset generation tools for experimentation.

## Technical Stack

- Backend API: Flask (Python)
- Vision and symbolic extraction: OpenCV + NumPy
- Frontend: HTML5 Canvas + vanilla JavaScript
- Styling: CSS
- Process control: launch.sh (start, stop, restart, status)
  - Optional realtime cap: start or restart with --realtime-max-seconds <N>
- Persistence:
  - Snapshot image and orientation metadata in snapshots/
  - Browser localStorage for UI state, acquired logs, processing history, and grounding scripts

## Design Reasoning

This system is designed around three goals:

1. Observable symbolic perception
- Show the rendered clock, symbolic output, and detailed telemetry together.
- Make detection fallbacks and error trends visible instead of hidden.

2. Controlled domain shift
- Support many visual theme variants and motion perturbations (rotation, flips, back view, artifacts).
- Allow users to move between clean and stress-test conditions.

3. Grounding-first experimentation
- Provide configurable grounding presets and script composition to teach and test robust symbolic grounding.
- Keep data generation and debug metrics close to the interactive loop.

## Current Feature Set

### Clock Rendering and Theme Engine
- Multiple era-inspired themes (including Roman numeral styles)
- Random theme selection and manual theme dropdown selection
- Face patterns and hand-style variants by era
- Reset to Plain Classic baseline face

### Perturbation and Stress Controls
- Randomize era theme every second
- Increment planar rotation every second
- Random flip every second
- 3D rotate with back-view simulation
- Extra distractor line toggle (default off)
- Artifact-safe mode toggle (default on)

### Realtime Capture and Symbolic Conversion
- Capture canvas snapshot to PNG
- Convert snapshot to symbolic clock fields
- Realtime loop for capture + conversion once per second
- Optional launch-time max duration for realtime loop (seconds)
- Non-fatal realtime handling for hand-detection shortages
- Buttons are locked while realtime capture is active to prevent destructive conflicts

### Symbolic and Telemetry Outputs
- Symbolic Output panel
- Probabilistic Logic Reasoning panel (MAP assignment, marginals, rule masses)
- Symbolic Output Log panel
- Export symbolic log JSON
- Live Processing Histogram with:
  - Candidate angle histogram
  - Average mask strength bars (hour, minute, second)
  - Perception error trend line
  - latest, avg, max row placed below graph plot area
  - Fallback usage summary

### Data and Workspace Controls
- Clear Data: clears logs, stats, dataset rows, and grounding script sequence while preserving face settings
- Clear Workspace: clears data and resets face settings to defaults

### Grounding Preset Studio
- Preset library for grounding strategies
- Drag-and-drop script generator
- Add All Presets action
- One-time preset insertion (each preset only once in a script)
- Generated grounding script JSON output with internal scroll pane
- Apply presets to live runtime state
- Copy and clear script actions
- Persisted script sequence across refresh

### Dataset Workflow
- Build dataset rows from symbolic frames
- Filter and search generated samples
- Collapse/expand groups
- Copy filtered JSONL rows
- Save filtered JSONL rows

### LLM Benchmark Workbench (Frontend + Backend)
- In-app LLM benchmark panel with:
  - model list input
  - per-condition question limit
  - timeout control
  - run + refresh actions
- Artifact browser for benchmark outputs with:
  - filter (all/answers/evaluations/questions)
  - sort (name/size)
  - text search
  - selected artifact count
  - single-file download buttons
  - filtered ZIP download
- Server-backed benchmark execution and artifact serving endpoints

### Persistence Behavior
- Persists face state, toggles, selected theme, histories, and generated data
- Realtime mode does not auto-start after refresh

## User Interface Sections

### Clock Canvas & Live Processing Histogram (Top Section)
**Purpose:** Visualize clock rendering and real-time symbolic inference metrics.

- **Left side (Canvas):** Interactive clock display with selected theme and perturbations.
  - Theme selector dropdown to choose era styles (Roman numerals, modern, etc.)
  - Toggles allow randomizing theme, rotating, flipping, and adding distractor artifacts.
  - Flip button manually mirrors the clock; useful for testing flip tolerance.
  - Reset Clock Face returns to Plain Classic baseline.

- **Right side (Histogram):** Shows three diagnostic plots:
  - **Angle bins (0–360°):** Histogram of detected hand angle candidates. Tall spikes = good consensus; low/flat = ambiguous detection.
  - **Mask strength bars (avg):** Average HSV mask pixel count for each hand (hour, minute, second). Taller bars = stronger color detection.
  - **Perception error trend:** Running average of hour/minute/second errors. Drooping trend = improving accuracy over frames; flat trend = inconsistent detection.
  - Updates every symbolic conversion during realtime capture.

### Symbolic Output & Telemetry (Center Panels)
**Purpose:** Inspect per-frame symbolic results and debug metrics.

- **Symbolic Output:** JSON object with HOUR_HAND, MINUTE_HAND, SECOND_HAND, and TIME string.
- **Probabilistic Logic Reasoning:** Formal weighted model-counting inference (MAP assignment, marginals, rule satisfaction mass).
- **Symbolic Output Log:** Rolling JSON log of all captured frames. Export with button for external analysis.

### Grounding Preset Studio (Collapsible Section)
**Purpose:** Compose reusable symbolic grounding strategies via drag-and-drop.

- **Preset Library (left column):** Available grounding rules (e.g., "hour > minute", "fallback detection").
- **Script Generator (right column):** Drag presets from library to build a sequence.
  - **Add All Presets:** Inserts all available rules in one action.
  - **Apply Presets:** Executes the script sequence on the current clock state (modifies symbolic output).
  - **Copy Script:** Exports generated script as JSON to clipboard.
  - **Clear Script:** Erases the current sequence.
- **Generated Script JSON:** Formatted output showing composed rule chain.
- Scripts persist across page refresh and are cleared only by "Clear Workspace".

### Reproducible Experiments – Run Sweep (Collapsible Section)
**Purpose:** Execute a full robustness benchmark with 10 degradation conditions and statistical analysis.

- **Parameters:**
  - **Samples/condition:** Number of synthetic clock images per condition (1–500; default 40).
  - **Stats iterations:** Bootstrap/permutation iterations for significance testing (100–50,000; default 2,000).

- **Run Sweep Button:** Launches background sweep thread on the server. Progress bar updates every 1.2s.
  - **Queued → Sampling → Stats → Completed** stage display.
  - Condition thumbnails (5 per row) show sample images for each degradation (rotate, blur, noise, occlusion, etc.).
  - Each thumbnail is clickable to expand and view full description and severity.

- **Progress & Results:**
  - Stats Summary Preview renders markdown report with two-way ANOVA and Tukey post-hoc tests.
  - Artifacts (CSV + markdown) are downloaded via buttons.
  - If page is refreshed during the sweep, UI will automatically reconnect to in-flight run (showing progress).

- **Thumbnails Persistence:** Condition sample thumbnails are cached in browser localStorage and persist across refresh (unless "Clear Workspace" is clicked).

### LLM Benchmark Comparison (Collapsible Section)
**Purpose:** Compare multiple LLM models on the same set of clock-reading questions derived from the symbolic experiment metrics.

- **Configuration:**
  - **Models:** Comma-separated list (e.g., `llama3.2:3b,gemma3:4b`). Models must be available in local Ollama.
  - **Questions/condition:** Questions sampled per severity level (1–100; default 20).
  - **Timeout (s):** Per-model per-question timeout (10–1800 seconds; default 120).

- **Run Benchmark Button:** Launches pipeline:
  1. **Generate questions** from metrics CSV (pulls representative rows per condition/severity).
  2. **Collect model responses** via Ollama REST API with temperature 0.1 (deterministic).
  3. **Score answers** (correctness, key factor extraction, completeness).
  4. **Run stats** (Two-way ANOVA, Tukey HSD). Parse success summary shows JSON parse rates per model.

- **Artifact Browser:**
  - **Filter, sort, search** across benchmark outputs (questions, answers, evaluations, stats).
  - **Download single artifacts** or **filtered ZIP** for external analysis.
  - Detailed CSV with per-sample model predictions and factor-matching scores.

- **Stats Output:** Markdown with parse success rates, descriptive tables, and ANOVA results. If all responses fail to parse, stats are skipped with a diagnostic message.

## Step-by-Step Workflow

1. **Open:** Run `launch.sh start`, then open `http://127.0.0.1:5014`.
2. **Set Baseline:** Pick Theme, optionally Reset Clock Face, keep Artifact-safe mode on for clean tests.
3. **Single Frame:** Click Capture Snapshot, then Convert to Symbolic. Review Symbolic Output and Log.
4. **Monitor Telemetry:** Watch Angle Histogram, Mask Strength bars, and Perception Error Trend update.
5. **Realtime Loop:** Start Realtime Capture + Process. Metrics accumulate. Click again to stop. Controls are locked during realtime.
6. **Build Grounding:** Drag presets into Script Generator (or Add All Presets), then Apply Presets or Copy Script.
7. **Run Sweep:** Adjust samples/condition and stats iterations, click Run Sweep. Watch thumbnail grid appear and progress update. Results download via buttons.
8. **LLM Benchmark:** Enter model list, set per-condition limit, click Run Benchmark (requires Ollama). Browse and download artifacts.
9. **Clear:** Clear Data resets logs/stats/script but preserves face. Clear Workspace also resets face to defaults.
10. **Validate:** Run `python self_test_symbolic.py` for reproducible offline checks.

## Symbolic Inference Pipeline

1. Detect clock circle center and radius (Hough + fallback).
2. Extract blue hour/minute candidates via HSV masks connected to center.
3. Extract red second-hand candidate via HSV mask.
4. Build line candidates via Canny + HoughLinesP for fallback.
5. Resolve missing hands using fallback modes:
   - infer-minute-from-hour
   - infer-hour-from-minute
   - infer-both-from-lines
6. Convert hand angles into symbolic HOUR_HAND, MINUTE_HAND, SECOND_HAND, TIME.
7. Emit debug_metrics with mask strengths, candidate angles, and fallback status.

## Processed Image Saving

When Convert to Symbolic is clicked, the frontend sends the current canvas image to the backend with `save_image: true`. The backend decodes the image, runs an OpenCV preprocessing pipeline (grayscale, blur, Canny/threshold, contour detection, and a display frame overlay), and writes the result to `static/processed_clock.png`. It also saves per-stage processing metadata to `static/processed_clock_data.json` for the processing stages panel and download link. The UI then shows a thumbnail preview of the saved PNG plus a download link for `/static/processed_clock.png` and the stage data artifact.

On non-fatal symbolic detection failures, the backend returns debug_metrics in 422 responses so the frontend can continue populating telemetry during realtime runs.

## API Endpoints

### POST /save-image
Body:

```json
{
  "imageData": "data:image/png;base64,...",
  "flipped": false
}
```

Behavior:
- Saves snapshots/clock_snapshot.png
- Saves orientation metadata in snapshots/snapshot_meta.json

### POST /symbolic-clock
Success response includes:
- HOUR_HAND, MINUTE_HAND, SECOND_HAND, TIME
- hour_angle, minute_angle, second_angle
- center_x, center_y, radius
- debug_metrics
- probabilistic_reasoning (formal weighted-model-counting style inference payload)
- was_mirrored, image_orientation

The `probabilistic_reasoning` field includes:
- `method`: reasoning algorithm used
- `world_count`: number of enumerated worlds
- `domain_sizes`: discrete domain sizes for hour/minute/second
- `map_assignment`: maximum a posteriori symbolic time
- `map_probability`: posterior probability of MAP assignment
- `marginals`: top posterior marginals per variable
- `rules`: weighted clause summary with posterior satisfaction mass

Failure response (422) includes:

```json
{
  "error": "Unable to detect enough clock hands in the snapshot.",
  "debug_metrics": {
    "pipeline": "color-mask + hough-fallback",
    "fallback_used": true,
    "fallback_mode": "candidate-shortage"
  }
}
```

### GET /project-readme
- Opens the in-app project documentation page.

### Reproducible Experiment Endpoints
- `POST /run-experiments`
- `GET /experiment-artifacts`
- `GET /experiment-summary`
- `GET /experiment-artifact/<artifact_name>`

### LLM Benchmark Endpoints
- `POST /run-llm-benchmark`
- `GET /llm-benchmark-artifacts`
- `GET /llm-benchmark-summary`
- `GET /llm-benchmark-artifact/<artifact_name>`
- `POST /llm-benchmark-artifacts-zip`

## Project Structure

- app.py: Flask routes and API response behavior
- symbolic_clock.py: OpenCV-based symbolic extraction and debug metrics
- static/app.js: UI logic, rendering, realtime loop, persistence, grounding studio, dataset tools
- static/style.css: main UI styling
- templates/index.html: application UI
- templates/project_readme.html: in-app documentation page
- self_test_symbolic.py: symbolic API self-test script
- launch.sh: process lifecycle helper
- snapshots/: saved capture assets and orientation metadata
- static/: generated image/JSON artifacts such as processed_clock.png and processed_clock_data.json
- results/: experiment sweep outputs, metrics, and benchmark artifacts

## Local Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./launch.sh start
```

Open:
- http://127.0.0.1:5014

Stop:

```bash
./launch.sh stop
```

Restart:

```bash
./launch.sh restart
```

## Self-Test

Run after server is up:

```bash
python self_test_symbolic.py
```

Custom target time:

```bash
python self_test_symbolic.py --hour 3 --minute 25 --second 40
```

## Reproducible Experiment Pipeline

Run a full condition sweep with metrics export and statistical summary:

```bash
python run_experiments.py --samples-per-condition 40
```

Artifacts are written to `results/`:
- `condition_matrix.csv`: fixed condition protocol used in the run
- `metrics_by_condition.csv`: per-sample evaluation rows
- `stats_by_condition.csv`: aggregate metrics per condition
- `stats_summary.md`: baseline-vs-condition significance summary

Optional controls:

```bash
# Faster smoke test
python run_experiments.py --samples-per-condition 5

# Change output folder and seed
python run_experiments.py --output-dir results_run2 --seed 12345

# Skip stats generation
python run_experiments.py --no-stats
```

Generate stats later from an existing metrics file:

```bash
python stats_summary.py --metrics-csv results/metrics_by_condition.csv
```

## Notes for AI Learning Workflows

- Use Artifact-safe mode for clean grounding baselines.
- Use perturbation toggles to build invariance training regimes.
- Use Grounding Preset Studio scripts to document experiment recipes.
- Use dataset generation tools to build instruction-style data from symbolic traces.
