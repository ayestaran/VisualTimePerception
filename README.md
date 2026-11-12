# Visual Time Perception & Symbolic Reasoning

## MSc AI Module 7WCM2018-0509-2025 Assignment - Custom Mini Project by Franz Ayestaran

This repository contains the implementation for a custom mini-project on real-time perceptual-to-symbolic conversion. The project uses visual clock scenes as a constrained environment for studying how image-based perception can be translated into structured symbolic representations and then stabilised through probabilistic reasoning.

The project was designed and authored by me. AI assisted with generating boiler‑plate Python code, but the conceptual structure, symbolic pipeline, probabilistic reasoning model, and experimental workflow were all created under my direction. I have gone through the core routines and documented them to ensure complete understanding.

The README is intended to sit alongside [CUSTON-MINI-PROJECT.md](CUSTON-MINI-PROJECT.md) and [ASSESSMENT-COMPONENTS.md](ASSESSMENT-COMPONENTS.md). Those documents explain the academic framing and marking emphasis, while this document explains how the implemented system is organised, what it demonstrates, and how to run it.

## Project Overview
-------------------

The core idea of the project is to connect three pieces of work into one coherent pipeline:

1. Visual perception of clock-hand geometry from generated clock images.
2. Conversion of those detections into symbolic time representations.
3. Probabilistic reasoning over the symbolic output to handle uncertainty, missing evidence, and perturbations.

The project is therefore not only a front-end demonstration. The main contribution is the perceptual-to-symbolic pipeline, supported by a probabilistic logic layer and a reproducible experimental workflow for robustness analysis.

## Objectives
-------------

The implementation in this repository supports the three project objectives described in [CUSTON-MINI-PROJECT.md](CUSTON-MINI-PROJECT.md):

* Objective 1: develop a real-time visual-to-symbolic conversion pipeline that can interpret clock images and translate them into symbolic representations.
* Objective 2: integrate probabilistic logic reasoning so that symbolic interpretation remains useful under uncertainty, ambiguous detections, or missing hands.
* Objective 3: analyse robustness under controlled perturbations such as rotation, blur, noise, flips, and occlusion through reproducible experimental sweeps.

## Main Components
------------------

The structure of the repository broadly follows the assessment emphasis described in [ASSESSMENT-COMPONENTS..md](ASSESSMENT-COMPONENTS..md).

### A. Core Perceptual-to-Symbolic Pipeline

* [symbolic_clock.py](symbolic_clock.py)
  Implements the visual analysis pipeline that detects the clock face and hand geometry, resolves fallbacks, and converts the result into symbolic clock fields.

* [project_utils.py](project_utils.py)
  Provides helper functionality used by the symbolic conversion process.

* [self_test_symbolic.py](self_test_symbolic.py)
  Provides a repeatable self-test for validating that the symbolic interpretation pipeline behaves as expected.

### B. Probabilistic Reasoning Layer

* [probabilistic_logic_reasoner.py](probabilistic_logic_reasoner.py)
  Implements the reasoning layer that computes MAP assignments, marginals, and rule-satisfaction mass over the symbolic output.

### C. Experimental Evaluation

* [run_experiments.py](run_experiments.py)
  Runs the perturbation sweep used to evaluate robustness across multiple degradation conditions.

* [stats_summary.py](stats_summary.py)
  Produces the statistical summaries used to interpret experimental outcomes.

* [results/](results/)
  Stores generated CSV outputs and markdown summaries from experiment runs.

### D. Integration and Demonstration Layer

* [app.py](app.py)
  Exposes the Flask routes and ties together the pipeline, reasoning layer, and experiment endpoints.

* [templates/index.html](templates/index.html)
  Defines the main interface layout.

* [static/app.js](static/app.js)
  Handles the interactive clock, realtime capture loop, grounding presets, and experiment controls.

* [static/style.css](static/style.css)
  Provides the visual styling for the interface.

## What the System Demonstrates
-------------------------------

The current implementation demonstrates the following capabilities:

* Generation of styled clock faces with multiple themes and perturbation controls.
* Single-frame and continuous capture of clock images.
* Conversion of visual detections into symbolic outputs including hour, minute, second, and time strings.
* Exposure of debug telemetry so that the symbolic perception process remains observable rather than hidden.
* Probabilistic logic reasoning over symbolic outputs to support interpretation under uncertainty.
* A grounding preset workflow for composing reusable symbolic grounding strategies.
* A reproducible experiment sweep with downloadable metrics and statistical summaries.

## System Workflow
------------------

In practice, the application follows this sequence:

1. A clock scene is rendered in the browser canvas.
2. The image is captured and sent to the backend.
3. The backend detects the clock face and candidate hand angles.
4. Missing or weak detections are handled through fallback logic.
5. The final detections are converted into symbolic fields.
6. The probabilistic reasoning layer evaluates the symbolic interpretation.
7. Telemetry, symbolic output, and experiment data are exposed through the interface.

## API Summary
--------------

The Flask application exposes the following main endpoints:

* `POST /save-image`
  Saves the current clock snapshot and associated orientation metadata.

* `POST /symbolic-clock`
  Runs the visual-to-symbolic pipeline and returns symbolic fields, geometric measurements, debug metrics, and probabilistic reasoning output.

* `GET /project-readme`
  Serves the in-app documentation view.

* Experiment endpoints:
  * `POST /run-experiments`
  * `GET /experiment-artifacts`
  * `GET /experiment-summary`
  * `GET /experiment-artifact/<artifact_name>`

## Running the Project
----------------------

For a fresh setup on a new machine, run:

```bash
chmod +x prerequisites.sh
./prerequisites.sh
./launch.sh start
```

Open the application at:

* `http://127.0.0.1:5014`

To stop the application:

```bash
./launch.sh stop
```

To restart it:

```bash
./launch.sh restart
```

To restart with a fixed realtime capture limit:

```bash
./launch.sh restart --realtime-max-seconds 30
```

## Validation and Testing
-------------------------

Run the symbolic self-test after the server is up:

```bash
python self_test_symbolic.py
```

You can also specify a custom target time:

```bash
python self_test_symbolic.py --hour 3 --minute 25 --second 40
```

## Reproducible Experiment Pipeline
-----------------------------------

Run a full condition sweep with metric export and statistical summary:

```bash
python run_experiments.py --samples-per-condition 40
```

Artifacts are written to [results/](results/):

* `condition_matrix.csv`: fixed condition protocol used in the run.
* `metrics_by_condition.csv`: per-sample evaluation rows.
* `stats_by_condition.csv`: aggregate metrics per condition.
* `stats_summary.md`: significance summary across conditions.

Useful variants:

```bash
python run_experiments.py --samples-per-condition 5
python run_experiments.py --output-dir results_run2 --seed 12345
python run_experiments.py --no-stats
python stats_summary.py --metrics-csv results/metrics_by_condition.csv
```

## Suggested Reading Order
--------------------------

If this repository is being read for assessment purposes, the most useful order is:

1. [CUSTON-MINI-PROJECT.md](CUSTON-MINI-PROJECT.md) for the project aims and constraints.
2. [ASSESSMENT-COMPONENTS..md](ASSESSMENT-COMPONENTS..md) for the intended marking focus.
3. [README.md](README.md) for the implemented system overview and run instructions.
4. [symbolic_clock.py](symbolic_clock.py), [probabilistic_logic_reasoner.py](probabilistic_logic_reasoner.py), and [run_experiments.py](run_experiments.py) for the main technical contribution.
