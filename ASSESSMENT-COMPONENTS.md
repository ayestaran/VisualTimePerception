# Components for Assessment

This document is meant to separate the parts of the project that are central to the actual research and implementation from the parts that are mainly there to support the project. I have used [Custom-Mini-Project.md](Custom-Mini-Project.md) as the main reference because it explains the aims of the project and the kind of work that should be considered most important.

The main value of this project is in the perceptual-to-symbolic pipeline, the probabilistic reasoning component, and the experimental evaluation. The web interface is useful because it lets the system be shown clearly and tested interactively, but it should not be treated as the main academic contribution.

---

## 1. Core Markable Components

These are the parts of the project that I would consider the most important for marking, because they are the most directly connected to the project aims and the main technical contribution.

### A. Visual-to-Symbolic Conversion Pipeline

This is the core of the project. It is where the system takes visual input from a clock image and turns it into structured symbolic information that can then be reasoned about.

- [symbolic_clock.py](symbolic_clock.py)
  - Contains the main symbolic interpretation logic.
  - Converts visual clock information into symbolic representations.
  - Forms the central part of the perceptual-to-symbolic pipeline.

- [project_utils.py](project_utils.py)
  - Provides supporting utilities used by the main conversion process.
  - Helps organise and process the information needed for symbolic interpretation.

- [self_test_symbolic.py](self_test_symbolic.py)
  - Used to validate the symbolic system and check that it behaves as expected.
  - Provides evidence that the core pipeline is working in a reliable way.

These files are important because they directly support the first objective of the project: building a system that can convert visual information into symbolic data for reasoning.

### B. Probabilistic Logic Reasoning

This part is important because it adds a reasoning layer to the system rather than just relying on perception alone. It allows the project to handle uncertainty and make more robust symbolic interpretations.

- [probabilistic_logic_reasoner.py](probabilistic_logic_reasoner.py)
  - Implements the probabilistic reasoning component.
  - Supports inference and symbolic interpretation under uncertainty.
  - Helps strengthen the system beyond simple visual detection.

This is one of the most academically interesting parts of the project because it connects perception with reasoning in a more structured and intelligent way.

### C. Experimental Evaluation and Robustness Testing

This part shows that the project is not only built, but also tested and analysed properly. It is important because the project includes a reproducible experimental component rather than just a prototype.

- [run_experiments.py](run_experiments.py)
  - Runs the experiment sweep and controls the testing conditions.
  - Allows the system to be evaluated across different degradation settings.

- [stats_summary.py](stats_summary.py)
  - Produces the summary statistics and results used to assess performance.
  - Supports the evaluation of robustness and significance.

- [results/](results/)
  - Contains the generated outputs and evidence from the experiments.
  - Useful for demonstrating that the project produced measurable results.

These files are important because they support the third objective of the project: analysing the system under controlled perturbations and showing how well it performs.

### D. Integration Layer

- [app.py](app.py)
  - Connects the backend processing pipeline to the application environment.
  - Helps the whole system run as an integrated application.

This is still relevant for marking, but it is more of an integration component rather than the main intellectual contribution of the project.

---

## 2. UI Harness / Demonstration Layer

These files are mainly there to present the project and make it easier to interact with. They are useful, but they are not the main focus of the work.

- [templates/index.html](templates/index.html)
  - The main HTML structure for the interface.

- [static/app.js](static/app.js)
  - The front-end logic and interaction behaviour.

- [static/style.css](static/style.css)
  - The styling and visual presentation of the demo.

- [launch.sh](launch.sh)
  - A helper for launching the application.

- [prerequisites.sh](prerequisites.sh)
  - A helper for setting up the environment.

- [requirements.txt](requirements.txt)
  - Lists the dependencies needed to run the project.

These files support the demonstration of the system, but they do not represent the main technical or academic contribution of the project.

---

## 3. Suggested Marking Focus

If I were explaining the project to a marker, I would highlight the following balance:

- Around 60–70%: the core symbolic conversion system
- Around 20–25%: the probabilistic reasoning component
- Around 10–15%: the experimental evaluation and robustness testing
- A smaller portion: integration and presentation quality

In short, the main mark should come from the core algorithmic and reasoning work in [symbolic_clock.py](symbolic_clock.py), [probabilistic_logic_reasoner.py](probabilistic_logic_reasoner.py), and the evaluation workflow in [run_experiments.py](run_experiments.py) and [stats_summary.py](stats_summary.py). The UI files should be seen as supporting infrastructure that helps present the work rather than as the main deliverable.
