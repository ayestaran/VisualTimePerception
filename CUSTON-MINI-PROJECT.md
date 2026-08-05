# Custom Mini-Project
=====================

If you prefer, you can choose a custom mini project, as long as this fulfils several
constraints:

## Constraints
--------------

1. The custom project must involve at least one of the topics covered in this
module, i.e. logic-based systems, cognitive architectures, symbol grounding
approaches, or probabilistic reasoning.

2. The project needs to contain a substantial practical element with one or more
measurable outcomes. So, a mere literature review on a topic would not qualify.

## Eligible Types of Project Tasks
----------------------------------

We currently envision the following three types of mini-projects:

* Development of your own small AI system, potentially using pre-existing libraries
and frameworks

* Integration of an existing AI implementation into a larger system, such as a
cognitive architecture of the like

* Analysis and testing of an existing AI implementation

However, if you come up with a topic that does fulfil the constraints and does not fall
into any of the two categories above, feel free to suggest it.

Whatever your idea, you will need to contact the module leader and discuss it
either via email or MS Teams call before starting the project. Assuming the
project idea is accepted, we will need to agree on the scope of the project as well as
on a set of tasks and measurable criteria and mark allocations to these.

# Visual Time Perception & Symbolic Reasoning - Single & Continuous Conversion
==============================================================================

## Objectives
-------------

This project explores real‑time perceptual‑to‑symbolic conversion, translating visual clock frames into structured symbolic data for reasoning tasks. It integrates probabilistic logic reasoning and dataset generation. The environment includes the Symbolic Canvas Clock, a Grounding Preset, and a Dataset Builder, forming a complete perceptual‑to‑symbolic reasoning pipeline.

* Objective 1 — Develop a real‑time visual‑to‑symbolic conversion pipeline  
Implement a system that captures clock images, detects clock‑hand geometry using OpenCV, and converts these detections into structured symbolic representations suitable for reasoning in both single‑frame and continuous modes.

* Objective 2 — Integrate probabilistic logic reasoning to stabilise symbolic interpretation  
Design and evaluate a probabilistic inference module that computes MAP assignments, posterior marginals, and rule‑satisfaction masses to ensure robust symbolic interpretation under uncertainty, perception shortages, and perturbation conditions.

* Objective 3 — Analyse robustness and grounding strategies across controlled perturbations  
Conduct a reproducible experiment sweep across multiple degradation conditions (rotation, blur, occlusion, noise, flips) and evaluate how grounding‑preset strategies influence detection accuracy, inference reliability, and error‑trend behaviour using descriptive statistics and ANOVA‑based significance testing.

## Expected Outcomes
--------------------

* A working perceptual-to-symbolic conversion system demonstrating symbol grounding in continuous computation.
* A dataset generator producing structured symbolic samples for reasoning analysis.
* A written evaluation of how perceptual grounding supports symbolic reasoning within cognitive AI frameworks.
* A reflection on how symbolic reasoning can be integrated into modern computational design and development practices.
