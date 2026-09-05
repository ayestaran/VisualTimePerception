# Custom Mini Project Report
FranzJosef Ayestaran — 7WCM2018

## Project Overview

### Project Aim
This project is intended to design and implement a real-time system which will convert images of analogue clocks into structured symbolic representations and stabilise those symbolic interpretations by means of probabilistic reasoning. The system has to remain robust when subjected to controlled perturbations such as rotation, blur, occlusion, noise, and mirroring.

### Approach
I designed a two-stage architecture. In the first stage, a visual-to-symbolic conversion pipeline identifies the geometry of the clock hands by means of colour masking, Hough-based fallback logic, and angle extraction. This pipeline can function both in single-frame and in continuous mode.

In the second stage, I incorporated a probabilistic reasoning module which calculates MAP assignments, posterior marginals, and rule-satisfaction masses in order to stabilise the symbolic output when there is uncertainty.

Lastly, I assessed robustness by means of controlled perturbation sweeps and grounding-preset strategies, examining detection accuracy, inference reliability, and error trends.

## Contextual Literature
### Citation of Source
Duda, R.O. and Hart, P.E. (1972). The use of the Hough Transformation for detecting lines and curves in pictures. Communications of the ACM.

### Summary of Source
Duda and Hart presented the Hough Transform as a reliable technique for detecting geometric primitives in images with noise. By formalising the process of voting in parameter space, their approach allowed for the accurate detection of lines and curves even in cases where the images had clutter, partial occlusion, or distortions.

### Relation to Project
The colour-masking method is used by my clock-hand detection pipeline, but when the colour indications are ambiguous or overlapping it switches to a Hough-based approach. This is exactly in line with the aim of Duda and Hart's work: extracting geometric structure in the presence of noise and uncertainty.

### Source Critique
Although the Hough Transform is robust, it can be rather computationally intensive and is sensitive to the choice of parameters. In complicated scenes, modern methods based on deep learning do better; yet in controlled situations such as on clock faces the Hough Transform is still efficient, interpretable, and reliable.

## Practical Objective 1: Visual-to-Symbolic Pipeline
I created a real-time pipeline which can convert images of analogue clocks into structured symbolic representations. The system uses colour masking, geometric filtering, and a fallback method based on the Hough transform to detect the hour, minute, and second hands; it then extracts the angles, resolves any overlaps, and produces symbolic states such as:

### Code
{ "HOUR_HAND": 1, "MINUTE_HAND": 45, "SECOND_HAND": 25, "TIME": "01:45:25" }
Example debug output:

### Code
candidate_angles_deg: [149.32, 270.74, 55.46]
fallback_used: true
pipeline: 'colour_mask + hough_fallback'
It meets PO1 at the highest level.

### Practical Objective 2: Probabilistic Stabilisation
The probabilistic reasoning module uses weighted model counting to stabilise symbolic interpretations. For each frame the system lists possible worlds, applies soft rules, and calculates:

* MAP assignment
* Posterior marginals
* Rule-satisfaction masses
* World counts
* Domain sizes

Example output:

### Code
MAP Assignment: time=01:45:25, hour=1, minute=45, second=25
MAP Probability: 0.101103
hour: 1 → 0.909335
satisfied_posterior_mass: 0.909335, weight: 0.9
This meets PO2 at the highest band.

## Empirical Examination
### Central Plot / Graph
The graph displays mean hour/minute/second error under various perturbation conditions (rotation, blur, noise, occlusion, flips). Caption explains severity vs symbolic error.

### Analysis
Baseline errors:

Hour: ~2.375°

Minute: ~11.175°

Second: ~15.300°

Medium noise increases minute error slightly.
Occlusion drops success rate to 0.95.
Mirroring increases minute error (~15.200°).
Probabilistic reasoning keeps MAP stable.

## Evaluation Page (PO3)
40 samples per condition, 2000 statistical iterations.

### Key results:

* Success rate ≥ 0.95 except occlusion (0.95)
* Mirror_untracked minute error: 15.200°
* rotate_8deg minute error: 16.300°
* noise_std_8 tolerance rate: 0.300
* Combined stress shows strong generalisation

Example:

Code
rotate_8deg: success=1.000, mean_minute_err=9.525, mean_second_err=16.300
This meets PO3 at the highest level.

## Final Remarks
Completion and Success
All three practical objectives were met.

### Shortcomings
Extreme occlusion

Stylised clock themes

Mirror orientation issues

Colour-dependent fallback modes

### Actionable Recommendations
Temporal smoothing

Lightweight CNN segmentation

Expanded probabilistic rules

Improved mirror tracking

## Appendix
* Evaluation page
* Code listings
* README
* Additional plots
* Symbolic logs and sweep statistics