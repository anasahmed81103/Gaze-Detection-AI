# Model card — gaze / head-direction estimation

Following the model-card format of Mitchell et al. (2019). All numbers are reproducible with
the scripts in [`../scripts/`](../scripts); sources are named for each claim.

## Model details

| | |
| --- | --- |
| Task | Single-frame 3-class classification: `front`, `left`, `right` |
| Version | 1.0.0 (2026-09) |
| Primary component | Geometric head-yaw estimator: 2 thresholds on one landmark-derived scalar |
| Fallback component | EfficientNetB0, 4 053 414 parameters, 128×128 RGB input |
| Upstream dependency | MediaPipe Face Mesh, 478 landmarks, `refine_landmarks=True` |
| Auxiliary outputs | Iris horizontal offset; eye-aspect-ratio blink signal |
| Artefacts | `models/geometric_calibration.json`, `models/head_dir_effnetb0_ft.keras`, `models/head_dir_effnetb0_lfr.keras` |
| Licence | MIT (code and weights) |
| Owners | Anas Ahmed, Ghulam Hussain, Ibrahim Junaid |

Label semantics are **image-space**: `left` means the face is turned towards the left edge of
the frame. For a non-mirrored capture that is the subject's own right.

## Intended use

- Coursework, teaching and research demonstrations of landmark-geometry versus CNN baselines.
- A starting point for attention-monitoring prototypes, given per-user calibration and a
  human in the loop.
- A reproducible small-data case study: what 400 images can and cannot support.

## Out-of-scope and prohibited use

- **Not an academic-dishonesty detector.** It reports where a head is pointed. Any inference
  about cheating is a human judgement with a real false-positive cost.
- **Not for consequential automated decisions** about any person — grading, discipline,
  hiring, insurance, or driver penalties.
- **Not a safety-critical driver monitor.** No validation exists at automotive standards, in
  vehicles, at night, or with infrared illumination.
- **Not a gaze-point estimator.** It cannot say where on a screen someone is looking.
- **Not validated for surveillance** of people who have not consented.

## Training and evaluation data

400 photographs of 3 consenting adult subjects, 2 phone cameras, 3 indoor rooms; class
balance 129 `front` / 142 `left` / 129 `right`; split 319 / 40 / 41. The geometric thresholds
are fitted only on the 319 training images. The CNN was pre-trained on the LFR face dataset
(98 505 images) and fine-tuned on the same 319. See [`../data/README.md`](../data/README.md).

## Quantitative performance

### Held-out random split (81 images, `val` + `test`)

| Method | Accuracy | Macro F1 |
| --- | --- | --- |
| Cascade (geometry + CNN fallback) | 96.30% | 0.963 |
| Geometric head-yaw | 95.06% | 0.951 |
| EfficientNetB0 fine-tuned | 80.25% | 0.794 |
| Geometric iris offset | 62.96% | 0.621 |
| EfficientNetB0 LFR pre-training only | 59.26% | 0.549 |
| Majority-class baseline | 35.80% | 0.176 |

Per class, geometric head-yaw — precision / recall / F1:

| Class | Precision | Recall | F1 |
| --- | --- | --- | --- |
| `front` | 0.867 | 1.000 | 0.929 |
| `left` | 1.000 | 0.931 | 0.964 |
| `right` | 1.000 | 0.923 | 0.960 |

The cascade's 1.24-point lead is a single frame out of 81 and is not meaningful at this
sample size. Restricted to the 78 frames where Face Mesh finds a face, the geometric
estimator is 98.72% accurate.

### Cross-validation (400 images, 5-fold stratified)

Geometric head-yaw **94.75% ± 1.66%**; geometric iris offset 55.50% ± 6.05%.

### Generalisation — leave-one-subject-out

| Held-out subject | Accuracy |
| --- | --- |
| A | 92.11% |
| B | 98.01% |
| C | 72.34% |
| **Mean** | **87.49% ± 10.98%** |

**87.5% is the figure that should be used to judge expected performance on a new person.**
The 96.30% headline reflects a random split in which every subject appears in training.

### Latency (CPU only, AMD Ryzen 5 3600, 640×480)

| Configuration | Median | p95 | FPS |
| --- | --- | --- | --- |
| Geometric pipeline | 5.94 ms | 19.01 ms | 168 |
| With CNN on every frame | 78.53 ms | 92.24 ms | 13 |

FP32 throughout. No GPU is used or required.

## Fairness and ethical considerations

- **The evaluation population is 3 young adult males.** Two of three wear glasses. Nothing
  here establishes performance across age, gender, ethnicity, facial hair, eyewear, head
  coverings or disability. Any deployment claim about fairness would be unsupported.
- **Per-subject spread is ±11%** with a worst case of 72.34%, so per-person variation is
  large and unmodelled.
- **The primary component inherits MediaPipe Face Mesh's biases.** Its detection behaviour
  across demographics is a property of Google's model, not of this work; we measure only an
  aggregate 98.0% detection rate on our own 3 subjects.
- **The images depict identifiable people** who consented to research and educational use.
- **Attention monitoring is a surveillance technology.** Used on students or drivers it
  should be disclosed, consented to, and never the sole basis for a consequential decision.

## Caveats and recommendations

- Assume roughly 87% on an unseen cooperative adult in good lighting, not 96%.
- **Calibrate per user.** Threshold transfer is the measured failure mode and the model is
  two scalars; a few seconds of frontal video should recover most of the gap.
- Prefer the geometric path. It is more accurate, 13× faster, resolution-invariant, and its
  errors are interpretable — you can read the score that produced any decision.
- Treat the eye-aspect-ratio blink output as uncalibrated: the threshold comes from the
  open-eye distribution because no labelled closed-eye split exists locally.
- Handle `label_name == 'unknown'` explicitly. It means no face was found, which is
  operationally different from `front`.
