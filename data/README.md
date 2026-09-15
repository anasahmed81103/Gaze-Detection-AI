# Data

## `personal_data_small/` — committed

400 labelled photographs captured by the authors, the only dataset redistributed here.

```
personal_data_small/
├── train/{front,left,right}/   319 images
├── val/{front,left,right}/      40 images
└── test/{front,left,right}/     41 images
```

| Class | Images | Meaning |
| --- | --- | --- |
| `front` | 129 | Face pointed at the camera |
| `left` | 142 | Face turned towards the **left edge of the frame** |
| `right` | 129 | Face turned towards the **right edge of the frame** |

Labels are defined in **image space**, not relative to the subject. For a non-mirrored
capture, `left` is the subject's own right. Use `scripts/run_video.py --mirror` if you want
subject-relative semantics from a webcam.

### Subjects and sessions

Three consenting adult subjects across three capture sessions. Subject identity is recoverable
from the filenames, which `scripts/cross_subject_eval.py` uses for leave-one-subject-out
evaluation:

| Group | Filename pattern | Images | Device | Setting |
| --- | --- | --- | --- | --- |
| A | `20250504_032*` | 152 | Galaxy S22 | patterned wall, warm light |
| B | `20250504_033*`, `20250504_034*` | 201 | Galaxy S22 | cream wall, warm light |
| C | `PXL_20250504_*` | 47 | Google Pixel | plain wall, cool light |

The committed `train` / `val` / `test` split is random over all images, so **all three
subjects appear in all three splits**. Accuracy on that split therefore measures "new photos
of a known person". For generalisation to a new person, use the leave-one-subject-out
protocol; it scores 87.49% ± 10.98% against 95.06% on the random split.

### Resolution

Images are downscaled to **512 px on the longest side** (JPEG quality 92) from originals of
roughly 2000×1500, which takes the folder from 110.7 MB to 19.7 MB.

This is a deliberate, measured trade-off. `scripts/resolution_sweep.py` shows the geometric
estimator is unaffected — 95.06% at every scale from 512 px down to 128 px — while the CNN
loses about 4 points relative to the originals (83.95% → 80.25% on the held-out set). All
numbers in the README and report are computed on the committed 512 px images unless
explicitly labelled as original-resolution.

### Privacy

These photographs depict identifiable people who consented to their use for research and
educational purposes. Please do not redistribute them separately from this repository, and do
not use them to train commercial biometric or surveillance systems.

## Datasets used but not committed

Used during earlier training and exploration. Too large to redistribute, and in most cases
covered by their own licences — download them from the original sources if you want to
reproduce the pre-training stages.

| Dataset | Size used | Role | Outcome |
| --- | --- | --- | --- |
| LFR face dataset | 98 505 / 21 108 / 21 111 | EfficientNetB0 3-class pre-training | 73.7% on its own val split; 59.26% transferred to our photos |
| [Kaggle: Drowsy Detection](https://www.kaggle.com/datasets) | — | Head-tilt and drowsiness exploration | Not in the final pipeline |
| [Kaggle: Open and Closed Eyes in the Wild](https://www.kaggle.com/datasets) | — | Eye-state CNN | Not in the final pipeline |
| [Kaggle: Eye Gaze](https://www.kaggle.com/datasets) | 8-class eye regions | Eye-region gaze CNN | 45.95% when mapped to our 3 classes — near chance |

The original course report records why the public sets disappointed: most were synthetic or
class-imbalanced, and the eye-gaze set contained single-eye crops incompatible with a model
that needs both eyes plus facial landmarks. Our measurements agree — see the negative-results
table in the [README](../README.md#what-did-not-work).

## Reproducing the split

The committed split came from a random 80/10/10 partition per class:

```python
train_images, temp = train_test_split(images, test_size=0.2, random_state=42)
val_images, test_images = train_test_split(temp, test_size=0.5, random_state=42)
```

If you want a harder and more informative protocol, prefer `scripts/cross_subject_eval.py`
over resplitting.
