# Notebooks

`legacy/` holds the original exploration notebooks **exactly as they were written during the
course**, including their outputs, dead ends and hard-coded absolute paths. They are kept as
an honest record of how the project actually developed, not as a supported interface.

They are not expected to run as-is: they reference `../models/` and `../../Datasets/` layouts
that no longer exist, and some need weights that are not committed
(see [`../models/README.md`](../models/README.md)).

One mechanical change was made: the 65 prediction panels embedded in
`face_dir_colab_finetune_eval.ipynb` were re-encoded from PNG to JPEG, taking the file from
10.5 MB to 1.9 MB so that GitHub renders it. Every cell and every output is still there.

**The maintained code lives in [`../gaze/`](../gaze) and [`../scripts/`](../scripts).**

| Notebook | What it did | What came of it |
| --- | --- | --- |
| `face_dir.ipynb` | First head-direction CNN on our own captures | Baseline; superseded |
| `face_dir_colab_baseline.ipynb` | Early Colab training runs | Superseded |
| `face_dir_colab_train_effnetb0.ipynb` | **EfficientNetB0 pre-training on LFR** (98 505 images, 10 + 6 epochs), then an abandoned EfficientNetB3 attempt at 300×300 | Produced `head_dir_effnetb0_lfr.keras`. Reached 73.7% on LFR val |
| `face_dir_colab_finetune_eval.ipynb` | **Fine-tuning on our 319 images** with augmentation and class weights, plus the first webcam loop | Produced `head_dir_effnetb0_ft.keras`, the shipped CNN |
| `final_predictor.ipynb` | The hybrid image + webcam predictor: custom CNN for head direction, 8-class eye-region CNN, MediaPipe with Haar fallback, hand-tuned fusion rules | The version demonstrated for the course. Reimplemented in `gaze/pipeline.py` after the class-order bug was found |
| `eye_detect_dir.ipynb` | 8-class eye-region gaze CNN | Measured at 45.95% on our data; dropped |
| `eyes_detect_closed.ipynb` | Open/closed eye classifier | Replaced by the landmark eye-aspect-ratio signal |
| `face_detect_drowsy.ipynb` | Drowsiness experiments on the Kaggle set | Outside the final scope |

## Why `final_predictor.ipynb` was rewritten

It was the working demo, and it did work — but for a reason worth recording. It declared its
classes as `['left', 'right', 'front']` while the model had been trained with Keras's
alphabetical `['front', 'left', 'right']`, and it fed OpenCV's native BGR to a model trained
on RGB. With both mistakes, the head-direction model scored 12.20% on our own data, below the
33% chance level.

The notebook compensated with hand-tuned rules such as:

```python
if (front >= 0.25 and 0.01 <= diff_front_right <= 0.18) or (right > 0.5 and left < 0.25) ...
```

Those thresholds were absorbing a systematic label permutation. Once the channel order and
class order were corrected the rules became unnecessary, and once the geometric estimator was
added the custom CNN became unnecessary too — it scores 54.50% against 95.06% for two
thresholds on a landmark ratio.

The lesson is in the [report](../docs/REPORT.md#10-what-changed-after-submission): a
below-chance model behind enough hand-tuned post-processing can still look like it works.

## Reproducing the CNN training

The two Colab notebooks document the training recipe. Nothing in this repository requires
re-running them, since both checkpoints are committed. If you do want to retrain, you will
need the LFR face dataset (see [`../data/README.md`](../data/README.md)); the pre-training
stage took roughly 10 minutes per epoch on the hardware available at the time.
