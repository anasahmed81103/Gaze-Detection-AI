# Models

Everything the pipeline needs is committed here — the repository is runnable straight after
`pip install -r requirements.txt`, with no downloads.

| File | Size | Role |
| --- | --- | --- |
| `geometric_calibration.json` | 2 KB | **The primary model.** Two thresholds plus the accuracies they achieved |
| `head_dir_effnetb0_ft.keras` | 16.3 MB | EfficientNetB0 fine-tuned on our 319 images. Fallback branch |
| `head_dir_effnetb0_lfr.keras` | 16.3 MB | EfficientNetB0 after LFR pre-training only. Kept for the ablation |

## `geometric_calibration.json`

The whole primary model is two floating-point numbers:

```json
"yaw": { "thresholds": { "low": -0.2356, "high": 0.2496 } }
```

`yaw_score > high` → `left`, `< low` → `right`, otherwise `front`. Fitted on the 319 training
images only. The file also carries the `iris` thresholds, the `ear` blink heuristic, the
dataset counts, the held-out accuracy and the full 5-fold cross-validation record, so the
provenance of every quoted number travels with the artefact.

Regenerate with:

```bash
python scripts/calibrate_geometric.py
```

## `head_dir_effnetb0_ft.keras`

ImageNet → LFR → our data. 4 053 414 parameters, 128×128 RGB input, softmax over
`['front', 'left', 'right']` (Keras's alphabetical order — see the warning below).

Two things will silently ruin your results if you load this yourself:

- **RGB, not BGR.** `cv2.imread` returns BGR; convert before predicting.
- **`efficientnet.preprocess_input`, then whole frames.** The model was trained on entire
  photographs, not face crops. Cropping to the detected face costs 6.2 accuracy points.

`gaze/cnn.py` handles both. Use it rather than calling `load_model` directly.

## Class order — the bug worth knowing about

Every split was built with `image_dataset_from_directory`, which sorts class folders
**alphabetically**: index 0 = `front`, 1 = `left`, 2 = `right`.

The original inference notebook declared `['left', 'right', 'front']` and fed BGR. With both
mistakes the model scored **12.20%** — below the 33% chance level. Correcting the channel
order and the class order took the same weights to 54.50%. The hand-tuned threshold rules in
`notebooks/legacy/final_predictor.ipynb` existed to paper over that mislabelling.

`gaze/config.py` defines `CLASSES` once, and every component imports it from there.

## Models trained during the project but not shipped

Available in the project's [Google Drive folder](https://drive.google.com/drive/folders/151msEwr06IfVHyn4Y62qvKnpHnutfUra?usp=drive_link).
None of them is needed to run anything here; each was measured and rejected.

| File | Size | Measured result | Why it is not shipped |
| --- | --- | --- | --- |
| `face_direction_model2.h5` | 66 MB | 54.50% at best | Custom 16.9 M-parameter CNN, four times the size of EfficientNetB0 for 40 points less accuracy |
| `eyes_dir_model.h5` | 15 MB | 45.95% mapped to 3 classes | 8-class eye-region CNN; near-uniform softmax on our eye crops, so it does not transfer |
| `open_close_eyes_model.h5` | 8 MB | not benchmarked | No labelled closed-eye split locally; the landmark EAR signal replaces it |
| `drowsy_detection_model.h5` | 128 MB | not benchmarked | Exceeds GitHub's 100 MB file limit and is outside the final scope |
| `final_model.pth` | 16 MB | not benchmarked | PyTorch head-pose experiment from the Hopenet reference track |

`notebooks/legacy/final_predictor.ipynb` references the first two by their original paths. It
is preserved as a historical record; the maintained equivalents are
`scripts/run_image.py` and `scripts/run_video.py`, which are faster and measurably more
accurate.
