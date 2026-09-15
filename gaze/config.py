"""Project-wide paths, class definitions and MediaPipe landmark indices."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "personal_data_small"
RESULTS_DIR = ROOT / "results"
ASSETS_DIR = ROOT / "assets"

#: Class order is alphabetical because every split was built with
#: ``image_dataset_from_directory``, which sorts class folders alphabetically.
CLASSES = ("front", "left", "right")
FRONT, LEFT, RIGHT = 0, 1, 2

CNN_INPUT_SIZE = (128, 128)
CALIBRATION_PATH = MODELS_DIR / "geometric_calibration.json"
CNN_FINETUNED = MODELS_DIR / "head_dir_effnetb0_ft.keras"
CNN_PRETRAINED = MODELS_DIR / "head_dir_effnetb0_lfr.keras"

# --- MediaPipe Face Mesh indices (canonical face model, subject-relative names) ---
NOSE_TIP = 1
CHEEK_LEFT = 234
CHEEK_RIGHT = 454

# Eye corners as (first, second); the vector first->second points along +x in image space.
EYE_CORNERS_LEFT = (33, 133)
EYE_CORNERS_RIGHT = (362, 263)

# Iris landmarks, only populated when Face Mesh runs with refine_landmarks=True.
IRIS_LEFT = tuple(range(468, 473))
IRIS_RIGHT = tuple(range(473, 478))

# Six-point rings used for the eye-aspect-ratio blink signal.
EAR_LEFT = (33, 160, 158, 133, 153, 144)
EAR_RIGHT = (362, 385, 387, 263, 373, 380)
