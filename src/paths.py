"""Repository path helpers."""

from pathlib import Path
import os


REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(os.environ.get("CONCEPT_DATASET_DIR", REPO_ROOT / "dataset")).resolve()
RESULTS_ROOT = Path(os.environ.get("CONCEPT_RESULTS_DIR", REPO_ROOT / "results")).resolve()
INTERMEDIARY_ROOT = Path(os.environ.get("CONCEPT_INTERMEDIARY_DIR", REPO_ROOT / "intermediary")).resolve()
MODELS_ROOT = Path(os.environ.get("CONCEPT_MODELS_DIR", REPO_ROOT / "models")).resolve()


def ensure_parent(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
