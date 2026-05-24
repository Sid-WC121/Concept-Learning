"""Prepare CheXpert-v1.0-small (ashery/chexpert) for the Concept-Learning repo.

Dataset: https://www.kaggle.com/datasets/ashery/chexpert

AUTO-DOWNLOAD (recommended)
  Requires kagglehub (pip install kagglehub) and a Kaggle API token.
  Get a token at https://www.kaggle.com -> Settings -> API -> Create New Token.

  Linux / macOS / WSL:
    export KAGGLE_API_TOKEN=KGAT_...
    python scripts/dataset_scripts/download_chexpert.py

  Windows:
    set KAGGLE_API_TOKEN=KGAT_... && python scripts/dataset_scripts/download_chexpert.py

MANUAL (if the zip is already downloaded and extracted):
    python scripts/dataset_scripts/download_chexpert.py \\
        --source-dir <folder-containing-train.csv-and-valid.csv>

OUTPUT
  dataset/chexpert/images/<patient>/<study>/<view>.jpg
  dataset/chexpert/preprocessed/train.pkl
  dataset/chexpert/preprocessed/val.pkl
  dataset/chexpert/preprocessed/test.pkl  (same as val)

Each pickle is a list of dicts matching the CUB/MNIST format:
  { 'id': int,
    'img_path': 'chexpert/images/<patient>/<study>/<view>.jpg',
    'class_label': 0 (Findings) or 1 (No Finding),
    'attribute_label': [int]*13 }

The 13 concepts (order matches Chexpert_Dataset.get_attributes()):
  Enlarged Cardiomediastinum, Cardiomegaly, Lung Lesion, Lung Opacity,
  Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax,
  Pleural Effusion, Pleural Other, Fracture, Support Devices

Uncertain labels (-1) are treated as 0 (negative).
"""

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.paths import DATASET_ROOT, ensure_dir

KAGGLE_DATASET = "ashery/chexpert"
CHEXPERT_DIR = DATASET_ROOT / "chexpert"
IMAGES_DIR = CHEXPERT_DIR / "images"
PREP_DIR = CHEXPERT_DIR / "preprocessed"

CONCEPT_COLS = [
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Lesion", "Lung Opacity",
    "Edema", "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax",
    "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices",
]


def kaggle_credentials_present() -> bool:
    return bool(os.environ.get("KAGGLE_API_TOKEN"))


KAGGLEHUB_CACHE = Path.home() / ".cache" / "kagglehub"


def download_via_kaggle(dest_dir: Path) -> Path:
    try:
        import kagglehub
    except ImportError:
        sys.exit("ERROR: Run 'pip install kagglehub' then set KAGGLE_API_TOKEN and retry.")

    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit(
            "ERROR: KAGGLE_API_TOKEN not set.\n"
            "export KAGGLE_API_TOKEN=KGAT_... && python scripts/dataset_scripts/download_chexpert.py\n"
            "Get a token at https://www.kaggle.com -> Settings -> API."
        )

    print(f"Downloading {KAGGLE_DATASET} from Kaggle (~11 GB) ...")
    kagglehub.dataset_download(KAGGLE_DATASET)

    found = sorted(KAGGLEHUB_CACHE.rglob("train.csv"))
    if not found:
        sys.exit("ERROR: train.csv not found after download.")
    return found[0].parent


def binarise(value) -> int:
    try:
        return 1 if float(value) == 1.0 else 0
    except (ValueError, TypeError):
        return 0


def resolve_image_path(csv_path_str: str, split_root: Path):
    p = Path(csv_path_str)
    for i, part in enumerate(p.parts):
        if part in ("train", "valid"):
            candidate = split_root.parent / Path(*p.parts[i:])
            if candidate.exists():
                return candidate
    candidate = split_root.parent / p
    return candidate if candidate.exists() else None


def process_split(csv_path: Path, split_root: Path, split_name: str, id_offset: int) -> list:
    df = pd.read_csv(csv_path)
    if "Frontal/Lateral" in df.columns:
        df = df[df["Frontal/Lateral"].str.lower() == "frontal"].copy()
    else:
        df = df[df["Path"].str.contains("frontal", case=False)].copy()
    df = df.reset_index(drop=True)

    records, skipped = [], 0
    for idx, row in df.iterrows():
        src = resolve_image_path(str(row["Path"]), split_root)
        if not src or not src.exists():
            skipped += 1
            continue

        try:
            rel = src.relative_to(split_root)
        except ValueError:
            parts = src.parts
            start = next((i for i, p in enumerate(parts) if p.startswith("patient")), None)
            if start is None:
                skipped += 1
                continue
            rel = Path(*parts[start:])

        dest = IMAGES_DIR / rel
        ensure_dir(dest.parent)
        if not dest.exists():
            try:
                os.link(src, dest)
            except (OSError, NotImplementedError):
                shutil.copy2(src, dest)

        records.append({
            "id":              id_offset + int(idx),
            "img_path":        (IMAGES_DIR / rel).relative_to(DATASET_ROOT).as_posix(),
            "class_label":     1 if binarise(row.get("No Finding", 0)) else 0,
            "attribute_label": [binarise(row.get(c, 0)) for c in CONCEPT_COLS],
        })

    if skipped:
        print(f"  [{split_name}] skipped {skipped} rows (image not found).")
    return records


def _cleanup_source(source_dir: Path, dl_dir):
    kaggle_cache = KAGGLEHUB_CACHE / "datasets" / KAGGLE_DATASET
    if kaggle_cache.exists():
        size = sum(f.stat().st_size for f in kaggle_cache.rglob("*") if f.is_file())
        print(f"  Removing kagglehub cache ({size / 1e9:.1f} GB) ...")
        shutil.rmtree(kaggle_cache)
    if dl_dir and dl_dir.exists():
        shutil.rmtree(dl_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Download and prepare CheXpert-v1.0-small.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--source-dir", type=Path, default=None,
        help="Path to already-extracted CheXpert folder (contains train.csv, valid.csv, train/, valid/). "
             "Omit to auto-download via Kaggle API.",
    )
    parser.add_argument(
        "--kaggle-download-dir", type=Path, default=None,
        help="Where to store the Kaggle zip during auto-download (default: dataset/chexpert/_download).",
    )
    args = parser.parse_args()

    source_dir = args.source_dir.resolve() if args.source_dir else None
    dl_dir = None

    if source_dir is not None:
        print(f"Using source: {source_dir}")
    elif kaggle_credentials_present():
        dl_dir = args.kaggle_download_dir or (CHEXPERT_DIR / "_download")
        source_dir = download_via_kaggle(dl_dir)
        print(f"Source extracted to: {source_dir}")
    else:
        sys.exit(
            "No Kaggle credentials found and no --source-dir given.\n\n"
            "Quickest setup:\n"
            "  pip install kagglehub\n"
            "  export KAGGLE_API_TOKEN=KGAT_...\n"
            "  python scripts/dataset_scripts/download_chexpert.py\n\n"
            "Or pass an already-extracted folder:\n"
            "  python scripts/dataset_scripts/download_chexpert.py --source-dir <path>"
        )

    train_csv = source_dir / "train.csv"
    valid_csv = source_dir / "valid.csv"
    train_root = source_dir / "train"
    valid_root = source_dir / "valid"

    for p in [train_csv, valid_csv, train_root, valid_root]:
        if not p.exists():
            sys.exit(f"ERROR: Expected path missing: {p}\n"
                     "Make sure --source-dir contains train.csv, valid.csv, train/, valid/.")

    ensure_dir(IMAGES_DIR)
    ensure_dir(PREP_DIR)

    print("Processing train ...")
    train_records = process_split(train_csv, train_root, "train", 0)
    print(f" {len(train_records)} frontal samples")

    print("Processing val ...")
    val_records = process_split(valid_csv, valid_root, "valid", len(train_records))
    print(f" {len(val_records)} frontal samples")

    print("Writing pickles ...")
    for name, data in [("train", train_records), ("val", val_records), ("test", val_records)]:
        with open(PREP_DIR / f"{name}.pkl", "wb") as f:
            pickle.dump(data, f)

    print(f"\nDone — {len(train_records)} train / {len(val_records)} val records.")

    print("Cleaning up downloaded source files ...")
    if args.source_dir is None and source_dir:
        _cleanup_source(source_dir, dl_dir)

    print('Verify: python -c "from src.dataset import Chexpert_Dataset; d=Chexpert_Dataset(); print(len(d.get_data()), len(d.get_attributes()))"')


if __name__ == "__main__":
    main()
