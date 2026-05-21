"""Prepare CheXpert-v1.0-small (ashery/chexpert) for the Concept-Learning repo.

Dataset: https://www.kaggle.com/datasets/ashery/chexpert

AUTO-DOWNLOAD (recommended)
  Requires a Kaggle account and the 'kaggle' package (pip install kaggle).
  Get a token at https://www.kaggle.com -> API -> Create New Token.

  Windows cmd.exe / conda prompt (default):
    set KAGGLE_API_TOKEN=KGAT_...
    python scripts/dataset_scripts/download_chexpert.py

  Windows PowerShell:
    $env:KAGGLE_API_TOKEN = "KGAT_..."
    python scripts/dataset_scripts/download_chexpert.py

  Linux / macOS:
    export KAGGLE_API_TOKEN=KGAT_...
    python scripts/dataset_scripts/download_chexpert.py

  Alternatively, place kaggle.json at ~/.kaggle/kaggle.json and run the
  script with no flags.

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
import zipfile
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
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return True
    config_dir = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (config_dir / "kaggle.json").exists()


def download_via_kaggle(dest_dir: Path) -> Path:
    try:
        import kaggle
        kaggle.api.authenticate()
    except ImportError:
        sys.exit("ERROR: Run 'pip install kaggle' then set KAGGLE_API_TOKEN and retry.")
    except Exception as exc:
        sys.exit(
            f"ERROR: Kaggle auth failed: {exc}\n"
            "Set $env:KAGGLE_API_TOKEN = 'KGAT_...' (PowerShell) or "
            "export KAGGLE_API_TOKEN=KGAT_... (bash) and retry.\n"
            "Get a token at https://www.kaggle.com -> Settings -> API."
        )

    zip_path = dest_dir / "chexpert.zip"
    if not zip_path.exists():
        print(f"Downloading {KAGGLE_DATASET} from Kaggle (~11 GB) ...")
        kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(dest_dir), unzip=False, quiet=False)
        zips = list(dest_dir.glob("*.zip"))
        if not zips:
            sys.exit("ERROR: Download completed but no zip file found.")
        if zips[0] != zip_path:
            zips[0].rename(zip_path)
    else:
        print(f"Zip already present at {zip_path}, skipping download.")

    print("Extracting ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)

    found = sorted(dest_dir.rglob("train.csv"))
    if not found:
        sys.exit("ERROR: train.csv not found after extraction.")
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

    if args.source_dir is not None:
        source_dir = args.source_dir.resolve()
        print(f"Using source: {source_dir}")
    elif kaggle_credentials_present():
        dl_dir = args.kaggle_download_dir or (CHEXPERT_DIR / "_download")
        ensure_dir(dl_dir)
        source_dir = download_via_kaggle(dl_dir)
        print(f"Source extracted to: {source_dir}")
    else:
        sys.exit(
            "No Kaggle credentials found and no --source-dir given.\n\n"
            "Quickest setup:\n"
            "  pip install kaggle\n"
            "  # Get token at https://www.kaggle.com -> Settings -> API -> Create New Token\n"
            "  set KAGGLE_API_TOKEN=KGAT_...         (cmd.exe / conda prompt)\n"
            "  $env:KAGGLE_API_TOKEN = 'KGAT_...'   (PowerShell)\n"
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
    print('Verify: python -c "from src.dataset import Chexpert_Dataset; d=Chexpert_Dataset(); print(len(d.get_data()), len(d.get_attributes()))"')


if __name__ == "__main__":
    main()
