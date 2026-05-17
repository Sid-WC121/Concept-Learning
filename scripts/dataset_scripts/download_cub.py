from pathlib import Path
import os
import pickle
import shutil
import sys
import tarfile
import urllib.request

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.paths import DATASET_ROOT, ensure_dir


DATASET_BASE = str(DATASET_ROOT)
CUB_DIR = DATASET_ROOT / "CUB"
cub_url = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
archive_path = CUB_DIR / "CUB_200_2011.tgz"
ensure_dir(CUB_DIR)

if not archive_path.exists():
    print("Downloading CUB dataset (~1.1 GB)...")
    urllib.request.urlretrieve(cub_url, archive_path)
else:
    print("CUB archive already exists, skipping download.")

print("Extracting...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(CUB_DIR)

extracted_folder = CUB_DIR / "CUB_200_2011"
if extracted_folder.exists():
    for item in os.listdir(extracted_folder):
        destination = CUB_DIR / item
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        shutil.move(str(extracted_folder / item), str(CUB_DIR))
    os.rmdir(extracted_folder)

ensure_dir(CUB_DIR / "preprocessed")
ensure_dir(CUB_DIR / "metadata")
shutil.copy(CUB_DIR / "attributes.txt", CUB_DIR / "metadata" / "attributes.txt")
shutil.copy(CUB_DIR / "classes.txt", CUB_DIR / "metadata" / "classes.txt")

print("Generating train/val/test splits...")
sys.path.append(str(Path(__file__).resolve().parents[2] / "cem"))
from cem.data.CUB200.data_processing import extract_data

orig_cwd = os.getcwd()
os.chdir(DATASET_BASE)
try:
    train_data, val_data, test_data = extract_data("CUB")
finally:
    os.chdir(orig_cwd)

def normalize_img_paths(rows):
    for row in rows:
        img_path = Path(row["img_path"])
        if img_path.is_absolute():
            row["img_path"] = img_path.relative_to(DATASET_ROOT).as_posix()
        else:
            row["img_path"] = img_path.as_posix()
    return rows

train_data = normalize_img_paths(train_data)
val_data = normalize_img_paths(val_data)
test_data = normalize_img_paths(test_data)

print("Saving pickle files...")
with open(CUB_DIR / "preprocessed" / "train.pkl", "wb") as f:
    pickle.dump(train_data, f)
with open(CUB_DIR / "preprocessed" / "val.pkl", "wb") as f:
    pickle.dump(val_data, f)
with open(CUB_DIR / "preprocessed" / "test.pkl", "wb") as f:
    pickle.dump(test_data, f)
print("Done. CUB dataset is prepared.")
