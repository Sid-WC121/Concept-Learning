from pathlib import Path
import sys
import urllib.request

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.dataset import write_ten_dsprites
from src.paths import DATASET_ROOT, ensure_dir


url = "https://github.com/google-deepmind/dsprites-dataset/raw/master/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"
output = DATASET_ROOT / "dsprites" / "dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz"

ensure_dir(output.parent)
ensure_dir(DATASET_ROOT / "dsprites" / "images")
ensure_dir(DATASET_ROOT / "dsprites" / "preprocessed")

if not output.exists():
    print("Downloading dSprites...")
    urllib.request.urlretrieve(url, output)
else:
    print("dSprites archive already exists, skipping download.")

write_ten_dsprites()
