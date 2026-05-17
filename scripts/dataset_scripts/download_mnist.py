from pathlib import Path
import sys
import tarfile

import gdown

sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.paths import DATASET_ROOT, ensure_dir
from scripts.dataset_scripts.unpack_colored_mnist import create_dataset


url = "https://drive.google.com/u/0/uc?id=1NSv4RCSHjcHois3dXjYw_PaLIoVlLgXu&export=download"
output = DATASET_ROOT / "colored_mnist" / "colored_mnist.tar.gz"

ensure_dir(output.parent)
gdown.download(url, str(output))

with tarfile.open(output) as tar:
    tar.extractall(path=DATASET_ROOT)

create_dataset(write_images=True)
