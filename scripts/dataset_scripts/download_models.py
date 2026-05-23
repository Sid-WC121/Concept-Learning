from pathlib import Path
import tarfile
import urllib.request
import zipfile

from src.paths import DATASET_ROOT, ensure_dir


INCEPTION_URL = "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
MOBILENET_URLS = [
    "https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz",
    "https://drive.usercontent.google.com/download?id=1qwY7W_yM4lXahvM8pOVPqU_bVRtVC4Qi&export=download&confirm=t&uuid=6b5fa58b-7bd9-461b-b816-9a08ed1bc170",
]


def download(url, output):
    output = Path(output)
    if output.exists() and output.stat().st_size > 0:
        print(f"{output} already exists")
        return
    print(f"Downloading {url}")
    urllib.request.urlretrieve(url, output)


def main():
    model_root = ensure_dir(DATASET_ROOT / "models")

    inception_dir = ensure_dir(model_root / "inception5h")
    inception_zip = model_root / "inception5h.zip"
    graph_path = inception_dir / "tensorflow_inception_graph.pb"
    labels_path = inception_dir / "imagenet_comp_graph_label_strings.txt"
    if not (graph_path.exists() and labels_path.exists()):
        download(INCEPTION_URL, inception_zip)
        with zipfile.ZipFile(inception_zip) as archive:
            archive.extractall(inception_dir)
        if inception_zip.exists():
            inception_zip.unlink()
    print(f"Inception5h ready: {graph_path}")

    mobilenet_dir = ensure_dir(model_root / "mobilenet_v2_1.0_224")
    mobilenet_tgz = model_root / "mobilenet_v2_1.0_224.tgz"
    if not any(mobilenet_dir.iterdir()):
        for url in MOBILENET_URLS:
            try:
                download(url, mobilenet_tgz)
                with tarfile.open(mobilenet_tgz) as archive:
                    archive.extractall(mobilenet_dir)
                if mobilenet_tgz.exists():
                    mobilenet_tgz.unlink()
                print(f"MobileNet ready: {mobilenet_dir}")
                break
            except Exception as exc:
                if mobilenet_tgz.exists():
                    mobilenet_tgz.unlink()
                print(f"Skipping MobileNet source {url}: {exc}")
        else:
            print("MobileNet not available; Inception5h is enough for TCAV.")
    else:
        print(f"MobileNet ready: {mobilenet_dir}")


if __name__ == "__main__":
    main()
