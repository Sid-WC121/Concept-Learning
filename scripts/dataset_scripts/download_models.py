from pathlib import Path
import tarfile
import urllib.request
import zipfile
import sys

sys.path.append(str(Path(__file__).resolve().parents[2]))
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


def resave_keras_models():
    """Generate VGG16 and ResNet50 Keras models with ImageNet weights."""
    from src.util import save_vgg_model, save_resnet_model
    keras_dir = ensure_dir(DATASET_ROOT / "models" / "keras")

    vgg16_path = keras_dir / "model_vgg16.h5"
    if not vgg16_path.exists():
        print(f"Generating {vgg16_path} (downloads ImageNet weights)...")
        save_vgg_model(lambda w: w, str(vgg16_path))
    print(f"VGG16 ready: {vgg16_path}")

    vgg16_robust_path = keras_dir / "model_vgg16_robust.h5"
    if not vgg16_robust_path.exists():
        print(f"Generating {vgg16_robust_path}...")
        from src.util import perturb_weights
        save_vgg_model(perturb_weights, str(vgg16_robust_path))
    print(f"VGG16 Robust ready: {vgg16_robust_path}")

    vgg16_responsive_path = keras_dir / "model_vgg16_responsive.h5"
    if not vgg16_responsive_path.exists():
        print(f"Generating {vgg16_responsive_path}...")
        from src.util import responsive_weights
        save_vgg_model(responsive_weights, str(vgg16_responsive_path))
    print(f"VGG16 Responsive ready: {vgg16_responsive_path}")

    resnet_path = keras_dir / "model_resnet.h5"
    if not resnet_path.exists():
        print(f"Generating {resnet_path} (downloads ImageNet weights)...")
        save_resnet_model(lambda w: w, str(resnet_path))
    print(f"ResNet50 ready: {resnet_path}")


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

    resave_keras_models()


if __name__ == "__main__":
    main()
