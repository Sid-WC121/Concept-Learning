import argparse
import json
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram
from scipy.spatial.distance import cosine

from scripts.dataset_scripts.download_models import main as download_models
from src.concept_vectors import (
    load_cem_vectors_simple,
    load_concept2vec_vectors_simple,
    load_label_vectors_simple,
    load_tcav_vectors_simple,
)
from src.create_vectors import create_concept2vec, create_tcav_dataset
from src.dataset import DSprites_Dataset, MNIST_Dataset
from src.hierarchy import create_hierarchy, create_ward_hierarchy
from src.metrics import embedding_distance, flat_distance_to_square, get_concept_distances
from src.paths import RESULTS_ROOT, ensure_dir


SEEDS = [43, 44, 45]
METHODS = {
    "label": load_label_vectors_simple,
    "concept2vec": load_concept2vec_vectors_simple,
    "tcav": load_tcav_vectors_simple,
    "cem": load_cem_vectors_simple,
}


def basis_exists(method, attribute, dataset, seed):
    try:
        vector = method(attribute, dataset, "", seed)
        return np.asarray(vector).size > 0
    except Exception:
        return False


def cem_expected_manifest(dataset, seed, epochs, validation_epochs, sample):
    return {
        "experiment_name": dataset.experiment_name,
        "seed": seed,
        "num_epochs": epochs,
        "validation_epochs": validation_epochs,
        "sample_train": sample,
        "sample_valid": sample,
        "sample_test": sample,
        "concept_pair_loss_weight": 0.0,
        "c_extractor_arch": "resnet34",
        "n_concepts": len(dataset.get_attributes()),
    }

def manifest_matches(path, expected):
    if not path.exists():
        return False
    try:
        actual = json.loads(path.read_text())
    except Exception:
        return False
    return all(actual.get(key) == value for key, value in expected.items())

def cem_basis_complete(dataset, seed, epochs, validation_epochs, sample):
    root = RESULTS_ROOT / "bases" / "cem" / dataset.experiment_name / str(seed)
    expected = cem_expected_manifest(dataset, seed, epochs, validation_epochs, sample)
    if not manifest_matches(root / "manifest.json", expected):
        return False
    for attribute in dataset.get_attributes():
        try:
            vector = load_cem_vectors_simple(attribute, dataset, "", seed)
        except Exception:
            return False
        if np.asarray(vector).size == 0:
            return False
    return True


def train_tcav(dataset, attributes, seeds, num_random_exp, images_per_folder):
    download_models()
    if dataset.experiment_name == "mnist":
        model_name = "GoogleNet"
        bottlenecks = ["mixed4c"]
    else:
        model_name = "VGG16"
        bottlenecks = ["block4_conv1"]

    for seed in seeds:
        for attribute in attributes:
            if dataset.experiment_name == "dsprites" and attribute in {"is_white", "is_scale_0.9"}:
                print(f"TCAV skips synthetic dSprites attribute {attribute} seed {seed}")
                continue
            if basis_exists(load_tcav_vectors_simple, attribute, dataset, seed):
                print(f"TCAV exists {attribute} seed {seed}")
                continue
            print(f"Training TCAV {attribute} seed {seed}")
            create_tcav_dataset(
                attribute,
                dataset,
                num_random_exp=num_random_exp,
                max_examples=images_per_folder,
                images_per_folder=images_per_folder,
                seed=seed,
                suffix="",
                model_name=model_name,
                bottlenecks=bottlenecks,
            )


def train_cem(dataset, seeds, epochs, validation_epochs, num_gpus, num_workers, sample):
    for seed in seeds:
        if cem_basis_complete(dataset, seed, epochs, validation_epochs, sample):
            print(f"CEM exists seed {seed}")
            continue
        print(f"Training CEM seed {seed}")
        command = [
            sys.executable,
            "-m",
            "scripts.cem_scripts.extract_cem_concepts",
            "--experiment_name",
            dataset.experiment_name,
            "--num_gpus",
            str(num_gpus),
            "--num_epochs",
            str(epochs),
            "--validation_epochs",
            str(validation_epochs),
            "--seed",
            str(seed),
            "--num_workers",
            str(num_workers),
            "--sample_train",
            str(sample),
            "--sample_valid",
            str(sample),
            "--sample_test",
            str(sample),
            "--concept_pair_loss_weight",
            "0",
        ]
        subprocess.run(command, check=True)

def dataset_from_name(name):
    if name == "mnist":
        return MNIST_Dataset()
    if name == "dsprites":
        return DSprites_Dataset()
    raise ValueError(f"Unsupported dataset {name}")


def concept_pair_agreement(method, dataset, seed):
    all_vectors = [
        np.mean(np.array(method(attribute, dataset, "", seed)), axis=0)
        for attribute in dataset.get_attributes()
    ]
    all_vectors = np.array(all_vectors)

    closest_vectors = []
    for i, current_vector in enumerate(all_vectors):
        similarities = [
            1 - cosine(current_vector, other_vector)
            for other_vector in all_vectors
        ]
        closest_index = np.argmax(similarities[:i] + similarities[i + 1:])
        if closest_index >= i:
            closest_index += 1
        closest_vectors.append(closest_index)

    attributes = dataset.get_attributes()
    if dataset.experiment_name == "mnist":
        correct_vectors = []
        for i in range(0, len(attributes), 2):
            correct_vectors.append(i + 1)
            correct_vectors.append(i)
        return float(np.mean(np.array(closest_vectors) == np.array(correct_vectors)))

    correct = []
    for i, closest_index in enumerate(closest_vectors):
        family = concept_family(attributes[i])
        same_family = [
            j for j, attribute in enumerate(attributes)
            if j != i and concept_family(attribute) == family
        ]
        if same_family:
            correct.append(closest_index in same_family)
    return float(np.mean(correct)) if correct else float("nan")


def concept_family(attribute):
    if attribute.endswith("_color") or attribute.endswith("_number"):
        return attribute.split("_", 1)[0]
    if attribute.startswith("is_scale"):
        return "scale"
    if attribute.startswith("is_orientation"):
        return "orientation"
    if attribute in {"is_square", "is_ellipse", "is_heart"}:
        return "shape"
    if attribute.startswith("is_x"):
        return "x"
    if attribute.startswith("is_y"):
        return "y"
    return attribute


def save_hierarchy_plot(method_name, method, dataset, attributes):
    linkage_matrix = create_ward_hierarchy(
        get_concept_distances(method, dataset, "", attributes, 43)
    )
    plt.figure(figsize=(12, 7))
    dendrogram(linkage_matrix, labels=attributes, leaf_rotation=90)
    plt.tight_layout()
    output_dir = ensure_dir(RESULTS_ROOT / "figures" / dataset.experiment_name)
    plt.savefig(output_dir / f"{method_name}_hierarchy.png", dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Run concept basis pipeline.")
    parser.add_argument("--dataset", choices=["mnist", "dsprites"], default="mnist")
    parser.add_argument("--cem-epochs", type=int, default=50)
    parser.add_argument("--cem-validation-epochs", type=int, default=25)
    parser.add_argument("--cem-num-gpus", type=int, default=1)
    parser.add_argument("--cem-num-workers", type=int, default=0)
    parser.add_argument("--cem-sample", type=float, default=1.0)
    parser.add_argument("--tcav-random", type=int, default=3)
    parser.add_argument("--tcav-images", type=int, default=100)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--skip-concept2vec", action="store_true")
    parser.add_argument("--skip-cem", action="store_true")
    parser.add_argument("--skip-tcav", action="store_true")
    parser.add_argument("--skip-cem-eval", action="store_true")
    parser.add_argument("--skip-tcav-eval", action="store_true")
    args = parser.parse_args()

    dataset = dataset_from_name(args.dataset)
    attributes = dataset.get_attributes()

    train = dataset.get_data(train=True)
    val = dataset.get_data(train=False)
    assert len(train) > 0
    assert len(val) > 0
    assert len(attributes) == len(train[0]["attribute_label"])

    if not args.eval_only and not args.skip_concept2vec:
        for seed in SEEDS:
            print(f"Training Concept2Vec seed {seed}")
            create_concept2vec(
                dataset,
                "",
                seed=seed,
                embedding_size=32,
                num_epochs=5,
                dataset_size=1000,
                initial_embedding=None,
            )

    if not args.eval_only and not args.skip_tcav:
        train_tcav(
            dataset,
            attributes,
            SEEDS,
            num_random_exp=args.tcav_random,
            images_per_folder=args.tcav_images,
        )

    if not args.eval_only and not args.skip_cem:
        train_cem(
            dataset,
            SEEDS,
            epochs=args.cem_epochs,
            validation_epochs=args.cem_validation_epochs,
            num_gpus=args.cem_num_gpus,
            num_workers=args.cem_num_workers,
            sample=args.cem_sample,
        )

    methods = dict(METHODS)
    if args.skip_tcav_eval:
        methods.pop("tcav", None)
    if args.skip_cem_eval:
        methods.pop("cem", None)
    if "cem" in methods:
        missing = [
            seed for seed in SEEDS
            if not cem_basis_complete(
                dataset,
                seed,
                args.cem_epochs,
                args.cem_validation_epochs,
                args.cem_sample,
            )
        ]
        if missing:
            raise RuntimeError(
                "CEM vectors do not match requested run settings for "
                f"{dataset.experiment_name} seeds {missing}. "
                "Run CEM training or pass --skip-cem-eval."
            )

    baseline_distances = np.zeros((len(attributes), len(attributes)))
    for i, attribute_1 in enumerate(attributes):
        for j, attribute_2 in enumerate(attributes):
            baseline_distances[i][j] = 1 - int(concept_family(attribute_1) == concept_family(attribute_2))

    distance_by_method = {}
    agreement_by_method = {}
    hierarchy_dir = ensure_dir(RESULTS_ROOT / "evaluation" / f"{dataset.experiment_name}_hierarchies")

    for method_name, method in methods.items():
        print(f"Evaluating {method_name}")
        distance_scores = []
        agreement_scores = []

        for seed in SEEDS:
            distances = flat_distance_to_square(
                get_concept_distances(method, dataset, "", attributes, seed)
            )
            distance_scores.append(
                embedding_distance(distances, baseline_distances, k=1)
            )
            agreement_scores.append(concept_pair_agreement(method, dataset, seed))

        distance_by_method[method_name] = [
            float(np.mean(distance_scores)),
            float(np.std(distance_scores)),
        ]
        agreement_by_method[method_name] = agreement_scores

        hierarchy = create_hierarchy(
            create_ward_hierarchy,
            method,
            dataset,
            "",
            attributes,
            43,
        )
        (hierarchy_dir / f"{method_name}.txt").write_text(str(hierarchy))
        save_hierarchy_plot(method_name, method, dataset, attributes)

    ablation_dir = ensure_dir(RESULTS_ROOT / "evaluation" / "ablation")
    (ablation_dir / f"distance_{dataset.experiment_name}.json").write_text(
        json.dumps(distance_by_method, indent=2)
    )
    (ablation_dir / f"agreement_{dataset.experiment_name}.json").write_text(
        json.dumps(agreement_by_method, indent=2)
    )

    print(f"{dataset.experiment_name} pipeline complete")
    print(json.dumps(distance_by_method, indent=2))
    print(json.dumps(agreement_by_method, indent=2))


if __name__ == "__main__":
    main()
