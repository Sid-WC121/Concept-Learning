#!/usr/bin/env python3
"""
Unified evaluation script for all datasets.
Run separately per dataset or with --all for all datasets + cross-dataset analyses.

Usage:
    python scripts/evaluation/evaluate.py --dataset dsprites
    python scripts/evaluation/evaluate.py --dataset mnist
    python scripts/evaluation/evaluate.py --dataset cub
    python scripts/evaluation/evaluate.py --dataset chexpert
    python scripts/evaluation/evaluate.py --all
"""

import os
import sys
import json
import time
import numpy as np
from collections import defaultdict
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

conda_lib = os.path.join(sys.prefix, 'lib')
os.environ["LD_LIBRARY_PATH"] = (
    f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"
)

from src.dataset import CUB_Dataset, MNIST_Dataset, DSprites_Dataset, Chexpert_Dataset
from src.concept_vectors import (
    load_cem_vectors_simple, load_concept2vec_vectors_simple,
    load_label_vectors_simple, load_tcav_vectors_simple,
    load_shapley_vectors_simple
)
from src.util import *
from src.hierarchy import *
from src.metrics import (
    compute_all_metrics, truthfulness_metric_shapley, stability_metric,
    robustness_image_metric, responsiveness_image_metric,
    concept_purity_metric, max_similarity_metric, embedding_distance,
    get_concept_distances
)
from src.hierarchy import flat_distance_to_square
from src.create_vectors import *

SEEDS = [43, 44, 45]
VECTOR_METHODS = [
    load_cem_vectors_simple,
    load_concept2vec_vectors_simple,
    load_label_vectors_simple,
    load_tcav_vectors_simple,
]
VECTOR_NAMES = ["CEM", "Concept2Vec", "Label", "TCAV"]

DATASET_MAP = {
    "mnist": MNIST_Dataset,
    "cub": CUB_Dataset,
    "dsprites": DSprites_Dataset,
    "chexpert": Chexpert_Dataset,
}


def evaluate_dataset(dataset_name):
    """Run per-dataset evaluation for all vector methods."""
    dataset_class = DATASET_MAP[dataset_name]
    dataset = dataset_class()
    attributes = dataset.get_attributes()
    out_dir = f"results/evaluation/{dataset_name}"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs("results/evaluation/ablation", exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Evaluate {dataset_name}")
    print(f"{'=' * 60}")

    for method, name in zip(VECTOR_METHODS, VECTOR_NAMES):
        print(f"\n  --- {name} ---")

        available_seeds = []
        for s in SEEDS:
            try:
                v = method(attributes[0], dataset, "", s)
                if np.asarray(v).size > 0:
                    available_seeds.append(s)
            except Exception:
                pass
        active_seeds = available_seeds if available_seeds else SEEDS
        print(f"    Active seeds: {active_seeds}")

        # Truthfulness (needs VGG16 model, gracefully skip if unavailable)
        try:
            t_result = truthfulness_metric_shapley(method, dataset, attributes, active_seeds)
            results = {"Truthfulness": t_result}
        except Exception as e:
            results = {}
            print(f"    Truthfulness skipped: {e}")

        # Image Robustness (needs _image_robustness dataset variant)
        try:
            r_result = robustness_image_metric(method, dataset, attributes, active_seeds)
            results["Robustness"] = r_result
        except Exception as e:
            print(f"    Robustness skipped: {e}")

        # Image Responsiveness (needs _image_responsiveness dataset variant)
        try:
            resp_result = responsiveness_image_metric(method, dataset, attributes, active_seeds)
            results["Responsiveness"] = resp_result
        except Exception as e:
            print(f"    Responsiveness skipped: {e}")

        # Stability (no external model needed)
        try:
            s_result = stability_metric(method, dataset, attributes, active_seeds)
            results["Stability"] = s_result
        except Exception as e:
            print(f"    Stability skipped: {e}")

        if results:
            with open(f"{out_dir}/{dataset_name}_{name.lower()}.txt", "w") as w:
                for key in results:
                    w.write(f"{key}: {results[key]}\n")
            print(f"    -> {out_dir}/{dataset_name}_{name.lower()}.txt")
        else:
            print(f"    No metrics computed for {name}")

        # concept purity (silhouette)
        try:
            purity = concept_purity_metric(method, dataset, attributes, active_seeds)
            with open(f"{out_dir}/{dataset_name}_{name.lower()}_purity.json", "w") as w:
                json.dump({"concept_purity": list(purity)}, w)
            if np.isnan(purity[0]):
                print(f"    concept purity (silhouette): N/A (only 1 vector per concept)")
            else:
                print(f"    concept purity (silhouette): {purity[0]:.4f} +/- {purity[1]:.4f}")
            print(f"    -> {out_dir}/{dataset_name}_{name.lower()}_purity.json")
        except Exception as e:
            print(f"    concept_purity skipped: {e}")

        # max similarity (best-pair alignment instead of centroid averaging)
        try:
            maxsim = max_similarity_metric(method, dataset, attributes, active_seeds)
            with open(f"{out_dir}/{dataset_name}_{name.lower()}_maxsim.json", "w") as w:
                json.dump({"max_similarity": list(maxsim)}, w)
            print(f"    max similarity stability: {maxsim[0]:.4f} +/- {maxsim[1]:.4f}")
            print(f"    -> {out_dir}/{dataset_name}_{name.lower()}_maxsim.json")
        except Exception as e:
            print(f"    max_similarity skipped: {e}")

        # GCN classification accuracy
        try:
            from src.gcn_eval import gcn_classification_accuracy
            gcn_acc = gcn_classification_accuracy(method, dataset, attributes, active_seeds)
            with open(f"{out_dir}/{dataset_name}_{name.lower()}_gcn.json", "w") as w:
                json.dump({"gcn_accuracy": list(gcn_acc)}, w)
            if np.isnan(gcn_acc[0]):
                print(f"    GCN accuracy: N/A (too few samples, needs >=100)")
            else:
                print(f"    GCN accuracy: {gcn_acc[0]:.4f} +/- {gcn_acc[1]:.4f}")
            print(f"    -> {out_dir}/{dataset_name}_{name.lower()}_gcn.json")
        except Exception as e:
            print(f"    GCN accuracy skipped: {e}")


def run_cross_dataset_analyses():
    """Run randomness analysis and hierarchy similarity (requires all datasets)."""
    print(f"\n{'=' * 60}")
    print("Cross-dataset: CEM/TCAV Randomness Analysis")
    print(f"{'=' * 60}")

    results_by_method = {
        "cem": {}, "tcav": {}, "label": {}, "concept2vec": {}
    }
    all_datasets = [CUB_Dataset(), MNIST_Dataset(), Chexpert_Dataset(), DSprites_Dataset()]

    for method, name in zip(
        [load_cem_vectors_simple, load_tcav_vectors_simple,
         load_label_vectors_simple, load_concept2vec_vectors_simple],
        ["cem", "tcav", "label", "concept2vec"]
    ):
        for dataset in all_datasets:
            a = dataset.get_attributes()
            similarities = []
            stds = []

            for seed in [43, 44, 45]:
                vectors = [
                    np.mean(method(attr, dataset, "", seed), axis=0)
                    for attr in a
                ]
                for v in vectors:
                    stds.append(np.std(v))
                cosine_similarities_max = []
                for i in range(len(vectors)):
                    sim = max([
                        1 - cosine(vectors[i], vectors[j])
                        for j in range(len(vectors)) if i != j
                    ])
                    cosine_similarities_max.append(sim)
                similarities.append(np.mean(cosine_similarities_max))

            d = len(vectors[0])
            std = np.mean(stds)
            mean_sim = np.mean(similarities)
            z_score = (mean_sim - 0) / (d * std**4 / (3**0.5))
            z_score *= len(a)**0.5

            results_by_method[name][dataset.experiment_name] = {
                "dimension": d,
                "std": float(std),
                "mean_similarity": float(mean_sim),
                "std_similarity": float(np.std(similarities)),
            }

    os.makedirs("results/evaluation/ablation", exist_ok=True)
    json.dump(
        results_by_method,
        open("results/evaluation/ablation/randomness_cem_tcav.json", "w"),
    )
    print("  -> results/evaluation/ablation/randomness_cem_tcav.json")

    print(f"\n{'=' * 60}")
    print("Cross-dataset: Hierarchy Similarity Analysis")
    print(f"{'=' * 60}")

    hierarchy_by_dataset = defaultdict(lambda: defaultdict(dict))
    for dataset_function, dataset_name in zip(
        [CUB_Dataset, MNIST_Dataset, DSprites_Dataset, Chexpert_Dataset],
        ["cub", "mnist", "dsprites", "chexpert"],
    ):
        dataset_obj = dataset_function()
        attributes = dataset_obj.get_attributes()

        for function, name in zip(
            [load_label_vectors_simple, load_shapley_vectors_simple,
             load_cem_vectors_simple, load_concept2vec_vectors_simple],
            ["label", "shapley", "cem", "concept2vec"],
        ):
            hierarchy_by_dataset[dataset_name][name] = {}
            for seed in [43, 44, 45]:
                hierarchy_by_dataset[dataset_name][name][seed] = (
                    flat_distance_to_square(
                        get_concept_distances(
                            function, dataset_obj, "", attributes, seed
                        )
                    )
                )

    distance_by_dataset = defaultdict(lambda: defaultdict(dict))
    for dataset_name in ["cub", "mnist", "dsprites", "chexpert"]:
        for name in ["cem", "shapley", "label", "concept2vec"]:
            for name2 in ["cem", "shapley", "label", "concept2vec"]:
                h1 = hierarchy_by_dataset[dataset_name][name]
                h2 = hierarchy_by_dataset[dataset_name][name2]
                distance_by_dataset[dataset_name][name][name2] = [
                    embedding_distance(h1[s], h2[s], k=3)
                    for s in [43, 44, 45]
                ]

    distances_cub = np.array([
        [
            distance_by_dataset["cub"][i][j]
            for j in distance_by_dataset["cub"][i]
        ]
        for i in distance_by_dataset["cub"]
    ])
    distances_cub = np.mean(distances_cub, axis=2)

    json.dump(
        distances_cub.tolist(),
        open("results/evaluation/ablation/distance_between_hierarchies.json", "w"),
    )
    print("  -> results/evaluation/ablation/distance_between_hierarchies.json")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate concept vector methods on one or all datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["mnist", "cub", "dsprites", "chexpert"],
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run evaluation on all datasets + cross-dataset analyses",
    )
    args = parser.parse_args()

    if args.dataset:
        evaluate_dataset(args.dataset)
    elif args.all:
        for ds in ["mnist", "cub", "dsprites", "chexpert"]:
            evaluate_dataset(ds)
        run_cross_dataset_analyses()
    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == "__main__":
    main()
