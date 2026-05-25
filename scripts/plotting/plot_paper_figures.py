"""
Paper Plotting Script
Extracted from Plotting.ipynb for paper replication.
Generates all figures: bar plots, shapley, OIS, correlation, intervention.
"""

import os
import sys
import json
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

conda_lib = os.path.join(sys.prefix, 'lib')
os.environ["LD_LIBRARY_PATH"] = f"{conda_lib}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as mtick
import matplotlib.colors as mcolors
import seaborn as sns
import pickle


def load_scores(dataset_name, methods, nice_names):
    dict_out = {}
    for m, m_nice in zip(methods, nice_names):
        dict_out[m_nice] = {}
        file_name = f"results/evaluation/{dataset_name}/{dataset_name}_{m}.txt"
        if not os.path.exists(file_name):
            continue
        data = open(file_name).read().strip().split("\n")
        for line in data:
            if ": " not in line:
                continue
            name, value = line.split(": ", 1)
            name = name.lower().split(" ")[-1]
            try:
                value = eval(value)
            except:
                continue
            dict_out[m_nice][name] = value
    return dict_out


def _read_metric(file_name, metric_key):
    if not os.path.exists(file_name):
        return None
    f = open(file_name).read().split("\n")
    matching = [i for i in f if metric_key in i]
    if not matching:
        return None
    try:
        val = eval(matching[0].split(": ", 1)[1])
        mean, std = val
        if np.isnan(mean):
            return None
        return (mean, std)
    except:
        return None


def bar_plot_metric(metric_label, metric_key, flip, filename):
    dataset_configs = [
        ("mnist", "MNIST"),
        ("cub", "CUB"),
        ("dsprites", "DSprites"),
        ("chexpert", "CheXpert"),
    ]
    methods = ['tcav', 'cem', 'concept2vec', 'label']
    nice_names = ['TCAV', 'CEM', 'Concept2Vec', 'Label']

    cb_palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_palette)

    fig, ax = plt.subplots(figsize=(6, 6))

    ind = np.arange(len(dataset_configs))
    width = 0.2
    for i in range(len(methods)):
        m = methods[i]
        group_means = []
        group_stds = []
        valid_idxs = []
        for j, (d, d_display) in enumerate(dataset_configs):
            file_name = f"results/evaluation/{d}/{d}_{m}.txt"
            val = _read_metric(file_name, metric_key)
            if val is not None:
                mean, std = val
                group_means.append(1 - mean if flip else mean)
                group_stds.append(std)
            else:
                group_means.append(0)
                group_stds.append(0)
            valid_idxs.append(j)

        x_pos = [ind[j] + i * width for j in valid_idxs]
        ax.bar(x_pos, group_means, width, label=nice_names[i], yerr=group_stds)

    ax.set_xticks([r + 1.5 * width for r in range(len(ind))])
    ax.set_xticklabels([d for _, d in dataset_configs], fontsize=16)
    ax.set_ylabel(metric_label, fontsize=20)
    ax.set_xlabel("Dataset", fontsize=20)

    fmt = mtick.PercentFormatter(xmax=1.0)
    ax.yaxis.set_major_formatter(fmt)
    plt.yticks(fontsize=16)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='y', length=0)

    ax.legend(nice_names, fontsize=16)
    plt.savefig(f"figures/paper/bar_plot_{filename}.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved figures/paper/bar_plot_{filename}.png")


def main():
    os.makedirs("figures/paper", exist_ok=True)
    os.makedirs("results/evaluation", exist_ok=True)

    sns.set_theme()
    methods = ['tcav', 'cem', 'concept2vec', 'label']
    nice_names = ['TCAV', 'CEM', 'Concept2Vec', 'Label']

    # ============================================================
    # Load data for tables
    # ============================================================
    print("Loading evaluation data...")
    ds_names = ['mnist', 'cub', 'dsprites', 'chexpert']
    all_dicts = {}
    for ds in ds_names:
        all_dicts[ds] = load_scores(ds, methods, nice_names)
        if all_dicts[ds]:
            print(f"  {ds}: { {m: list(v.keys()) for m, v in all_dicts[ds].items()} }")
        else:
            print(f"  {ds}: (no data)")

    # ============================================================
    # Tables
    # ============================================================
    print("\nGenerating tables...")
    columns = ['truthfulness', 'robustness', 'responsiveness', 'stability']

    for table_name, datasets in [("table_main", ['mnist', 'cub']), ("table_extended", ['dsprites', 'chexpert'])]:
        print(f"\n--- {table_name} ---")
        for m in nice_names:
            row_vals = []
            for d in datasets:
                for c in columns:
                    try:
                        val = all_dicts[d][m][c]
                        if c in ('robustness', 'stability'):
                            val = (1 - val[0], val[1])
                        row_vals.append(val)
                    except KeyError:
                        row_vals.append(None)
            if all(v is None for v in row_vals):
                continue
            parts = [m]
            for v in row_vals:
                if v is None:
                    parts.append("--")
                else:
                    fmt_str = f"${v[0]:0.2f} \\pm {v[1]:0.2f}$"
                    if m == "Label":
                        fmt_str = "\\textbf{" + fmt_str + "}"
                    parts.append(fmt_str)
            print(" & ".join(parts) + " \\\\")
            if m == "Label":
                print(" \\bottomrule")

    # ============================================================
    # Individual Bar Plots
    # ============================================================
    print("\nGenerating bar plots...")
    bar_plot_metric("Faithfulness", "Truthfulness", flip=False, filename="faithfulness")
    bar_plot_metric("Robustness", "Robustness", flip=True, filename="robustness")
    bar_plot_metric("Responsiveness", "Responsiveness", flip=False, filename="responsiveness")
    bar_plot_metric("Stability", "Stability", flip=True, filename="stability")

    # ============================================================
    # Combined 4-panel bar plot
    # ============================================================
    print("\nGenerating combined bar plot...")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    metrics_def = [
        ('Truthfulness', "Faithfulness", False),
        ('Robustness', "Robustness", True),
        ('Responsiveness', "Responsiveness", False),
        ('Stability', "Stability", True),
    ]
    dataset_entries = [
        ("mnist", "MNIST"),
        ("cub", "CUB"),
        ("dsprites", "DSprit."),
        ("chexpert", "CheX."),
    ]

    cb_palette = sns.color_palette("colorblind")
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_palette)

    for i, (metric_key, label, flip) in enumerate(metrics_def):
        datasets_display = [d for _, d in dataset_entries]
        ind = np.arange(len(datasets_display))
        axes[i].set_xlabel(label, fontsize=20)

        width = 0.2
        for m_idx in range(len(methods)):
            m = methods[m_idx]
            group_means = []
            group_stds = []
            for j, (d, _) in enumerate(dataset_entries):
                file_name = f"results/evaluation/{d}/{d}_{m}.txt"
                val = _read_metric(file_name, metric_key)
                if val is not None:
                    mean, std = val
                    group_means.append(1 - mean if flip else mean)
                    group_stds.append(std)
                else:
                    group_means.append(0)
                    group_stds.append(0)
            x_pos = [ind[j] + m_idx * width for j in range(len(dataset_entries))]
            axes[i].bar(x_pos, group_means, width, label=nice_names[m_idx],
                        yerr=group_stds, capsize=3)

        axes[i].set_xticks([r + 1.5 * width for r in range(len(ind))])
        axes[i].set_xticklabels(datasets_display, fontsize=12)

        fmt = mtick.PercentFormatter(xmax=1.0)
        axes[i].yaxis.set_major_formatter(fmt)
        axes[i].spines['top'].set_visible(False)
        axes[i].spines['right'].set_visible(False)
        axes[i].tick_params(axis='y', length=0)
        axes[i].tick_params(axis='y', labelsize=16)

    fig.legend(nice_names, loc='upper left', bbox_to_anchor=(0.05, 1.12),
               ncol=4, fontsize=14)
    plt.tight_layout()
    fig.savefig("figures/paper/bar_plot_all.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print("Saved figures/paper/bar_plot_all.png")

    # ============================================================
    # Shapley Plots
    # ============================================================
    print("\nGenerating Shapley plot...")
    shapley_exists = any(os.path.exists(f"results/evaluation/{d}/{d}_shapley.txt") for d in ds_names)
    if shapley_exists:
        metrics_list = ['Stability', 'Robustness', 'Responsiveness', 'Truthfulness']
        metrics_short = ['Stability', 'Robustness', 'Responsiveness', 'Truthfulness']

        truth_data = {}
        for d, d_display in dataset_entries:
            file_name = f"results/evaluation/{d}/{d}_shapley.txt"
            if not os.path.exists(file_name):
                continue
            truth_data[d_display] = []
            for m_name in metrics_list:
                f = open(file_name).read().split("\n")
                matching = [i for i in f if m_name in i]
                if not matching:
                    truth_data[d_display].append(None)
                    continue
                try:
                    f_val = eval(matching[0].split(": ", 1)[1])
                    mean, std = f_val
                    if m_name in ['Stability', 'Robustness', 'Responsiveness']:
                        mean = 1 - mean
                    truth_data[d_display].append((mean, std))
                except:
                    truth_data[d_display].append(None)

        datasets_display = ['MNIST', 'CUB', 'DSprites', 'CheXpert']

        cb_palette = sns.color_palette("colorblind")
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_palette)

        fig, ax = plt.subplots(figsize=(6, 6))

        ind = np.arange(len(datasets_display))
        ax.set_xticks(ind)
        ax.set_xticklabels(datasets_display)
        ax.set_ylabel('Score for Shapley', fontsize=20)
        ax.set_xlabel("Dataset", fontsize=20)

        width = 0.2
        for i in range(len(metrics_list)):
            group_means = []
            group_stds = []
            valid_idxs = []
            for j, d in enumerate(datasets_display):
                if d in truth_data and truth_data[d] is not None and i < len(truth_data[d]) and truth_data[d][i] is not None:
                    group_means.append(truth_data[d][i][0])
                    group_stds.append(truth_data[d][i][1])
                    valid_idxs.append(j)
            if not group_means:
                continue
            x_pos = [ind[j] + i * width for j in valid_idxs]
            ax.bar(x_pos, group_means, width, label=metrics_short[i], yerr=group_stds)

        ax.set_xticks([r + width for r in range(len(ind))])
        ax.set_xticklabels(datasets_display, fontsize=16)

        fmt = mtick.PercentFormatter(xmax=1.0)
        ax.yaxis.set_major_formatter(fmt)
        plt.yticks(fontsize=16)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='y', length=0)

        ax.legend(metrics_short, fontsize=16)
        plt.savefig("figures/paper/shapley_metrics.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        print("Saved figures/paper/shapley_metrics.png")
    else:
        print("  Skipped (no shapley files)")

    # ============================================================
    # OIS Experiments
    # ============================================================
    print("\nGenerating OIS plots...")

    if os.path.exists("results/evaluation/ois"):
        fig, ax = plt.subplots(figsize=(4.5, 3))
        datasets_ois = ["cub", "mnist", "dsprites", "chexpert"]
        ois_values = {}
        for d in datasets_ois:
            path = f"results/evaluation/ois/{d}.json"
            if os.path.exists(path):
                obj = json.load(open(path))
                ois_values[d] = obj

        if ois_values:
            avg_by_dataset_value = {}
            for d in ois_values:
                avg_by_dataset_value[d] = {'cem': [], 'label': []}
                for seed in [43, 44, 45]:
                    for key in avg_by_dataset_value[d]:
                        if str(seed) in ois_values[d] and key in ois_values[d][str(seed)]:
                            avg_by_dataset_value[d][key].append(ois_values[d][str(seed)][key])
                for key in avg_by_dataset_value[d]:
                    if avg_by_dataset_value[d][key]:
                        avg_by_dataset_value[d][key] = (
                            np.mean(avg_by_dataset_value[d][key]),
                            np.std(avg_by_dataset_value[d][key])
                        )
                    else:
                        avg_by_dataset_value[d][key] = (0, 0)

            dataset_to_nice = {
                'cub': 'CUB',
                'dsprites': 'dSprites',
                'mnist': 'MNIST',
                'chexpert': 'CheXpert',
            }
            methods_to_nice = {'cem': 'CEM', 'label': 'Label'}
            datasets_ordered = ['mnist', 'cub', 'dsprites', 'chexpert']
            methods_ois = ['cem', 'label']

            width = 0.2
            plt.ylabel("OIS Score", fontsize=14)
            plt.xlabel("Dataset", fontsize=14)
            plt.yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5], fontsize=14)
            plt.xticks(
                list(range(len(datasets_ordered))),
                [dataset_to_nice[d] for d in datasets_ordered],
                fontsize=14
            )

            for i in range(len(methods_ois)):
                x_coords = np.array(list(range(len(datasets_ordered)))) + width * i
                y_coords = [
                    avg_by_dataset_value.get(d, {}).get(methods_ois[i], (0,))[0]
                    for d in datasets_ordered
                ]
                plt.bar(x_coords, y_coords, width=width,
                        label=methods_to_nice[methods_ois[i]])
            plt.legend(fontsize=14)
            plt.savefig('figures/paper/ois_scores.png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            print("Saved figures/paper/ois_scores.png")

            # OIS vs Robustness scatter
            fig_s, ax_s = plt.subplots(figsize=(5, 2.5))
            datasets_order_scatter = ['mnist', 'cub', 'dsprites', 'chexpert']

            avg_by_dataset_value_robustness = {}
            for d in datasets_order_scatter:
                avg_by_dataset_value_robustness[d] = {'cem': 0, 'label': 0}
                for m in methods_ois:
                    file_name = f"results/evaluation/{d}/{d}_{m}.txt"
                    if os.path.exists(file_name):
                        f = open(file_name).read().split("\n")
                        matching = [i for i in f if 'Robustness' in i]
                        if matching:
                            try:
                                f_inner = eval(matching[0].split(": ", 1)[1])
                                mean, _ = f_inner
                                avg_by_dataset_value_robustness[d][m] = 1 - mean
                            except:
                                pass

            colors = ['red', 'blue', 'black', 'orange']
            labels_scatter = ['MNIST', 'CUB', 'DSprites', 'CheXpert']

            for i, d in enumerate(datasets_order_scatter):
                scatter_x = []
                scatter_y = []
                for m in methods_ois:
                    if d in avg_by_dataset_value and m in avg_by_dataset_value[d]:
                        val = avg_by_dataset_value.get(d, {}).get(m, (0, 0))
                        if isinstance(val, tuple):
                            scatter_x.append(val[0])
                        else:
                            scatter_x.append(0)
                        scatter_y.append(avg_by_dataset_value_robustness.get(d, {}).get(m, 0))
                if scatter_x:
                    plt.scatter(scatter_x, scatter_y, c=colors[i], label=labels_scatter[i])

            plt.ylabel("Robustness", fontsize=14)
            plt.xticks(fontsize=14)
            plt.xlabel("OIS Score", fontsize=14)
            plt.yticks([0, 0.5, 1.0], fontsize=14)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig('figures/paper/ois_scatter.png', dpi=300, bbox_inches='tight')
            plt.close(fig_s)
            print("Saved figures/paper/ois_scatter.png")
    else:
        print("  Skipped (no results/evaluation/ois)")

    # ============================================================
    # CEM/TCAV Randomness Z-Score plot
    # ============================================================
    print("\nGenerating randomness plot...")
    if os.path.exists("results/evaluation/ablation/randomness_cem_tcav.json"):
        randomness_cem_tcav = json.load(
            open("results/evaluation/ablation/randomness_cem_tcav.json")
        )
        datasets_r = ['mnist', 'cub', 'dsprites', 'chexpert']
        methods_r = ['cem', 'tcav']
        method_names_r = ["CEM", "TCAV"]

        truth_data_r = {}
        for d in datasets_r:
            truth_data_r[d] = []
            for m in methods_r:
                val = randomness_cem_tcav.get(m, {}).get(d, {}).get('z_score', 0)
                truth_data_r[d].append(abs(val))

        datasets_display_r = ['MNIST', 'CUB', 'DSprites', 'CheXpert']

        cb_palette = sns.color_palette("colorblind")
        plt.rcParams['axes.prop_cycle'] = plt.cycler(color=cb_palette)

        means_r = []
        for group in truth_data_r.keys():
            group_data = truth_data_r[group]
            group_means = [t for t in group_data]
            means_r.append(group_means)

        means_r = list(np.array(means_r).T)

        fig_r, ax_r = plt.subplots(figsize=(6, 2))

        ind_r = np.arange(len(datasets_display_r))
        ax_r.set_xticks(ind_r)
        ax_r.set_xticklabels(datasets_display_r)
        ax_r.set_ylabel('Z-Score', fontsize=20)
        ax_r.set_xlabel("Dataset", fontsize=20)

        width = 0.2
        for i in range(len(means_r)):
            group_means = [j for j in means_r[i]]
            ax_r.bar(ind_r + i * width, group_means, width,
                     label=datasets_display_r)

        ax_r.set_xticks([r + width for r in range(len(ind_r))])
        ax_r.set_xticklabels(datasets_display_r, fontsize=16)
        plt.yticks([0, 1, 2, 3, 4], fontsize=16)

        ax_r.spines['top'].set_visible(False)
        ax_r.spines['right'].set_visible(False)
        ax_r.tick_params(axis='y', length=0)

        ax_r.legend(method_names_r, fontsize=16)
        plt.savefig('figures/paper/tcav_cem_randomness.png', dpi=300,
                    bbox_inches='tight')
        plt.close(fig_r)
        print("Saved figures/paper/tcav_cem_randomness.png")
    else:
        print("  Skipped (no randomness_cem_tcav.json)")

    # ============================================================
    # Comparison With Ground Truth (Heuristic distances)
    # ============================================================
    print("\nGenerating heuristic distances plot...")
    if all(os.path.exists(f) for f in [
        "results/evaluation/ablation/distance_mnist.json",
        "results/evaluation/ablation/distance_cub_first_part.json",
        "results/evaluation/ablation/distance_cub_second_part.json"
    ]):
        mnist_distances = json.load(
            open("results/evaluation/ablation/distance_mnist.json")
        )
        cub_distances_part_1 = json.load(
            open("results/evaluation/ablation/distance_cub_first_part.json")
        )
        cub_distances_part_2 = json.load(
            open("results/evaluation/ablation/distance_cub_second_part.json")
        )

        labels_x = ['TCAV', 'CEM', 'Concept2Vec', 'Label']
        labels_x_keys = ['tcav', 'cem', 'concept2vec', 'label']
        labels_y = ['MNIST', 'CUB Group 1', 'CUB Group 2']

        distances_as_matrix = [
            [
                j.get(i, [0])[0] if isinstance(j.get(i), list) else 0
                for j in [mnist_distances, cub_distances_part_1, cub_distances_part_2]
            ]
            for i in labels_x_keys
        ]

        fig_h, ax_h = plt.subplots(figsize=(6, 5))
        plt.imshow(np.array(distances_as_matrix).T, cmap='viridis',
                   interpolation='nearest')
        plt.xticks(np.arange(len(labels_x)), labels_x, rotation=45)
        plt.yticks(np.arange(len(labels_y)), labels_y)
        colorbar = plt.colorbar()
        colorbar.set_label('Basis Distance')
        plt.savefig("figures/paper/heuristic_distances.png", dpi=300,
                    bbox_inches='tight')
        plt.close(fig_h)
        print("Saved figures/paper/heuristic_distances.png")
    else:
        print("  Skipped (missing distance files)")

    # ============================================================
    # Hierarchy distance heatmap
    # ============================================================
    print("\nGenerating hierarchy distance heatmap...")
    if os.path.exists("results/evaluation/ablation/distance_between_hierarchies.json"):
        distance_by_hierarchy = json.load(
            open("results/evaluation/ablation/distance_between_hierarchies.json")
        )
        labels_heat = ['cem', 'shapley', 'label', 'concept2vec']
        method_to_nice = {
            'cem': 'CEM',
            'concept2vec': 'Concept2Vec',
            'label': 'Label',
            'shapley': 'Shapley',
        }

        fig_heat, ax_heat = plt.subplots(figsize=(6, 5))
        plt.imshow(distance_by_hierarchy, cmap='viridis', interpolation='nearest')
        plt.xticks(np.arange(len(labels_heat)),
                   [method_to_nice[i] for i in labels_heat], rotation=45)
        plt.yticks(np.arange(len(labels_heat)),
                   [method_to_nice[i] for i in labels_heat])
        colorbar = plt.colorbar()
        colorbar.set_label('Basis Distance')
        plt.savefig("figures/paper/hierarchy_distances.png", dpi=300,
                    bbox_inches='tight')
        plt.close(fig_heat)
        print("Saved figures/paper/hierarchy_distances.png")
    else:
        print("  Skipped (no distance_between_hierarchies.json)")

    print("\nAll plots generated successfully!")


if __name__ == "__main__":
    main()
