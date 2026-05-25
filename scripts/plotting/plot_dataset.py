#!/usr/bin/env python3
"""
Per-dataset evaluation plotting script.
Generates figures from JSON outputs produced by evaluate.py --dataset <name>.

Usage:
    python scripts/plotting/plot_dataset.py --dataset dsprites
    python scripts/plotting/plot_dataset.py --dataset mnist
    python scripts/plotting/plot_dataset.py --dataset cub
    python scripts/plotting/plot_dataset.py --dataset chexpert
    python scripts/plotting/plot_dataset.py --all
"""

import os
import sys
import json
import argparse
import numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

METHODS = ['cem', 'concept2vec', 'label', 'tcav']
NICE_NAMES = ['CEM', 'Concept2Vec', 'Label', 'TCAV']


def plot_dataset(dataset_name):
    out_dir = f'results/evaluation/{dataset_name}'
    fig_dir = f'figures/{dataset_name}'
    os.makedirs(fig_dir, exist_ok=True)

    if not os.path.exists(out_dir):
        print(f"  No data for {dataset_name} at {out_dir}/")
        return

    def load_json(metric_name):
        data = {}
        for m in METHODS:
            path = f'{out_dir}/{dataset_name}_{m}_{metric_name}.json'
            if os.path.exists(path):
                with open(path) as f:
                    data[m] = json.load(f)
        return data

    purity = load_json('purity')
    maxsim = load_json('maxsim')
    gcn = load_json('gcn')

    print(f"\n{'=' * 60}")
    print(f"{dataset_name.capitalize()} Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"{'Method':<15} {'Purity':<20} {'MaxSim':<20} {'GCN Acc':<20}")
    print(f"{'-' * 15} {'-' * 20} {'-' * 20} {'-' * 20}")
    for m, name in zip(METHODS, NICE_NAMES):
        p = purity.get(m, {}).get('concept_purity', [float('nan'), float('nan')])
        ms = maxsim.get(m, {}).get('max_similarity', [float('nan'), float('nan')])
        g = gcn.get(m, {}).get('gcn_accuracy', [float('nan'), float('nan')])
        p_str = f"{p[0]:.4f} +/- {p[1]:.4f}" if not np.isnan(p[0]) else "N/A"
        ms_str = f"{ms[0]:.4f} +/- {ms[1]:.4f}" if not np.isnan(ms[0]) else "N/A"
        g_str = f"{g[0]:.4f} +/- {g[1]:.4f}" if not np.isnan(g[0]) else "N/A"
        print(f"{name:<15} {p_str:<20} {ms_str:<20} {g_str:<20}")

    # GCN accuracy bar plot
    gcn_vals = [(name, gcn[m]['gcn_accuracy'][0], gcn[m]['gcn_accuracy'][1])
                for m, name in zip(METHODS, NICE_NAMES)
                if m in gcn and not np.isnan(gcn[m]['gcn_accuracy'][0])]
    if gcn_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [v[0] for v in gcn_vals]
        means = [v[1] for v in gcn_vals]
        stds = [v[2] for v in gcn_vals]
        colors = sns.color_palette("colorblind", len(names))
        bars = ax.bar(names, means, yerr=stds, color=colors, capsize=5)
        ax.set_ylabel('GCN Test Accuracy', fontsize=14)
        ax.set_title(f'{dataset_name.capitalize()} - GCN Classification Accuracy', fontsize=14)
        ax.set_ylim(0, 1.05)
        for bar, m, _ in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{m:.3f}', ha='center', va='bottom', fontsize=11)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{fig_dir}/gcn_accuracy.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {fig_dir}/gcn_accuracy.png")

    # Max-similarity stability bar plot
    ms_vals = [(name, maxsim[m]['max_similarity'][0], maxsim[m]['max_similarity'][1])
               for m, name in zip(METHODS, NICE_NAMES)
               if m in maxsim and not np.isnan(maxsim[m]['max_similarity'][0])]
    if ms_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [v[0] for v in ms_vals]
        means = [v[1] for v in ms_vals]
        stds = [v[2] for v in ms_vals]
        colors = sns.color_palette("colorblind", len(names))
        ax.bar(names, means, yerr=stds, color=colors, capsize=5)
        ax.set_ylabel('Max-Similarity Stability', fontsize=14)
        ax.set_title(f'{dataset_name.capitalize()} - Max-Similarity Stability', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{fig_dir}/maxsim_stability.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {fig_dir}/maxsim_stability.png")

    # Concept purity bar plot
    p_vals = [(name, purity[m]['concept_purity'][0], purity[m]['concept_purity'][1])
              for m, name in zip(METHODS, NICE_NAMES)
              if m in purity and not np.isnan(purity[m]['concept_purity'][0])]
    if p_vals:
        fig, ax = plt.subplots(figsize=(6, 4))
        names = [v[0] for v in p_vals]
        means = [v[1] for v in p_vals]
        stds = [v[2] for v in p_vals]
        colors = sns.color_palette("colorblind", len(names))
        ax.bar(names, means, yerr=stds, color=colors, capsize=5)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.set_ylabel('Silhouette Score', fontsize=14)
        ax.set_title(f'{dataset_name.capitalize()} - Concept Purity', fontsize=12)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        fig.savefig(f'{fig_dir}/concept_purity.png', dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  -> {fig_dir}/concept_purity.png")


def main():
    parser = argparse.ArgumentParser(description='Plot per-dataset evaluation figures.')
    parser.add_argument('--dataset', type=str, choices=['mnist', 'cub', 'dsprites', 'chexpert'])
    parser.add_argument('--all', action='store_true', help='Plot all datasets')
    args = parser.parse_args()

    if args.dataset:
        plot_dataset(args.dataset)
    elif args.all:
        for ds in ['mnist', 'cub', 'dsprites', 'chexpert']:
            plot_dataset(ds)
    else:
        parser.print_help()
        sys.exit(1)

    print("\nDone.")


if __name__ == '__main__':
    main()
