#!/usr/bin/env python3
"""
Single comparison figure: bias severity (y-axis) by bias name (x-axis),
with two bars per bias comparing SD-1.5 vs SD-XL for the same prompt.
"""

import json
import re
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def entropy(x):
    """Normalized entropy (0=concentrated, 1=uniform). Same as make_plots.py."""
    eps = 1e-10
    x_smoothed = np.array(x, dtype=float) + eps
    x_smoothed = x_smoothed / np.sum(x_smoothed)
    return -np.sum(x_smoothed * np.log(x_smoothed)) / np.log(len(x))


def severity_per_bias(counts_dict, exclude_classes=None):
    """
    Compute bias severity for each bias in the data counts.
    Returns dict: { "person gender": 0.37, "person race": 0.58, ... }
    """
    if exclude_classes is None:
        exclude_classes = {'unknown', 'other', 'non-binary'}

    result = {}
    for bias_cluster in counts_dict:
        for bias_name in counts_dict[bias_cluster]:
            for class_cluster in counts_dict[bias_cluster][bias_name]:
                class_counts = counts_dict[bias_cluster][bias_name][class_cluster]
                local_classes = [c for c in class_counts.keys() if c not in exclude_classes]
                pred_counts = np.array([class_counts[c] for c in local_classes], dtype=float)

                if np.sum(pred_counts) == 0:
                    continue

                pred_counts = pred_counts / np.sum(pred_counts)
                h = entropy(pred_counts)
                result[bias_name] = 1 - h
    return result


def parse_data_count_from_file(filepath):
    """
    Parse a result file (e.g. lap_coffeeshop_SDXL.txt) and extract
    the single //data count JSON block. Returns the parsed dict or None.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    data_count_idx = content.lower().find('// data count')
    if data_count_idx == -1:
        data_count_idx = content.lower().find('//data count')
    if data_count_idx == -1:
        return None

    after = content[data_count_idx:]
    start = after.find('{')
    if start == -1:
        return None

    brace_count = 0
    for i, c in enumerate(after[start:], start):
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = after[start:start + i + 1]
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    return None
    return None


# Display names for biases (capitalized, readable)
BIAS_DISPLAY_NAMES = {
    'person gender': 'Person gender',
    'person race': 'Person race',
    'person age': 'Person age',
    'person occupation': 'Person occupation',
    'laptop brand': 'Laptop brand',
}


def plot_generator_comparison(
    severity_sd15,
    severity_sdxl,
    output_path,
    title="Bias severity by generator (same prompt)",
):
    """
    Grouped bar chart: x = bias names, y = severity, two bars per bias (SD-1.5, SD-XL).
    """
    # Consistent order and labels
    bias_order = [
        'person gender',
        'person race',
        'person age',
        'person occupation',
        'laptop brand',
    ]
    all_keys = set(severity_sd15.keys()) | set(severity_sdxl.keys())
    all_bias = [b for b in bias_order if b in all_keys]
    all_bias += sorted(all_keys - set(bias_order))  # any extra biases at the end
    if not all_bias:
        print("No bias keys to plot.")
        return

    labels = [BIAS_DISPLAY_NAMES.get(b, b.replace('_', ' ').title()) for b in all_bias]
    x = np.arange(len(labels))
    width = 0.35

    vals_sd15 = [severity_sd15.get(b, 0) for b in all_bias]
    vals_sdxl = [severity_sdxl.get(b, 0) for b in all_bias]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, vals_sd15, width, label='SD-1.5', color='#3498db', alpha=0.9, edgecolor='#2c3e50')
    bars2 = ax.bar(x + width / 2, vals_sdxl, width, label='SD-XL', color='#e74c3c', alpha=0.9, edgecolor='#2c3e50')

    ax.set_ylabel('Bias severity (1 − entropy)', fontsize=12)
    ax.set_xlabel('Bias', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.axhline(y=0, color='#95a5a6', linestyle='-', linewidth=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: add value labels on bars
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f'{h:.2f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords='offset points', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot SD-1.5 vs SD-XL bias severity comparison (same prompt)'
    )
    parser.add_argument(
        '--sd15',
        default='proposed_biases/coco/3/lap_coffeeshop_SD1_5.txt',
        help='Path to SD-1.5 result file (with //data count)'
    )
    parser.add_argument(
        '--sdxl',
        default='proposed_biases/coco/3/lap_coffeeshop_SDXL.txt',
        help='Path to SD-XL result file (with //data count)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/VQA/coco/generated/generator_comparison.png',
        help='Output plot path'
    )
    parser.add_argument(
        '--title',
        default='Bias severity by generator (prompt: person in coffee shop)',
        help='Plot title'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_sd15 = os.path.join(script_dir, args.sd15)
    path_sdxl = os.path.join(script_dir, args.sdxl)
    output_path = os.path.join(script_dir, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counts_sd15 = parse_data_count_from_file(path_sd15)
    counts_sdxl = parse_data_count_from_file(path_sdxl)

    if not counts_sd15:
        print(f"Error: Could not parse data count from {path_sd15}")
        return 1
    if not counts_sdxl:
        print(f"Error: Could not parse data count from {path_sdxl}")
        return 1

    severity_sd15 = severity_per_bias(counts_sd15)
    severity_sdxl = severity_per_bias(counts_sdxl)

    print("Severity per bias:")
    for bias in sorted(set(severity_sd15.keys()) | set(severity_sdxl.keys())):
        s1 = severity_sd15.get(bias, 0)
        s2 = severity_sdxl.get(bias, 0)
        print(f"  {bias}: SD-1.5 = {s1:.4f}, SD-XL = {s2:.4f}")

    plot_generator_comparison(severity_sd15, severity_sdxl, output_path, title=args.title)
    return 0


if __name__ == '__main__':
    exit(main())
