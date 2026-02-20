#!/usr/bin/env python3
"""
Plot bias severity from a result file with one or two // data count blocks.
- Two blocks (e.g. SD-1.5 then SD-XL): grouped bar chart comparing severity per bias.
- One block: single bar chart of severity per bias.
"""

import json
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
    """Compute bias severity for each bias. Returns dict: { "person gender": 0.37, ... }."""
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


def _extract_one_data_count(content, start_from=0):
    """Find next '// data count' from start_from; return (parsed_dict, end_index) or (None, -1)."""
    idx = content.lower().find('// data count', start_from)
    if idx == -1:
        idx = content.lower().find('//data count', start_from)
    if idx == -1:
        return None, -1

    after = content[idx:]
    start = after.find('{')
    if start == -1:
        return None, -1

    brace_count = 0
    for i, c in enumerate(after[start:], start):
        if c == '{':
            brace_count += 1
        elif c == '}':
            brace_count -= 1
            if brace_count == 0:
                json_str = after[start : i + 1]
                try:
                    return json.loads(json_str), idx + i + 1
                except json.JSONDecodeError:
                    return None, -1
    return None, -1


def get_gender_counts(counts_dict):
    """
    Extract female and male counts from data count dict (person -> person gender -> cluster_* -> female/male).
    Returns (female, male) or (None, None) if not found.
    """
    for cluster in counts_dict:
        if not isinstance(counts_dict[cluster], dict):
            continue
        if "person gender" not in counts_dict[cluster]:
            continue
        for bias_name, class_data in counts_dict[cluster]["person gender"].items():
            if not isinstance(class_data, dict):
                continue
            female = class_data.get("female", 0)
            male = class_data.get("male", 0)
            return (int(female), int(male))
    return None, None


def plot_gender_only(female_a, male_a, female_b, male_b, label_a, label_b, output_path, title, use_proportion=True):
    """
    Grouped bar chart: x = model (label_a, label_b), two bars per model (Female, Male).
    Y = count or proportion (default proportion).
    """
    x = np.arange(2)  # two models
    width = 0.35
    total_a = female_a + male_a
    total_b = female_b + male_b
    if use_proportion and total_a and total_b:
        f_a, m_a = female_a / total_a, male_a / total_a
        f_b, m_b = female_b / total_b, male_b / total_b
        ylabel = "Proportion"
        ymax = 1.05
    else:
        f_a, m_a = female_a, male_a
        f_b, m_b = female_b, male_b
        ylabel = "Count"
        ymax = max(f_a, m_a, f_b, m_b) * 1.15 or 1
    fig, ax = plt.subplots(figsize=(7, 5))
    bars1 = ax.bar(x - width / 2, [f_a, f_b], width, label="Female", color="#e91e63", alpha=0.9, edgecolor="#2c3e50")
    bars2 = ax.bar(x + width / 2, [m_a, m_b], width, label="Male", color="#2196f3", alpha=0.9, edgecolor="#2c3e50")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlabel("Model", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([label_a, label_b])
    ax.set_ylim(0, ymax)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bar in bars1:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    for bar in bars2:
        h = bar.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                        xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def parse_two_data_counts(filepath):
    """
    Parse a file with one or two // data count blocks.
    Returns (counts_first, counts_second). counts_second is None if only one block exists.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    first, end1 = _extract_one_data_count(content, 0)
    if first is None or end1 < 0:
        return None, None
    second, _ = _extract_one_data_count(content, end1)
    return first, second


BIAS_DISPLAY_NAMES = {
    'person gender': 'Person gender',
    'person race': 'Person race',
    'person age': 'Person age',
    'person occupation': 'Person occupation',
    'person activity': 'Person activity',
    'laptop brand': 'Laptop brand',
    'oven type': 'Oven type',
    'motorcycle type': 'Motorcycle type',
    'environment': 'Environment',
    'gender': 'Gender',
    'race': 'Race',
    'age': 'Age',
    'attire': 'Attire',
    'occasion': 'Occasion',
}

BIAS_ORDER = [
    'person gender', 'person race', 'person age', 'person occupation',
    'person activity', 'laptop brand', 'oven type', 'motorcycle type', 'environment',
    'gender', 'race', 'age', 'attire', 'occasion',
]


def plot_comparison(severity_a, severity_b, label_a, label_b, output_path, title):
    """Grouped bar chart: x = bias, y = severity, two bars per bias."""
    all_keys = set(severity_a.keys()) | set(severity_b.keys())
    all_bias = [b for b in BIAS_ORDER if b in all_keys]
    all_bias += sorted(all_keys - set(BIAS_ORDER))
    if not all_bias:
        print("No bias keys to plot.")
        return

    labels = [BIAS_DISPLAY_NAMES.get(b, b.replace('_', ' ').title()) for b in all_bias]
    x = np.arange(len(labels))
    width = 0.35

    vals_a = [severity_a.get(b, 0) for b in all_bias]
    vals_b = [severity_b.get(b, 0) for b in all_bias]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, vals_a, width, label=label_a, color='#3498db', alpha=0.9, edgecolor='#2c3e50')
    bars2 = ax.bar(x + width / 2, vals_b, width, label=label_b, color='#e74c3c', alpha=0.9, edgecolor='#2c3e50')

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


def plot_single_block(severity, output_path, title):
    """Single set of bars: x = bias, y = severity (one model / one block)."""
    all_bias = [b for b in BIAS_ORDER if b in severity]
    all_bias += sorted(set(severity.keys()) - set(BIAS_ORDER))
    if not all_bias:
        print("No bias keys to plot.")
        return
    labels = [BIAS_DISPLAY_NAMES.get(b, b.replace('_', ' ').title()) for b in all_bias]
    x = np.arange(len(labels))
    vals = [severity.get(b, 0) for b in all_bias]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, vals, width=0.6, color='#3498db', alpha=0.9, edgecolor='#2c3e50')
    ax.set_ylabel('Bias severity (1 − entropy)', fontsize=12)
    ax.set_xlabel('Bias', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha='right')
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    for bar in bars:
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
        description='Plot severity comparison from one file with two // data count blocks (e.g. SD-1.5 vs SD-XL)'
    )
    parser.add_argument(
        '--input', '-i',
        required=True,
        help='Path to result file with two blocks (each with // data count)'
    )
    parser.add_argument(
        '--label1',
        default='SD-1.5',
        help='Label for first block (e.g. SD-1.5)'
    )
    parser.add_argument(
        '--label2',
        default='SD-XL',
        help='Label for second block (e.g. SD-XL)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/VQA/Tested_results/two_blocks_comparison.png',
        help='Output plot path'
    )
    parser.add_argument(
        '--title',
        default='Bias severity: first block vs second block',
        help='Plot title'
    )
    parser.add_argument(
        '--gender-only',
        action='store_true',
        help='Plot only gender (female vs male) per model'
    )
    parser.add_argument(
        '--counts',
        action='store_true',
        dest='gender_counts',
        help='With --gender-only: use raw counts instead of proportions'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    counts_a, counts_b = parse_two_data_counts(input_path)
    if counts_a is None:
        print("Error: Could not parse at least one data count block from input file.")
        return 1

    # Single-block file: plot severity per bias only
    if counts_b is None:
        severity = severity_per_bias(counts_a)
        print("Severity per bias (single block):")
        for bias in sorted(severity.keys()):
            print(f"  {bias}: {severity[bias]:.4f}")
        plot_single_block(severity, output_path, title=args.title)
        return 0

    if args.gender_only:
        fa, ma = get_gender_counts(counts_a)
        fb, mb = get_gender_counts(counts_b)
        if (fa is None and ma is None) or (fb is None and mb is None):
            print("Error: Could not find person gender counts in one or both blocks.")
            return 1
        fa = fa or 0
        ma = ma or 0
        fb = fb or 0
        mb = mb or 0
        print(f"Gender counts: {args.label1} Female={fa} Male={ma}, {args.label2} Female={fb} Male={mb}")
        plot_gender_only(
            fa, ma, fb, mb,
            args.label1, args.label2,
            output_path, title=args.title,
            use_proportion=not args.gender_counts,
        )
        return 0

    severity_a = severity_per_bias(counts_a)
    severity_b = severity_per_bias(counts_b)

    print("Severity per bias:")
    for bias in sorted(set(severity_a.keys()) | set(severity_b.keys())):
        s1 = severity_a.get(bias, 0)
        s2 = severity_b.get(bias, 0)
        print(f"  {bias}: {args.label1} = {s1:.4f}, {args.label2} = {s2:.4f}")

    plot_comparison(
        severity_a, severity_b,
        args.label1, args.label2,
        output_path, title=args.title
    )
    return 0


if __name__ == '__main__':
    exit(main())
