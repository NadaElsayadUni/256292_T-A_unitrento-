#!/usr/bin/env python3
"""
Plot prompt specificity: severity (y-axis) vs prompt type (x-axis).
Compares bias severity across different prompts (e.g. Working vs Coffee shop).
Uses the same entropy-based severity score as make_plots.py.
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


def compute_severity(counts_dict, exclude_classes=None):
    """
    Compute bias severity from data counts.
    Severity = 1 - entropy (higher = more biased).
    """
    if exclude_classes is None:
        exclude_classes = {'unknown', 'other', 'non-binary'}

    for bias_cluster in counts_dict:
        for bias_name in counts_dict[bias_cluster]:
            for class_cluster in counts_dict[bias_cluster][bias_name]:
                class_counts = counts_dict[bias_cluster][bias_name][class_cluster]
                local_classes = [c for c in class_counts.keys() if c not in exclude_classes]
                pred_counts = np.array([class_counts[c] for c in local_classes], dtype=float)

                if np.sum(pred_counts) == 0:
                    return None

                pred_counts = pred_counts / np.sum(pred_counts)
                h = entropy(pred_counts)
                return 1 - h

    return None


def infer_prompt_type(comment_line):
    """
    Infer prompt type from the results comment.
    E.g. "person working on a laptop" -> Working
         "person working on a laptop in a coffee shop" -> Coffee shop
    """
    if comment_line is None:
        return 'Unknown'
    comment_lower = comment_line.lower()
    if 'in a coffee shop' in comment_lower or 'coffee shop' in comment_lower:
        return 'Coffee shop'
    if 'working on a laptop' in comment_lower or 'working' in comment_lower:
        return 'Working'
    # Fallback: use first 40 chars of the prompt description
    match = re.search(r'results of the (.+?) bias with', comment_line, re.IGNORECASE)
    if match:
        label = match.group(1).strip()
        if len(label) > 25:
            return label[:22] + '...'
        return label
    return 'Unknown'


def parse_prompt_specificity_file(filepath, prompt_type_labels=None):
    """
    Parse person_working_vs_person_coffeeShop.txt to extract (prompt_type, data_counts) pairs.
    Each block is separated by //////////////////////////////////////////////////////////////////////
    and has a comment like "// results of the person working on a laptop bias with sample = 30 //"
    followed by VQA JSON and "// data count" + data counts JSON.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    results = []
    blocks = re.split(r'\n//////////////////////////////////////////////////////////////////////\n', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Find the line containing "results of the" for prompt description (skip separator lines)
        comment_line = None
        for line in block.split('\n'):
            if 'results of the' in line.lower():
                comment_line = line
                break
        prompt_type = infer_prompt_type(comment_line or block)
        if prompt_type_labels and prompt_type in prompt_type_labels:
            prompt_type = prompt_type_labels[prompt_type]

        # Find "// data count" and the JSON after it
        data_count_idx = block.lower().find('// data count')
        if data_count_idx == -1:
            data_count_idx = block.lower().find('//data count')
        if data_count_idx == -1:
            continue

        after_data_count = block[data_count_idx:]
        start = after_data_count.find('{')
        if start == -1:
            continue

        brace_count = 0
        json_str = None
        for i, c in enumerate(after_data_count[start:], start):
            if c == '{':
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0:
                    json_str = after_data_count[start:start + i + 1]
                    break

        if json_str:
            try:
                data_counts = json.loads(json_str)
                severity = compute_severity(data_counts)
                if severity is not None:
                    results.append({
                        'prompt_type': prompt_type,
                        'data_counts': data_counts,
                        'severity': severity
                    })
            except json.JSONDecodeError:
                pass

    return results


def plot_prompt_specificity(results, output_path, title="Prompt Specificity: Severity by Prompt Type"):
    """Bar plot: prompt type on x-axis, severity on y-axis."""
    if not results:
        print("No results to plot.")
        return

    prompt_types = [r['prompt_type'] for r in results]
    severities = [r['severity'] for r in results]

    plt.figure(figsize=(8, 6))
    x_pos = np.arange(len(prompt_types))
    bars = plt.bar(x_pos, severities, color=['#3498db', '#e74c3c'], alpha=0.9, edgecolor='#2c3e50', width=0.5)

    # Use distinct colors if more than 2 prompt types
    if len(prompt_types) > 2:
        colors = plt.cm.Set2(np.linspace(0, 1, len(prompt_types)))
        for bar, c in zip(bars, colors):
            bar.set_facecolor(c)

    plt.xticks(x_pos, prompt_types, fontsize=12)
    plt.ylabel('Bias Severity (1 - entropy)', fontsize=14)
    plt.xlabel('Prompt Type', fontsize=14)
    plt.title(title, fontsize=14)
    plt.ylim(0, 1)
    plt.axhline(y=0, color='#95a5a6', linestyle='-', linewidth=0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    for i, (pt, s) in enumerate(zip(prompt_types, severities)):
        plt.annotate(f'{s:.3f}', (i, s), textcoords="offset points", xytext=(0, 5),
                     ha='center', fontsize=11, fontweight='bold')

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot prompt specificity: severity vs prompt type'
    )
    parser.add_argument(
        '--input', '-i',
        default='proposed_biases/coco/3/person_working_vs_person_coffeeShop.txt',
        help='Path to results file (Working vs Coffee shop)'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/VQA/coco/generated/sd-xl/blip-large/prompt_specificity.png',
        help='Output plot path'
    )
    parser.add_argument(
        '--title',
        default='Prompt Specificity: Laptop Brand Bias (Apple Brand) by Prompt Type',
        help='Plot title'
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = parse_prompt_specificity_file(input_path)

    if not results:
        print("Error: No valid results found in input file.")
        return 1

    print(f"Parsed {len(results)} prompt types:")
    for r in results:
        print(f"  {r['prompt_type']}: severity = {r['severity']:.4f}")

    plot_prompt_specificity(results, output_path, title=args.title)
    return 0


if __name__ == '__main__':
    exit(main())
