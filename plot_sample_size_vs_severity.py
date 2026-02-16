#!/usr/bin/env python3
"""
Plot sample size vs bias severity to determine if a finding is a random occurrence
or a true model belief. Uses the same entropy-based severity score as make_plots.py.
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

    # Flatten nested structure: bias_cluster -> bias_name -> class_cluster -> {class: count}
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


def parse_results_file(filepath):
    """
    Parse person_working_laptop_results.txt to extract (sample_size, data_counts) pairs.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    # Split by block separator
    blocks = re.split(r'/+', content)

    results = []
    current_sample = None
    in_data_count = False
    json_buffer = []
    brace_count = 0

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Check for sample size in comment
        sample_match = re.search(r'sample\s*=\s*(\d+)', block, re.IGNORECASE)
        if sample_match:
            current_sample = int(sample_match.group(1))

        # Find data count JSON - look for {...} that contains "laptop" and count structure
        # The data count block typically follows "// data count" or "//data count"
        if 'data count' in block.lower():
            # Extract JSON object - find first { and match braces
            start = block.find('{')
            if start != -1:
                brace_count = 0
                for i, c in enumerate(block[start:], start):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = block[start:i+1]
                            try:
                                data_counts = json.loads(json_str)
                                if current_sample is not None:
                                    severity = compute_severity(data_counts)
                                    if severity is not None:
                                        results.append({
                                            'sample_size': current_sample,
                                            'data_counts': data_counts,
                                            'severity': severity
                                        })
                            except json.JSONDecodeError:
                                pass
                            break
            current_sample = None

    # Alternative: split by "// data count" and parse the JSON that follows
    if not results:
        # Try splitting by "// data count" pattern
        parts = re.split(r'//\s*data count\s*', content, flags=re.IGNORECASE)
        for i, part in enumerate(parts[1:], 1):  # Skip first part (no data count before it)
            # Find sample size from the preceding block
            preceding = parts[i-1] if i > 0 else content
            sample_match = re.search(r'sample\s*=\s*(\d+)', preceding, re.IGNORECASE)
            sample_size = int(sample_match.group(1)) if sample_match else None

            # Extract JSON from this part
            start = part.find('{')
            if start != -1 and sample_size is not None:
                brace_count = 0
                for j, c in enumerate(part[start:], start):
                    if c == '{':
                        brace_count += 1
                    elif c == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            json_str = part[start:start+j+1]
                            try:
                                data_counts = json.loads(json_str)
                                severity = compute_severity(data_counts)
                                if severity is not None:
                                    results.append({
                                        'sample_size': sample_size,
                                        'data_counts': data_counts,
                                        'severity': severity
                                    })
                            except json.JSONDecodeError:
                                pass
                            break

    return results


def parse_results_file_v2(filepath):
    """
    Simpler parser: split by "// data count" and extract sample + JSON from each block.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    results = []
    # Split by the separator line to get blocks
    blocks = re.split(r'\n//////////////////////////////////////////////////////////////////////\n', content)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Get sample size from first line/comment
        sample_match = re.search(r'sample\s*=\s*(\d+)', block, re.IGNORECASE)
        if not sample_match:
            continue
        sample_size = int(sample_match.group(1))

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
                    json_str = after_data_count[start:start+i+1]
                    break

        if json_str:
            try:
                data_counts = json.loads(json_str)
                severity = compute_severity(data_counts)
                if severity is not None:
                    results.append({
                        'sample_size': sample_size,
                        'data_counts': data_counts,
                        'severity': severity
                    })
            except json.JSONDecodeError:
                pass

    return results


def plot_sample_size_vs_severity(results, output_path, title="Sample Size vs Bias Severity"):
    """Create plot with sample size on x-axis and severity on y-axis."""
    if not results:
        print("No results to plot.")
        return

    # Sort by sample size
    results = sorted(results, key=lambda x: x['sample_size'])
    sample_sizes = [r['sample_size'] for r in results]
    severities = [r['severity'] for r in results]

    plt.figure(figsize=(10, 6))
    plt.plot(sample_sizes, severities, 'o-', color='#2ecc71', linewidth=2, markersize=10)

    # Add horizontal line at severity=0 (random/uniform)
    plt.axhline(y=0, color='#95a5a6', linestyle='--', alpha=0.7, label='Random (uniform)')

    plt.xlabel('Sample Size', fontsize=14)
    plt.ylabel('Bias Severity', fontsize=14)
    plt.title(title, fontsize=16)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Annotate points with severity values
    for x, y in zip(sample_sizes, severities):
        plt.annotate(f'{y:.3f}', (x, y), textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, format='png', bbox_inches='tight', dpi=150)
    plt.close()
    print(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Plot sample size vs bias severity from VQA results'
    )
    parser.add_argument(
        '--input', '-i',
        default='proposed_biases/coco/3/person_working_laptop_results.txt',
        help='Path to results file'
    )
    parser.add_argument(
        '--output', '-o',
        default='results/VQA/coco/generated/sd-xl/blip-large/sample_size_vs_severity.png',
        help='Output plot path'
    )
    parser.add_argument(
        '--title',
        default='Laptop Brand Bias (Apple): Sample Size vs Severity',
        help='Plot title'
    )
    args = parser.parse_args()

    # Resolve paths relative to script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(script_dir, args.input)
    output_path = os.path.join(script_dir, args.output)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results = parse_results_file_v2(input_path)

    if not results:
        print("Could not parse any results. Trying alternative parser...")
        results = parse_results_file(input_path)

    if not results:
        print("Error: No valid results found in input file.")
        return 1

    print(f"Parsed {len(results)} sample sizes: {[r['sample_size'] for r in results]}")
    for r in results:
        print(f"  Sample {r['sample_size']}: severity = {r['severity']:.4f}")

    plot_sample_size_vs_severity(results, output_path, title=args.title)
    return 0


if __name__ == '__main__':
    exit(main())
