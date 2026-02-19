"""Compute empirically validated thresholds for chaos indicators.

This script analyzes all indicator data to determine optimal thresholds
for classifying systems as chaotic vs non-chaotic based on Zero-One K,
Permutation Entropy, and MEGNO indicators.
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np


def load_system_truth_assignments(systems_dir: str = "systems") -> Dict[str, bool]:
    """Load ground truth chaotic/non-chaotic labels for all systems.

    Args:
        systems_dir: Directory containing system JSON files.

    Returns:
        Dict mapping system_id to True (chaotic) or False (non-chaotic).
    """
    truth_map = {}
    for fname in os.listdir(systems_dir):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(systems_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            truth = data.get("truth_assignment", {})
            truth_map[sid] = truth.get("Chaotic", False)
    return truth_map


def load_all_indicators(indicators_dir: str = "systems/indicators") -> Dict[str, Dict]:
    """Load all indicator JSON files.

    Args:
        indicators_dir: Directory containing indicator JSON files.

    Returns:
        Dict mapping system_id to indicator data.
    """
    indicators = {}
    for fname in os.listdir(indicators_dir):
        if not fname.endswith("_indicators.json"):
            continue
        fpath = os.path.join(indicators_dir, fname)
        with open(fpath) as f:
            data = json.load(f)
        sid = data.get("system_id")
        if sid:
            indicators[sid] = data
    return indicators


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute descriptive statistics for a list of values.

    Args:
        values: List of numerical values.

    Returns:
        Dict with mean, median, std, min, max, Q1, Q3.
    """
    if not values:
        return {}

    arr = np.array(values)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "Q1": float(np.percentile(arr, 25)),
        "Q3": float(np.percentile(arr, 75)),
        "count": len(values),
    }


def find_optimal_threshold(
    chaotic_values: List[float],
    non_chaotic_values: List[float],
) -> Tuple[float, float]:
    """Find threshold that maximizes separation between chaotic and non-chaotic.

    Uses a simple approach: finds the threshold that minimizes classification error.

    Args:
        chaotic_values: Indicator values for chaotic systems.
        non_chaotic_values: Indicator values for non-chaotic systems.

    Returns:
        Tuple of (optimal_threshold, accuracy).
    """
    if not chaotic_values or not non_chaotic_values:
        return None, 0.0

    # Try all possible thresholds at midpoints between values
    all_values = sorted(chaotic_values + non_chaotic_values)
    thresholds = [(all_values[i] + all_values[i + 1]) / 2
                  for i in range(len(all_values) - 1)]

    # Add min and max as candidate thresholds
    thresholds = [min(all_values) - 0.1] + thresholds + [max(all_values) + 0.1]

    best_threshold = None
    best_accuracy = 0.0

    for threshold in thresholds:
        # Count correct classifications (chaotic >= threshold, non-chaotic < threshold)
        correct_chaotic = sum(1 for v in chaotic_values if v >= threshold)
        correct_non_chaotic = sum(1 for v in non_chaotic_values if v < threshold)
        accuracy = (correct_chaotic + correct_non_chaotic) / (len(chaotic_values) + len(non_chaotic_values))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold

    return best_threshold, best_accuracy


def analyze_indicator(
    indicator_name: str,
    indicator_key: str,
    truth_map: Dict[str, bool],
    indicators: Dict[str, Dict],
) -> Dict:
    """Analyze a single indicator and compute recommended threshold.

    Args:
        indicator_name: Display name of the indicator.
        indicator_key: Key name in indicator JSON files.
        truth_map: Ground truth chaotic/non-chaotic labels.
        indicators: All indicator data.

    Returns:
        Dict with analysis results.
    """
    chaotic_values = []
    non_chaotic_values = []
    missing_count = 0

    for sid, is_chaotic in truth_map.items():
        if sid not in indicators:
            continue

        value = indicators[sid].get(indicator_key)
        if value is None or not np.isfinite(value):
            missing_count += 1
            continue

        if is_chaotic:
            chaotic_values.append(value)
        else:
            non_chaotic_values.append(value)

    # Compute statistics
    chaotic_stats = compute_statistics(chaotic_values)
    non_chaotic_stats = compute_statistics(non_chaotic_values)

    # Find optimal threshold
    optimal_threshold, accuracy = find_optimal_threshold(chaotic_values, non_chaotic_values)

    return {
        "indicator": indicator_name,
        "key": indicator_key,
        "chaotic": chaotic_stats,
        "non_chaotic": non_chaotic_stats,
        "missing_count": missing_count,
        "optimal_threshold": optimal_threshold,
        "classification_accuracy": accuracy,
    }


def print_analysis_report(analysis: Dict) -> None:
    """Print a formatted analysis report for an indicator.

    Args:
        analysis: Analysis results from analyze_indicator().
    """
    print(f"\n{'='*70}")
    print(f"{analysis['indicator']}")
    print(f"{'='*70}")

    print(f"\nChaotic systems ({analysis['chaotic'].get('count', 0)}):")
    if analysis['chaotic']:
        print(f"  Mean:   {analysis['chaotic']['mean']:.4f}")
        print(f"  Median: {analysis['chaotic']['median']:.4f}")
        print(f"  Std:    {analysis['chaotic']['std']:.4f}")
        print(f"  Range:  [{analysis['chaotic']['min']:.4f}, {analysis['chaotic']['max']:.4f}]")
        print(f"  Q1-Q3:  [{analysis['chaotic']['Q1']:.4f}, {analysis['chaotic']['Q3']:.4f}]")

    print(f"\nNon-chaotic systems ({analysis['non_chaotic'].get('count', 0)}):")
    if analysis['non_chaotic']:
        print(f"  Mean:   {analysis['non_chaotic']['mean']:.4f}")
        print(f"  Median: {analysis['non_chaotic']['median']:.4f}")
        print(f"  Std:    {analysis['non_chaotic']['std']:.4f}")
        print(f"  Range:  [{analysis['non_chaotic']['min']:.4f}, {analysis['non_chaotic']['max']:.4f}]")
        print(f"  Q1-Q3:  [{analysis['non_chaotic']['Q1']:.4f}, {analysis['non_chaotic']['Q3']:.4f}]")

    if analysis['missing_count'] > 0:
        print(f"\nMissing/invalid values: {analysis['missing_count']}")

    if analysis['optimal_threshold'] is not None:
        print(f"\nRecommended threshold: {analysis['optimal_threshold']:.4f}")
        print(f"Classification accuracy: {analysis['classification_accuracy']:.2%}")
        print(f"(Chaotic >= threshold, Non-chaotic < threshold)")
    else:
        print("\nCannot compute threshold (insufficient data)")


def main():
    """Main function to compute and display indicator thresholds."""
    print("ChaosBench-Logic v2: Indicator Threshold Analysis")
    print("=" * 70)

    # Load data
    truth_map = load_system_truth_assignments()
    indicators = load_all_indicators()

    print(f"\nLoaded {len(truth_map)} systems:")
    print(f"  Chaotic: {sum(truth_map.values())}")
    print(f"  Non-chaotic: {sum(not v for v in truth_map.values())}")
    print(f"\nLoaded indicators for {len(indicators)} systems")

    # Analyze each indicator
    indicators_to_analyze = [
        ("Zero-One K Test", "zero_one_K"),
        ("Permutation Entropy", "permutation_entropy"),
        ("MEGNO", "megno"),
    ]

    results = []
    for name, key in indicators_to_analyze:
        analysis = analyze_indicator(name, key, truth_map, indicators)
        results.append(analysis)
        print_analysis_report(analysis)

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY - Recommended Thresholds")
    print(f"{'='*70}\n")

    for result in results:
        if result['optimal_threshold'] is not None:
            print(f"{result['indicator']:25s} {result['optimal_threshold']:7.4f} "
                  f"(accuracy: {result['classification_accuracy']:.1%})")
        else:
            print(f"{result['indicator']:25s} N/A (insufficient data)")

    print("\nUsage guidelines:")
    print("  - Values >= threshold suggest chaotic behavior")
    print("  - Values < threshold suggest non-chaotic behavior")
    print("  - Consider using multiple indicators for robust classification")
    print("  - Some systems may have ambiguous values near the threshold")


if __name__ == "__main__":
    main()
