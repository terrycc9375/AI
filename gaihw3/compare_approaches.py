#!/usr/bin/env python3
"""
Comparison script for evaluating all class imbalance approaches.
Run this after training and evaluating all four approaches.
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict, List

def load_results(approach: str) -> Dict:
    """Load evaluation results for an approach"""
    results_file = f"evaluation_results_{approach}.json"
    if not Path(results_file).exists():
        print(f"Warning: {results_file} not found. Skipping {approach}.")
        return None
    
    with open(results_file) as f:
        return json.load(f)

def compare_approaches():
    """Compare results across all approaches"""
    approaches = ["standard", "oversample", "undersample", "combined"]
    results = {}
    
    # Load all results
    for approach in approaches:
        result = load_results(approach)
        if result:
            results[approach] = result
    
    if not results:
        print("No evaluation results found. Please run test.py for each approach first.")
        return
    
    print("\n" + "="*100)
    print("CLASS IMBALANCE APPROACH COMPARISON")
    print("="*100)
    
    # Overall metrics comparison
    print("\n1. OVERALL METRICS")
    print("-" * 100)
    
    df_metrics = []
    for approach in results.keys():
        r = results[approach]
        df_metrics.append({
            "Approach": approach,
            "Accuracy": f"{r['accuracy']:.4f}",
            "F1 (Weighted)": f"{r['f1_weighted']:.4f}",
            "F1 (Macro)": f"{r['f1_macro']:.4f}",
            "Precision": f"{r['precision_weighted']:.4f}",
            "Recall": f"{r['recall_weighted']:.4f}",
            "Samples": r['num_valid_predictions']
        })
    
    df = pd.DataFrame(df_metrics)
    print(df.to_string(index=False))
    
    # Best approach for each metric
    print("\n2. BEST PERFORMANCE BY METRIC")
    print("-" * 100)
    
    metrics_to_track = {
        "accuracy": "Accuracy",
        "f1_weighted": "F1 (Weighted)",
        "f1_macro": "F1 (Macro)",
        "precision_weighted": "Precision (Weighted)",
        "recall_weighted": "Recall (Weighted)"
    }
    
    for metric_key, metric_name in metrics_to_track.items():
        best_approach = max(results.keys(), key=lambda a: results[a][metric_key])
        best_value = results[best_approach][metric_key]
        print(f"  {metric_name:25} → {best_approach:12} ({best_value:.4f})")
    
    # Per-class analysis
    print("\n3. PER-CLASS ACCURACY COMPARISON")
    print("-" * 100)
    
    # Get all class names from first result
    first_result = next(iter(results.values()))
    class_names = first_result['class_names']
    
    for class_name in class_names:
        print(f"\n  {class_name}:")
        for approach in results.keys():
            if class_name in results[approach].get('per_class_metrics', {}):
                metrics = results[approach]['per_class_metrics'][class_name]
                count = metrics['count']
                accuracy = metrics['accuracy']
                print(f"    {approach:12} - {count:4d} samples, {accuracy:.4f} accuracy")
    
    # Recommendation
    print("\n4. RECOMMENDATION")
    print("-" * 100)
    
    best_overall = max(results.keys(), key=lambda a: results[a]['f1_weighted'])
    best_macro = max(results.keys(), key=lambda a: results[a]['f1_macro'])
    
    print(f"\n  Best Overall (F1-Weighted): {best_overall}")
    print(f"  Best Balanced (F1-Macro):   {best_macro}")
    
    if best_overall == best_macro:
        print(f"\n  ✓ {best_overall.upper()} is the recommended approach.")
    else:
        print(f"\n  → Choose between {best_overall} (overall) or {best_macro} (balanced).")
    
    # Summary statistics
    print("\n5. SUMMARY")
    print("-" * 100)
    
    for approach in results.keys():
        r = results[approach]
        print(f"\n  {approach.upper()}:")
        print(f"    - Processed: {r['num_samples']} samples, {r['num_valid_predictions']} valid predictions")
        print(f"    - Accuracy: {r['accuracy']:.4f} | F1 (W): {r['f1_weighted']:.4f} | F1 (M): {r['f1_macro']:.4f}")
        
        # Find worst performing class
        worst_class = min(
            r['per_class_metrics'].keys(),
            key=lambda k: r['per_class_metrics'][k]['accuracy']
        )
        worst_acc = r['per_class_metrics'][worst_class]['accuracy']
        print(f"    - Worst class: {worst_class} ({worst_acc:.4f} accuracy)")
    
    print("\n" + "="*100 + "\n")
    
    # Save comparison to file
    comparison = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "approaches_compared": list(results.keys()),
        "overall_metrics": {
            approach: {
                "accuracy": results[approach]["accuracy"],
                "f1_weighted": results[approach]["f1_weighted"],
                "f1_macro": results[approach]["f1_macro"],
                "num_valid_predictions": results[approach]["num_valid_predictions"]
            }
            for approach in results.keys()
        },
        "best_approaches": {
            "overall_f1_weighted": best_overall,
            "balanced_f1_macro": best_macro
        }
    }
    
    with open("comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)
    
    print("Comparison results saved to comparison_results.json")

if __name__ == "__main__":
    compare_approaches()
