#!/usr/bin/env python3
"""
Quick start script for training all approaches and comparing results.
Run: python run_all_approaches.py
"""

import subprocess
import sys
import time
import json
from pathlib import Path

def run_command(cmd: str, description: str):
    """Run a command and report status"""
    print(f"\n{'='*70}")
    print(f"[{time.strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print(f"\n❌ FAILED: {description}")
        return False
    else:
        print(f"\n✓ COMPLETED: {description}")
        return True

def main():
    print("\n" + "="*70)
    print("HALLUCINATION CLASSIFICATION - CLASS IMBALANCE APPROACHES")
    print("="*70)
    
    approaches = ["standard", "oversample", "undersample", "combined"]
    results_summary = {}
    
    # Check if indices are cached
    if not Path("indices_cache.pkl").exists():
        print("\n⚠️  WARNING: indices_cache.pkl not found.")
        print("The first training will index all papers (5-7 hours).")
        response = input("Continue? (y/n): ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return
    
    # Train all approaches
    print("\n" + "="*70)
    print("PHASE 1: TRAINING ALL APPROACHES")
    print("="*70)
    
    train_results = {}
    for approach in approaches:
        cmd = f"python train.py {approach}"
        success = run_command(cmd, f"Training with {approach} approach")
        train_results[approach] = success
        
        if not success:
            print(f"\n⚠️  Training failed for {approach}. Continuing with others...")
    
    completed_approaches = [a for a, s in train_results.items() if s]
    
    if not completed_approaches:
        print("\n❌ No training approaches completed successfully!")
        return
    
    print(f"\n✓ Training completed for: {', '.join(completed_approaches)}")
    
    # Evaluate all trained approaches
    print("\n" + "="*70)
    print("PHASE 2: EVALUATING ON dev.csv")
    print("="*70)
    
    eval_results = {}
    for approach in completed_approaches:
        cmd = f"python test.py {approach}"
        success = run_command(cmd, f"Evaluation with {approach} approach")
        eval_results[approach] = success
        
        if not success:
            print(f"\n⚠️  Evaluation failed for {approach}.")
    
    evaluated_approaches = [a for a, s in eval_results.items() if s]
    
    if not evaluated_approaches:
        print("\n❌ No evaluations completed successfully!")
        return
    
    print(f"\n✓ Evaluation completed for: {', '.join(evaluated_approaches)}")
    
    # Compare results
    print("\n" + "="*70)
    print("PHASE 3: COMPARING RESULTS")
    print("="*70)
    
    if len(evaluated_approaches) > 1:
        cmd = "python compare_approaches.py"
        run_command(cmd, "Comparing all approaches")
        
        # Load and display comparison summary
        if Path("comparison_results.json").exists():
            with open("comparison_results.json") as f:
                comparison = json.load(f)
            
            print("\n" + "="*70)
            print("QUICK SUMMARY")
            print("="*70)
            print(f"\nBest Overall (F1-Weighted): {comparison['best_approaches']['overall_f1_weighted']}")
            print(f"Best Balanced (F1-Macro):   {comparison['best_approaches']['balanced_f1_macro']}")
            
            print("\nDetailed Results by Approach:")
            for approach, metrics in comparison['overall_metrics'].items():
                print(f"\n  {approach}:")
                print(f"    Accuracy:     {metrics['accuracy']:.4f}")
                print(f"    F1 (Weighted):{metrics['f1_weighted']:.4f}")
                print(f"    F1 (Macro):   {metrics['f1_macro']:.4f}")
                print(f"    Samples:      {metrics['num_valid_predictions']}")
    else:
        print("\n⚠️  Only one approach evaluated. Skipping comparison.")
    
    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"\nTraining Results: {sum(1 for v in train_results.values() if v)}/{len(approaches)} completed")
    print(f"Evaluation Results: {len(evaluated_approaches)}/{len(completed_approaches)} completed")
    
    if evaluated_approaches:
        print(f"\nGenerated Files:")
        for approach in evaluated_approaches:
            print(f"  - evaluation_results_{approach}.json")
            print(f"  - predictions_{approach}.json")
        if len(evaluated_approaches) > 1:
            print(f"  - comparison_results.json")
    
    print("\n" + "="*70)
    print("✓ ALL PHASES COMPLETED!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Review evaluation_results_*.json files for detailed metrics")
    print("  2. Check CLASS_IMBALANCE_GUIDE.md for approach descriptions")
    print("  3. Run 'python compare_approaches.py' anytime to regenerate comparison")
    print("\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Process interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
