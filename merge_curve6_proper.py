#!/usr/bin/env python3
"""
Properly Merge Only Curve 6 (Problem 6) Data

This script specifically merges only the Problem 6 mechanisms from both sources.
"""

import numpy as np

def main():
    """Main execution function."""
    print("🚀 PROPERLY MERGING ONLY CURVE 6 (PROBLEM 6) DATA")
    print("="*60)

    # Load overnight run submission
    overnight_submission = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/overnight_run_20250929_204331/submission.npy', allow_pickle=True).item()
    print("✅ Loaded overnight submission")

    # Load optimized data
    optimized_data = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/optimized_curve_6.npy', allow_pickle=True).item()
    print("✅ Loaded optimized data")

    # Get Problem 6 mechanisms from both sources
    overnight_problem6 = overnight_submission.get('Problem 6', [])
    optimized_problem6 = optimized_data.get('Problem 6', [])

    print(f"\n📊 PROBLEM 6 COMPARISON:")
    print(f"Overnight Problem 6: {len(overnight_problem6)} mechanisms")
    print(f"Optimized Problem 6: {len(optimized_problem6)} mechanisms")

    # Combine only Problem 6 mechanisms
    combined_problem6 = overnight_problem6 + optimized_problem6
    print(f"Combined Problem 6: {len(combined_problem6)} mechanisms")

    # Create updated submission with merged Problem 6
    updated_submission = overnight_submission.copy()
    updated_submission['Problem 6'] = combined_problem6

    # Save the properly merged submission
    np.save('submission_merged_problem6_only.npy', updated_submission)
    print(f"✅ SAVED: submission_merged_problem6_only.npy")

    # Also save just the Problem 6 mechanisms for further analysis
    np.save('problem6_merged_mechanisms.npy', combined_problem6)
    print(f"✅ SAVED: problem6_merged_mechanisms.npy ({len(combined_problem6)} mechanisms)")

    print(f"\n📈 FINAL SUBMISSION SUMMARY:")
    for key in sorted(updated_submission.keys()):
        mechanism_count = len(updated_submission[key])
        change = ""
        if key == 'Problem 6':
            change = f" (↑ from {len(overnight_problem6)})"
        print(f"  {key}: {mechanism_count} mechanisms{change}")

    print(f"\n✅ Successfully merged Problem 6 with {len(optimized_problem6)} additional mechanisms!")

if __name__ == "__main__":
    main()