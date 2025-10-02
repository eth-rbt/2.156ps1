#!/usr/bin/env python3
"""
Simple Submission Merger (No Evaluation)

This script merges two submission files by simply combining mechanisms
without LINKS evaluation. Useful for structure validation and quick merging.

Usage:
    python merge_submissions_simple.py submission1.npy submission2.npy [output_name]
"""

import numpy as np
import sys
import os
import json

def load_submission(file_path):
    """Load a submission file and validate its structure."""
    try:
        submission = np.load(file_path, allow_pickle=True).item()
        print(f"✅ Loaded: {os.path.basename(file_path)}")

        # Validate structure
        expected_problems = [f'Problem {i}' for i in range(1, 7)]
        for problem in expected_problems:
            if problem not in submission:
                print(f"⚠️ Warning: {problem} not found")
            else:
                count = len(submission[problem])
                print(f"   {problem}: {count} mechanisms")

        return submission
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def simple_merge_mechanisms(mechanisms1, mechanisms2, max_count=1000):
    """
    Simple merge by combining mechanisms and limiting to max_count.

    Strategy:
    1. Combine all mechanisms
    2. If over limit, take roughly equal portions from each source
    3. Prefer mechanisms from the larger set (assuming it's better optimized)
    """
    total_count = len(mechanisms1) + len(mechanisms2)

    if total_count <= max_count:
        # If under limit, take all
        return mechanisms1 + mechanisms2

    # Need to limit - distribute proportionally but ensure representation from both
    ratio1 = len(mechanisms1) / total_count
    ratio2 = len(mechanisms2) / total_count

    # Ensure minimum representation (at least 10% of final or 10 mechanisms, whichever is smaller)
    min_rep = min(max_count // 10, 10)

    count1 = max(min_rep, int(max_count * ratio1))
    count2 = max(min_rep, max_count - count1)

    # Adjust if over limit
    if count1 + count2 > max_count:
        if len(mechanisms1) > len(mechanisms2):
            count1 = max_count - min_rep
            count2 = min_rep
        else:
            count2 = max_count - min_rep
            count1 = min_rep

    # Take mechanisms (first N from each - could be randomized for better sampling)
    selected1 = mechanisms1[:count1] if count1 <= len(mechanisms1) else mechanisms1
    selected2 = mechanisms2[:count2] if count2 <= len(mechanisms2) else mechanisms2

    return selected1 + selected2

def merge_submissions_simple(file1, file2, output_name=None):
    """Main function to merge two submission files simply."""
    print("🚀 SIMPLE SUBMISSION MERGER (NO EVALUATION)")
    print("="*55)

    # Load both submissions
    print("📥 LOADING SUBMISSIONS...")
    submission1 = load_submission(file1)
    submission2 = load_submission(file2)

    if submission1 is None or submission2 is None:
        print("❌ Failed to load submissions. Exiting.")
        return

    # Create output name
    if output_name is None:
        base1 = os.path.splitext(os.path.basename(file1))[0]
        base2 = os.path.splitext(os.path.basename(file2))[0]
        output_name = f"merged_{base1}_{base2}"

    # Initialize merged submission
    merged_submission = {}
    merge_summary = []

    # Merge each problem
    print("\n🔄 MERGING ALL PROBLEMS...")
    for i in range(6):
        problem_key = f'Problem {i + 1}'

        mechanisms1 = submission1.get(problem_key, [])
        mechanisms2 = submission2.get(problem_key, [])

        print(f"\n{problem_key}:")
        print(f"   Submission 1: {len(mechanisms1)} mechanisms")
        print(f"   Submission 2: {len(mechanisms2)} mechanisms")

        # Simple merge with 1000 mechanism limit
        merged_mechanisms = simple_merge_mechanisms(mechanisms1, mechanisms2, max_count=1000)
        merged_submission[problem_key] = merged_mechanisms

        print(f"   Merged result: {len(merged_mechanisms)} mechanisms")

        merge_summary.append({
            'problem': problem_key,
            'submission1_count': len(mechanisms1),
            'submission2_count': len(mechanisms2),
            'merged_count': len(merged_mechanisms)
        })

    # Save merged submission
    merged_path = f'{output_name}.npy'
    np.save(merged_path, merged_submission)
    print(f"\n✅ SAVED: {merged_path}")

    # Create and save summary
    print(f"\n📊 MERGE SUMMARY:")
    print(f"{'Problem':<12} {'Sub1':<6} {'Sub2':<6} {'Merged':<8} {'Strategy':<20}")
    print("-" * 55)

    total1 = total2 = total_merged = 0

    for summary in merge_summary:
        count1 = summary['submission1_count']
        count2 = summary['submission2_count']
        merged = summary['merged_count']

        total1 += count1
        total2 += count2
        total_merged += merged

        # Determine strategy used
        if count1 + count2 <= 1000:
            strategy = "All combined"
        elif count1 == 0:
            strategy = "Sub2 only"
        elif count2 == 0:
            strategy = "Sub1 only"
        else:
            strategy = "Proportional limit"

        print(f"{summary['problem']:<12} {count1:<6} {count2:<6} {merged:<8} {strategy:<20}")

    print("-" * 55)
    print(f"{'TOTAL':<12} {total1:<6} {total2:<6} {total_merged:<8}")

    # Save summary to JSON
    summary_data = {
        'input_files': {
            'submission1': file1,
            'submission2': file2
        },
        'output_file': merged_path,
        'merge_strategy': 'simple_combination_with_1000_limit',
        'problems': merge_summary,
        'totals': {
            'submission1_total': total1,
            'submission2_total': total2,
            'merged_total': total_merged
        }
    }

    summary_path = f'{output_name}_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary_data, f, indent=2)
    print(f"✅ SAVED: {summary_path}")

    print(f"\n🎯 SIMPLE MERGE COMPLETE!")
    print(f"Merged submission: {merged_path}")
    print(f"Total mechanisms: {total_merged} (from {total1} + {total2})")

    # Validate final submission structure
    print(f"\n🔍 VALIDATING FINAL SUBMISSION...")
    try:
        final_submission = np.load(merged_path, allow_pickle=True).item()
        all_valid = True

        for i in range(6):
            problem_key = f'Problem {i + 1}'
            if problem_key not in final_submission:
                print(f"❌ Missing {problem_key}")
                all_valid = False
            else:
                mechanisms = final_submission[problem_key]
                if not isinstance(mechanisms, list):
                    print(f"❌ {problem_key} is not a list")
                    all_valid = False
                elif len(mechanisms) > 0:
                    # Check first mechanism structure
                    mech = mechanisms[0]
                    required_keys = ['x0', 'edges', 'fixed_joints', 'motor', 'target_joint']
                    for key in required_keys:
                        if key not in mech:
                            print(f"❌ {problem_key}: Missing '{key}' in mechanism")
                            all_valid = False
                            break

        if all_valid:
            print("✅ Final submission structure is valid!")
        else:
            print("⚠️ Final submission has structural issues")

    except Exception as e:
        print(f"❌ Error validating final submission: {e}")

def main():
    """Main execution with command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python merge_submissions_simple.py <submission1.npy> <submission2.npy> [output_name]")
        print("\nExample:")
        print("  python merge_submissions_simple.py overnight_run.npy optimized_results.npy")
        print("  python merge_submissions_simple.py sub1.npy sub2.npy final_merged")
        return

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    output_name = sys.argv[3] if len(sys.argv) > 3 else None

    # Validate input files exist
    if not os.path.exists(file1):
        print(f"❌ File not found: {file1}")
        return

    if not os.path.exists(file2):
        print(f"❌ File not found: {file2}")
        return

    merge_submissions_simple(file1, file2, output_name)

if __name__ == "__main__":
    main()