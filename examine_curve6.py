#!/usr/bin/env python3
"""
Examine Curve 6 Data Structure

This script examines the structure of both curve 6 datasets without requiring JAX.
"""

import numpy as np
import json

def examine_data():
    """Load and examine both curve 6 datasets."""
    print("🔍 EXAMINING CURVE 6 DATA STRUCTURE...")

    # Load overnight run submission
    try:
        overnight_submission = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/overnight_run_20250929_204331/submission.npy', allow_pickle=True).item()
        print("✅ Loaded overnight submission")
        print(f"Submission keys: {list(overnight_submission.keys())}")
    except Exception as e:
        print(f"❌ Error loading overnight submission: {e}")
        return

    # Examine Problem 6 from overnight run
    if 'Problem 6' in overnight_submission:
        problem6_overnight = overnight_submission['Problem 6']
        print(f"\nProblem 6 (overnight): {len(problem6_overnight)} mechanisms")
        if len(problem6_overnight) > 0:
            sample_mech = problem6_overnight[0]
            print(f"Sample mechanism keys: {list(sample_mech.keys())}")
            print(f"Sample x0 shape: {np.array(sample_mech['x0']).shape}")
            print(f"Sample edges shape: {np.array(sample_mech['edges']).shape}")
            print(f"Sample target_joint: {sample_mech.get('target_joint', 'Not found')}")
    else:
        print("❌ No Problem 6 in overnight submission")

    # Load separate optimized curve 6
    try:
        optimized_curve6 = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/optimized_curve_6.npy', allow_pickle=True).item()
        print(f"\n✅ Loaded optimized curve 6")
        print(f"Type: {type(optimized_curve6)}")
    except Exception as e:
        print(f"❌ Error loading optimized curve 6: {e}")
        return

    # Examine optimized curve 6 structure
    if isinstance(optimized_curve6, dict):
        print(f"Dictionary keys: {list(optimized_curve6.keys())}")
        for key, value in optimized_curve6.items():
            print(f"  {key}: {type(value)}, length: {len(value) if hasattr(value, '__len__') else 'N/A'}")
            if isinstance(value, list) and len(value) > 0:
                print(f"    Sample item type: {type(value[0])}")
                if isinstance(value[0], dict):
                    print(f"    Sample item keys: {list(value[0].keys())}")
    elif isinstance(optimized_curve6, list):
        print(f"List with {len(optimized_curve6)} items")
        if len(optimized_curve6) > 0:
            sample_item = optimized_curve6[0]
            print(f"Sample item type: {type(sample_item)}")
            if isinstance(sample_item, dict):
                print(f"Sample item keys: {list(sample_item.keys())}")
    else:
        print(f"Unknown structure: {type(optimized_curve6)}")

    return overnight_submission, optimized_curve6

def extract_mechanisms_from_optimized(optimized_curve6):
    """Extract mechanisms from the optimized curve 6 data."""
    mechanisms = []

    if isinstance(optimized_curve6, list):
        # If it's directly a list of mechanisms
        mechanisms = optimized_curve6
    elif isinstance(optimized_curve6, dict):
        # If it's a dictionary, look for mechanism lists
        for key, value in optimized_curve6.items():
            print(f"Checking key '{key}': {type(value)}")
            if isinstance(value, list) and len(value) > 0:
                # Check if this looks like a list of mechanisms
                if isinstance(value[0], dict) and 'x0' in value[0]:
                    print(f"  Found mechanisms in key '{key}': {len(value)} items")
                    mechanisms.extend(value)

    return mechanisms

def create_merged_submission(overnight_submission, optimized_mechanisms):
    """Create a merged submission with the new mechanisms."""
    print(f"\n🔄 CREATING MERGED SUBMISSION...")

    # Get existing Problem 6 mechanisms
    existing_problem6 = overnight_submission.get('Problem 6', [])
    print(f"Existing Problem 6 mechanisms: {len(existing_problem6)}")
    print(f"Optimized mechanisms to add: {len(optimized_mechanisms)}")

    # Combine mechanisms (simple concatenation for now)
    combined_mechanisms = existing_problem6 + optimized_mechanisms
    print(f"Total combined mechanisms: {len(combined_mechanisms)}")

    # Create updated submission
    updated_submission = overnight_submission.copy()
    updated_submission['Problem 6'] = combined_mechanisms

    # Save the merged submission
    np.save('submission_merged_curve6_simple.npy', updated_submission)
    print(f"✅ SAVED: submission_merged_curve6_simple.npy")

    # Also save just the combined Problem 6 for analysis
    np.save('problem6_combined_mechanisms.npy', combined_mechanisms)
    print(f"✅ SAVED: problem6_combined_mechanisms.npy ({len(combined_mechanisms)} mechanisms)")

    return updated_submission

def main():
    """Main execution function."""
    print("🚀 EXAMINING AND MERGING CURVE 6 DATA")
    print("="*50)

    # Examine data structures
    overnight_submission, optimized_curve6 = examine_data()

    if overnight_submission is None or optimized_curve6 is None:
        print("❌ Failed to load data")
        return

    # Extract mechanisms from optimized data
    optimized_mechanisms = extract_mechanisms_from_optimized(optimized_curve6)

    if len(optimized_mechanisms) == 0:
        print("❌ No mechanisms found in optimized data")
        return

    print(f"\n📊 SUMMARY:")
    print(f"Overnight Problem 6: {len(overnight_submission.get('Problem 6', []))} mechanisms")
    print(f"Optimized mechanisms: {len(optimized_mechanisms)} mechanisms")

    # Create merged submission
    merged_submission = create_merged_submission(overnight_submission, optimized_mechanisms)

    print(f"✅ Merged submission created with {len(merged_submission['Problem 6'])} total mechanisms for Problem 6")

    # Print some info about the merged submission
    print(f"\nFinal submission structure:")
    for key in merged_submission.keys():
        mechanism_count = len(merged_submission[key])
        print(f"  {key}: {mechanism_count} mechanisms")

if __name__ == "__main__":
    main()