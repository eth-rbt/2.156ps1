#!/usr/bin/env python3
"""
Merge Curve 6 Data from Overnight Run and Optimized Results

This script:
1. Loads curve 6 data from the overnight run
2. Loads the separate optimized_curve_6.npy file
3. Combines them and extracts the Pareto frontier
4. Updates the submission with the merged Pareto frontier for Problem 6
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from utils import setup_environment, load_target_curves
from LINKS.Optimization import Tools

def load_and_examine_data():
    """Load both curve 6 datasets and examine their structure."""
    print("🔍 LOADING CURVE 6 DATA...")

    # Load overnight run submission
    overnight_submission = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/overnight_run_20250929_204331/submission.npy', allow_pickle=True).item()

    # Load separate optimized curve 6
    optimized_curve6 = np.load('/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/optimized_curve_6.npy', allow_pickle=True).item()

    print(f"Overnight submission keys: {list(overnight_submission.keys())}")
    print(f"Optimized curve 6 type: {type(optimized_curve6)}")

    # Examine Problem 6 from overnight run
    if 'Problem 6' in overnight_submission:
        problem6_overnight = overnight_submission['Problem 6']
        print(f"Problem 6 (overnight): {len(problem6_overnight)} mechanisms")
        if len(problem6_overnight) > 0:
            print(f"Sample mechanism keys: {list(problem6_overnight[0].keys())}")

    # Examine optimized curve 6 structure
    if isinstance(optimized_curve6, dict):
        print(f"Optimized curve 6 keys: {list(optimized_curve6.keys())}")
    elif isinstance(optimized_curve6, list):
        print(f"Optimized curve 6: {len(optimized_curve6)} items")
        if len(optimized_curve6) > 0:
            print(f"Sample item type: {type(optimized_curve6[0])}")
            if isinstance(optimized_curve6[0], dict):
                print(f"Sample item keys: {list(optimized_curve6[0].keys())}")

    return overnight_submission, optimized_curve6

def evaluate_mechanisms(mechanisms, target_curve):
    """Evaluate a list of mechanisms and return F values."""
    if len(mechanisms) == 0:
        return np.array([]).reshape(0, 2)

    # Setup evaluation tools
    tools = Tools(device='cpu')
    tools.compile()

    # Extract mechanism data
    x0s = [np.array(mech['x0']) for mech in mechanisms]
    edges = [np.array(mech['edges']) for mech in mechanisms]
    fixed_joints = [np.array(mech['fixed_joints']) for mech in mechanisms]
    motors = [np.array(mech['motor']) for mech in mechanisms]
    target_idxs = [mech['target_joint'] for mech in mechanisms]

    print(f"   Evaluating {len(mechanisms)} mechanisms...")

    try:
        distances, materials = tools(x0s, edges, fixed_joints, motors, target_curve, target_idxs)
        return np.column_stack([distances, materials])
    except Exception as e:
        print(f"   Error evaluating mechanisms: {e}")
        return np.array([]).reshape(0, 2)

def merge_and_extract_pareto(overnight_submission, optimized_curve6, target_curve):
    """Merge both datasets and extract the combined Pareto frontier."""
    print("\n🔄 MERGING CURVE 6 DATA...")

    # Get mechanisms from overnight run
    overnight_mechanisms = overnight_submission.get('Problem 6', [])
    print(f"Overnight run mechanisms: {len(overnight_mechanisms)}")

    # Get mechanisms from optimized file
    optimized_mechanisms = []
    if isinstance(optimized_curve6, list):
        optimized_mechanisms = optimized_curve6
    elif isinstance(optimized_curve6, dict):
        # If it's a dict, try to extract mechanisms
        for key, value in optimized_curve6.items():
            if isinstance(value, list):
                optimized_mechanisms.extend(value)

    print(f"Optimized mechanisms: {len(optimized_mechanisms)}")

    # Combine all mechanisms
    all_mechanisms = overnight_mechanisms + optimized_mechanisms
    print(f"Total combined mechanisms: {len(all_mechanisms)}")

    if len(all_mechanisms) == 0:
        print("❌ No mechanisms found to merge!")
        return [], np.array([]).reshape(0, 2)

    # Evaluate all mechanisms
    all_F = evaluate_mechanisms(all_mechanisms, target_curve)

    if len(all_F) == 0:
        print("❌ No valid evaluations!")
        return [], np.array([]).reshape(0, 2)

    print(f"Evaluation results: {len(all_F)} F values")

    # Filter for feasible solutions
    feasible_mask = np.logical_and(all_F[:, 0] <= 0.75, all_F[:, 1] <= 10.0)
    feasible_count = np.sum(feasible_mask)
    print(f"Feasible solutions: {feasible_count}/{len(all_F)}")

    if feasible_count == 0:
        print("⚠️ No feasible solutions found!")
        return all_mechanisms, all_F

    # Extract feasible mechanisms and F values
    feasible_indices = np.where(feasible_mask)[0]
    feasible_F = all_F[feasible_indices]
    feasible_mechanisms = [all_mechanisms[i] for i in feasible_indices]

    # Apply non-dominated sorting to get Pareto frontier
    nds = NonDominatedSorting()
    fronts = nds.do(feasible_F)

    if len(fronts) > 0:
        pareto_indices = fronts[0]
        pareto_F = feasible_F[pareto_indices]
        pareto_mechanisms = [feasible_mechanisms[i] for i in pareto_indices]

        print(f"✅ Extracted Pareto frontier: {len(pareto_mechanisms)} mechanisms")

        # Calculate hypervolume
        hv_indicator = HV(np.array([0.75, 10.0]))
        hv = hv_indicator(pareto_F)
        print(f"Hypervolume: {hv:.6f}")

        return pareto_mechanisms, pareto_F
    else:
        print("⚠️ No Pareto front found, returning all feasible")
        return feasible_mechanisms, feasible_F

def plot_comparison(overnight_f, optimized_f, merged_f):
    """Plot comparison of the three datasets."""
    plt.figure(figsize=(15, 5))

    # Plot overnight data
    plt.subplot(1, 3, 1)
    if len(overnight_f) > 0:
        plt.scatter(overnight_f[:, 0], overnight_f[:, 1], alpha=0.6, s=20, color='blue')
        feasible_mask = np.logical_and(overnight_f[:, 0] <= 0.75, overnight_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(overnight_f[feasible_mask])
            plt.title(f'Overnight Run\n{len(overnight_f)} mechanisms\nHV: {hv:.4f}')
        else:
            plt.title(f'Overnight Run\n{len(overnight_f)} mechanisms\nNo feasible')
    else:
        plt.title('Overnight Run\n(No data)')
    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Material')
    plt.grid(True, alpha=0.3)

    # Plot optimized data
    plt.subplot(1, 3, 2)
    if len(optimized_f) > 0:
        plt.scatter(optimized_f[:, 0], optimized_f[:, 1], alpha=0.6, s=20, color='green')
        feasible_mask = np.logical_and(optimized_f[:, 0] <= 0.75, optimized_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(optimized_f[feasible_mask])
            plt.title(f'Optimized\n{len(optimized_f)} mechanisms\nHV: {hv:.4f}')
        else:
            plt.title(f'Optimized\n{len(optimized_f)} mechanisms\nNo feasible')
    else:
        plt.title('Optimized\n(No data)')
    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Material')
    plt.grid(True, alpha=0.3)

    # Plot merged Pareto frontier
    plt.subplot(1, 3, 3)
    if len(merged_f) > 0:
        plt.scatter(merged_f[:, 0], merged_f[:, 1], alpha=0.7, s=30, color='red')
        feasible_mask = np.logical_and(merged_f[:, 0] <= 0.75, merged_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(merged_f[feasible_mask])
            plt.title(f'Merged Pareto\n{len(merged_f)} mechanisms\nHV: {hv:.4f}')
        else:
            plt.title(f'Merged Pareto\n{len(merged_f)} mechanisms\nNo feasible')
    else:
        plt.title('Merged Pareto\n(No data)')
    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Material')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('curve6_merge_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SAVED: curve6_merge_comparison.png")

def update_submission(overnight_submission, new_problem6_mechanisms):
    """Update the submission with the new Problem 6 mechanisms."""
    print("\n💾 UPDATING SUBMISSION...")

    # Create updated submission
    updated_submission = overnight_submission.copy()
    updated_submission['Problem 6'] = new_problem6_mechanisms

    # Save updated submission
    np.save('submission_merged_curve6.npy', updated_submission)
    print(f"✅ SAVED: submission_merged_curve6.npy")
    print(f"Updated Problem 6 with {len(new_problem6_mechanisms)} mechanisms")

    return updated_submission

def main():
    """Main execution function."""
    print("🚀 MERGING CURVE 6 DATA FROM OVERNIGHT RUN AND OPTIMIZED RESULTS")
    print("="*70)

    setup_environment()
    target_curves = load_target_curves()
    target_curve_6 = target_curves[5]  # Curve 6 is index 5

    # Load data
    overnight_submission, optimized_curve6 = load_and_examine_data()

    # Get overnight Problem 6 mechanisms for comparison
    overnight_mechanisms = overnight_submission.get('Problem 6', [])
    overnight_f = evaluate_mechanisms(overnight_mechanisms, target_curve_6) if len(overnight_mechanisms) > 0 else np.array([]).reshape(0, 2)

    # Get optimized mechanisms for comparison
    optimized_mechanisms = []
    if isinstance(optimized_curve6, list):
        optimized_mechanisms = optimized_curve6
    elif isinstance(optimized_curve6, dict):
        for key, value in optimized_curve6.items():
            if isinstance(value, list):
                optimized_mechanisms.extend(value)

    optimized_f = evaluate_mechanisms(optimized_mechanisms, target_curve_6) if len(optimized_mechanisms) > 0 else np.array([]).reshape(0, 2)

    # Merge and extract Pareto frontier
    merged_mechanisms, merged_f = merge_and_extract_pareto(overnight_submission, optimized_curve6, target_curve_6)

    # Plot comparison
    plot_comparison(overnight_f, optimized_f, merged_f)

    # Update submission if we have improved results
    if len(merged_mechanisms) > 0:
        updated_submission = update_submission(overnight_submission, merged_mechanisms)

        print("\n📊 SUMMARY:")
        print(f"Overnight Problem 6: {len(overnight_mechanisms)} mechanisms")
        print(f"Optimized data: {len(optimized_mechanisms)} mechanisms")
        print(f"Merged Pareto: {len(merged_mechanisms)} mechanisms")

        if len(merged_f) > 0:
            feasible_mask = np.logical_and(merged_f[:, 0] <= 0.75, merged_f[:, 1] <= 10.0)
            if np.any(feasible_mask):
                hv = HV(np.array([0.75, 10.0]))(merged_f[feasible_mask])
                print(f"Final hypervolume: {hv:.6f}")
    else:
        print("❌ No valid merged results to save")

if __name__ == "__main__":
    main()