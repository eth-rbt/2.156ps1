#!/usr/bin/env python3
"""
Merge Problem 6 with Actual LINKS Evaluation

This script properly merges Problem 6 mechanisms from both sources and extracts
the Pareto frontier using actual LINKS evaluation.

Run this in the proper environment with JAX/LINKS available.
"""

import numpy as np
import matplotlib.pyplot as plt
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from utils import setup_environment, load_target_curves
from LINKS.Optimization import Tools

def evaluate_mechanisms(mechanisms, target_curve):
    """Evaluate a list of mechanisms using LINKS Tools."""
    if len(mechanisms) == 0:
        return np.array([]).reshape(0, 2)

    print(f"   🔧 Evaluating {len(mechanisms)} mechanisms...")

    # Setup evaluation tools
    tools = Tools(device='cpu')
    tools.compile()

    # Extract mechanism data
    x0s = [np.array(mech['x0']) for mech in mechanisms]
    edges = [np.array(mech['edges']) for mech in mechanisms]
    fixed_joints = [np.array(mech['fixed_joints']) for mech in mechanisms]
    motors = [np.array(mech['motor']) for mech in mechanisms]
    target_idxs = [mech['target_joint'] for mech in mechanisms]

    try:
        distances, materials = tools(x0s, edges, fixed_joints, motors, target_curve, target_idxs)
        F = np.column_stack([distances, materials])
        print(f"   ✅ Evaluation complete: {len(F)} F values")
        return F
    except Exception as e:
        print(f"   ❌ Error evaluating mechanisms: {e}")
        return np.array([]).reshape(0, 2)

def extract_pareto_frontier(mechanisms, f_values, feasibility_threshold=(0.75, 10.0)):
    """Extract Pareto frontier from mechanisms based on F values."""
    print(f"📊 EXTRACTING PARETO FRONTIER...")
    print(f"Input: {len(mechanisms)} mechanisms")

    if len(f_values) == 0:
        print("❌ No F values to process")
        return [], np.array([]).reshape(0, 2)

    # Filter for feasible solutions
    feasible_mask = np.logical_and(
        f_values[:, 0] <= feasibility_threshold[0],
        f_values[:, 1] <= feasibility_threshold[1]
    )
    feasible_count = np.sum(feasible_mask)
    print(f"Feasible solutions: {feasible_count}/{len(mechanisms)}")

    if feasible_count == 0:
        print("⚠️ No feasible solutions found! Using all mechanisms.")
        return mechanisms, f_values

    # Get feasible mechanisms and F values
    feasible_indices = np.where(feasible_mask)[0]
    feasible_f = f_values[feasible_indices]
    feasible_mechanisms = [mechanisms[i] for i in feasible_indices]

    # Apply non-dominated sorting to get Pareto frontier
    nds = NonDominatedSorting()
    fronts = nds.do(feasible_f)

    if len(fronts) > 0:
        pareto_indices = fronts[0]
        pareto_f = feasible_f[pareto_indices]
        pareto_mechanisms = [feasible_mechanisms[i] for i in pareto_indices]

        print(f"✅ Pareto frontier: {len(pareto_mechanisms)} mechanisms")

        # Calculate hypervolume
        hv_indicator = HV(np.array([0.75, 10.0]))
        hv = hv_indicator(pareto_f)
        print(f"📈 Hypervolume: {hv:.6f}")

        return pareto_mechanisms, pareto_f
    else:
        print("⚠️ No Pareto front found, using all feasible")
        return feasible_mechanisms, feasible_f

def plot_comparison(overnight_f, optimized_f, merged_pareto_f):
    """Plot comparison of the datasets."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot overnight data
    ax = axes[0]
    if len(overnight_f) > 0:
        ax.scatter(overnight_f[:, 0], overnight_f[:, 1], alpha=0.6, s=20, color='blue', label='Overnight')
        feasible_mask = np.logical_and(overnight_f[:, 0] <= 0.75, overnight_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(overnight_f[feasible_mask])
            ax.set_title(f'Overnight Problem 6\\n{len(overnight_f)} mechanisms\\nHV: {hv:.4f}')
        else:
            ax.set_title(f'Overnight Problem 6\\n{len(overnight_f)} mechanisms\\nNo feasible')
    else:
        ax.set_title('Overnight Problem 6\\n(No data)')
    ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, label='Constraints')
    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Material')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot optimized data
    ax = axes[1]
    if len(optimized_f) > 0:
        ax.scatter(optimized_f[:, 0], optimized_f[:, 1], alpha=0.6, s=20, color='green', label='Optimized')
        feasible_mask = np.logical_and(optimized_f[:, 0] <= 0.75, optimized_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(optimized_f[feasible_mask])
            ax.set_title(f'Optimized Problem 6\\n{len(optimized_f)} mechanisms\\nHV: {hv:.4f}')
        else:
            ax.set_title(f'Optimized Problem 6\\n{len(optimized_f)} mechanisms\\nNo feasible')
    else:
        ax.set_title('Optimized Problem 6\\n(No data)')
    ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, label='Constraints')
    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Material')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Plot merged Pareto frontier
    ax = axes[2]
    if len(merged_pareto_f) > 0:
        ax.scatter(merged_pareto_f[:, 0], merged_pareto_f[:, 1], alpha=0.8, s=40, color='red', label='Merged Pareto')
        feasible_mask = np.logical_and(merged_pareto_f[:, 0] <= 0.75, merged_pareto_f[:, 1] <= 10.0)
        if np.any(feasible_mask):
            hv = HV(np.array([0.75, 10.0]))(merged_pareto_f[feasible_mask])
            ax.set_title(f'Merged Pareto Frontier\\n{len(merged_pareto_f)} mechanisms\\nHV: {hv:.4f}')
        else:
            ax.set_title(f'Merged Pareto Frontier\\n{len(merged_pareto_f)} mechanisms\\nNo feasible')
    else:
        ax.set_title('Merged Pareto Frontier\\n(No data)')
    ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, label='Constraints')
    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Distance')
    ax.set_ylabel('Material')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig('problem6_merge_evaluation_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ SAVED: problem6_merge_evaluation_comparison.png")

def main():
    """Main execution function."""
    print("🚀 MERGING PROBLEM 6 WITH ACTUAL LINKS EVALUATION")
    print("="*65)

    # Setup environment
    setup_environment()
    target_curves = load_target_curves()
    target_curve_6 = target_curves[5]  # Problem 6 is curve index 5

    # Load overnight run submission
    print("📥 LOADING DATA...")
    overnight_path = '/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/overnight_run_20250929_204331/submission.npy'
    overnight_submission = np.load(overnight_path, allow_pickle=True).item()
    overnight_problem6 = overnight_submission.get('Problem 6', [])
    print(f"✅ Overnight Problem 6: {len(overnight_problem6)} mechanisms")

    # Load optimized curve 6 data
    optimized_path = '/Users/ethrbt/code/2.156/2155-Optimization-Challenge-Problem/optimized_curve_6.npy'
    optimized_data = np.load(optimized_path, allow_pickle=True).item()
    optimized_problem6 = optimized_data.get('Problem 6', [])
    print(f"✅ Optimized Problem 6: {len(optimized_problem6)} mechanisms")

    # Evaluate both datasets
    print("\\n🔧 EVALUATING MECHANISMS...")
    overnight_f = evaluate_mechanisms(overnight_problem6, target_curve_6)
    optimized_f = evaluate_mechanisms(optimized_problem6, target_curve_6)

    # Combine mechanisms and F values
    print("\\n🔄 COMBINING DATASETS...")
    combined_mechanisms = overnight_problem6 + optimized_problem6
    combined_f = np.vstack([overnight_f, optimized_f]) if len(overnight_f) > 0 and len(optimized_f) > 0 else (overnight_f if len(overnight_f) > 0 else optimized_f)

    print(f"Combined: {len(combined_mechanisms)} mechanisms")

    # Extract Pareto frontier from combined data
    pareto_mechanisms, pareto_f = extract_pareto_frontier(combined_mechanisms, combined_f)

    # Create final submission with merged Pareto frontier
    print("\\n💾 CREATING FINAL SUBMISSION...")
    final_submission = overnight_submission.copy()
    final_submission['Problem 6'] = pareto_mechanisms

    # Save final submission
    np.save('submission_final_merged_problem6.npy', final_submission)
    print(f"✅ SAVED: submission_final_merged_problem6.npy")

    # Save analysis data
    np.save('problem6_final_pareto_mechanisms.npy', pareto_mechanisms)
    np.save('problem6_final_pareto_f_values.npy', pareto_f)
    print(f"✅ SAVED: problem6_final_pareto_mechanisms.npy ({len(pareto_mechanisms)} mechanisms)")
    print(f"✅ SAVED: problem6_final_pareto_f_values.npy")

    # Plot comparison
    plot_comparison(overnight_f, optimized_f, pareto_f)

    # Summary
    print(f"\\n📊 FINAL SUMMARY:")
    print(f"Overnight Problem 6: {len(overnight_problem6)} → {len(overnight_f)} evaluated")
    print(f"Optimized Problem 6: {len(optimized_problem6)} → {len(optimized_f)} evaluated")
    print(f"Combined total: {len(combined_mechanisms)} mechanisms")
    print(f"Final Pareto frontier: {len(pareto_mechanisms)} mechanisms")

    if len(pareto_f) > 0:
        feasible_mask = np.logical_and(pareto_f[:, 0] <= 0.75, pareto_f[:, 1] <= 10.0)
        feasible_count = np.sum(feasible_mask)
        print(f"Feasible in final Pareto: {feasible_count}/{len(pareto_mechanisms)}")

        if feasible_count > 0:
            hv = HV(np.array([0.75, 10.0]))(pareto_f[feasible_mask])
            print(f"Final hypervolume: {hv:.6f}")

    print(f"\\n🎯 Final submission has been created with optimized Problem 6!")

if __name__ == "__main__":
    main()