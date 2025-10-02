#!/usr/bin/env python3
"""
Merge Two Complete Submission Files

This script merges two complete submission .npy files by:
1. Loading both submissions
2. Combining mechanisms for each problem (1-6)
3. Extracting Pareto frontier for each problem from combined mechanisms
4. Creating a final optimized submission with the best from both

Usage:
    python merge_submissions.py submission1.npy submission2.npy [output_name]
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.indicators.hv import HV
from utils import setup_environment, load_target_curves
from LINKS.Optimization import Tools

def load_submission(file_path):
    """Load a submission file and validate its structure."""
    try:
        submission = np.load(file_path, allow_pickle=True).item()
        print(f"✅ Loaded: {file_path}")

        # Validate structure
        expected_problems = [f'Problem {i}' for i in range(1, 7)]
        for problem in expected_problems:
            if problem not in submission:
                print(f"⚠️ Warning: {problem} not found in {file_path}")
            else:
                count = len(submission[problem])
                print(f"   {problem}: {count} mechanisms")

        return submission
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

def evaluate_mechanisms(mechanisms, target_curve, problem_name):
    """Evaluate a list of mechanisms using LINKS Tools."""
    if len(mechanisms) == 0:
        print(f"   ⚠️ {problem_name}: No mechanisms to evaluate")
        return np.array([]).reshape(0, 2)

    print(f"   🔧 {problem_name}: Evaluating {len(mechanisms)} mechanisms...")

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
        print(f"   ✅ {problem_name}: Evaluation complete ({len(F)} F values)")
        return F
    except Exception as e:
        print(f"   ❌ {problem_name}: Error evaluating mechanisms: {e}")
        return np.array([]).reshape(0, 2)

def extract_pareto_frontier(mechanisms, f_values, problem_name, feasibility_threshold=(0.75, 10.0)):
    """Extract Pareto frontier from mechanisms based on F values."""
    if len(mechanisms) == 0 or len(f_values) == 0:
        print(f"   ⚠️ {problem_name}: No data to process")
        return [], np.array([]).reshape(0, 2), 0.0

    print(f"   📊 {problem_name}: Extracting Pareto frontier from {len(mechanisms)} mechanisms")

    # Filter for feasible solutions
    feasible_mask = np.logical_and(
        f_values[:, 0] <= feasibility_threshold[0],
        f_values[:, 1] <= feasibility_threshold[1]
    )
    feasible_count = np.sum(feasible_mask)
    print(f"   🎯 {problem_name}: {feasible_count}/{len(mechanisms)} feasible solutions")

    if feasible_count == 0:
        print(f"   ⚠️ {problem_name}: No feasible solutions! Using all mechanisms")
        return mechanisms, f_values, 0.0

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

        # Calculate hypervolume
        hv_indicator = HV(np.array([0.75, 10.0]))
        hv = hv_indicator(pareto_f)

        print(f"   ✅ {problem_name}: Pareto frontier with {len(pareto_mechanisms)} mechanisms (HV: {hv:.6f})")
        return pareto_mechanisms, pareto_f, hv
    else:
        print(f"   ⚠️ {problem_name}: No Pareto front found, using all feasible")
        hv_indicator = HV(np.array([0.75, 10.0]))
        hv = hv_indicator(feasible_f)
        return feasible_mechanisms, feasible_f, hv

def merge_single_problem(submission1, submission2, problem_key, target_curve):
    """Merge mechanisms for a single problem from both submissions."""
    print(f"\n🔄 MERGING {problem_key}...")

    # Get mechanisms from both submissions
    mechanisms1 = submission1.get(problem_key, [])
    mechanisms2 = submission2.get(problem_key, [])

    print(f"   Submission 1: {len(mechanisms1)} mechanisms")
    print(f"   Submission 2: {len(mechanisms2)} mechanisms")

    # Combine mechanisms
    combined_mechanisms = mechanisms1 + mechanisms2
    print(f"   Combined: {len(combined_mechanisms)} mechanisms")

    if len(combined_mechanisms) == 0:
        print(f"   ❌ {problem_key}: No mechanisms to merge!")
        return [], np.array([]).reshape(0, 2), 0.0

    # Evaluate combined mechanisms
    combined_f = evaluate_mechanisms(combined_mechanisms, target_curve, problem_key)

    if len(combined_f) == 0:
        print(f"   ❌ {problem_key}: Evaluation failed!")
        return [], np.array([]).reshape(0, 2), 0.0

    # Extract Pareto frontier
    pareto_mechanisms, pareto_f, hv = extract_pareto_frontier(combined_mechanisms, combined_f, problem_key)

    return pareto_mechanisms, pareto_f, hv

def create_comparison_plots(submission1, submission2, merged_submission, target_curves, output_dir):
    """Create comparison plots showing before/after for each problem."""
    print(f"\n📊 CREATING COMPARISON PLOTS...")

    # Setup evaluation tools
    tools = Tools(device='cpu')
    tools.compile()

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()

    hv_results = {'submission1': [], 'submission2': [], 'merged': []}

    for i in range(6):
        problem_key = f'Problem {i + 1}'
        ax = axes[i]
        target_curve = target_curves[i]

        # Evaluate mechanisms from each submission
        for submission, label, color in [(submission1, 'Submission 1', 'blue'),
                                       (submission2, 'Submission 2', 'green'),
                                       (merged_submission, 'Merged', 'red')]:
            mechanisms = submission.get(problem_key, [])
            if len(mechanisms) > 0:
                try:
                    f_values = evaluate_mechanisms(mechanisms, target_curve, f"{problem_key}-{label}")
                    if len(f_values) > 0:
                        # Plot all points
                        alpha = 0.4 if label != 'Merged' else 0.8
                        size = 15 if label != 'Merged' else 25
                        ax.scatter(f_values[:, 0], f_values[:, 1], alpha=alpha, s=size,
                                 color=color, label=f'{label} ({len(mechanisms)})')

                        # Calculate hypervolume for feasible solutions
                        feasible_mask = np.logical_and(f_values[:, 0] <= 0.75, f_values[:, 1] <= 10.0)
                        if np.any(feasible_mask):
                            hv = HV(np.array([0.75, 10.0]))(f_values[feasible_mask])
                            if label == 'Submission 1':
                                hv_results['submission1'].append(hv)
                            elif label == 'Submission 2':
                                hv_results['submission2'].append(hv)
                            else:
                                hv_results['merged'].append(hv)
                        else:
                            if label == 'Submission 1':
                                hv_results['submission1'].append(0.0)
                            elif label == 'Submission 2':
                                hv_results['submission2'].append(0.0)
                            else:
                                hv_results['merged'].append(0.0)
                except Exception as e:
                    print(f"   ⚠️ Error plotting {problem_key}-{label}: {e}")
            else:
                if label == 'Submission 1':
                    hv_results['submission1'].append(0.0)
                elif label == 'Submission 2':
                    hv_results['submission2'].append(0.0)
                else:
                    hv_results['merged'].append(0.0)

        # Add constraint lines
        ax.axvline(x=0.75, color='black', linestyle='--', alpha=0.7, label='Constraints')
        ax.axhline(y=10.0, color='black', linestyle='--', alpha=0.7)

        # Formatting
        ax.set_xlabel('Distance')
        ax.set_ylabel('Material')
        ax.set_title(f'{problem_key}')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'submission_merge_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SAVED: {plot_path}")

    # Create hypervolume summary plot
    plt.figure(figsize=(12, 6))
    problems = [f'P{i+1}' for i in range(6)]
    x = np.arange(len(problems))

    plt.bar(x - 0.25, hv_results['submission1'], 0.25, label='Submission 1', color='blue', alpha=0.7)
    plt.bar(x, hv_results['submission2'], 0.25, label='Submission 2', color='green', alpha=0.7)
    plt.bar(x + 0.25, hv_results['merged'], 0.25, label='Merged', color='red', alpha=0.7)

    plt.xlabel('Problems')
    plt.ylabel('Hypervolume')
    plt.title('Hypervolume Comparison: Submission Merging Results')
    plt.xticks(x, problems)
    plt.legend()
    plt.grid(True, alpha=0.3)

    hv_plot_path = os.path.join(output_dir, 'hypervolume_comparison.png')
    plt.savefig(hv_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ SAVED: {hv_plot_path}")

    return hv_results

def merge_submissions(file1, file2, output_name=None):
    """Main function to merge two submission files."""
    print("🚀 MERGING TWO COMPLETE SUBMISSION FILES")
    print("="*60)

    # Setup environment
    setup_environment()
    target_curves = load_target_curves()

    # Load both submissions
    print("📥 LOADING SUBMISSIONS...")
    submission1 = load_submission(file1)
    submission2 = load_submission(file2)

    if submission1 is None or submission2 is None:
        print("❌ Failed to load submissions. Exiting.")
        return

    # Create output directory
    if output_name is None:
        output_name = "merged_submission"

    output_dir = f"{output_name}_{int(np.random.random() * 10000):04d}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Created output directory: {output_dir}")

    # Initialize merged submission
    merged_submission = {}
    merge_summary = []

    # Merge each problem
    print("\n🔄 MERGING ALL PROBLEMS...")
    for i in range(6):
        problem_key = f'Problem {i + 1}'
        target_curve = target_curves[i]

        pareto_mechanisms, pareto_f, hv = merge_single_problem(
            submission1, submission2, problem_key, target_curve
        )

        merged_submission[problem_key] = pareto_mechanisms

        # Store summary info
        count1 = len(submission1.get(problem_key, []))
        count2 = len(submission2.get(problem_key, []))
        merged_count = len(pareto_mechanisms)

        merge_summary.append({
            'problem': problem_key,
            'submission1_count': count1,
            'submission2_count': count2,
            'combined_count': count1 + count2,
            'merged_count': merged_count,
            'hypervolume': hv
        })

    # Save merged submission
    merged_path = os.path.join(output_dir, f'{output_name}.npy')
    np.save(merged_path, merged_submission)
    print(f"\n✅ SAVED: {merged_path}")

    # Create comparison plots
    hv_results = create_comparison_plots(submission1, submission2, merged_submission, target_curves, output_dir)

    # Save summary report
    print(f"\n📊 MERGE SUMMARY REPORT:")
    print(f"{'Problem':<12} {'Sub1':<6} {'Sub2':<6} {'Combined':<10} {'Merged':<8} {'HV':<10}")
    print("-" * 60)

    total_hv1 = sum(hv_results['submission1'])
    total_hv2 = sum(hv_results['submission2'])
    total_hv_merged = sum(hv_results['merged'])

    for summary in merge_summary:
        print(f"{summary['problem']:<12} {summary['submission1_count']:<6} {summary['submission2_count']:<6} "
              f"{summary['combined_count']:<10} {summary['merged_count']:<8} {summary['hypervolume']:<10.4f}")

    print("-" * 60)
    print(f"{'TOTAL HV':<12} {total_hv1:<6.3f} {total_hv2:<6.3f} {'N/A':<10} {'N/A':<8} {total_hv_merged:<10.4f}")

    # Save summary to file
    summary_path = os.path.join(output_dir, 'merge_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("SUBMISSION MERGE SUMMARY\\n")
        f.write("="*50 + "\\n\\n")
        f.write(f"Input Files:\\n")
        f.write(f"  Submission 1: {file1}\\n")
        f.write(f"  Submission 2: {file2}\\n\\n")
        f.write(f"Output Files:\\n")
        f.write(f"  Merged Submission: {merged_path}\\n")
        f.write(f"  Comparison Plots: {output_dir}/\\n\\n")

        f.write(f"{'Problem':<12} {'Sub1':<6} {'Sub2':<6} {'Combined':<10} {'Merged':<8} {'HV':<10}\\n")
        f.write("-" * 60 + "\\n")
        for summary in merge_summary:
            f.write(f"{summary['problem']:<12} {summary['submission1_count']:<6} {summary['submission2_count']:<6} "
                   f"{summary['combined_count']:<10} {summary['merged_count']:<8} {summary['hypervolume']:<10.4f}\\n")
        f.write("-" * 60 + "\\n")
        f.write(f"{'TOTAL HV':<12} {total_hv1:<6.3f} {total_hv2:<6.3f} {'N/A':<10} {'N/A':<8} {total_hv_merged:<10.4f}\\n")

    print(f"✅ SAVED: {summary_path}")

    print(f"\\n🎯 MERGE COMPLETE!")
    print(f"Final merged submission: {merged_path}")
    print(f"Total hypervolume improvement: {total_hv_merged:.4f} vs {max(total_hv1, total_hv2):.4f}")

def main():
    """Main execution with command line arguments."""
    if len(sys.argv) < 3:
        print("Usage: python merge_submissions.py <submission1.npy> <submission2.npy> [output_name]")
        print("\\nExample:")
        print("  python merge_submissions.py overnight_run.npy optimized_results.npy final_submission")
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

    merge_submissions(file1, file2, output_name)

if __name__ == "__main__":
    main()