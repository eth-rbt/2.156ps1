#!/usr/bin/env python3
"""
Main script for running advanced linkage mechanism optimization.

This script replicates the functionality from Advanced_Starter_Notebook.ipynb
using the utilities from utils.py for advanced mechanism synthesis.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    AdvancedMechanismOptimizer,
    optimize_all_targets_advanced,
    setup_environment,
    load_target_curves,
    save_submission
)
from LINKS.CP import evaluate_submission


def main():
    """Main function that runs the advanced optimization workflow."""

    # Setup environment
    print("Setting up environment...")
    setup_environment()

    # Load target curves
    print("Loading target curves...")
    target_curves = load_target_curves()

    # Plot all target curves
    print("Plotting target curves...")
    plot_target_curves(target_curves)

    # Initialize optimizer
    print("Initializing advanced mechanism optimizer...")
    optimizer = AdvancedMechanismOptimizer(device='cpu')

    # Demonstrate single target optimization
    print("\nDemonstrating single target optimization (Problem 2)...")
    single_target_demo(optimizer, target_curves[1])

    # Run optimization for all targets
    print("\nRunning optimization for all target curves...")
    submission, all_results = optimize_all_targets_advanced(
        target_curves,
        N=7,
        pop_size=100,
        n_gen=50,
        use_gradient=True
    )

    # Evaluate and save submission
    print("\nEvaluating submission...")
    scores = evaluate_submission(submission)
    print(f"Submission scores: {scores}")

    # Save submission
    print("Saving submission...")
    filepath = save_submission(submission)
    print(f"Submission saved to: {filepath}")

    return submission, scores


def plot_target_curves(target_curves):
    """Plot all target curves in a 2x3 grid."""
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    for i in range(6):
        x_coords = np.array(target_curves[i])[:, 0]
        y_coords = np.array(target_curves[i])[:, 1]

        axs[i // 3, i % 3].plot(x_coords, y_coords, color='black', linewidth=3)
        axs[i // 3, i % 3].set_title(f'Egg {i + 1}')
        axs[i // 3, i % 3].axis('equal')
        axs[i // 3, i % 3].axis('off')

    plt.tight_layout()
    plt.show()


def single_target_demo(optimizer, target_curve):
    """Demonstrate optimization for a single target curve."""

    # Generate random mechanisms
    print("Generating random mechanisms...")
    initial_mechs = optimizer.generate_random_mechanisms(count=100, size=7)

    # Run GA optimization
    print("Running GA optimization...")
    results, problem = optimizer.run_mixed_variable_optimization(
        target_curve,
        N=7,
        pop_size=100,
        n_gen=50,
        initial_mechanisms=initial_mechs,
        verbose=False
    )

    if results.X is not None:
        # Apply gradient optimization
        print("Applying gradient optimization...")
        grad_results, original_mechs = optimizer.apply_gradient_optimization(
            results, problem, target_curve
        )

        # Evaluate hypervolume
        if grad_results is not None:
            combined_x0s = original_mechs[0] + grad_results
            combined_edges = original_mechs[1] + original_mechs[1]
            combined_fixed_joints = original_mechs[2] + original_mechs[2]
            combined_motors = original_mechs[3] + original_mechs[3]
            combined_target_idxs = original_mechs[4] + original_mechs[4]

            combined_mechs = (combined_x0s, combined_edges, combined_fixed_joints,
                            combined_motors, combined_target_idxs)

            hv_before, F_before = optimizer.evaluate_hypervolume(original_mechs, target_curve)
            hv_after, F_after = optimizer.evaluate_hypervolume(combined_mechs, target_curve)

            print(f"Hypervolume before gradient optimization: {hv_before:.4f}")
            print(f"Hypervolume after gradient optimization: {hv_after:.4f}")

        # Visualize best solutions
        print("Visualizing best solutions...")
        optimizer.visualize_best_solutions(results, problem, target_curve)

    else:
        print("No feasible solutions found!")


if __name__ == "__main__":
    submission, scores = main()
    print("\nOptimization complete!")
    print(f"Overall score: {scores['Overall Score']:.4f}")