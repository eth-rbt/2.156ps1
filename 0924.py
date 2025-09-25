#!/usr/bin/env python3
"""
0924.py - Run single curve optimization on all 6 curves and output a score.

This script applies the single_curve.py algorithm to all target curves
and creates a submission for scoring.
"""

import numpy as np
import matplotlib.pyplot as plt
from single_curve import main as single_curve_main
from utils import (
    setup_environment,
    load_target_curves,
    AdvancedMechanismOptimizer
)
from LINKS.CP import make_empty_submission, evaluate_submission


def create_submission_from_single_curve_results(results, problem, problem_number):
    """Create submission list from single curve optimization results."""
    submission_list = []

    if results is None or results.X is None:
        return submission_list

    optimizer = AdvancedMechanismOptimizer()
    submission_list = optimizer.create_submission_from_results(results, problem, problem_number)

    return submission_list


def run_all_curves_optimization(num_nodes=6, num_mechanisms=400, pop_size=200, n_gen=100):
    """
    Run single curve optimization on all 6 target curves.

    Args:
        num_nodes: Number of nodes in mechanisms
        num_mechanisms: Number of random mechanisms to generate
        pop_size: Population size for NSGA-II
        n_gen: Number of generations for NSGA-II

    Returns:
        submission: Complete submission dictionary
        all_results: Results for each curve
        all_problems: Problem instances for each curve
    """
    print("=== Running Single Curve Optimization on All 6 Curves ===")
    print(f"Parameters: {num_nodes} nodes, {num_mechanisms} mechanisms, {pop_size} pop_size, {n_gen} generations")
    print("="*60)

    # Setup environment
    setup_environment()

    # Load target curves
    target_curves = load_target_curves()

    # Initialize submission
    submission = make_empty_submission()
    all_results = {}
    all_problems = {}
    all_hypervolumes = {}

    # Process each curve
    for curve_idx in range(len(target_curves)):
        print(f"\n{'='*20} CURVE {curve_idx + 1} {'='*20}")

        try:
            # Temporarily modify the single_curve main function to work with different curves
            # We'll create a modified version that accepts curve_idx as parameter
            results, problem = optimize_single_curve(
                curve_idx,
                target_curves[curve_idx],
                num_nodes=num_nodes,
                num_mechanisms=num_mechanisms,
                pop_size=pop_size,
                n_gen=n_gen
            )

            if results is not None:
                # Store results
                all_results[f'Problem {curve_idx + 1}'] = results
                all_problems[f'Problem {curve_idx + 1}'] = problem

                # Create submission for this curve
                submission_list = create_submission_from_single_curve_results(
                    results, problem, curve_idx + 1
                )
                submission[f'Problem {curve_idx + 1}'] = submission_list

                # Calculate hypervolume
                if results.F is not None:
                    from pymoo.indicators.hv import HV
                    ind = HV(np.array([0.75, 10.0]))
                    hypervolume = ind(results.F)
                    all_hypervolumes[f'Problem {curve_idx + 1}'] = hypervolume

                    print(f"\nCurve {curve_idx + 1} Summary:")
                    print(f"- Solutions found: {len(results.F)}")
                    print(f"- Hypervolume: {hypervolume:.4f}")
                    print(f"- Best distance: {min(results.F[:, 0]):.4f}")
                    print(f"- Best material: {min(results.F[:, 1]):.4f}")
                    print(f"- Mechanisms in submission: {len(submission_list)}")
                else:
                    print(f"Curve {curve_idx + 1}: No solutions found!")
            else:
                print(f"Curve {curve_idx + 1}: Optimization failed!")

        except Exception as e:
            print(f"Error optimizing curve {curve_idx + 1}: {str(e)}")
            submission[f'Problem {curve_idx + 1}'] = []

    return submission, all_results, all_problems, all_hypervolumes


def optimize_single_curve(curve_idx, target_curve, num_nodes=7, num_mechanisms=200, pop_size=100, n_gen=100):
    """
    Modified version of single_curve main function for a specific curve.

    This replicates the logic from single_curve.py but for any curve index.
    """
    from LINKS.Optimization import MechanismRandomizer
    from utils import (
        filter_feasible_mechanisms,
        run_nsga2_optimization,
        plot_hypervolume_results,
        analyze_initial_population,
        AdvancedMechanismOptimizer
    )

    print(f"Target: Curve {curve_idx + 1}")
    print(f"Mechanism nodes: {num_nodes}")
    print(f"Random mechanisms to generate: {num_mechanisms}")
    print(f"NSGA-II population size: {pop_size}")
    print(f"NSGA-II generations: {n_gen}")

    # Generate random mechanisms
    print(f"\nGenerating {num_mechanisms} random mechanisms with {num_nodes} nodes...")
    randomizer = MechanismRandomizer(
        min_size=num_nodes,
        max_size=num_nodes,
        device='cpu'
    )

    mechanisms = []
    for i in range(num_mechanisms):
        try:
            mech = randomizer(n=num_nodes)
            mechanisms.append(mech)
        except:
            continue

    print(f"Successfully generated {len(mechanisms)} mechanisms")

    # Analyze initial population
    print("\nAnalyzing initial population...")
    filterthreshold = (2, 12.0)
    analyze_initial_population(mechanisms, target_curve)

    # Filter mechanisms
    print(f"\nFiltering mechanisms for feasibility...")
    print(f"Feasibility criteria: distance <= {filterthreshold[0]}, material <= {filterthreshold[1]}")

    feasible_mechanisms, performance_metrics = filter_feasible_mechanisms(
        mechanisms,
        target_curve,
        feasibility_threshold=filterthreshold
    )

    print(f"Feasible mechanisms: {len(feasible_mechanisms)}/{len(mechanisms)} "
          f"({100*len(feasible_mechanisms)/len(mechanisms):.1f}%)")

    if len(feasible_mechanisms) == 0:
        print("ERROR: No feasible mechanisms found!")
        return None, None

    # Print statistics
    if performance_metrics:
        distances = [p[0] for p in performance_metrics]
        materials = [p[1] for p in performance_metrics]
        print(f"Distance range: {min(distances):.4f} - {max(distances):.4f}")
        print(f"Material range: {min(materials):.4f} - {max(materials):.4f}")

    # Run NSGA-II optimization
    print(f"\nRunning NSGA-II optimization...")
    print(f"Population size: {pop_size}")
    print(f"Generations: {n_gen}")

    results, problem = run_nsga2_optimization(
        feasible_mechanisms,
        target_curve,
        N=num_nodes,
        pop_size=min(pop_size, len(feasible_mechanisms)),
        n_gen=n_gen,
        verbose=True
    )

    if results.X is not None:
        # Plot hypervolume results
        print(f"\nPlotting optimization results...")
        hypervolume = plot_hypervolume_results(
            results,
            ref_point=np.array([0.75, 10.0]),
            title=f"NSGA-II Results for Curve {curve_idx + 1} ({num_nodes} nodes)"
        )

        # Apply gradient optimization
        print(f"\nApplying gradient optimization...")
        optimizer = AdvancedMechanismOptimizer()

        grad_results, original_mechs = optimizer.apply_gradient_optimization(
            results, problem, target_curve,
            step_size=1e-4, n_steps=1000
        )

        if grad_results is not None and original_mechs is not None:
            # Evaluate gradient optimization improvement
            combined_x0s = original_mechs[0] + grad_results
            combined_edges = original_mechs[1] + original_mechs[1]
            combined_fixed_joints = original_mechs[2] + original_mechs[2]
            combined_motors = original_mechs[3] + original_mechs[3]
            combined_target_idxs = original_mechs[4] + original_mechs[4]

            combined_mechs = (combined_x0s, combined_edges, combined_fixed_joints,
                            combined_motors, combined_target_idxs)

            grad_hypervolume, grad_F = optimizer.evaluate_hypervolume(
                combined_mechs, target_curve, ref_point=np.array([0.75, 10.0])
            )

            print(f"\nGradient Optimization Results:")
            print(f"- Original hypervolume: {hypervolume:.4f}")
            print(f"- Gradient-enhanced hypervolume: {grad_hypervolume:.4f}")
            print(f"- Improvement: {grad_hypervolume - hypervolume:.4f}")
        else:
            print("Gradient optimization failed to produce results.")

        return results, problem
    else:
        print("No solutions found in NSGA-II optimization!")
        return None, None


def main():
    """Main function to run optimization on all curves and get final score."""

    # Run optimization on all curves
    submission, all_results, all_problems, all_hypervolumes = run_all_curves_optimization(
        num_nodes=7,
        num_mechanisms=200,
        pop_size=100,
        n_gen=100
    )

    print("\n" + "="*60)
    print("FINAL RESULTS SUMMARY")
    print("="*60)

    # Print hypervolume summary
    print("\nHypervolume Results:")
    total_hv = 0
    valid_curves = 0
    for problem, hv in all_hypervolumes.items():
        print(f"- {problem}: {hv:.4f}")
        total_hv += hv
        valid_curves += 1

    if valid_curves > 0:
        avg_hv = total_hv / valid_curves
        print(f"\nAverage Hypervolume: {avg_hv:.4f}")

    # Print submission summary
    print(f"\nSubmission Summary:")
    for problem, mechanisms in submission.items():
        print(f"- {problem}: {len(mechanisms)} mechanisms")

    # Evaluate submission and get final score
    print("\n" + "="*60)
    print("EVALUATING SUBMISSION...")
    print("="*60)

    try:
        scores = evaluate_submission(submission)
        print(f"\nFINAL SCORES:")
        print(f"Raw scores: {scores}")  # Debug print

        for problem, score in scores.items():
            if problem != 'Overall Score':
                try:
                    print(f"- {problem}: {float(score):.4f}")
                except (ValueError, TypeError):
                    print(f"- {problem}: {score}")

        # Handle overall score safely
        if 'Overall Score' in scores:
            try:
                overall_score = float(scores['Overall Score'])
                print(f"\n🎯 OVERALL SCORE: {overall_score:.4f}")
            except (ValueError, TypeError):
                print(f"\n🎯 OVERALL SCORE: {scores['Overall Score']}")
        else:
            print(f"\n🎯 OVERALL SCORE: Not available")

        # Save submission
        np.save('0924_submission.npy', submission)
        print(f"\nSubmission saved to: 0924_submission.npy")

        return submission, scores, all_hypervolumes

    except Exception as e:
        print(f"Error evaluating submission: {str(e)}")
        return submission, None, all_hypervolumes


if __name__ == "__main__":
    submission, scores, hypervolumes = main()

    if scores is not None:
        try:
            overall_score = float(scores.get('Overall Score', 0))
            print(f"\n✅ Optimization complete! Final score: {overall_score:.4f}")
        except (ValueError, TypeError):
            print(f"\n✅ Optimization complete! Final score: {scores.get('Overall Score', 'N/A')}")
    else:
        print(f"\n❌ Optimization completed but scoring failed.")