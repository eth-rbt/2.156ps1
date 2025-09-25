#!/usr/bin/env python3
"""
Single curve optimization script for curve 2.

This script follows the specific workflow:
1. Load and plot curve 2
2. Generate N=200 random mechanisms with specified node count
3. Filter mechanisms based on feasibility
4. Run NSGA-II algorithm
5. Plot results on hypervolume graph
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_environment,
    load_target_curves,
    plot_single_curve,
    filter_feasible_mechanisms,
    run_nsga2_optimization,
    plot_hypervolume_results,
    plot_initial_distribution,
    analyze_initial_population,
    AdvancedMechanismOptimizer
)
from LINKS.Optimization import MechanismRandomizer


def apply_distance_only_gradient_optimization(ga_results, problem, target_curve, optimizer,
                                            step_size=2e-5, n_steps=1000):
    """
    Apply gradient-based optimization that only optimizes for distance (not material).

    Args:
        ga_results: Results from GA optimization
        problem: The optimization problem instance
        target_curve: Target curve for optimization
        optimizer: AdvancedMechanismOptimizer instance
        step_size: Gradient descent step size
        n_steps: Maximum number of optimization steps

    Returns:
        Optimized mechanisms and their performance
    """
    if ga_results.X is None:
        return None, None

    # Extract mechanisms from GA results (similar to regular gradient optimization)
    x0s, edges, fixed_joints, motors, target_idxs = [], [], [], [], []

    if not isinstance(ga_results.X, dict):
        for i in range(ga_results.X.shape[0]):
            x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member = \
                problem.convert_1D_to_mech(ga_results.X[i])
            x0s.append(x0_member)
            edges.append(edges_member)
            fixed_joints.append(fixed_joints_member)
            motors.append(motor_member)
            target_idxs.append(target_idx_member)
    else:
        x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member = \
            problem.convert_1D_to_mech(ga_results.X)
        x0s.append(x0_member)
        edges.append(edges_member)
        fixed_joints.append(fixed_joints_member)
        motors.append(motor_member)
        target_idxs.append(target_idx_member)

    x = x0s.copy()
    done_optimizing = np.zeros(len(x), dtype=bool)
    x_last = x.copy()

    print(f"Starting distance-only gradient optimization with {len(x)} mechanisms...")

    for step in range(n_steps):
        distances, materials, distance_grads, material_grads = optimizer.diff_tools(
            x, edges, fixed_joints, motors, target_curve, target_idxs
        )

        # Only update valid members (feasible solutions)
        valids = np.where(np.logical_and(distances <= 0.75, materials <= 10.0))[0]
        invalids = np.where(~np.logical_and(distances <= 0.75, materials <= 10.0))[0]

        # Revert invalid members and mark as done
        for i in invalids:
            done_optimizing[i] = True
            x[i] = x_last[i]

        x_last = x.copy()

        # Update valid members using ONLY distance gradients (ignore material gradients)
        for i in valids:
            if done_optimizing[i]:
                continue
            # Only use distance gradient, ignore material gradient
            x[i] = x[i] - step_size * distance_grads[i]

        if np.all(done_optimizing):
            print(f'All members done optimizing at step {step}')
            break

        # Print progress every 100 steps
        if step % 100 == 0 and step > 0:
            valid_distances = [distances[i] for i in valids if not done_optimizing[i]]
            if valid_distances:
                print(f'Step {step}: Best distance = {min(valid_distances):.6f}, Active mechanisms = {len(valid_distances)}')

    return x, (x0s, edges, fixed_joints, motors, target_idxs)


def main(num_nodes=7, num_mechanisms=200, pop_size=100, n_gen=100):
    """
    Main workflow for single curve optimization.

    Args:
        num_nodes: Number of nodes in the mechanisms
        num_mechanisms: Number of random mechanisms to generate
        pop_size: Population size for NSGA-II
        n_gen: Number of generations for NSGA-II
    """

    print("=== Single Curve Optimization Workflow ===")
    print(f"Target: Curve 2 (index 1)")
    print(f"Mechanism nodes: {num_nodes}")
    print(f"Random mechanisms to generate: {num_mechanisms}")
    print(f"NSGA-II population size: {pop_size}")
    print(f"NSGA-II generations: {n_gen}")
    print("="*50)

    # Step 1: Setup environment
    print("\n1. Setting up environment...")
    setup_environment()

    # Step 2: Load and plot curve 2
    print("\n2. Loading target curves...")
    target_curves = load_target_curves()
    target_curve = target_curves[1]  # Curve 2 (index 1)

    print("3. Plotting target curve 2...")
    plot_single_curve(target_curve, curve_index=1, title="Target Curve")

    # Step 3: Generate random mechanisms
    print(f"\n4. Generating {num_mechanisms} random mechanisms with {num_nodes} nodes...")
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
            # Skip mechanisms that fail to generate
            continue

    print(f"Successfully generated {len(mechanisms)} mechanisms")

    # Analyze initial population before filtering
    print("\n4.5. Analyzing initial population...")
    filterthreshold = (3, 10.0)
    analyze_initial_population(mechanisms, target_curve)
    plot_initial_distribution(mechanisms, target_curve, ref_point=np.array(filterthreshold))
    # Step 4: Filter mechanisms based on feasibility
    print(f"\n5. Filtering mechanisms for feasibility...")
    print(f"   Feasibility criteria: distance <= {filterthreshold[0]}, material <= {filterthreshold[1]}")

    feasible_mechanisms, performance_metrics = filter_feasible_mechanisms(
        mechanisms,
        target_curve,
        feasibility_threshold=filterthreshold
    )

    print(f"   Feasible mechanisms: {len(feasible_mechanisms)}/{len(mechanisms)} "
          f"({100*len(feasible_mechanisms)/len(mechanisms):.1f}%)")

    if len(feasible_mechanisms) == 0:
        print("   ERROR: No feasible mechanisms found! Try:")
        print("   - Increasing num_mechanisms")
        print("   - Relaxing feasibility constraints")
        print("   - Using different mechanism size")
        return None, None

    # Print some statistics about feasible mechanisms
    if performance_metrics:
        distances = [p[0] for p in performance_metrics]
        materials = [p[1] for p in performance_metrics]
        print(f"   Distance range: {min(distances):.4f} - {max(distances):.4f}")
        print(f"   Material range: {min(materials):.4f} - {max(materials):.4f}")

        # Calculate and print initial hypervolume

        # Plot initial distribution of feasible mechanisms
        print("   Plotting initial feasible mechanism distribution...")
        #plot_initial_distribution(performance_metrics, target_curve, ref_point=np.array([6, 10.0]))

    # Step 5: Run NSGA-II optimization
    print(f"\n6. Running NSGA-II optimization...")
    print(f"   Population size: {pop_size}")
    print(f"   Generations: {n_gen}")

    results, problem = run_nsga2_optimization(
        feasible_mechanisms,
        target_curve,
        N=num_nodes,
        pop_size=min(pop_size, len(feasible_mechanisms)),
        n_gen=n_gen,
        verbose=True
    )

    # Step 6: Plot hypervolume results
    print(f"\n7. Plotting optimization results...")

    if results.X is not None:
        hypervolume = plot_hypervolume_results(
            results,
            ref_point=np.array([0.75, 10.0]),
            title=f"NSGA-II Results for Curve 2 ({num_nodes} nodes)"
        )

        print(f"\nOptimization Summary:")
        print(f"- Solutions found: {len(results.F)}")
        print(f"- Hypervolume: {hypervolume:.4f}")
        print(f"- Best distance: {min(results.F[:, 0]):.4f}")
        print(f"- Best material: {min(results.F[:, 1]):.4f}")

        # Step 7: Apply gradient optimization
        print(f"\n8. Applying gradient optimization...")
        optimizer = AdvancedMechanismOptimizer()

        grad_results, original_mechs = optimizer.apply_gradient_optimization(
            results, problem, target_curve,
            step_size=4e-4, n_steps=1000
        )

        if grad_results is not None and original_mechs is not None:
            # Evaluate hypervolume for gradient-optimized results
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

            # Plot comparison
            plt.figure(figsize=(15, 5))

            # Original results
            plt.subplot(1, 3, 1)
            plt.scatter(results.F[:, 0], results.F[:, 1], alpha=0.7, s=30, c='blue', label='GA Results')
            plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Distance')
            plt.ylabel('Material')
            plt.title(f'GA Results\nHV: {hypervolume:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Gradient results
            plt.subplot(1, 3, 2)
            plt.scatter(grad_F[:, 0], grad_F[:, 1], alpha=0.7, s=30, c='green', label='GA + Gradient')
            plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Distance')
            plt.ylabel('Material')
            plt.title(f'GA + Gradient Results\nHV: {grad_hypervolume:.4f}')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Combined comparison
            plt.subplot(1, 3, 3)
            plt.scatter(results.F[:, 0], results.F[:, 1], alpha=0.5, s=20, c='blue', label='GA Only')
            plt.scatter(grad_F[:, 0], grad_F[:, 1], alpha=0.7, s=30, c='green', label='GA + Gradient')
            plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
            plt.xlabel('Distance')
            plt.ylabel('Material')
            plt.title('Comparison')
            plt.legend()
            plt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.show()

            # Step 8: Perturbation around best solutions
            print(f"\n9. Generating perturbed mechanisms around best solutions...")

            # Get the best solutions from gradient-optimized results
            best_indices = np.argsort(grad_F[:, 0])[:20]  # Top 20 by distance
            best_x0s = [combined_x0s[i] for i in best_indices]
            best_edges = [combined_edges[i] for i in best_indices]
            best_fixed_joints = [combined_fixed_joints[i] for i in best_indices]
            best_motors = [combined_motors[i] for i in best_indices]
            best_target_idxs = [combined_target_idxs[i] for i in best_indices]

            # Generate perturbed mechanisms
            perturbed_mechanisms = []
            perturbation_scale = 0.05  # 5% perturbation
            num_perturbations_per_best = 10  # Generate 10 variants per best solution

            for idx in range(len(best_x0s)):
                base_x0 = best_x0s[idx]
                base_edges = best_edges[idx]
                base_fixed_joints = best_fixed_joints[idx]
                base_motor = best_motors[idx]
                base_target_idx = best_target_idxs[idx]

                for _ in range(num_perturbations_per_best):
                    # Perturb positions with small random noise
                    noise = np.random.normal(0, perturbation_scale, base_x0.shape)
                    perturbed_x0 = base_x0 + noise

                    # Ensure positions stay within reasonable bounds [0, 1]
                    perturbed_x0 = np.clip(perturbed_x0, 0.0, 1.0)

                    # Create mechanism dictionary
                    perturbed_mech = {
                        'x0': perturbed_x0,
                        'edges': base_edges,
                        'fixed_joints': base_fixed_joints,
                        'motor': base_motor,
                        'target_joint': base_target_idx
                    }
                    perturbed_mechanisms.append(perturbed_mech)

            print(f"Generated {len(perturbed_mechanisms)} perturbed mechanisms")

            # Step 9: Filter perturbed mechanisms for feasibility
            print(f"\n10. Filtering perturbed mechanisms...")
            feasible_perturbed, perturbed_metrics = filter_feasible_mechanisms(
                perturbed_mechanisms,
                target_curve,
                feasibility_threshold=(0.75, 10.0)  # Use final scoring thresholds
            )

            print(f"Feasible perturbed mechanisms: {len(feasible_perturbed)}/{len(perturbed_mechanisms)} "
                  f"({100*len(feasible_perturbed)/len(perturbed_mechanisms):.1f}%)")

            if len(feasible_perturbed) > 0:
                # Step 10: Run second round of NSGA-II on perturbed mechanisms
                print(f"\n11. Running second NSGA-II on perturbed mechanisms...")

                second_results, second_problem = run_nsga2_optimization(
                    feasible_perturbed,
                    target_curve,
                    N=num_nodes,
                    pop_size=min(pop_size, len(feasible_perturbed)),
                    n_gen=n_gen // 2,  # Use fewer generations for the second round
                    verbose=True
                )

                if second_results.X is not None:
                    # Calculate hypervolume for second GA results
                    second_hypervolume = plot_hypervolume_results(
                        second_results,
                        ref_point=np.array([0.75, 10.0]),
                        title=f"Second NSGA-II Results (Perturbed) for Curve 2 ({num_nodes} nodes)"
                    )

                    print(f"\nSecond GA Results:")
                    print(f"- Solutions found: {len(second_results.F)}")
                    print(f"- Hypervolume: {second_hypervolume:.4f}")
                    print(f"- Best distance: {min(second_results.F[:, 0]):.4f}")
                    print(f"- Best material: {min(second_results.F[:, 1]):.4f}")

                    # Step 11: Apply distance-only gradient optimization to perturbed GA results
                    print(f"\n12. Applying distance-only gradient optimization to perturbed GA results...")

                    # Custom distance-only gradient optimization
                    second_grad_results, second_original_mechs = apply_distance_only_gradient_optimization(
                        second_results, second_problem, target_curve, optimizer,
                        step_size=2e-5, n_steps=1000
                    )

                    if second_grad_results is not None and second_original_mechs is not None:
                        # Evaluate second gradient optimization
                        second_combined_x0s = second_original_mechs[0] + second_grad_results
                        second_combined_edges = second_original_mechs[1] + second_original_mechs[1]
                        second_combined_fixed_joints = second_original_mechs[2] + second_original_mechs[2]
                        second_combined_motors = second_original_mechs[3] + second_original_mechs[3]
                        second_combined_target_idxs = second_original_mechs[4] + second_original_mechs[4]

                        second_combined_mechs = (second_combined_x0s, second_combined_edges, second_combined_fixed_joints,
                                               second_combined_motors, second_combined_target_idxs)

                        second_grad_hypervolume, second_grad_F = optimizer.evaluate_hypervolume(
                            second_combined_mechs, target_curve, ref_point=np.array([0.75, 10.0])
                        )

                        print(f"\nSecond Gradient Optimization Results:")
                        print(f"- Second GA hypervolume: {second_hypervolume:.4f}")
                        print(f"- Second gradient-enhanced hypervolume: {second_grad_hypervolume:.4f}")
                        print(f"- Improvement: {second_grad_hypervolume - second_hypervolume:.4f}")

                        # Step 12: Generate mechanisms for third GA from distance-optimized results
                        print(f"\n13. Generating mechanisms for third GA from distance-optimized results...")

                        # Get best distance-optimized solutions
                        best_distance_indices = np.argsort(second_grad_F[:, 0])[:15]  # Top 15 by distance
                        best_distance_x0s = [second_combined_x0s[i] for i in best_distance_indices]
                        best_distance_edges = [second_combined_edges[i] for i in best_distance_indices]
                        best_distance_fixed_joints = [second_combined_fixed_joints[i] for i in best_distance_indices]
                        best_distance_motors = [second_combined_motors[i] for i in best_distance_indices]
                        best_distance_target_idxs = [second_combined_target_idxs[i] for i in best_distance_indices]

                        # Generate small perturbations around distance-optimized solutions
                        third_ga_mechanisms = []
                        third_perturbation_scale = 0.02  # Smaller perturbation for refined solutions
                        num_third_perturbations = 15  # Generate 15 variants per best solution

                        for idx in range(len(best_distance_x0s)):
                            base_x0 = best_distance_x0s[idx]
                            base_edges = best_distance_edges[idx]
                            base_fixed_joints = best_distance_fixed_joints[idx]
                            base_motor = best_distance_motors[idx]
                            base_target_idx = best_distance_target_idxs[idx]

                            for _ in range(num_third_perturbations):
                                # Small perturbation around distance-optimized solutions
                                noise = np.random.normal(0, third_perturbation_scale, base_x0.shape)
                                perturbed_x0 = base_x0 + noise
                                perturbed_x0 = np.clip(perturbed_x0, 0.0, 1.0)

                                third_mech = {
                                    'x0': perturbed_x0,
                                    'edges': base_edges,
                                    'fixed_joints': base_fixed_joints,
                                    'motor': base_motor,
                                    'target_joint': base_target_idx
                                }
                                third_ga_mechanisms.append(third_mech)

                        print(f"Generated {len(third_ga_mechanisms)} mechanisms for third GA")

                        # Filter mechanisms for third GA
                        print(f"\n14. Filtering mechanisms for third GA...")
                        feasible_third_ga, third_ga_metrics = filter_feasible_mechanisms(
                            third_ga_mechanisms,
                            target_curve,
                            feasibility_threshold=(0.75, 10.0)  # Use final scoring thresholds
                        )

                        print(f"Feasible third GA mechanisms: {len(feasible_third_ga)}/{len(third_ga_mechanisms)} "
                              f"({100*len(feasible_third_ga)/len(third_ga_mechanisms):.1f}%)")

                        if len(feasible_third_ga) > 0:
                            # Step 13: Run third GA
                            print(f"\n15. Running third NSGA-II on distance-optimized mechanisms...")

                            third_results, third_problem = run_nsga2_optimization(
                                feasible_third_ga,
                                target_curve,
                                N=num_nodes,
                                pop_size=min(pop_size, len(feasible_third_ga)),
                                n_gen=n_gen // 3,  # Use even fewer generations for the third round
                                verbose=True
                            )

                            if third_results.X is not None:
                                # Calculate hypervolume for third GA results
                                third_hypervolume = plot_hypervolume_results(
                                    third_results,
                                    ref_point=np.array([0.75, 10.0]),
                                    title=f"Third NSGA-II Results (Distance-Optimized) for Curve 2 ({num_nodes} nodes)"
                                )

                                print(f"\nThird GA Results:")
                                print(f"- Solutions found: {len(third_results.F)}")
                                print(f"- Hypervolume: {third_hypervolume:.4f}")
                                print(f"- Best distance: {min(third_results.F[:, 0]):.4f}")
                                print(f"- Best material: {min(third_results.F[:, 1]):.4f}")

                                third_ga_F = third_results.F
                            else:
                                print("Third NSGA-II failed to find solutions!")
                                third_ga_F = None
                        else:
                            print("No feasible mechanisms for third GA!")
                            third_ga_F = None
                    else:
                        print("Second gradient optimization failed to produce results.")
                        second_grad_F = None
                        third_ga_F = None

                    # Step 15: Combine ALL solutions for final hypervolume calculation
                    print(f"\n16. Calculating final hypervolume from ALL processes...")

                    # Collect all F values from all processes
                    all_F_arrays = [results.F]  # Original GA
                    process_names = ['Original GA']

                    if grad_F is not None:
                        all_F_arrays.append(grad_F)  # First gradient
                        process_names.append('First Gradient')

                    all_F_arrays.append(second_results.F)  # Second GA
                    process_names.append('Second GA')

                    if second_grad_F is not None:
                        all_F_arrays.append(second_grad_F)  # Distance-only gradient
                        process_names.append('Distance-only Gradient')

                    if third_ga_F is not None:
                        all_F_arrays.append(third_ga_F)  # Third GA
                        process_names.append('Third GA')

                    # Combine all solutions
                    combined_all_F = np.vstack(all_F_arrays)

                    # Filter for feasible solutions (distance < 0.75, material < 10.0)
                    feasible_mask = np.logical_and(combined_all_F[:, 0] < 0.75, combined_all_F[:, 1] < 10.0)
                    feasible_combined_F = combined_all_F[feasible_mask]

                    if len(feasible_combined_F) > 0:
                        # Calculate final hypervolume using all feasible solutions
                        from pymoo.indicators.hv import HV
                        ind = HV(np.array([0.75, 10.0]))
                        final_hypervolume = ind(feasible_combined_F)

                        print(f"Final Combined Results:")
                        print(f"- Total solutions from all processes: {len(combined_all_F)}")
                        print(f"- Feasible solutions: {len(feasible_combined_F)}")
                        print(f"- FINAL HYPERVOLUME (ALL PROCESSES): {final_hypervolume:.4f}")
                    else:
                        final_hypervolume = 0.0
                        print("No feasible solutions found across all processes!")

                    # Enhanced visualization showing all stages
                    # Calculate number of plots needed
                    num_stages = 4  # Original GA, Second GA, All methods combined, Final Feasible (always present)
                    if grad_F is not None:
                        num_stages += 1
                    if second_grad_F is not None:
                        num_stages += 1
                    if third_ga_F is not None:
                        num_stages += 1

                    fig_width = max(20, num_stages * 4)  # Dynamic width based on stages
                    plt.figure(figsize=(fig_width, 6))  # Slightly taller for better readability

                    plot_idx = 1

                    # Original GA results
                    plt.subplot(1, num_stages, plot_idx)
                    plt.scatter(results.F[:, 0], results.F[:, 1], alpha=0.7, s=30, c='blue', label='Original GA')
                    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel('Distance')
                    plt.ylabel('Material')
                    plt.title(f'1. Original GA\nHV: {hypervolume:.4f}')
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)
                    plot_idx += 1

                    # First Gradient results
                    if grad_F is not None:
                        plt.subplot(1, num_stages, plot_idx)
                        plt.scatter(grad_F[:, 0], grad_F[:, 1], alpha=0.7, s=30, c='green', label='First Gradient')
                        plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                        plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                        plt.xlabel('Distance')
                        plt.ylabel('Material')
                        plt.title(f'2. First Gradient\nHV: {grad_hypervolume:.4f}')
                        plt.legend(fontsize=8)
                        plt.grid(True, alpha=0.3)
                        plot_idx += 1

                    # Second GA results
                    plt.subplot(1, num_stages, plot_idx)
                    plt.scatter(second_results.F[:, 0], second_results.F[:, 1], alpha=0.7, s=30, c='purple', label='Second GA')
                    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel('Distance')
                    plt.ylabel('Material')
                    plt.title(f'3. Second GA\nHV: {second_hypervolume:.4f}')
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)
                    plot_idx += 1

                    # Distance-only Gradient results (if available)
                    if second_grad_F is not None:
                        plt.subplot(1, num_stages, plot_idx)
                        plt.scatter(second_grad_F[:, 0], second_grad_F[:, 1], alpha=0.7, s=30, c='orange', label='Distance Gradient')
                        plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                        plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                        plt.xlabel('Distance')
                        plt.ylabel('Material')
                        plt.title(f'4. Distance Gradient\nHV: {second_grad_hypervolume:.4f}')
                        plt.legend(fontsize=8)
                        plt.grid(True, alpha=0.3)
                        plot_idx += 1

                    # Third GA results (if available)
                    if third_ga_F is not None:
                        plt.subplot(1, num_stages, plot_idx)
                        plt.scatter(third_ga_F[:, 0], third_ga_F[:, 1], alpha=0.7, s=30, c='cyan', label='Third GA')
                        plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                        plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                        plt.xlabel('Distance')
                        plt.ylabel('Material')
                        plt.title(f'5. Third GA\nHV: {third_hypervolume:.4f}')
                        plt.legend(fontsize=8)
                        plt.grid(True, alpha=0.3)
                        plot_idx += 1

                    # All methods combined
                    plt.subplot(1, num_stages, plot_idx)
                    colors = ['blue', 'green', 'purple', 'orange', 'cyan']
                    labels = ['Original GA', 'First Gradient', 'Second GA', 'Distance Gradient', 'Third GA']
                    alphas = [0.3, 0.4, 0.5, 0.7, 0.8]
                    sizes = [8, 10, 12, 15, 18]

                    plot_data = [(results.F, 0), (grad_F, 1), (second_results.F, 2), (second_grad_F, 3), (third_ga_F, 4)]

                    for F_data, idx in plot_data:
                        if F_data is not None:
                            plt.scatter(F_data[:, 0], F_data[:, 1],
                                      alpha=alphas[idx], s=sizes[idx], c=colors[idx], label=labels[idx])

                    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel('Distance')
                    plt.ylabel('Material')
                    plt.title('All Methods Combined')
                    plt.legend(fontsize=7)
                    plt.grid(True, alpha=0.3)
                    plot_idx += 1

                    # Final feasible solutions only
                    plt.subplot(1, num_stages, plot_idx)
                    if len(feasible_combined_F) > 0:
                        plt.scatter(feasible_combined_F[:, 0], feasible_combined_F[:, 1], alpha=0.7, s=30, c='red', label='All Feasible')
                    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
                    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
                    plt.xlabel('Distance')
                    plt.ylabel('Material')
                    plt.title(f'Final Feasible\nHV: {final_hypervolume:.4f}')
                    plt.legend(fontsize=8)
                    plt.grid(True, alpha=0.3)

                    plt.tight_layout()
                    plt.show()

                    print(f"\nComprehensive 6-Stage Optimization Results:")
                    print(f"- Stage 1 - Original GA hypervolume: {hypervolume:.4f}")
                    if grad_F is not None:
                        print(f"- Stage 2 - First gradient hypervolume: {grad_hypervolume:.4f}")
                    print(f"- Stage 3 - Second GA (perturbed) hypervolume: {second_hypervolume:.4f}")
                    if second_grad_F is not None:
                        print(f"- Stage 4 - Distance-only gradient hypervolume: {second_grad_hypervolume:.4f}")
                    if third_ga_F is not None:
                        print(f"- Stage 5 - Third GA (distance-refined) hypervolume: {third_hypervolume:.4f}")
                    print(f"- Stage 6 - FINAL COMBINED HYPERVOLUME: {final_hypervolume:.4f}")

                    # Calculate improvements
                    total_improvement = final_hypervolume - hypervolume
                    print(f"\nImprovement Analysis:")
                    print(f"- Total improvement from original: {total_improvement:.4f}")

                    # Calculate individual stage improvements
                    stage_hvs = [hypervolume]
                    if grad_F is not None:
                        stage_hvs.append(grad_hypervolume)
                    stage_hvs.append(second_hypervolume)
                    if second_grad_F is not None:
                        stage_hvs.append(second_grad_hypervolume)
                    if third_ga_F is not None:
                        stage_hvs.append(third_hypervolume)

                    best_single_hv = max(stage_hvs)
                    print(f"- Best individual stage HV: {best_single_hv:.4f}")
                    print(f"- Combined approach advantage: {final_hypervolume - best_single_hv:.4f}")

                    # Summary statistics
                    print(f"\nFinal Summary:")
                    print(f"- Total optimization stages: {len(process_names)}")
                    print(f"- Total solutions evaluated: {len(combined_all_F)}")
                    print(f"- Feasible solutions: {len(feasible_combined_F)}")
                    print(f"- Final hypervolume improvement: {((final_hypervolume/hypervolume) - 1)*100:.2f}%")

                    # Return results - we'll return the original results but the final HV represents all processes
                    print(f"\n✅ 6-Stage optimization complete! Final HV incorporates ALL {len(process_names)} processes!")
                    return results, problem
                else:
                    print("Second NSGA-II failed to find solutions!")
            else:
                print("No feasible perturbed mechanisms found!")
        else:
            print("Gradient optimization failed to produce results.")

        return results, problem
    else:
        print("No solutions found in NSGA-II optimization!")
        return None, None

    




if __name__ == "__main__":
    # Run with default parameters
    results, problem = main(
        num_nodes=7,
        num_mechanisms=4000,
        pop_size=400,
        n_gen=200
    )

    if results is not None:
        print("\n=== Optimization completed successfully! ===")
    else:
        print("\n=== Optimization failed ===")