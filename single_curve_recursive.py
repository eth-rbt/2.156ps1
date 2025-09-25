#!/usr/bin/env python3
"""
Recursive Single Curve Optimization Script

This script implements a recursive optimization approach with the following steps:
1. Sample 4000 mechanisms, filter at (3,12), plot initial distribution at (0.75,10)
2. For each iteration (5 total):
   (1) Run GA on current cluster
   (2) Apply material-only gradient descent on frontier
   (3) Apply distance-only gradient descent on original frontier
   (4) Combine all 3 groups, plot and calculate HV
   (5) Use combined cluster for next iteration
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_environment,
    load_target_curves,
    filter_feasible_mechanisms,
    run_nsga2_optimization,
    analyze_initial_population,
    plot_initial_distribution,
    AdvancedMechanismOptimizer
)
from LINKS.Optimization import MechanismRandomizer
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


def extract_pareto_frontier(all_mechanisms, target_curve):
    """
    Extract the Pareto frontier mechanisms from a combined set.

    Args:
        all_mechanisms: List of mechanism dictionaries
        target_curve: Target curve for evaluation

    Returns:
        List of Pareto frontier mechanisms
    """
    if len(all_mechanisms) == 0:
        return []

    # Evaluate all mechanisms
    optimizer = AdvancedMechanismOptimizer()

    # Convert mechanisms to evaluation format
    x0s, edges, fixed_joints, motors, target_idxs = [], [], [], [], []
    for mech in all_mechanisms:
        x0s.append(mech['x0'])
        edges.append(mech['edges'])
        fixed_joints.append(mech['fixed_joints'])
        motors.append(mech['motor'])
        target_idxs.append(mech.get('target_joint', mech['x0'].shape[0] - 1))

    try:
        # Evaluate all mechanisms
        distances, materials = optimizer.tools(x0s, edges, fixed_joints, motors, target_curve, target_idxs)
        F = np.column_stack([distances, materials])

        # Filter for feasible solutions
        feasible_mask = np.logical_and(F[:, 0] <= 0.75, F[:, 1] <= 10.0)
        if not np.any(feasible_mask):
            return all_mechanisms[:min(50, len(all_mechanisms))]  # Return first 50 if none feasible

        feasible_indices = np.where(feasible_mask)[0]
        feasible_F = F[feasible_indices]

        # Apply non-dominated sorting to get Pareto frontier
        nds = NonDominatedSorting()
        fronts = nds.do(feasible_F)

        # Get the first (best) front
        if len(fronts) > 0:
            pareto_indices = fronts[0]
            global_pareto_indices = feasible_indices[pareto_indices]
            pareto_mechanisms = [all_mechanisms[i] for i in global_pareto_indices]

            print(f"Extracted {len(pareto_mechanisms)} Pareto frontier mechanisms from {len(all_mechanisms)} total")
            return pareto_mechanisms
        else:
            return all_mechanisms[:min(50, len(all_mechanisms))]

    except Exception as e:
        print(f"Error in Pareto frontier extraction: {e}")
        return all_mechanisms[:min(50, len(all_mechanisms))]


def apply_material_only_gradient_optimization(ga_results, problem, target_curve, optimizer,
                                            step_size=2e-5, n_steps=500):
    """
    Apply gradient-based optimization that only optimizes for material (not distance).
    """
    if ga_results.X is None:
        return None, None

    # Extract mechanisms from GA results
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

        # Update valid members using ONLY material gradients (ignore distance gradients)
        for i in valids:
            if done_optimizing[i]:
                continue
            # Only use material gradient, ignore distance gradient
            x[i] = x[i] - step_size * material_grads[i]

        if np.all(done_optimizing):
            break

    return x, (x0s, edges, fixed_joints, motors, target_idxs)


def apply_distance_only_gradient_optimization(ga_results, problem, target_curve, optimizer,
                                            step_size=2e-5, n_steps=500):
    """
    Apply gradient-based optimization that only optimizes for distance (not material).
    """
    if ga_results.X is None:
        return None, None

    # Extract mechanisms from GA results
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
            break

    return x, (x0s, edges, fixed_joints, motors, target_idxs)


def mechanisms_from_results(results, problem):
    """Convert optimization results back to mechanism format."""
    mechanisms = []

    if results.X is None:
        return mechanisms

    if not isinstance(results.X, dict):
        for i in range(results.X.shape[0]):
            x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member = \
                problem.convert_1D_to_mech(results.X[i])

            mech = {
                'x0': x0_member,
                'edges': edges_member,
                'fixed_joints': fixed_joints_member,
                'motor': motor_member,
                'target_joint': target_idx_member
            }
            mechanisms.append(mech)
    else:
        x0_member, edges_member, fixed_joints_member, motor_member, target_idx_member = \
            problem.convert_1D_to_mech(results.X)

        mech = {
            'x0': x0_member,
            'edges': edges_member,
            'fixed_joints': fixed_joints_member,
            'motor': motor_member,
            'target_joint': target_idx_member
        }
        mechanisms.append(mech)

    return mechanisms


def mechanisms_from_gradient_results(grad_results, original_mechs):
    """Convert gradient optimization results back to mechanism format."""
    mechanisms = []

    if grad_results is None or original_mechs is None:
        return mechanisms

    x0s_grad, edges, fixed_joints, motors, target_idxs = original_mechs

    for i in range(len(grad_results)):
        mech = {
            'x0': grad_results[i],
            'edges': edges[i],
            'fixed_joints': fixed_joints[i],
            'motor': motors[i],
            'target_joint': target_idxs[i]
        }
        mechanisms.append(mech)

    return mechanisms


def recursive_optimization(target_curve, num_mechanisms=4000, num_iterations=5,
                         num_nodes=7, pop_size=200, n_gen=100,
                         step_sizes_material=[2e-5, 1e-5, 5e-6, 2e-6, 1e-6],
                         step_sizes_distance=[2e-5, 1e-5, 5e-6, 2e-6, 1e-6]):
    """
    Run recursive optimization with specified parameters.

    Args:
        target_curve: Target curve to optimize for
        num_mechanisms: Initial number of mechanisms to generate
        num_iterations: Number of recursive iterations
        num_nodes: Number of nodes in mechanisms
        pop_size: Population size for GA
        n_gen: Number of generations for GA
        step_sizes_material: Array of step sizes for material gradient descent
        step_sizes_distance: Array of step sizes for distance gradient descent

    Returns:
        List of hypervolumes for each iteration, final mechanisms
    """

    print("=== Recursive Single Curve Optimization ===")
    print(f"Target: Curve 2 (index 1)")
    print(f"Initial mechanisms: {num_mechanisms}")
    print(f"Iterations: {num_iterations}")
    print(f"Material step sizes: {step_sizes_material}")
    print(f"Distance step sizes: {step_sizes_distance}")
    print("="*50)

    # Step 1: Initial sampling and filtering
    print(f"\n1. Generating {num_mechanisms} initial mechanisms...")

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
    print("\n2. Analyzing initial population...")
    analyze_initial_population(mechanisms, target_curve)

    # Filter at (3, 12)
    print("\n3. Filtering mechanisms at (3, 12)...")
    feasible_mechanisms, performance_metrics = filter_feasible_mechanisms(
        mechanisms,
        target_curve,
        feasibility_threshold=(3, 12.0)
    )

    print(f"Feasible mechanisms: {len(feasible_mechanisms)}/{len(mechanisms)} "
          f"({100*len(feasible_mechanisms)/len(mechanisms):.1f}%)")

    # Plot initial distribution at (0.75, 10)
    print("\n4. Plotting initial distribution at (0.75, 10)...")
    plot_initial_distribution(feasible_mechanisms, target_curve,
                            ref_point=np.array([0.75, 10.0]))

    if len(feasible_mechanisms) == 0:
        print("ERROR: No feasible mechanisms found!")
        return [], []

    # Initialize for recursive optimization
    current_mechanisms = feasible_mechanisms
    hypervolumes = []
    all_iteration_results = []
    optimizer = AdvancedMechanismOptimizer()
    hv_indicator = HV(np.array([0.75, 10.0]))

    # Track all mechanisms from all previous iterations
    all_previous_mechanisms = []

    # Recursive optimization iterations
    for iteration in range(num_iterations):
        print(f"\n{'='*60}")
        print(f"ITERATION {iteration + 1}/{num_iterations}")
        print(f"{'='*60}")

        # Step (-1): From iteration 2 onwards, use Pareto frontier of all previous mechanisms
        if iteration >= 1:  # Starting from iteration 2 (iteration index 1)
            print(f"\n(-1) Extracting Pareto frontier from all previous iterations...")
            print(f"Total accumulated mechanisms from previous iterations: {len(all_previous_mechanisms)}")

            if len(all_previous_mechanisms) > 0:
                pareto_mechanisms = extract_pareto_frontier(all_previous_mechanisms, target_curve)
                print(f"Using {len(pareto_mechanisms)} Pareto frontier mechanisms as starting point")
                current_mechanisms = pareto_mechanisms
            else:
                print("No previous mechanisms available, using current mechanisms")

        print(f"Starting with {len(current_mechanisms)} mechanisms")

        # Step (0): Apply perturbation to current mechanisms before GA
        print(f"\n(0) Applying perturbation to current mechanisms...")

        # Generate perturbations around current mechanisms
        perturbed_mechanisms = []
        perturbation_scale = max(0.01, 0.1 - iteration * 0.015)  # Decreasing perturbation over iterations
        num_perturbations_per_mech = 5  # Generate 5 variants per mechanism

        for mech in current_mechanisms:
            base_x0 = mech['x0']
            base_edges = mech['edges']
            base_fixed_joints = mech['fixed_joints']
            base_motor = mech['motor']
            base_target_idx = mech.get('target_joint', mech['x0'].shape[0] - 1)

            # Add the original mechanism
            perturbed_mechanisms.append(mech)

            # Generate perturbed variants
            for _ in range(num_perturbations_per_mech):
                # Perturb positions with small random noise
                noise = np.random.normal(0, perturbation_scale, base_x0.shape)
                perturbed_x0 = base_x0 + noise

                # Ensure positions stay within reasonable bounds [0, 1]
                perturbed_x0 = np.clip(perturbed_x0, 0.0, 1.0)

                # Create perturbed mechanism dictionary
                perturbed_mech = {
                    'x0': perturbed_x0,
                    'edges': base_edges,
                    'fixed_joints': base_fixed_joints,
                    'motor': base_motor,
                    'target_joint': base_target_idx
                }
                perturbed_mechanisms.append(perturbed_mech)

        print(f"Generated {len(perturbed_mechanisms)} perturbed mechanisms (scale: {perturbation_scale:.3f})")

        # Filter perturbed mechanisms for feasibility
        feasible_perturbed, _ = filter_feasible_mechanisms(
            perturbed_mechanisms,
            target_curve,
            feasibility_threshold=(0.75, 10.0)
        )

        print(f"Feasible perturbed mechanisms: {len(feasible_perturbed)}/{len(perturbed_mechanisms)} "
              f"({100*len(feasible_perturbed)/len(perturbed_mechanisms):.1f}%)")

        # Use perturbed mechanisms for GA if we have enough, otherwise fall back to original
        ga_input_mechanisms = feasible_perturbed if len(feasible_perturbed) >= 10 else current_mechanisms

        # Step (1): Run GA on perturbed cluster
        print(f"\n(1) Running GA on perturbed cluster ({len(ga_input_mechanisms)} mechanisms)...")
        ga_results, ga_problem = run_nsga2_optimization(
            ga_input_mechanisms,
            target_curve,
            N=num_nodes,
            pop_size=min(pop_size, len(ga_input_mechanisms)),
            n_gen=n_gen,
            verbose=False
        )

        if ga_results.X is None:
            print(f"GA failed in iteration {iteration + 1}")
            break

        ga_mechanisms = mechanisms_from_results(ga_results, ga_problem)

        # Step (2): Material-only gradient descent on frontier
        print(f"\n(2) Applying material-only gradient descent...")
        material_grad_results, material_original_mechs = apply_material_only_gradient_optimization(
            ga_results, ga_problem, target_curve, optimizer,
            step_size=step_sizes_material[iteration], n_steps=500
        )

        material_mechanisms = mechanisms_from_gradient_results(material_grad_results, material_original_mechs)

        # Step (3): Distance-only gradient descent on original frontier
        print(f"\n(3) Applying distance-only gradient descent...")
        distance_grad_results, distance_original_mechs = apply_distance_only_gradient_optimization(
            ga_results, ga_problem, target_curve, optimizer,
            step_size=step_sizes_distance[iteration], n_steps=500
        )

        distance_mechanisms = mechanisms_from_gradient_results(distance_grad_results, distance_original_mechs)

        # Step (4): Combine all 3 groups
        print(f"\n(4) Combining all 3 groups...")
        combined_mechanisms = ga_mechanisms + material_mechanisms + distance_mechanisms

        # Remove duplicates and invalid mechanisms
        valid_combined = []
        for mech in combined_mechanisms:
            if mech is not None and 'x0' in mech:
                valid_combined.append(mech)

        # Filter combined mechanisms for feasibility at (0.75, 10.0)
        final_feasible, final_metrics = filter_feasible_mechanisms(
            valid_combined,
            target_curve,
            feasibility_threshold=(0.75, 10.0)
        )

        if len(final_feasible) == 0:
            print(f"No feasible mechanisms in iteration {iteration + 1}")
            break

        # Calculate hypervolume
        distances = [m[0] for m in final_metrics]
        materials = [m[1] for m in final_metrics]
        F = np.column_stack([distances, materials])

        hypervolume = hv_indicator(F)
        hypervolumes.append(hypervolume)

        # Store results for plotting
        all_iteration_results.append({
            'F': F,
            'hypervolume': hypervolume,
            'num_solutions': len(F)
        })

        print(f"\nIteration {iteration + 1} Results:")
        print(f"- GA solutions: {len(ga_mechanisms)}")
        print(f"- Material gradient solutions: {len(material_mechanisms)}")
        print(f"- Distance gradient solutions: {len(distance_mechanisms)}")
        print(f"- Combined valid solutions: {len(valid_combined)}")
        print(f"- Final feasible solutions: {len(final_feasible)}")
        print(f"- Hypervolume: {hypervolume:.4f}")
        print(f"- Best distance: {min(distances):.4f}")
        print(f"- Best material: {min(materials):.4f}")

        # Plot current iteration
        plt.figure(figsize=(8, 6))
        plt.scatter(distances, materials, alpha=0.7, s=30, c='blue')
        plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
        plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Distance')
        plt.ylabel('Material')
        plt.title(f'Iteration {iteration + 1} - HV: {hypervolume:.4f}')
        plt.grid(True, alpha=0.3)
        plt.show()

        # Accumulate all mechanisms from current iteration for future Pareto frontier extraction
        current_iteration_mechanisms = ga_mechanisms + material_mechanisms + distance_mechanisms
        all_previous_mechanisms.extend(current_iteration_mechanisms)

        print(f"Accumulated {len(current_iteration_mechanisms)} mechanisms from iteration {iteration + 1}")
        print(f"Total accumulated mechanisms: {len(all_previous_mechanisms)}")

        # Prepare for next iteration
        current_mechanisms = final_feasible

        print(f"Prepared {len(current_mechanisms)} mechanisms for next iteration")

    return hypervolumes, all_iteration_results


def main():
    """Main function for recursive optimization."""

    # Setup environment
    setup_environment()

    # Load target curves
    target_curves = load_target_curves()
    target_curve = target_curves[1]  # Curve 2 (index 1)

    # Define step sizes for each iteration (adjustable)
    step_sizes_material = [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]
    step_sizes_distance = [2e-4, 1e-4, 5e-5, 2e-5, 1e-5]

    # Run recursive optimization
    hypervolumes, iteration_results = recursive_optimization(
        target_curve,
        num_mechanisms=4000,
        num_iterations=5,
        num_nodes=7,
        pop_size=200,
        n_gen=100,
        step_sizes_material=step_sizes_material,
        step_sizes_distance=step_sizes_distance
    )

    # Final visualization of all 5 iterations
    if len(iteration_results) > 0:
        print(f"\n{'='*60}")
        print("FINAL RESULTS - ALL ITERATIONS")
        print(f"{'='*60}")

        # Create two plots: individual subplots and combined plot

        # Plot 1: Individual subplots for each iteration
        fig1, axes = plt.subplots(1, len(iteration_results), figsize=(4*len(iteration_results), 5))
        if len(iteration_results) == 1:
            axes = [axes]

        for i, result in enumerate(iteration_results):
            ax = axes[i]
            F = result['F']
            hv = result['hypervolume']

            ax.scatter(F[:, 0], F[:, 1], alpha=0.7, s=30, c='blue')
            ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Distance')
            ax.set_ylabel('Material')
            ax.set_title(f'Iter {i+1}\nHV: {hv:.4f}')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # Plot 2: All 5 clusters in different colors on single plot
        plt.figure(figsize=(10, 8))

        # Define colors for each iteration
        colors = ['blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta']
        alphas = [0.4, 0.5, 0.6, 0.7, 0.8]  # Increasing alpha for later iterations
        sizes = [20, 25, 30, 35, 40]  # Increasing size for later iterations

        for i, result in enumerate(iteration_results):
            F = result['F']
            hv = result['hypervolume']

            plt.scatter(F[:, 0], F[:, 1],
                       alpha=alphas[min(i, len(alphas)-1)],
                       s=sizes[min(i, len(sizes)-1)],
                       c=colors[i],
                       label=f'Iter {i+1} (HV: {hv:.4f})',
                       edgecolors='black',
                       linewidth=0.5)

        # Add constraint lines
        plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Distance constraint')
        plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Material constraint')

        plt.xlabel('Distance', fontsize=12)
        plt.ylabel('Material', fontsize=12)
        plt.title('All Iterations Combined - Recursive Optimization Evolution', fontsize=14, fontweight='bold')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)

        # Set axis limits to focus on feasible region
        if len(iteration_results) > 0:
            all_distances = []
            all_materials = []
            for result in iteration_results:
                all_distances.extend(result['F'][:, 0])
                all_materials.extend(result['F'][:, 1])

            if all_distances and all_materials:
                plt.xlim(0, min(max(all_distances) * 1.1, 1.0))
                plt.ylim(0, min(max(all_materials) * 1.1, 12.0))

        plt.tight_layout()
        plt.show()

        # Plot 3: Final Pareto frontier dots across all batches
        print(f"\nExtracting final Pareto frontier across all iterations...")

        # Collect all mechanisms from all iterations
        all_final_mechanisms = []
        for i, result in enumerate(iteration_results):
            # Convert F back to mechanism-like format (we only need the F values)
            F = result['F']
            for j in range(len(F)):
                all_final_mechanisms.append({'F': F[j], 'iteration': i + 1})

        # Extract Pareto frontier from all F values
        all_F_values = np.array([mech['F'] for mech in all_final_mechanisms])
        nds = NonDominatedSorting()
        fronts = nds.do(all_F_values)

        if len(fronts) > 0:
            pareto_indices = fronts[0]
            pareto_F = all_F_values[pareto_indices]
            pareto_iterations = [all_final_mechanisms[i]['iteration'] for i in pareto_indices]

            plt.figure(figsize=(10, 8))

            # Color by iteration
            for i in range(1, len(iteration_results) + 1):
                iter_mask = np.array(pareto_iterations) == i
                if np.any(iter_mask):
                    iter_F = pareto_F[iter_mask]
                    plt.scatter(iter_F[:, 0], iter_F[:, 1],
                               alpha=0.8, s=60, c=colors[i-1],
                               label=f'Iter {i} Pareto ({np.sum(iter_mask)} points)',
                               edgecolors='black', linewidth=1)

            plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Distance constraint')
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Material constraint')

            plt.xlabel('Distance', fontsize=12)
            plt.ylabel('Material', fontsize=12)
            plt.title(f'Final Pareto Frontier Across All Iterations ({len(pareto_F)} points)', fontsize=14, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)

            if all_distances and all_materials:
                plt.xlim(0, min(max(all_distances) * 1.1, 1.0))
                plt.ylim(0, min(max(all_materials) * 1.1, 12.0))

            plt.tight_layout()
            plt.show()

            print(f"Final Pareto frontier contains {len(pareto_F)} solutions")
        else:
            print("Could not extract Pareto frontier from final results")

        # Plot 4: Comparison plot with fixed x-y limits for all 5 batches
        print(f"\nCreating comparison plot with fixed axis limits...")

        # Calculate global axis limits
        global_x_max = max(all_distances) * 1.1 if all_distances else 1.0
        global_y_max = max(all_materials) * 1.1 if all_materials else 12.0
        global_x_max = min(global_x_max, 1.0)
        global_y_max = min(global_y_max, 12.0)

        fig2, axes2 = plt.subplots(1, len(iteration_results), figsize=(5*len(iteration_results), 5))
        if len(iteration_results) == 1:
            axes2 = [axes2]

        for i, result in enumerate(iteration_results):
            ax = axes2[i]
            F = result['F']
            hv = result['hypervolume']

            ax.scatter(F[:, 0], F[:, 1], alpha=0.7, s=30, c=colors[i])
            ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7)
            ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7)
            ax.set_xlabel('Distance')
            ax.set_ylabel('Material')
            ax.set_title(f'Iter {i+1}\nHV: {hv:.4f}\n({len(F)} points)')
            ax.grid(True, alpha=0.3)

            # Set fixed axis limits for comparison
            ax.set_xlim(0, global_x_max)
            ax.set_ylim(0, global_y_max)

        plt.suptitle('Iteration Comparison - Fixed Axis Limits', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

        # Print summary
        print(f"\nHypervolume Evolution:")
        for i, hv in enumerate(hypervolumes):
            print(f"- Iteration {i+1}: {hv:.4f}")

        if len(hypervolumes) > 1:
            total_improvement = hypervolumes[-1] - hypervolumes[0]
            percent_improvement = ((hypervolumes[-1]/hypervolumes[0]) - 1) * 100
            print(f"\nTotal Improvement: {total_improvement:.4f} ({percent_improvement:.2f}%)")

    return hypervolumes, iteration_results


if __name__ == "__main__":
    hypervolumes, results = main()
    print("\n=== Recursive Optimization Complete! ===")