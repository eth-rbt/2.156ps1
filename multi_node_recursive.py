#!/usr/bin/env python3
"""
Multi-Node Recursive Single Curve Optimization Script

This script runs the recursive optimization approach across different node configurations
(5, 6, 7 nodes) and combines all Pareto fronts into a single visualization.

For each node configuration:
1. Sample 4000 mechanisms, filter at (3,12), plot initial distribution at (0.75,10)
2. For each iteration (5 total):
   (1) Run GA on current cluster
   (2) Apply material-only gradient descent on frontier
   (3) Apply distance-only gradient descent on original frontier
   (4) Combine all 3 groups, plot and calculate HV
   (5) Use combined cluster for next iteration
3. Extract final Pareto frontier for that node configuration
4. Combine all node configurations into single plot
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
            # Check shape compatibility before applying gradient
            if x[i].shape == material_grads[i].shape:
                x[i] = x[i] - step_size * material_grads[i]
            else:
                # Shape mismatch, mark as done to avoid further errors
                done_optimizing[i] = True

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
            # Check shape compatibility before applying gradient
            if x[i].shape == distance_grads[i].shape:
                x[i] = x[i] - step_size * distance_grads[i]
            else:
                # Shape mismatch, mark as done to avoid further errors
                done_optimizing[i] = True

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


def recursive_optimization_single_node(target_curve, num_nodes, num_mechanisms=4000,
                                     num_iterations=5, pop_size=200, n_gen=100,
                                     step_sizes_material=[2e-4, 1e-4, 5e-5, 2e-5, 1e-5],
                                     step_sizes_distance=[2e-4, 1e-4, 5e-5, 2e-5, 1e-5]):
    """
    Run recursive optimization for a single node configuration.

    Args:
        target_curve: Target curve to optimize for
        num_nodes: Number of nodes in mechanisms
        num_mechanisms: Initial number of mechanisms to generate
        num_iterations: Number of recursive iterations
        pop_size: Population size for GA
        n_gen: Number of generations for GA
        step_sizes_material: Array of step sizes for material gradient descent
        step_sizes_distance: Array of step sizes for distance gradient descent

    Returns:
        List of hypervolumes for each iteration, final Pareto frontier F values
    """

    print(f"\n{'='*70}")
    print(f"RECURSIVE OPTIMIZATION FOR {num_nodes} NODES")
    print(f"{'='*70}")
    print(f"Target: Curve 2 (index 1)")
    print(f"Initial mechanisms: {num_mechanisms}")
    print(f"Iterations: {num_iterations}")
    print(f"Material step sizes: {step_sizes_material}")
    print(f"Distance step sizes: {step_sizes_distance}")

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

    if len(feasible_mechanisms) == 0:
        print("ERROR: No feasible mechanisms found!")
        return [], None

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
        print(f"ITERATION {iteration + 1}/{num_iterations} ({num_nodes} nodes)")
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

        # Accumulate all mechanisms from current iteration for future Pareto frontier extraction
        current_iteration_mechanisms = ga_mechanisms + material_mechanisms + distance_mechanisms
        all_previous_mechanisms.extend(current_iteration_mechanisms)

        print(f"Accumulated {len(current_iteration_mechanisms)} mechanisms from iteration {iteration + 1}")
        print(f"Total accumulated mechanisms: {len(all_previous_mechanisms)}")

        # Prepare for next iteration
        current_mechanisms = final_feasible

        print(f"Prepared {len(current_mechanisms)} mechanisms for next iteration")

    # Extract final Pareto frontier for this node configuration
    if len(all_iteration_results) > 0:
        print(f"\nExtracting final Pareto frontier for {num_nodes} nodes...")

        # Collect all F values from all iterations
        all_F_values = []
        for result in all_iteration_results:
            all_F_values.extend(result['F'])

        if len(all_F_values) > 0:
            all_F_array = np.array(all_F_values)
            nds = NonDominatedSorting()
            fronts = nds.do(all_F_array)

            if len(fronts) > 0:
                pareto_indices = fronts[0]
                final_pareto_F = all_F_array[pareto_indices]
                print(f"Final Pareto frontier for {num_nodes} nodes: {len(final_pareto_F)} solutions")
                return hypervolumes, final_pareto_F

    return hypervolumes, None


def multi_node_recursive_optimization():
    """
    Run recursive optimization across multiple node configurations and combine results.
    """
    print("=== MULTI-NODE RECURSIVE OPTIMIZATION ===")
    print("Running optimization for 7, 8, and 9 nodes (3 runs each = 9 total epochs)")
    print("="*50)

    # Setup environment
    setup_environment()

    # Load target curves
    target_curves = load_target_curves()
    target_curve = target_curves[1]  # Curve 2 (index 1)

    # Define step sizes for each iteration (adjustable) - 16 iterations
    step_sizes_material = [2e-4, 2e-4, 1e-4, 1e-4, 8e-5, 8e-5, 5e-5, 5e-5, 2e-5, 2e-5]
    step_sizes_distance = [2e-4, 2e-4, 1e-4, 1e-4, 8e-5, 8e-5, 5e-5, 5e-5, 2e-5, 2e-5]

    node_configurations = [5, 8, 9]
    num_runs_per_config = 3
    colors = ['blue', 'green', 'red']
    markers = ['o', 's', '^']

    all_results = {}
    all_epochs = []  # Store all 9 epoch results

    # Run optimization for each node configuration, multiple times each
    for config_idx, num_nodes in enumerate(node_configurations):
        node_results = []

        for run_idx in range(num_runs_per_config):
            print(f"\n\n{'#'*80}")
            print(f"STARTING OPTIMIZATION FOR {num_nodes} NODES - RUN {run_idx + 1}/{num_runs_per_config}")
            print(f"{'#'*80}")

            hypervolumes, final_pareto_F = recursive_optimization_single_node(
                target_curve,
                num_nodes=num_nodes,
                num_mechanisms=8000,
                num_iterations=10,
                pop_size=100,
                n_gen=200,
                step_sizes_material=step_sizes_material,
                step_sizes_distance=step_sizes_distance
            )

            run_result = {
                'hypervolumes': hypervolumes,
                'pareto_front': final_pareto_F,
                'num_nodes': num_nodes,
                'run_id': run_idx + 1,
                'epoch_id': f"{num_nodes}nodes_run{run_idx + 1}",
                'color': colors[config_idx],
                'marker': markers[config_idx]
            }

            node_results.append(run_result)
            all_epochs.append(run_result)

            print(f"Completed {num_nodes} nodes, run {run_idx + 1}")
            if final_pareto_F is not None:
                print(f"  - Pareto front size: {len(final_pareto_F)}")
                print(f"  - Best distance: {min(final_pareto_F[:, 0]):.6f}")
                print(f"  - Best material: {min(final_pareto_F[:, 1]):.4f}")

        all_results[num_nodes] = node_results

        print(f"\nCompleted optimization for {num_nodes} nodes")
        if final_pareto_F is not None:
            print(f"Final Pareto frontier: {len(final_pareto_F)} solutions")
            print(f"Best distance: {min(final_pareto_F[:, 0]):.4f}")
            print(f"Best material: {min(final_pareto_F[:, 1]):.4f}")
        else:
            print("No final Pareto frontier found")

    # Create comprehensive visualization
    print(f"\n{'='*80}")
    print("CREATING COMPREHENSIVE VISUALIZATION - 9 EPOCHS")
    print(f"{'='*80}")

    hv_indicator = HV(np.array([0.75, 10.0]))

    # (1) FINAL COMBINED PLOT - All 9 epochs
    print("Creating Plot 1: Final combined plot with all 9 epochs...")
    plt.figure(figsize=(14, 10))

    all_distances = []
    all_materials = []

    # Define colors and markers for each run
    run_colors = {7: ['lightblue', 'blue', 'darkblue'],
                  8: ['lightgreen', 'green', 'darkgreen'],
                  9: ['lightcoral', 'red', 'darkred']}
    run_markers = ['o', 's', '^']

    for epoch in all_epochs:
        if epoch['pareto_front'] is not None:
            F = epoch['pareto_front']
            distances = F[:, 0]
            materials = F[:, 1]

            all_distances.extend(distances)
            all_materials.extend(materials)

            # Use run-specific color and marker
            run_color = run_colors[epoch['num_nodes']][epoch['run_id'] - 1]

            plt.scatter(distances, materials,
                       alpha=0.6, s=40,
                       c=run_color,
                       marker=run_markers[epoch['run_id'] - 1],
                       edgecolors='black',
                       linewidth=0.5,
                       label=f"{epoch['num_nodes']} nodes, Run {epoch['run_id']} ({len(F)} pts)")

    # Add constraint lines
    plt.axvline(x=0.75, color='red', linestyle='--', alpha=0.8, linewidth=3, label='Distance constraint')
    plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.8, linewidth=3, label='Material constraint')

    # Calculate combined hypervolume
    if all_distances and all_materials:
        combined_F = np.column_stack([all_distances, all_materials])
        combined_hv = hv_indicator(combined_F)

        # Extract combined Pareto frontier
        nds = NonDominatedSorting()
        fronts = nds.do(combined_F)
        if len(fronts) > 0:
            combined_pareto_indices = fronts[0]
            combined_pareto_F = combined_F[combined_pareto_indices]
            combined_pareto_hv = hv_indicator(combined_pareto_F)
            title_text = f'All 9 Epochs: 7, 8, 9 Node Configurations (3 runs each)\nCombined HV: {combined_hv:.3f} | Pareto HV: {combined_pareto_hv:.3f} ({len(combined_pareto_F)} frontier pts)'
        else:
            title_text = f'All 9 Epochs: 7, 8, 9 Node Configurations (3 runs each)\nCombined HV: {combined_hv:.3f}'

    plt.xlabel('Distance', fontsize=12)
    plt.ylabel('Material', fontsize=12)
    plt.title(title_text, fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)

    if all_distances and all_materials:
        plt.xlim(0, min(max(all_distances) * 1.1, 1.0))
        plt.ylim(0, min(max(all_materials) * 1.1, 12.0))

    plt.tight_layout()
    plt.show()

    # (2) 3×3 GRID PLOTS - Individual results
    print("Creating Plot 2: 3×3 grid showing individual epoch results...")
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    for i, epoch in enumerate(all_epochs):
        row = i // 3
        col = i % 3
        ax = axes[row, col]

        if epoch['pareto_front'] is not None:
            F = epoch['pareto_front']
            distances = F[:, 0]
            materials = F[:, 1]

            # Calculate HV for this epoch
            hv_value = hv_indicator(F) if len(F) > 0 else 0

            ax.scatter(distances, materials,
                      alpha=0.7, s=30,
                      c=epoch['color'],
                      marker=epoch['marker'],
                      edgecolors='black',
                      linewidth=0.5)

            ax.axvline(x=0.75, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.set_xlabel('Distance', fontsize=10)
            ax.set_ylabel('Material', fontsize=10)
            ax.set_title(f'{epoch["num_nodes"]} nodes, Run {epoch["run_id"]}\n{len(F)} pts, HV: {hv_value:.3f}', fontsize=10)
            ax.grid(True, alpha=0.3)

            if all_distances and all_materials:
                ax.set_xlim(0, min(max(all_distances) * 1.1, 1.0))
                ax.set_ylim(0, min(max(all_materials) * 1.1, 12.0))

    plt.suptitle('Individual Epoch Results: 3×3 Grid (9 Total Runs)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

    # (3) AVERAGED HV GROWTH CURVES
    print("Creating Plot 3: Averaged HV growth curves for 7, 8, 9 nodes...")
    plt.figure(figsize=(12, 8))

    for num_nodes in node_configurations:
        node_runs = all_results[num_nodes]

        # Collect hypervolume curves for this node configuration
        hv_curves = []
        for run_result in node_runs:
            if run_result['hypervolumes']:
                hv_curves.append(run_result['hypervolumes'])

        if hv_curves:
            # Find minimum length to ensure all curves have same length
            min_length = min(len(curve) for curve in hv_curves)
            hv_array = np.array([curve[:min_length] for curve in hv_curves])

            # Calculate mean and std
            hv_mean = np.mean(hv_array, axis=0)
            hv_std = np.std(hv_array, axis=0)

            iterations = range(1, len(hv_mean) + 1)

            # Plot mean curve with std band
            color_map = {7: 'blue', 8: 'green', 9: 'red'}
            color = color_map[num_nodes]

            plt.plot(iterations, hv_mean,
                    color=color, linewidth=3,
                    marker='o', markersize=6,
                    label=f'{num_nodes} nodes (avg of 3 runs)')

            plt.fill_between(iterations,
                           hv_mean - hv_std, hv_mean + hv_std,
                           color=color, alpha=0.2)

            # Plot individual runs as thin lines
            for i, curve in enumerate(hv_curves):
                plt.plot(range(1, len(curve[:min_length]) + 1), curve[:min_length],
                        color=color, alpha=0.4, linewidth=1, linestyle='--')

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Hypervolume', fontsize=12)
    plt.title('Hypervolume Evolution: Averaged Across 3 Runs Per Node Configuration', fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # (4) HV PLOT WITH FRONTIER HIGHLIGHTED
    print("Creating Plot 4: HV comparison with frontier solutions highlighted...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Final HV values for each epoch
    hv_values = []
    epoch_labels = []
    colors_list = []

    for epoch in all_epochs:
        if epoch['pareto_front'] is not None:
            hv_val = hv_indicator(epoch['pareto_front'])
            hv_values.append(hv_val)
            epoch_labels.append(f"{epoch['num_nodes']}n R{epoch['run_id']}")
            colors_list.append(epoch['color'])

    bars = ax1.bar(range(len(hv_values)), hv_values, color=colors_list, alpha=0.7, edgecolor='black')

    # Highlight the best performing epoch
    if hv_values:
        best_idx = np.argmax(hv_values)
        bars[best_idx].set_color('gold')
        bars[best_idx].set_edgecolor('red')
        bars[best_idx].set_linewidth(3)

    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Final Hypervolume', fontsize=12)
    ax1.set_title('Final Hypervolume by Epoch\n(Gold bar = Best frontier)', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(len(epoch_labels)))
    ax1.set_xticklabels(epoch_labels, rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')

    # Right plot: Pareto front of the best performing epoch
    if hv_values:
        best_epoch = all_epochs[best_idx]
        if best_epoch['pareto_front'] is not None:
            F_best = best_epoch['pareto_front']

            # Plot all other epochs in background
            for i, epoch in enumerate(all_epochs):
                if i != best_idx and epoch['pareto_front'] is not None:
                    F = epoch['pareto_front']
                    ax2.scatter(F[:, 0], F[:, 1],
                              alpha=0.2, s=20, c='lightgray',
                              edgecolors='none')

            # Plot best epoch prominently
            ax2.scatter(F_best[:, 0], F_best[:, 1],
                       alpha=0.8, s=60, c='gold',
                       edgecolors='red', linewidth=2,
                       label=f'Best: {best_epoch["epoch_id"]}')

            ax2.axvline(x=0.75, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax2.axhline(y=10.0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            ax2.set_xlabel('Distance', fontsize=12)
            ax2.set_ylabel('Material', fontsize=12)
            ax2.set_title(f'Best Performing Pareto Front\nHV: {hv_values[best_idx]:.3f}', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)

            if all_distances and all_materials:
                ax2.set_xlim(0, min(max(all_distances) * 1.1, 1.0))
                ax2.set_ylim(0, min(max(all_materials) * 1.1, 12.0))

    plt.tight_layout()
    plt.show()

    # Print summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    for num_nodes, results in all_results.items():
        print(f"\n{num_nodes} Nodes:")
        if results['pareto_front'] is not None:
            F = results['pareto_front']
            distances = F[:, 0]
            materials = F[:, 1]

            print(f"  - Pareto front size: {len(F)} solutions")
            print(f"  - Best distance: {min(distances):.4f}")
            print(f"  - Best material: {min(materials):.4f}")
            print(f"  - Distance range: {min(distances):.4f} - {max(distances):.4f}")
            print(f"  - Material range: {min(materials):.4f} - {max(materials):.4f}")

            if len(results['hypervolumes']) > 0:
                print(f"  - Final hypervolume: {results['hypervolumes'][-1]:.4f}")
                if len(results['hypervolumes']) > 1:
                    improvement = results['hypervolumes'][-1] - results['hypervolumes'][0]
                    print(f"  - Hypervolume improvement: {improvement:.4f}")
        else:
            print(f"  - No solutions found")

    # Save results in submittable format
    print(f"\n{'='*80}")
    print("SAVING RESULTS FOR SUBMISSION")
    print(f"{'='*80}")

    # Collect all final mechanisms from both configurations
    all_final_mechanisms = []

    for num_nodes, results in all_results.items():
        if results['pareto_front'] is not None:
            F = results['pareto_front']
            print(f"\n{num_nodes} nodes: {len(F)} Pareto optimal solutions")

            # For submission, we need mechanism representations, not just F values
            # Since we only have F values from the final Pareto front,
            # we'll save those and note the configuration
            for i, f_vals in enumerate(F):
                mechanism_data = {
                    'distance': float(f_vals[0]),
                    'material': float(f_vals[1]),
                    'nodes': int(num_nodes),
                    'mechanism_id': f"{num_nodes}nodes_{i:03d}"
                }
                all_final_mechanisms.append(mechanism_data)

    # Sort by distance (best first)
    all_final_mechanisms.sort(key=lambda x: x['distance'])

    print(f"\nTotal mechanisms for submission: {len(all_final_mechanisms)}")
    print(f"Best distance: {all_final_mechanisms[0]['distance']:.6f}")
    print(f"Best material: {min(m['material'] for m in all_final_mechanisms):.4f}")

    # Save as JSON for analysis
    import json
    with open('curve2_submission_results.json', 'w') as f:
        json.dump(all_final_mechanisms, f, indent=2)

    # Save distances and materials as numpy arrays for direct submission
    distances = np.array([m['distance'] for m in all_final_mechanisms])
    materials = np.array([m['material'] for m in all_final_mechanisms])

    # Create submission array (assuming submission format needs distance, material pairs)
    submission_array = np.column_stack([distances, materials])

    # Save as .npy file for submission
    np.save('curve2_submission_pareto_front.npy', submission_array)

    print(f"\nSaved results:")
    print(f"- curve2_submission_results.json (detailed results)")
    print(f"- curve2_submission_pareto_front.npy (submission array shape: {submission_array.shape})")

    # Display top 10 solutions
    print(f"\nTop 10 solutions:")
    print(f"{'Rank':<4} {'Distance':<10} {'Material':<10} {'Nodes':<6} {'ID'}")
    print("-" * 50)
    for i in range(min(10, len(all_final_mechanisms))):
        m = all_final_mechanisms[i]
        print(f"{i+1:<4} {m['distance']:<10.6f} {m['material']:<10.4f} {m['nodes']:<6} {m['mechanism_id']}")

    return all_results


if __name__ == "__main__":
    results = multi_node_recursive_optimization()
    print("\n=== Multi-Node Recursive Optimization Complete! ===")