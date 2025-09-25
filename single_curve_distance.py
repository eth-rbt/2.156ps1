#!/usr/bin/env python3
"""
Single-Objective Distance Minimization for Curve 2

This script focuses purely on minimizing distance to the target curve
with the constraint that material usage must be < 10.

Process:
1. Generate random mechanisms
2. Filter for material < 10 constraint
3. Run single-objective GA to minimize distance
4. Plot results
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import (
    setup_environment,
    load_target_curves,
    filter_feasible_mechanisms,
    analyze_initial_population,
    plot_initial_distribution,
    AdvancedMechanismOptimizer
)
from LINKS.Optimization import MechanismRandomizer
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination


class DistanceMinimizationProblem(ElementwiseProblem):
    """
    Single-objective optimization problem for distance minimization.
    Constraint: material < 10
    """

    def __init__(self, target_curve, N=7):
        self.N = N
        self.target_curve = target_curve

        # Global optimization tools
        global PROBLEM_TOOLS
        if 'PROBLEM_TOOLS' not in globals():
            from LINKS.Optimization import Tools
            PROBLEM_TOOLS = Tools(device='cpu')
            PROBLEM_TOOLS.compile()

        variables = dict()

        # Connectivity matrix variables (upper triangular, excluding motor connection)
        for i in range(N):
            for j in range(i):
                variables[f"C{j}_{i}"] = Binary()

        # Remove known motor connection
        if "C0_1" in variables:
            del variables["C0_1"]

        # Position variables
        for i in range(2 * N):
            variables[f"X0{i}"] = Real(bounds=(0.0, 1.0))

        # Fixed nodes variables
        for i in range(N):
            variables[f"fixed_nodes{i}"] = Binary()

        # Target node variable
        variables["target"] = Integer(bounds=(1, N-1))

        # Single objective: minimize distance
        # Single constraint: material < 10
        super().__init__(vars=variables, n_obj=1, n_constr=1)

    def convert_1D_to_mech(self, x):
        """Convert 1D optimization variables to mechanism representation."""
        N = self.N
        target_idx = x["target"]

        # Build connectivity matrix
        C = np.zeros((N, N))
        x["C0_1"] = 1  # Motor connection

        for i in range(N):
            for j in range(i):
                C[j, i] = x[f"C{j}_{i}"]

        edges = np.array(np.where(C == 1)).T

        # Reshape position matrix
        x0 = np.array([x[f"X0{i}"] for i in range(2 * N)]).reshape([N, 2])

        # Extract fixed joints
        fixed_joints = np.where(np.array([x[f"fixed_nodes{i}"] for i in range(N)]))[0].astype(int)

        # Motor is fixed as [0, 1]
        motor = np.array([0, 1])

        return x0, edges, fixed_joints, motor, target_idx

    def convert_mech_to_1D(self, x0, edges, fixed_joints, target_joint=None, **kwargs):
        """Convert mechanism representation to 1D optimization variables."""
        N = self.N
        x = {}

        # Target joint
        if target_joint is None:
            target_joint = x0.shape[0] - 1
        x["target"] = target_joint

        # Connectivity matrix
        C = np.zeros((N, N), dtype=bool)
        C[edges[:, 0], edges[:, 1]] = 1
        C[edges[:, 1], edges[:, 0]] = 1

        for i in range(N):
            for j in range(i):
                x[f"C{j}_{i}"] = C[i, j]

        if "C0_1" in x:
            del x["C0_1"]

        # Position matrix
        if x0.shape[0] != N:
            x0 = np.pad(x0, ((0, N - x0.shape[0]), (0, 0)), 'constant', constant_values=0)

        for i in range(2 * N):
            x[f"X0{i}"] = x0.flatten()[i]

        # Fixed nodes
        for i in range(N):
            x[f"fixed_nodes{i}"] = (i in fixed_joints) or (i >= N)

        return x

    def _evaluate(self, x, out, *args, **kwargs):
        """Evaluate mechanism for distance objective and material constraint."""
        x0, edges, fixed_joints, motor, target_idx = self.convert_1D_to_mech(x)

        distance, material = PROBLEM_TOOLS(
            x0, edges, fixed_joints, motor, self.target_curve, target_idx=target_idx
        )

        # Single objective: minimize distance
        out["F"] = np.array([distance])

        # Single constraint: material < 10
        out["G"] = np.array([material - 10.0])


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


def run_distance_optimization(target_curve, initial_mechanisms, num_nodes=7, pop_size=200, n_gen=100):
    """
    Run single-objective GA optimization to minimize distance.

    Args:
        target_curve: Target curve to optimize for
        initial_mechanisms: List of initial mechanism dictionaries
        num_nodes: Number of nodes in mechanism
        pop_size: Population size
        n_gen: Number of generations

    Returns:
        Optimization results and problem instance
    """
    problem = DistanceMinimizationProblem(target_curve, num_nodes)

    # Convert mechanisms to 1D representation
    initial_population = []
    for mech in initial_mechanisms:
        try:
            x_1d = problem.convert_mech_to_1D(**mech)
            initial_population.append(x_1d)
        except:
            continue

    if len(initial_population) == 0:
        print("No valid initial mechanisms for optimization!")
        return None, None

    # Pad population if needed
    while len(initial_population) < pop_size:
        initial_population.extend(initial_population[:min(len(initial_population), pop_size - len(initial_population))])

    initial_population = initial_population[:pop_size]

    from pymoo.operators.sampling.rnd import Sampling

    class SampleFromInitial(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return np.array([initial_population[i % len(initial_population)] for i in range(n_samples)])

    # Use NSGA2 with single objective (it handles mixed variables better)
    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SampleFromInitial(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        mutation=PolynomialMutation(prob=0.5),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )

    print(f"Running single-objective GA with {len(initial_population)} initial mechanisms...")
    print(f"Population size: {pop_size}, Generations: {n_gen}")
    print("Objective: Minimize distance")
    print("Constraint: Material < 10")

    results = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=True,
        save_history=True,
        seed=123
    )

    return results, problem


def main():
    """Main function for distance-only optimization."""

    print("=== Single-Objective Distance Minimization ===")
    print("Target: Curve 2 (index 1)")
    print("Objective: Minimize distance to target curve")
    print("Constraint: Material usage < 10")
    print("="*50)

    # Setup environment
    setup_environment()

    # Load target curves
    target_curves = load_target_curves()
    target_curve = target_curves[1]  # Curve 2 (index 1)

    # Step 1: Generate random mechanisms
    print(f"\n1. Generating random mechanisms...")
    num_mechanisms = 4000
    num_nodes = 7

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

    # Step 2: Analyze initial population
    print("\n2. Analyzing initial population...")
    analyze_initial_population(mechanisms, target_curve)

    # Step 3: Filter for material < 10 constraint
    print("\n3. Filtering mechanisms for material < 10...")
    material_feasible, material_metrics = filter_feasible_mechanisms(
        mechanisms,
        target_curve,
        feasibility_threshold=(float('inf'), 10.0)  # Only material constraint
    )

    print(f"Material-feasible mechanisms: {len(material_feasible)}/{len(mechanisms)} "
          f"({100*len(material_feasible)/len(mechanisms):.1f}%)")

    if len(material_feasible) == 0:
        print("ERROR: No mechanisms satisfy material < 10 constraint!")
        return None

    # Plot initial distribution
    print("\n4. Plotting initial distribution...")
    plot_initial_distribution(material_feasible, target_curve,
                            ref_point=np.array([2.0, 10.0]))

    # Print statistics about material-feasible mechanisms
    if material_metrics:
        distances = [p[0] for p in material_metrics]
        materials = [p[1] for p in material_metrics]
        print(f"Distance range: {min(distances):.4f} - {max(distances):.4f}")
        print(f"Material range: {min(materials):.4f} - {max(materials):.4f}")

    # Step 4: Run single-objective GA optimization
    print(f"\n5. Running distance minimization GA...")

    results, problem = run_distance_optimization(
        target_curve,
        material_feasible,
        num_nodes=num_nodes,
        pop_size=200,
        n_gen=100
    )

    if results is None:
        print("Optimization failed!")
        return None

    # Step 5: Analyze and plot results
    print(f"\n6. Analyzing optimization results...")

    if results.X is not None and results.F is not None:
        # Convert results back to mechanisms for evaluation
        final_mechanisms = mechanisms_from_results(results, problem)

        # Evaluate final mechanisms
        optimizer = AdvancedMechanismOptimizer()

        final_distances = []
        final_materials = []

        for mech in final_mechanisms:
            try:
                distance, material = optimizer.tools(
                    mech['x0'], mech['edges'], mech['fixed_joints'],
                    mech['motor'], target_curve, mech['target_joint']
                )
                final_distances.append(distance)
                final_materials.append(material)
            except:
                continue

        if final_distances:
            print(f"\nOptimization Results:")
            print(f"- Solutions found: {len(final_distances)}")
            print(f"- Best distance: {min(final_distances):.6f}")
            print(f"- Average distance: {np.mean(final_distances):.6f}")
            print(f"- Distance range: {min(final_distances):.6f} - {max(final_distances):.6f}")
            print(f"- Material range: {min(final_materials):.4f} - {max(final_materials):.4f}")
            print(f"- Material constraint violations: {sum(1 for m in final_materials if m >= 10)}")

            # Enhanced plotting with multiple solutions

            # Plot 1: Overview plots (4 subplots)
            plt.figure(figsize=(20, 10))

            # Plot 1a: Distance vs Material scatter with quality zones
            plt.subplot(2, 4, 1)
            colors = []
            for d, m in zip(final_distances, final_materials):
                if d < 0.1:  # Very good distance
                    colors.append('green')
                elif d < 0.2:  # Good distance
                    colors.append('blue')
                elif d < 0.5:  # Medium distance
                    colors.append('orange')
                else:  # Poor distance
                    colors.append('red')

            plt.scatter(final_distances, final_materials, alpha=0.7, s=40, c=colors)
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Material constraint')
            plt.axvline(x=0.1, color='green', linestyle=':', alpha=0.7, label='Excellent distance')
            plt.axvline(x=0.2, color='blue', linestyle=':', alpha=0.7, label='Good distance')
            plt.xlabel('Distance')
            plt.ylabel('Material')
            plt.title('Solution Quality Zones')
            plt.legend(fontsize=8)
            plt.grid(True, alpha=0.3)

            # Plot 1b: Distance histogram
            plt.subplot(2, 4, 2)
            plt.hist(final_distances, bins=30, alpha=0.7, color='green', edgecolor='black')
            plt.axvline(x=min(final_distances), color='red', linestyle='--', alpha=0.7,
                       label=f'Best: {min(final_distances):.4f}')
            plt.xlabel('Distance')
            plt.ylabel('Count')
            plt.title('Distance Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 1c: Material histogram
            plt.subplot(2, 4, 3)
            plt.hist(final_materials, bins=30, alpha=0.7, color='orange', edgecolor='black')
            plt.axvline(x=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Constraint')
            plt.xlabel('Material')
            plt.ylabel('Count')
            plt.title('Material Distribution')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 1d: Solution ranking
            plt.subplot(2, 4, 4)
            sorted_indices = np.argsort(final_distances)
            sorted_distances = [final_distances[i] for i in sorted_indices]
            plt.plot(range(len(sorted_distances)), sorted_distances, 'b-', linewidth=2, alpha=0.7)
            plt.fill_between(range(len(sorted_distances)), sorted_distances, alpha=0.3)
            plt.xlabel('Solution Rank')
            plt.ylabel('Distance')
            plt.title('Sorted Distance Values')
            plt.grid(True, alpha=0.3)

            # Plot 2: Best solutions detailed view
            n_best = min(10, len(final_mechanisms))
            best_indices = np.argsort(final_distances)[:n_best]

            # Plot 2a: Best solutions scatter
            plt.subplot(2, 4, 5)
            best_distances = [final_distances[i] for i in best_indices]
            best_materials = [final_materials[i] for i in best_indices]

            plt.scatter(best_distances, best_materials, alpha=0.8, s=80, c='red',
                       edgecolors='black', linewidth=1, label=f'Top {n_best} Solutions')
            plt.scatter(final_distances, final_materials, alpha=0.3, s=20, c='lightblue', label='All Solutions')
            plt.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, linewidth=2)
            plt.xlabel('Distance')
            plt.ylabel('Material')
            plt.title(f'Top {n_best} Best Solutions')
            plt.legend()
            plt.grid(True, alpha=0.3)

            # Plot 2b: Performance metrics
            plt.subplot(2, 4, 6)
            metrics = ['Best', 'Top 5 Avg', 'Top 10 Avg', 'Overall Avg']
            values = [
                min(final_distances),
                np.mean(sorted_distances[:min(5, len(final_distances))]),
                np.mean(sorted_distances[:min(10, len(final_distances))]),
                np.mean(final_distances)
            ]
            colors_bar = ['red', 'orange', 'yellow', 'lightblue']
            bars = plt.bar(metrics, values, color=colors_bar, alpha=0.7, edgecolor='black')
            plt.ylabel('Distance')
            plt.title('Performance Summary')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3, axis='y')

            # Add value labels on bars
            for bar, value in zip(bars, values):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01,
                        f'{value:.4f}', ha='center', va='bottom', fontsize=8)

            # Plot 2c: Constraint satisfaction
            plt.subplot(2, 4, 7)
            satisfied = sum(1 for m in final_materials if m < 10.0)
            violated = len(final_materials) - satisfied

            plt.pie([satisfied, violated], labels=['Satisfied', 'Violated'],
                   colors=['green', 'red'], autopct='%1.1f%%', startangle=90)
            plt.title(f'Material Constraint\n(< 10.0)')

            # Plot 2d: Solution diversity
            plt.subplot(2, 4, 8)
            distance_std = np.std(final_distances)
            material_std = np.std(final_materials)
            distance_range = max(final_distances) - min(final_distances)
            material_range = max(final_materials) - min(final_materials)

            diversity_metrics = ['Distance\nStd Dev', 'Material\nStd Dev', 'Distance\nRange', 'Material\nRange']
            diversity_values = [distance_std, material_std, distance_range, material_range]
            bars = plt.bar(diversity_metrics, diversity_values, color=['blue', 'orange', 'lightblue', 'lightyellow'],
                          alpha=0.7, edgecolor='black')
            plt.ylabel('Value')
            plt.title('Solution Diversity')
            plt.xticks(rotation=45, fontsize=8)
            plt.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.show()

            # Plot 3: Individual best solutions details
            print(f"\nDetailed Analysis of Top {n_best} Solutions:")
            print("-" * 60)

            plt.figure(figsize=(20, 4))

            for i in range(min(n_best, 5)):  # Show top 5 in detail
                idx = best_indices[i]
                distance = final_distances[idx]
                material = final_materials[idx]

                plt.subplot(1, 5, i+1)

                # Create a simple representation of the solution
                mechanism = final_mechanisms[idx]
                x0 = mechanism['x0']

                # Plot mechanism nodes
                plt.scatter(x0[:, 0], x0[:, 1], s=100, alpha=0.7, c='blue', edgecolors='black')

                # Add node numbers
                for j, (x, y) in enumerate(x0):
                    plt.text(x, y, str(j), ha='center', va='center', fontweight='bold', color='white', fontsize=8)

                plt.xlim(0, 1)
                plt.ylim(0, 1)
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.title(f'Rank {i+1}\nDist: {distance:.4f}\nMat: {material:.2f}')
                plt.grid(True, alpha=0.3)
                plt.gca().set_aspect('equal', adjustable='box')

            plt.tight_layout()
            plt.show()

            # Print detailed statistics
            print(f"\nDetailed Solution Statistics:")
            print(f"{'Rank':<4} {'Distance':<10} {'Material':<10} {'Constraint':<12}")
            print("-" * 40)

            for i in range(min(10, len(best_indices))):
                idx = best_indices[i]
                distance = final_distances[idx]
                material = final_materials[idx]
                constraint_status = "✓ Satisfied" if material < 10.0 else "✗ Violated"
                print(f"{i+1:<4} {distance:<10.6f} {material:<10.4f} {constraint_status}")

            print(f"\nSummary:")
            print(f"- Best distance achieved: {min(final_distances):.6f}")
            print(f"- Solutions with distance < 0.1: {sum(1 for d in final_distances if d < 0.1)}")
            print(f"- Solutions with distance < 0.2: {sum(1 for d in final_distances if d < 0.2)}")
            print(f"- Constraint satisfaction rate: {satisfied}/{len(final_materials)} ({100*satisfied/len(final_materials):.1f}%)")

            return final_distances, final_materials, final_mechanisms

    else:
        print("No solutions found in optimization!")
        return None


def comprehensive_evaluation(distances, materials, mechanisms):
    """
    Comprehensive evaluation of the optimization results.

    Args:
        distances: List of distances for all solutions
        materials: List of materials for all solutions
        mechanisms: List of mechanism dictionaries

    Returns:
        Dictionary with evaluation metrics
    """
    print(f"\n{'='*80}")
    print("COMPREHENSIVE EVALUATION")
    print(f"{'='*80}")

    # Basic statistics
    n_solutions = len(distances)
    best_distance = min(distances)
    worst_distance = max(distances)
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)
    median_distance = np.median(distances)

    best_material = min(materials)
    worst_material = max(materials)
    avg_material = np.mean(materials)
    std_material = np.std(materials)
    median_material = np.median(materials)

    print(f"\n1. BASIC STATISTICS")
    print(f"   Total Solutions Found: {n_solutions}")
    print(f"   Distance Metrics:")
    print(f"     - Best (minimum):     {best_distance:.6f}")
    print(f"     - Worst (maximum):    {worst_distance:.6f}")
    print(f"     - Average:            {avg_distance:.6f}")
    print(f"     - Median:             {median_distance:.6f}")
    print(f"     - Standard Deviation: {std_distance:.6f}")
    print(f"   Material Metrics:")
    print(f"     - Best (minimum):     {best_material:.4f}")
    print(f"     - Worst (maximum):    {worst_material:.4f}")
    print(f"     - Average:            {avg_material:.4f}")
    print(f"     - Median:             {median_material:.4f}")
    print(f"     - Standard Deviation: {std_material:.4f}")

    # Quality assessment
    excellent_count = sum(1 for d in distances if d < 0.1)
    good_count = sum(1 for d in distances if d < 0.2)
    decent_count = sum(1 for d in distances if d < 0.5)
    poor_count = n_solutions - decent_count

    constraint_satisfied = sum(1 for m in materials if m < 10.0)
    constraint_violated = n_solutions - constraint_satisfied

    print(f"\n2. QUALITY ASSESSMENT")
    print(f"   Distance Quality Distribution:")
    print(f"     - Excellent (< 0.1):  {excellent_count:4d} ({100*excellent_count/n_solutions:5.1f}%)")
    print(f"     - Good (< 0.2):       {good_count:4d} ({100*good_count/n_solutions:5.1f}%)")
    print(f"     - Decent (< 0.5):     {decent_count:4d} ({100*decent_count/n_solutions:5.1f}%)")
    print(f"     - Poor (≥ 0.5):       {poor_count:4d} ({100*poor_count/n_solutions:5.1f}%)")
    print(f"   Constraint Compliance:")
    print(f"     - Satisfied (< 10.0): {constraint_satisfied:4d} ({100*constraint_satisfied/n_solutions:5.1f}%)")
    print(f"     - Violated (≥ 10.0):  {constraint_violated:4d} ({100*constraint_violated/n_solutions:5.1f}%)")

    # Performance tiers
    sorted_indices = np.argsort(distances)
    top_1_pct = max(1, n_solutions // 100)
    top_5_pct = max(1, n_solutions // 20)
    top_10_pct = max(1, n_solutions // 10)

    top_1_distances = [distances[sorted_indices[i]] for i in range(top_1_pct)]
    top_5_distances = [distances[sorted_indices[i]] for i in range(top_5_pct)]
    top_10_distances = [distances[sorted_indices[i]] for i in range(top_10_pct)]

    print(f"\n3. PERFORMANCE TIERS")
    print(f"   Top 1% ({top_1_pct} solutions):")
    print(f"     - Average Distance: {np.mean(top_1_distances):.6f}")
    print(f"     - Best Distance:    {min(top_1_distances):.6f}")
    print(f"   Top 5% ({top_5_pct} solutions):")
    print(f"     - Average Distance: {np.mean(top_5_distances):.6f}")
    print(f"     - Worst in Tier:    {max(top_5_distances):.6f}")
    print(f"   Top 10% ({top_10_pct} solutions):")
    print(f"     - Average Distance: {np.mean(top_10_distances):.6f}")
    print(f"     - Worst in Tier:    {max(top_10_distances):.6f}")

    # Optimization effectiveness
    improvement_potential = worst_distance - best_distance
    convergence_ratio = std_distance / avg_distance
    elite_ratio = excellent_count / n_solutions if n_solutions > 0 else 0

    print(f"\n4. OPTIMIZATION EFFECTIVENESS")
    print(f"   Improvement Range:     {improvement_potential:.6f}")
    print(f"   Convergence Ratio:     {convergence_ratio:.4f} (lower = better convergence)")
    print(f"   Elite Solution Ratio:  {elite_ratio:.4f} (higher = better)")

    # Success criteria assessment
    success_criteria = {
        'best_distance_under_01': best_distance < 0.1,
        'avg_distance_under_03': avg_distance < 0.3,
        'constraint_satisfaction_over_90': (constraint_satisfied / n_solutions) > 0.9,
        'elite_ratio_over_10': elite_ratio > 0.1,
        'convergence_good': convergence_ratio < 0.5
    }

    success_count = sum(success_criteria.values())

    print(f"\n5. SUCCESS CRITERIA ASSESSMENT")
    print(f"   ✓ Best distance < 0.1:           {'PASS' if success_criteria['best_distance_under_01'] else 'FAIL'}")
    print(f"   ✓ Average distance < 0.3:        {'PASS' if success_criteria['avg_distance_under_03'] else 'FAIL'}")
    print(f"   ✓ Constraint satisfaction > 90%: {'PASS' if success_criteria['constraint_satisfaction_over_90'] else 'FAIL'}")
    print(f"   ✓ Elite solutions > 10%:         {'PASS' if success_criteria['elite_ratio_over_10'] else 'FAIL'}")
    print(f"   ✓ Good convergence:              {'PASS' if success_criteria['convergence_good'] else 'FAIL'}")
    print(f"   Overall Success Score: {success_count}/5")

    # Performance grade
    if success_count >= 4:
        grade = "A - Excellent"
    elif success_count >= 3:
        grade = "B - Good"
    elif success_count >= 2:
        grade = "C - Fair"
    elif success_count >= 1:
        grade = "D - Poor"
    else:
        grade = "F - Failed"

    print(f"\n6. OVERALL PERFORMANCE GRADE: {grade}")

    # Recommendations
    print(f"\n7. RECOMMENDATIONS")
    if not success_criteria['best_distance_under_01']:
        print(f"   • Increase population size or generations to find better solutions")
    if not success_criteria['constraint_satisfaction_over_90']:
        print(f"   • Review constraint handling - many solutions violate material < 10")
    if not success_criteria['elite_ratio_over_10']:
        print(f"   • Consider different selection pressure or elitism strategies")
    if not success_criteria['convergence_good']:
        print(f"   • Solutions are too spread out - improve convergence mechanisms")
    if success_count == 5:
        print(f"   • Excellent performance! Consider this configuration for production")

    # Create evaluation report
    evaluation_report = {
        'n_solutions': n_solutions,
        'best_distance': best_distance,
        'avg_distance': avg_distance,
        'std_distance': std_distance,
        'best_material': best_material,
        'avg_material': avg_material,
        'excellent_count': excellent_count,
        'constraint_satisfied': constraint_satisfied,
        'success_criteria': success_criteria,
        'success_count': success_count,
        'grade': grade,
        'elite_ratio': elite_ratio,
        'convergence_ratio': convergence_ratio,
        'constraint_satisfaction_rate': constraint_satisfied / n_solutions if n_solutions > 0 else 0
    }

    print(f"\n{'='*80}")

    return evaluation_report


if __name__ == "__main__":
    result = main()
    if result is not None:
        distances, materials, mechanisms = result

        print(f"\n=== Distance Optimization Complete! ===")
        print(f"Best distance achieved: {min(distances):.6f}")
        print(f"Number of solutions: {len(distances)}")

        # Run comprehensive evaluation
        evaluation_report = comprehensive_evaluation(distances, materials, mechanisms)

        # Save evaluation report
        import json
        with open('distance_optimization_evaluation.json', 'w') as f:
            json.dump({k: str(v) for k, v in evaluation_report.items()}, f, indent=2)
        print(f"\nEvaluation report saved to: distance_optimization_evaluation.json")

    else:
        print(f"\n=== Distance Optimization Failed! ===")