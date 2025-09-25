"""
Advanced Linkage Synthesis Optimization Utilities

This module contains clean utility functions extracted from the Advanced_Starter_Notebook.ipynb
for advanced linkage mechanism synthesis and optimization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.auto import tqdm, trange

# Core imports for optimization
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.variable import Real, Integer, Choice, Binary
from pymoo.core.mixed import MixedVariableMating, MixedVariableGA, MixedVariableSampling, MixedVariableDuplicateElimination
from pymoo.operators.sampling.rnd import FloatRandomSampling, Sampling
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.optimize import minimize
from pymoo.indicators.hv import HV

# LINKS library imports
from LINKS.Optimization import DifferentiableTools, Tools, MechanismRandomizer
from LINKS.Visualization import MechanismVisualizer, GAVisualizer
from LINKS.Kinematics import MechanismSolver
from LINKS.Geometry import CurveEngine
from LINKS.CP import make_empty_submission, evaluate_submission


class AdvancedMechanismOptimizer:
    """
    Advanced mechanism synthesis optimizer that handles full mechanism generation
    including both structure and position optimization.
    """

    def __init__(self, device='cpu'):
        """Initialize the optimizer with specified device."""
        self.device = device
        self.tools = Tools(device=device)
        self.tools.compile()
        self.diff_tools = DifferentiableTools(device=device)
        self.diff_tools.compile()
        self.randomizer = MechanismRandomizer(min_size=6, max_size=14, device=device)
        self.visualizer = MechanismVisualizer()
        self.ga_visualizer = GAVisualizer()
        self.solver = MechanismSolver(device=device)
        self.curve_engine = CurveEngine(normalize_scale=False, device=device)

    def generate_random_mechanisms(self, count=100, size=7):
        """Generate a population of random mechanisms."""
        return [self.randomizer(n=size) for _ in trange(count, desc="Generating mechanisms")]

    def create_mixed_variable_problem(self, target_curve, N=7):
        """Create a mixed variable optimization problem for full mechanism synthesis."""
        return MechanismSynthesisOptimization(target_curve, N)

    def run_mixed_variable_optimization(self, target_curve, N=7, pop_size=100, n_gen=100,
                                      initial_mechanisms=None, verbose=True):
        """
        Run mixed variable GA optimization for mechanism synthesis.

        Args:
            target_curve: Target curve to optimize for
            N: Number of nodes in mechanism
            pop_size: Population size
            n_gen: Number of generations
            initial_mechanisms: Optional list of initial mechanisms
            verbose: Whether to show optimization progress

        Returns:
            Optimization results
        """
        problem = self.create_mixed_variable_problem(target_curve, N)

        if initial_mechanisms is None:
            initial_mechanisms = self.generate_random_mechanisms(pop_size, N)

        initial_population = [problem.convert_mech_to_1D(**mech) for mech in initial_mechanisms]

        class SampleFromRandom(Sampling):
            def _do(self, problem, n_samples, **kwargs):
                return np.array([initial_population[i % len(initial_population)] for i in range(n_samples)])

        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=SampleFromRandom(),
            mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
            mutation=PolynomialMutation(prob=0.5),
            eliminate_duplicates=MixedVariableDuplicateElimination()
        )

        results = minimize(
            problem,
            algorithm,
            ('n_gen', n_gen),
            verbose=verbose,
            save_history=True,
            seed=123
        )

        return results, problem

    def apply_gradient_optimization(self, ga_results, problem, target_curve,
                                  step_size=4e-4, n_steps=1000):
        """
        Apply gradient-based optimization to GA results.

        Args:
            ga_results: Results from GA optimization
            problem: The optimization problem instance
            target_curve: Target curve for optimization
            step_size: Gradient descent step size
            n_steps: Maximum number of optimization steps

        Returns:
            Optimized mechanisms and their performance
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

        for step in trange(n_steps, desc="Gradient optimization"):
            distances, materials, distance_grads, material_grads = self.diff_tools(
                x, edges, fixed_joints, motors, target_curve, target_idxs
            )

            # Only update valid members
            valids = np.where(np.logical_and(distances <= 0.75, materials <= 10.0))[0]
            invalids = np.where(~np.logical_and(distances <= 0.75, materials <= 10.0))[0]

            # Revert invalid members and mark as done
            for i in invalids:
                done_optimizing[i] = True
                x[i] = x_last[i]

            x_last = x.copy()

            # Update valid members using distance gradients
            for i in valids:
                if done_optimizing[i]:
                    continue
                x[i] = x[i] - step_size * distance_grads[i] - step_size * material_grads[i]

            if np.all(done_optimizing):
                print(f'All members done optimizing at step {step}')
                break

        return x, (x0s, edges, fixed_joints, motors, target_idxs)

    def evaluate_hypervolume(self, mechanisms, target_curve, ref_point=None):
        """Evaluate hypervolume for a set of mechanisms."""
        if ref_point is None:
            ref_point = np.array([0.75, 10.0])

        if isinstance(mechanisms, tuple):
            x0s, edges, fixed_joints, motors, target_idxs = mechanisms
            F = np.array(self.tools(x0s, edges, fixed_joints, motors, target_curve, target_idxs)).T
        else:
            # Convert mechanisms to evaluation format
            x0s, edges, fixed_joints, motors, target_idxs = [], [], [], [], []
            for mech in mechanisms:
                x0s.append(mech['x0'])
                edges.append(mech['edges'])
                fixed_joints.append(mech['fixed_joints'])
                motors.append(mech['motor'])
                target_idxs.append(mech.get('target_joint', mech['x0'].shape[0] - 1))

            F = np.array(self.tools(x0s, edges, fixed_joints, motors, target_curve, target_idxs)).T

        ind = HV(ref_point)
        return ind(F), F

    def create_submission_from_results(self, results, problem, problem_number):
        """Create submission dictionary from optimization results."""
        submission_list = []

        if results.X is None:
            return submission_list

        if not isinstance(results.X, dict):
            for i in range(results.X.shape[0]):
                x0_member, edges, fixed_joints, motor, target_idx = \
                    problem.convert_1D_to_mech(results.X[i])

                mech = {
                    'x0': x0_member,
                    'edges': edges,
                    'fixed_joints': fixed_joints,
                    'motor': motor,
                    'target_joint': target_idx
                }
                submission_list.append(mech)
        else:
            x0_member, edges, fixed_joints, motor, target_idx = \
                problem.convert_1D_to_mech(results.X)

            mech = {
                'x0': x0_member,
                'edges': edges,
                'fixed_joints': fixed_joints,
                'motor': motor,
                'target_joint': target_idx
            }
            submission_list.append(mech)

        return submission_list

    def visualize_best_solutions(self, results, problem, target_curve, target_index=0):
        """Visualize the best solutions for distance and material objectives."""
        if results.X is None:
            print('No solutions found!')
            return

        # Best distance solution
        best_distance_idx = np.argmin(results.F[:, 0])
        best_sol_dist, edges_dist, fixed_joints_dist, motor_dist, target_idx_dist = \
            problem.convert_1D_to_mech(results.X[best_distance_idx])

        plt.figure(figsize=(16, 8))

        # Plot best distance mechanism
        plt.subplot(2, 4, 1)
        self.visualizer(best_sol_dist, edges_dist, fixed_joints_dist, motor_dist, ax=plt.gca())
        plt.title(f'Best Distance Mechanism\nDist: {results.F[best_distance_idx, 0]:.4f}')

        # Plot best distance curve
        plt.subplot(2, 4, 2)
        traced_curve_dist = self.solver(best_sol_dist, edges_dist, fixed_joints_dist, motor_dist)[target_idx_dist]
        self.curve_engine.visualize_comparison(traced_curve_dist, target_curve)
        plt.title('Best Distance Curve Match')

        # Best material solution
        best_material_idx = np.argmin(results.F[:, 1])
        best_sol_mat, edges_mat, fixed_joints_mat, motor_mat, target_idx_mat = \
            problem.convert_1D_to_mech(results.X[best_material_idx])

        # Plot best material mechanism
        plt.subplot(2, 4, 3)
        self.visualizer(best_sol_mat, edges_mat, fixed_joints_mat, motor_mat, ax=plt.gca())
        plt.title(f'Best Material Mechanism\nMat: {results.F[best_material_idx, 1]:.4f}')

        # Plot best material curve
        plt.subplot(2, 4, 4)
        traced_curve_mat = self.solver(best_sol_mat, edges_mat, fixed_joints_mat, motor_mat)[target_idx_mat]
        self.curve_engine.visualize_comparison(traced_curve_mat, target_curve)
        plt.title('Best Material Curve Match')

        # Hypervolume plot
        plt.subplot(2, 2, 3)
        ref_point = np.array([0.75, 10.0])
        self.ga_visualizer.plot_HV(results.F, ref_point, objective_labels=['Distance', 'Material'], ax=plt.gca())
        plt.title('Pareto Front')

        plt.tight_layout()
        plt.show()


class MechanismSynthesisOptimization(ElementwiseProblem):
    """
    Mixed variable optimization problem for mechanism synthesis.
    Optimizes both mechanism structure and joint positions.
    """

    def __init__(self, target_curve, N=5):
        self.N = N
        self.target_curve = target_curve

        # Global optimization tools (defined outside class due to pymoo deepcopy limitations)
        global PROBLEM_TOOLS
        if 'PROBLEM_TOOLS' not in globals():
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

        super().__init__(vars=variables, n_obj=2, n_constr=2)

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
        """Evaluate mechanism for objectives and constraints."""
        x0, edges, fixed_joints, motor, target_idx = self.convert_1D_to_mech(x)

        distance, material = PROBLEM_TOOLS(
            x0, edges, fixed_joints, motor, self.target_curve, target_idx=target_idx
        )

        out["F"] = np.array([distance, material])
        out["G"] = out["F"] - np.array([0.75, 10.0])  # Constraints


def optimize_all_targets_advanced(target_curves, N=7, pop_size=100, n_gen=50, use_gradient=True):
    """
    Run advanced optimization for all target curves.

    Args:
        target_curves: Array of target curves
        N: Number of nodes in mechanisms
        pop_size: Population size for GA
        n_gen: Number of generations
        use_gradient: Whether to apply gradient optimization

    Returns:
        Complete submission dictionary and results
    """
    optimizer = AdvancedMechanismOptimizer()
    submission = make_empty_submission()
    all_results = {}

    for i, target_curve in enumerate(target_curves):
        print(f"\nOptimizing Problem {i+1}...")

        # Generate initial mechanisms
        initial_mechs = optimizer.generate_random_mechanisms(pop_size, N)

        # Run GA optimization
        results, problem = optimizer.run_mixed_variable_optimization(
            target_curve, N, pop_size, n_gen, initial_mechs, verbose=False
        )

        if results.X is not None:
            # Apply gradient optimization if requested
            if use_gradient:
                print(f"Applying gradient optimization for Problem {i+1}...")
                grad_results, original_mechs = optimizer.apply_gradient_optimization(
                    results, problem, target_curve
                )

                if grad_results is not None:
                    # Combine GA and gradient results
                    combined_x0s = original_mechs[0] + grad_results
                    combined_edges = original_mechs[1] + original_mechs[1]
                    combined_fixed_joints = original_mechs[2] + original_mechs[2]
                    combined_motors = original_mechs[3] + original_mechs[3]
                    combined_target_idxs = original_mechs[4] + original_mechs[4]

                    combined_mechs = (combined_x0s, combined_edges, combined_fixed_joints,
                                    combined_motors, combined_target_idxs)

                    hv, F = optimizer.evaluate_hypervolume(combined_mechs, target_curve)
                    print(f"Problem {i+1} hypervolume after gradient optimization: {hv:.4f}")

            # Create submission
            submission_list = optimizer.create_submission_from_results(results, problem, i+1)
            submission[f'Problem {i+1}'] = submission_list
            all_results[f'Problem {i+1}'] = results
        else:
            print(f"No solutions found for Problem {i+1}")

    return submission, all_results


def setup_environment():
    """Setup environment for optimization."""
    os.environ["JAX_PLATFORMS"] = "cpu"
    np.random.seed(0)
    random.seed(0)


def load_target_curves(filepath='target_curves.npy'):
    """Load target curves from file."""
    return np.load(filepath)


def save_submission(submission, filepath='advanced_submission.npy'):
    """Save submission to file."""
    np.save(filepath, submission)
    return filepath


def filter_feasible_mechanisms(mechanisms, target_curve, feasibility_threshold=(0.75, 10.0)):
    """
    Filter mechanisms based on feasibility constraints.

    Args:
        mechanisms: List of mechanism dictionaries
        target_curve: Target curve to evaluate against
        feasibility_threshold: Tuple of (max_distance, max_material)

    Returns:
        List of feasible mechanisms and their performance metrics
    """
    tools = Tools(device='cpu')
    tools.compile()

    feasible_mechanisms = []
    performance_metrics = []

    for mech in tqdm(mechanisms, desc="Filtering feasible mechanisms"):
        try:
            distance, material = tools(
                mech['x0'],
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                target_idx=mech.get('target_joint', mech['x0'].shape[0] - 1)
            )

            # Check feasibility constraints
            if distance <= feasibility_threshold[0] and material <= feasibility_threshold[1]:
                feasible_mechanisms.append(mech)
                performance_metrics.append((distance, material))
        except:
            # Skip mechanisms that cause simulation errors
            continue

    return feasible_mechanisms, performance_metrics


def plot_single_curve(target_curve, curve_index=0, title="Target Curve"):
    """Plot a single target curve."""
    plt.figure(figsize=(8, 6))
    x_coords = np.array(target_curve)[:, 0]
    y_coords = np.array(target_curve)[:, 1]

    plt.plot(x_coords, y_coords, color='black', linewidth=3)
    plt.title(f'{title} {curve_index + 1}')
    plt.axis('equal')
    plt.axis('off')
    plt.show()


def run_nsga2_optimization(mechanisms, target_curve, N=7, pop_size=100, n_gen=100, verbose=True):
    """
    Run NSGA-II optimization using pre-filtered feasible mechanisms.

    Args:
        mechanisms: List of feasible mechanism dictionaries
        target_curve: Target curve to optimize for
        N: Number of nodes in mechanism
        pop_size: Population size
        n_gen: Number of generations
        verbose: Whether to show optimization progress

    Returns:
        Optimization results and problem instance
    """
    problem = MechanismSynthesisOptimization(target_curve, N)

    # Convert mechanisms to 1D representation
    initial_population = [problem.convert_mech_to_1D(**mech) for mech in mechanisms]

    # Pad population if needed
    while len(initial_population) < pop_size:
        initial_population.extend(initial_population[:min(len(initial_population), pop_size - len(initial_population))])

    initial_population = initial_population[:pop_size]

    class SampleFromFeasible(Sampling):
        def _do(self, problem, n_samples, **kwargs):
            return np.array([initial_population[i % len(initial_population)] for i in range(n_samples)])

    algorithm = NSGA2(
        pop_size=pop_size,
        sampling=SampleFromFeasible(),
        mating=MixedVariableMating(eliminate_duplicates=MixedVariableDuplicateElimination()),
        mutation=PolynomialMutation(prob=0.5),
        eliminate_duplicates=MixedVariableDuplicateElimination()
    )

    results = minimize(
        problem,
        algorithm,
        ('n_gen', n_gen),
        verbose=verbose,
        save_history=True,
        seed=123
    )

    return results, problem


def plot_hypervolume_results(results, ref_point=None, title="Optimization Results"):
    """Plot hypervolume results from optimization."""
    if results.X is None:
        print("No solutions found!")
        return

    if ref_point is None:
        ref_point = np.array([0.75, 10.0])

    ga_visualizer = GAVisualizer()

    # Calculate hypervolume
    ind = HV(ref_point)
    hypervolume = ind(results.F)

    plt.figure(figsize=(10, 6))
    ga_visualizer.plot_HV(results.F, ref_point, objective_labels=['Distance', 'Material'], ax=plt.gca())
    plt.title(f'{title}\nHypervolume: {hypervolume:.4f}')
    plt.show()

    return hypervolume


def plot_initial_distribution(mechanisms, target_curve, ref_point=None, constraints=[0.75, 10.0]):
    """Plot the distribution of initial random mechanisms."""
    tools = Tools(device='cpu')
    tools.compile()

    distances = []
    materials = []

    for mech in tqdm(mechanisms, desc="Analyzing initial population"):
        try:
            distance, material = tools(
                mech['x0'],
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                target_idx=mech.get('target_joint', mech['x0'].shape[0] - 1)
            )
            distances.append(distance)
            materials.append(material)
        except:
            continue

    if not distances:
        print("No mechanisms to plot in initial distribution.")
        return

    if ref_point is None:
        ref_point = np.array(constraints)

    plt.figure(figsize=(12, 5))

    # Scatter plot of feasible mechanisms
    plt.subplot(1, 2, 1)
    plt.scatter(distances, materials, alpha=0.6, s=30, c='blue', label='Feasible mechanisms')
    plt.axvline(x=ref_point[0], color='red', linestyle='--', alpha=0.7, label=f'Distance constraint ({ref_point[0]})')
    plt.axhline(y=ref_point[1], color='red', linestyle='--', alpha=0.7, label=f'Material constraint ({ref_point[1]})')
    plt.xlabel('Distance')
    plt.ylabel('Material')
    plt.title(f'Initial Feasible Mechanisms Distribution\n({len(distances)} mechanisms)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Histograms
    plt.subplot(2, 2, 2)
    plt.hist(distances, bins=20, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=ref_point[0], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.title('Distance Distribution')
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 2, 4)
    plt.hist(materials, bins=20, alpha=0.7, color='green', edgecolor='black')
    plt.axvline(x=ref_point[1], color='red', linestyle='--', alpha=0.7)
    plt.xlabel('Material')
    plt.ylabel('Count')
    plt.title('Material Distribution')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_initial_population(mechanisms, target_curve):
    """Analyze the initial random population before optimization."""
    print("\n=== Initial Population Analysis ===")

    tools = Tools(device='cpu')
    tools.compile()

    distances = []
    materials = []

    for mech in tqdm(mechanisms, desc="Analyzing initial population"):
        try:
            distance, material = tools(
                mech['x0'],
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                target_idx=mech.get('target_joint', mech['x0'].shape[0] - 1)
            )
            distances.append(distance)
            materials.append(material)
        except:
            continue

    if distances:
        print(f"Initial population statistics:")
        print(f"- Distance: mean={np.mean(distances):.4f}, std={np.std(distances):.4f}")
        print(f"- Distance: min={min(distances):.4f}, max={max(distances):.4f}")
        print(f"- Material: mean={np.mean(materials):.4f}, std={np.std(materials):.4f}")
        print(f"- Material: min={min(materials):.4f}, max={max(materials):.4f}")

        # Count feasible vs infeasible
        feasible_count = sum(1 for d, m in zip(distances, materials) if d <= 6 and m <= 20)
        print(f"- Feasible mechanisms: {feasible_count}/{len(distances)} ({100*feasible_count/len(distances):.1f}%)")

        # Plot initial population
        plt.figure(figsize=(12, 5))

        # Scatter plot
        plt.subplot(1, 2, 1)
        feasible_mask = [(d <= 6 and m <= 20) for d, m in zip(distances, materials)]
        infeasible_mask = [not mask for mask in feasible_mask]

        if sum(feasible_mask) > 0:
            feasible_d = [d for d, mask in zip(distances, feasible_mask) if mask]
            feasible_m = [m for m, mask in zip(materials, feasible_mask) if mask]
            plt.scatter(feasible_d, feasible_m, alpha=0.6, s=20, c='green', label=f'Feasible ({sum(feasible_mask)})')

        if sum(infeasible_mask) > 0:
            infeasible_d = [d for d, mask in zip(distances, infeasible_mask) if mask]
            infeasible_m = [m for m, mask in zip(materials, infeasible_mask) if mask]
            plt.scatter(infeasible_d, infeasible_m, alpha=0.3, s=20, c='red', label=f'Infeasible ({sum(infeasible_mask)})')

        plt.axvline(x=6, color='red', linestyle='--', label='Distance constraint (6)')
        plt.axhline(y=20.0, color='red', linestyle='--', label='Material constraint (20)')
        plt.xlabel('Distance')
        plt.ylabel('Material')
        plt.title('Initial Random Population')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Histogram of distances
        plt.subplot(2, 2, 2)
        plt.hist(distances, bins=30, alpha=0.7, color='blue', edgecolor='black')
        plt.axvline(x=6, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Distance')
        plt.ylabel('Count')
        plt.title('Distance Distribution')
        plt.grid(True, alpha=0.3)

        # Histogram of materials
        plt.subplot(2, 2, 4)
        plt.hist(materials, bins=30, alpha=0.7, color='orange', edgecolor='black')
        plt.axvline(x=20, color='red', linestyle='--', alpha=0.7)
        plt.xlabel('Material')
        plt.ylabel('Count')
        plt.title('Material Distribution')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    return distances, materials

def analyze_initial_population(mechanisms, target_curve):
    """Analyze the initial random population before optimization."""
    print("\n=== Initial Population Analysis ===")

    from LINKS.Optimization import Tools
    tools = Tools(device='cpu')
    tools.compile()

    distances = []
    materials = []

    for mech in mechanisms:
        try:
            distance, material = tools(
                mech['x0'],
                mech['edges'],
                mech['fixed_joints'],
                mech['motor'],
                target_curve,
                target_idx=mech.get('target_joint', mech['x0'].shape[0] - 1)
            )
            distances.append(distance)
            materials.append(material)
        except:
            continue

    if distances:
        print(f"Initial population statistics:")
        print(f"- Distance: mean={np.mean(distances):.4f}, std={np.std(distances):.4f}")
        print(f"- Distance: min={min(distances):.4f}, max={max(distances):.4f}")
        print(f"- Material: mean={np.mean(materials):.4f}, std={np.std(materials):.4f}")
        print(f"- Material: min={min(materials):.4f}, max={max(materials):.4f}")
    return distances, materials