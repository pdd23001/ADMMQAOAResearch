
from docplex.mp.model import Model
from qiskit_algorithms import QAOA,NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.translators import from_docplex_mp
import networkx as nx
import numpy as np

class ConstraintAwareSampler(Sampler):
    def __init__(self, graph_complement, objective_function, options=None):
        """
        Custom sampler to sample only feasible solutions, prioritizing the best-cost solutions.
        
        Args:
            graph_complement: The complement graph of the original problem graph.
            objective_function: A callable function that evaluates the cost of a solution.
            options: Default options.
        """
        super().__init__(options=options)
        self.graph_complement = graph_complement
        self.objective_function = objective_function  # Used to evaluate solution costs

    def _is_feasible(self, solution):
        """
        Check if a solution satisfies the constraints of the complement graph.
        
        Args:
            solution: A binary string representing a potential solution.

        Returns:
            True if the solution satisfies the constraints, False otherwise.
        """
        for u, v in self.graph_complement.edges:
            if int(solution[u - 1]) + int(solution[v - 1]) > 1:  # Adjust indexing if necessary
                return False
        return True

    def _filter_and_score_solutions(self, prob_dict):
        """
        Filter solutions for feasibility and prioritize those with better costs.

        Args:
            prob_dict: Dictionary of solution probabilities.

        Returns:
            Dictionary of feasible solutions with probabilities adjusted by their costs.
        """
        scored_prob_dict = {}
        for solution, prob in prob_dict.items():
            if self._is_feasible(solution):
                # Evaluate the cost using the objective function
                cost = self.objective_function(solution)
                scored_prob_dict[solution] = prob * cost  # Scale probability by cost

        total_score = sum(scored_prob_dict.values())
        if total_score == 0:
            raise ValueError("No feasible solutions found.")

        # Normalize the scored probabilities
        return {key: score / total_score for key, score in scored_prob_dict.items()}

    def _run(self, circuits, parameter_values, shots=1024, seed=None):
        """
        Sample only feasible solutions, prioritizing those with the best costs.

        Args:
            circuits: Sequence of circuit indices.
            parameter_values: Sequence of parameter values for each circuit.
            shots: Number of shots to sample.
            seed: Random seed for reproducibility.

        Returns:
            Dictionary of feasible solutions and their probabilities.
        """
        rng = np.random.default_rng(seed)
        sampler_result = super()._run(circuits, parameter_values, shots=shots).result()

        feasible_results = []
        for quasi_dist in sampler_result.quasi_dists:
            # Filter and score feasible solutions
            scored_prob_dict = self._filter_and_score_solutions(quasi_dist.binary_probabilities())
            
            # Resample based on scored probabilities
            feasible_counts = rng.multinomial(shots, list(scored_prob_dict.values()))
            feasible_results.append({
                solution: count / shots
                for solution, count in zip(scored_prob_dict.keys(), feasible_counts)
                if count > 0
            })

        return feasible_results

cobyla = CobylaOptimizer()


G = nx.Graph()
G.add_edges_from([(2, 3), (2, 5), (2, 1), (1, 4), (1, 5), (3, 4), (3, 6), (3, 5), (4, 5), (4, 6), (5, 6)])
#G.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 2), (3, 4), (3, 5), (3, 13), (2, 5), (2, 14), (2, 13), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), (5, 13), (5, 12), (6, 7), (6, 8), (6, 10), (6, 11), (8, 7), (8, 10), (8, 11), (7, 11), (7, 10), (7, 12), (10, 11), (9, 8), (9, 10), (10, 21), (13, 12), (13, 14), (13, 16), (12, 11), (12, 16), (12, 15), (11, 21), (11, 18), (11, 15), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (18, 19), (18, 20), (18, 21), (16, 17), (17, 19), (19, 20), (20, 21)])

G_complement = nx.complement(G)

mdl = Model("MaxClique Problem-3")
x = {i: mdl.binary_var(name=f"x{i}") for i in G.nodes}
mdl.maximize(mdl.sum(x[i] for i in G.nodes))

for u, v in G_complement.edges:
    mdl.add_constraint(x[u] + x[v] <= 1)

qaoa = MinimumEigenOptimizer(QAOA(sampler=ConstraintAwareSampler(graph_complement=G_complement, objective_function= mdl.sum(x[i] for i in G.nodes)), optimizer=COBYLA()))
qp = from_docplex_mp(mdl)
print("QUBO Problem:")
print(qp.prettyprint())
qubo_optimizer= qaoa

admm_params = ADMMParameters(rho_initial=1, beta=10, factor_c=1, maxiter=50, three_block=True, tol=1.0e-4)
admm = ADMMOptimizer(params=admm_params, qubo_optimizer=qaoa, continuous_optimizer=CobylaOptimizer())


result = admm.solve(qp)
print("Optimization Result:")
print(result.prettyprint())




