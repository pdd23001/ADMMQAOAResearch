# -*- coding: utf-8 -*-
"""
Modified on Jan 26 2025

@author: Parth Danve
"""
from docplex.mp.model import Model
from qiskit_algorithms import QAOA, NumPyMinimumEigensolver
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.translators import from_docplex_mp
import networkx as nx
import numpy as np

# Define graph
G = nx.Graph()
G.add_edges_from([
    (1, 2), (1, 3), (1, 4), (3, 2), (3, 4), (3, 5), (3, 13), (2, 5), 
    (2, 14), (2, 13), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), 
    (5, 13), (5, 12), (6, 7), (6, 8), (6, 10), (6, 11), (8, 7), (8, 10), 
    (8, 11), (7, 11), (7, 10), (7, 12), (10, 11), (9, 8), (9, 10), 
    (10, 21), (13, 12), (13, 14), (13, 16), (12, 11), (12, 16), (12, 15), 
    (11, 21), (11, 18), (11, 15), (14, 16), (15, 16), (15, 17), (15, 18), 
    (15, 19), (18, 19), (18, 20), (18, 21), (16, 17), (17, 19), (19, 20), (20, 21)
])

G_complement = nx.complement(G)

# Create MaxClique model
mdl = Model("MaxClique Problem-3")
x = {i: mdl.binary_var(name=f"x{i}") for i in G.nodes}
mdl.maximize(mdl.sum(x[i] for i in G.nodes))
for u, v in G_complement.edges:
    mdl.add_constraint(x[u] + x[v] <= 1)

# Convert to QUBO
qp = from_docplex_mp(mdl)

# Define custom sampler for QAOA
class FeasibleSampler(Sampler):
    def __init__(self, qp, graph_complement):
        super().__init__()
        self.qp = qp
        self.graph_complement = graph_complement
    
    def sample(self, parameters):
        # Simulate sampling from the QUBO problem (placeholder for QAOA execution)
        solutions = np.random.randint(2, size=(100, len(self.qp.variables)))
        feasible_solutions = []
        
        # Filter feasible solutions based on constraints
        for sol in solutions:
            feasible = True
            for u, v in self.graph_complement.edges:
                if sol[u - 1] + sol[v - 1] > 1:  # Constraint violation
                    feasible = False
                    break
            if feasible:
                feasible_solutions.append(sol)
        
        # Evaluate costs of feasible solutions
        feasible_costs = [self.qp.objective.evaluate(sol) for sol in feasible_solutions]
        
        # Return the best feasible solution
        if feasible_solutions:
            best_solution = feasible_solutions[np.argmax(feasible_costs)]
            return {"solution": best_solution, "cost": max(feasible_costs)}
        else:
            return {"solution": None, "cost": None}

# Initialize QAOA with the FeasibleSampler
feasible_sampler = FeasibleSampler(qp, G_complement)
qaoa = MinimumEigenOptimizer(QAOA(sampler=feasible_sampler, optimizer=COBYLA()))

# Solve the problem using ADMM
admm_params = ADMMParameters(rho_initial=1, beta=10, factor_c=1, maxiter=50, three_block=False, tol=1.0e-4)
admm = ADMMOptimizer(params=admm_params, continuous_optimizer=MinimumEigenOptimizer(NumPyMinimumEigensolver()))

result = admm.solve(qp)

print("QUBO Problem:")
print(qp.prettyprint())

print("Optimization Result:")
print(result.prettyprint())
