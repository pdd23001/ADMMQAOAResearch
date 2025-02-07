import networkx as nx
import numpy as np
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler

# Define Graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 2), (3, 4), (3, 5), (3, 13), (2, 5), (2, 14), (2, 13), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), (5, 13), (5, 12), (6, 7), (6, 8), (6, 10), (6, 11), (8, 7), (8, 10), (8, 11), (7, 11), (7, 10), (7, 12), (10, 11), (9, 8), (9, 10), (10, 21), (13, 12), (13, 14), (13, 16), (12, 11), (12, 16), (12, 15), (11, 21), (11, 18), (11, 15), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (18, 19), (18, 20), (18, 21), (16, 17), (17, 19), (19, 20), (20, 21)])


# Create QUBO Model
qubo = QuadraticProgram()
n = G.number_of_nodes()

# Add binary decision variables x_i
for i in G.nodes:
    qubo.binary_var(f"x{i}")

# Objective Function: Maximize clique size
linear_terms = {f"x{i}": -1 for i in G.nodes}  # -sum(x_i)

# Clique Constraints: Penalize non-edges
quad_terms = {}
penalty = 5  # Adjust penalty coefficient if needed
G_complement = nx.complement(G)

for u, v in G_complement.edges:
    quad_terms[(f"x{u}", f"x{v}")] = penalty  # x_u * x_v penalty for non-edges

# Set objective function
qubo.minimize(linear=linear_terms, quadratic=quad_terms)

# Convert to QUBO format
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qubo)

# Define QAOA Solver
qaoa_solver = MinimumEigenOptimizer(QAOA(sampler=Sampler(), optimizer=SPSA()))

# Solve using QAOA
qaoa_result = qaoa_solver.solve(qubo)

# Print solution
print("QAOA Optimization Result:")
print(qaoa_result)
