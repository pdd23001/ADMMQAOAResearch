import networkx as nx
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import MinimumEigenOptimizer, ADMMOptimizer, ADMMParameters
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import SPSA
from qiskit.primitives import Sampler

# Define Graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 2), (3, 4), (3, 5), (3, 13), (2, 5), (2, 14), (2, 13), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), (5, 13), (5, 12), (6, 7), (6, 8), (6, 10), (6, 11), (8, 7), (8, 10), (8, 11), (7, 11), (7, 10), (7, 12), (10, 11), (9, 8), (9, 10), (10, 21), (13, 12), (13, 14), (13, 16), (12, 11), (12, 16), (12, 15), (11, 21), (11, 18), (11, 15), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (18, 19), (18, 20), (18, 21), (16, 17), (17, 19), (19, 20), (20, 21)])

from qiskit.primitives import SamplerResult



class feasible_sampler(Sampler):
    def _call(self,circuits, parameter_values,**run_options):
        a=super()._call(circuits, parameter_values, **run_options)
        feasible_quasi_dists=[]
        quasi_dists=a.quasi_dists
        for i in quasi_dists:
            feasible_quasi_dists.append({k:v for k,v in i.items() if k.is_feasible()})

        return SamplerResult(quasi_dists=feasible_quasi_dists, metadata=a.metadata)  
    
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
penalty = 1.5  # Reduced penalty
G_complement = nx.complement(G)

for u, v in G_complement.edges:
    quad_terms[(f"x{u}", f"x{v}")] = penalty  # x_u * x_v penalty for non-edges

# Set objective function
qubo.minimize(linear=linear_terms, quadratic=quad_terms)

# Convert to QUBO format
qubo_converter = QuadraticProgramToQubo()
qubo = qubo_converter.convert(qubo)

# ========== Define Improved QAOA ==========
qaoa = MinimumEigenOptimizer(QAOA(sampler=feasible_sampler(), optimizer=SPSA(), reps=3))

# ========== Solve using ADMM with QAOA ==========
admm_params = ADMMParameters(
    rho_initial=0.5,  # Reduced penalty strength
    beta=5,           # Reduced beta (to avoid overpowering QAOA)
    factor_c=1,
    maxiter=1,      # More iterations
    three_block=True, # Improve handling of discrete variables
    tol=1e-5          # Force smaller updates
)

admm_qaoa_solver = ADMMOptimizer(params=admm_params, qubo_optimizer=qaoa)

admm_qaoa_result = admm_qaoa_solver.solve(qubo)

print("\n===== ADMM + QAOA Optimization Result =====")
print(admm_qaoa_result.prettyprint())

