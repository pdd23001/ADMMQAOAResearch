from docplex.mp.model import Model
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from .minimum_eigen_optimizer import MinimumEigenOptimizer
from .admm_optimizer import ADMMParameters, ADMMOptimizer
from ..translators import from_docplex_mp
import networkx as nx
from .cobyla_optimizer import CobylaOptimizer


cobyla = CobylaOptimizer()
qaoa = MinimumEigenOptimizer(min_eigen_solver=QAOA(sampler=Sampler(), optimizer=COBYLA()))

G = nx.Graph()
G.add_edges_from([(2, 3), (2, 5), (2, 1), (1, 4), (1, 5), (3, 4), (3, 6), (3, 5), (4, 5), (4, 6), (5, 6)])
#G.add_edges_from([(1, 2), (1, 3), (1, 4), (3, 2), (3, 4), (3, 5), (3, 13), (2, 5), (2, 14), (2, 13), (4, 5), (4, 6), (4, 8), (4, 9), (5, 6), (5, 7), (5, 13), (5, 12), (6, 7), (6, 8), (6, 10), (6, 11), (8, 7), (8, 10), (8, 11), (7, 11), (7, 10), (7, 12), (10, 11), (9, 8), (9, 10), (10, 21), (13, 12), (13, 14), (13, 16), (12, 11), (12, 16), (12, 15), (11, 21), (11, 18), (11, 15), (14, 16), (15, 16), (15, 17), (15, 18), (15, 19), (18, 19), (18, 20), (18, 21), (16, 17), (17, 19), (19, 20), (20, 21)])

G_complement = nx.complement(G)
mdl = Model("MaxClique Problem-3")
x = {i: mdl.binary_var(name=f"x{i}") for i in G.nodes}
mdl.maximize(mdl.sum(x[i] for i in G.nodes))
for u, v in G_complement.edges:
    mdl.add_constraint(x[u] + x[v] <= 1)

qp = from_docplex_mp(mdl)
print("QUBO Problem:")
print(qp.prettyprint())

admm_params = ADMMParameters(rho_initial=1, beta=10, factor_c=1, maxiter=50, three_block=False, tol=1.0e-4)
admm = ADMMOptimizer(params=admm_params, qubo_optimizer=qaoa, continuous_optimizer=CobylaOptimizer())

result = admm.solve(qp)
print("Optimization Result:")
print(result.prettyprint())
