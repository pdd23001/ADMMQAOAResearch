from docplex.mp.model import Model
from qiskit_algorithms.minimum_eigensolvers import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_optimization.algorithms import CobylaOptimizer, MinimumEigenOptimizer
from qiskit_optimization.algorithms.admm_optimizer import ADMMParameters, ADMMOptimizer
from qiskit_optimization.translators import from_docplex_mp
import networkx as nx
from qiskit import QuantumCircuit



class feasible_sample(Sampler):
    def _call(self,circuits, parameter_values,**run_options):
        a=super._call(circuits, parameter_values, **run_options)

        