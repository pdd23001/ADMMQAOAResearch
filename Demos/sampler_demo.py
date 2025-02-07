from qiskit.primitives import Sampler
from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes


qc=QuantumCircuit(2)
qc.h(0)
qc.cx(0,1)
qc.measure_all()

# two parameterized circuits
pqc = RealAmplitudes(num_qubits=2, reps=2)
pqc.measure_all()
pqc2 = RealAmplitudes(num_qubits=2, reps=3)
pqc2.measure_all()
 
theta1 = [0, 1, 1, 2, 3, 5]
theta2 = [0, 1, 2, 3, 4, 5, 6, 7]
 
# initialization of the sampler
sampler = Sampler()
 
# Sampler runs a job on the Bell circuit
job = sampler.run(circuits=[qc], parameter_values=[[]], parameters=[[]])
job_result = job.result()
print(job_result)
 
# Sampler runs a job on the parameterized circuits
job2 = sampler.run(
    circuits=[pqc],
    parameter_values=[theta1],
    parameters=[pqc.parameters])
job_result = job2.result()
print(job_result)