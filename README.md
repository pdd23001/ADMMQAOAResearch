# ADMMQAOAResearch

This is my code for solving max-clique graph problems as mixed binary optimizations using ADMM formulations and QAOA.
Currently dealing with 2 graphs of size 6 nodes and 21 nodes. Using the MBO approach 6 node gives feasible solutions but 21 node does not.
I have changed the default sampling strategy in qiskit and changed self.update_x0 function which updates the parameters QUBO block of the ADMM (Qiskit_Optimization_Modification/admm_optimizer.py) such that it checks for feasibility of the solution to th QUBO subproblem generated by QAOA while also minimizing cost of the overall objective function when updating parameters at every ADMM iteration. 

I have solved the 21 node using the pure QUBO solution approach instead of MBO (PossibleSolutions/pure_admm.py). However even after updatations in qiskit source code mentioned above a timeout error seems to occur with QAOA and mixed solutions which I am working on currently with my professor. I apologize I have not documented code very well in this repo since I would be formatting the entire repo at the end of this project. 

I highly recommend having knowledge of both ADMM and VQAs to better undersatnd the code. Qiskit tutorials at https://qiskit-community.github.io/qiskit-optimization/tutorials/05_admm_optimizer.html and https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.QAOA.html will help for sure!

Thanks so much for looking at my repo!
