Notes on 21 node graph:


Week 2-9 Feb
1. Sampler Returns an object called SamplerResult which contains 2 parameters quasi_dists and metadata(not imp)
quasi_dists is a dictionary consisting of states as keys and quasi probs as values

2. Problem easily solvable using pure ADMM, but not with a mix of QAOA and ADMM or pure QAOA (Problem in QAOA?)

3. Zakir's code couldn't solve it using pure ADMM giving wrong answer 21.0



Week 10-17 Feb

1. Problem with QAOA
2. Modify in optimization_algorithm.py so that _interpret_samples changed to for feasability check. Then this is used in _solve_internal  