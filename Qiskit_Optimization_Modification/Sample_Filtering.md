1. optimization_algorithm's fxn _interpret_samples returns sorted_samples, best_raw (tuple)

2. optimization_algorithm's fxn _interpret returns 
        result_class(
            x=x,
            fval=problem.objective.evaluate(x),
            variables=problem.variables,
            status=cls._get_feasibility_status(problem, x),
            **kwargs,
        )

3. minimum_eigen_optimizer's solve_internal method 
   assigns samples, best_raw to the result of _interpret_samples

4. minimum_eigen_optimizer's solve internal returns 
     return cast(
            MinimumEigenOptimizationResult,
            self._interpret(
                x=best_raw.x,
                converters=self._converters,
                problem=original_problem,
                result_class=MinimumEigenOptimizationResult,
                samples=samples,
                raw_samples=raw_samples,
                min_eigen_solver_result=eigen_result,
            ),
        )

    which basically casts  the result of _interpret to MinimumEigenOptimizer Result

5. solve_internal used in actual solve method of solve of the min eigen optimizer

6. _update_x0 uses the solve method in the following way
x0_all_binaries = np.zeros(len(self._state.binary_indices))
        x0_qubo = np.asarray(self._qubo_optimizer.solve(op1).x)
        x0_all_binaries[self._state.step1_relative_indices] = x0_qubo
        return x0_all_binaries

