import numpy as np
from scipy.optimize import minimize


class EnhancedAdaptiveQuantumPSO:
    def __init__(self, budget=10000, population_size=50):
        self.budget = budget
        self.population_size = population_size
        self.dim = 5
        self.bounds = (-5.0, 5.0)
        self.inertia_weight = 0.729
        self.cognitive_weight = 1.49445
        self.social_weight = 1.49445
        self.quantum_weight = 0.2
        self.elite_fraction = 0.25
        self.memory_size = 5  # Memory size for tracking performance
        self.local_search_probability = 0.3  # Probability of local search
        self.convergence_threshold = 1e-6  # Convergence threshold for local search
        self.stagnation_threshold = 10  # No improvement iterations before triggering local search

    def __call__(self, func):
        def evaluate(individual):
            return func(np.clip(individual, self.bounds[0], self.bounds[1]))

        # Initialize population and velocities
        population = np.random.uniform(self.bounds[0], self.bounds[1], (self.population_size, self.dim))
        fitness = np.array([evaluate(ind) for ind in population])
        eval_count = self.population_size

        velocities = np.zeros((self.population_size, self.dim))
        personal_best_positions = np.copy(population)
        personal_best_scores = np.copy(fitness)

        best_individual = population[np.argmin(fitness)]
        best_fitness = np.min(fitness)
        performance_memory = [best_fitness] * self.memory_size
        adaptive_factor = 1.0
        no_improvement_count = 0

        while eval_count < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                velocities[i] = (
                    self.inertia_weight * velocities[i]
                    + self.cognitive_weight * r1 * (personal_best_positions[i] - population[i])
                    + self.social_weight * r2 * (best_individual - population[i])
                )

                # Quantum behavior with adaptive step size
                if np.random.rand() < self.quantum_weight:
                    quantum_step = np.random.normal(0, 1, self.dim)
                    step_size = np.linalg.norm(velocities[i])
                    population[i] = best_individual + step_size * quantum_step
                else:
                    population[i] = np.clip(population[i] + velocities[i], self.bounds[0], self.bounds[1])

                trial_fitness = evaluate(population[i])
                eval_count += 1

                if trial_fitness < fitness[i]:
                    fitness[i] = trial_fitness
                    personal_best_positions[i] = population[i]
                    personal_best_scores[i] = trial_fitness

                    if trial_fitness < best_fitness:
                        best_individual = population[i]
                        best_fitness = trial_fitness
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                else:
                    no_improvement_count += 1

                if eval_count >= self.budget:
                    break

            performance_memory.append(best_fitness)
            if len(performance_memory) > self.memory_size:
                performance_memory.pop(0)

            mean_recent_performance = np.mean(performance_memory)
            if best_fitness < mean_recent_performance * 0.95:
                adaptive_factor *= 0.9
                self.quantum_weight = min(1.0, self.quantum_weight * adaptive_factor)
            else:
                adaptive_factor *= 1.1
                self.quantum_weight = max(0.0, self.quantum_weight * adaptive_factor)

            # Trigger local search after a certain number of iterations without improvement
            if no_improvement_count >= self.stagnation_threshold:
                elite_indices = np.argsort(fitness)[: int(self.population_size * self.elite_fraction)]
                for idx in elite_indices:
                    res = self.local_search(func, population[idx])
                    eval_count += res[2]["nit"]
                    if res[1] < fitness[idx]:
                        population[idx] = res[0]
                        fitness[idx] = res[1]
                        personal_best_positions[idx] = res[0]
                        personal_best_scores[idx] = res[1]
                        if res[1] < best_fitness:
                            best_individual = res[0]
                            best_fitness = res[1]
                            no_improvement_count = 0  # Reset the counter on improvement

                    if eval_count >= self.budget:
                        break

                # Reset no improvement count after local search
                no_improvement_count = 0

        self.f_opt = best_fitness
        self.x_opt = best_individual
        return self.f_opt, self.x_opt

    def local_search(self, func, x_start, tol=1e-6, max_iter=50):
        res = minimize(
            func,
            x_start,
            method="L-BFGS-B",
            bounds=[self.bounds] * self.dim,
            tol=tol,
            options={"maxiter": max_iter, "ftol": self.convergence_threshold},
        )
        return res.x, res.fun, res


# Example usage
# optimizer = EnhancedAdaptiveQuantumPSO(budget=10000)
# best_fitness, best_solution = optimizer(some_black_box_function)
