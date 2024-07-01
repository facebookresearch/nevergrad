import numpy as np


class RefinedPrecisionEvolutionaryThermalOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Enhanced initial temperature and adjusted cooling rate for improved exploration-exploitation balance
        T = 1.5
        T_min = 0.001
        alpha = 0.98

        # Optimal mutation and crossover parameters derived from prior performance analysis
        F = 0.75  # Dynamically adjusted mutation factor
        CR = 0.88  # Crossover probability to maintain genetic diversity

        population_size = 80  # Adjusted population size for better initial search scope
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce dynamic mutation and sophisticated simulated annealing acceptance conditions
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * (0.5 + 0.5 * np.sin(np.pi * T)) * (0.6 + 0.4 * (evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion considering the magnitude of fitness improvements
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.log(1 + np.abs(delta_fitness))))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling rate based on the optimization stage
            adaptive_cooling = alpha - 0.015 * (evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
