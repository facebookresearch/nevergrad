import numpy as np


class EnhancedOptimalPrecisionEvolutionaryThermalOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Adjust initial temperature and refinement of cooling and exploitation strategies
        T = 1.0  # Reduced initial temperature to prevent premature convergence
        T_min = 0.001  # Lower minimum temperature for prolonged fine-tuning
        alpha = 0.98  # Slower cooling rate to extend the exploration phase significantly

        # Mutation and crossover parameters optimized further
        F = 0.75  # Mutation factor adjusted for deeper exploration
        CR = 0.88  # Adjusted crossover probability to increase genetic diversity

        population_size = 80  # Adjusted population size for better initial space exploration
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhanced mutation factor adjustments and more responsive annealing acceptance conditions
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation factor influenced by temperature and linear progress
                dynamic_F = F * np.exp(-T) * (0.75 + 0.25 * np.cos(np.pi * evaluation_count / self.budget))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion incorporating a more responsive strategy to temperature and fitness changes
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Progressive and adaptive cooling strategy that adjusts more dynamically based on optimization progress
            adaptive_cooling = alpha - 0.01 * np.sin(np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
