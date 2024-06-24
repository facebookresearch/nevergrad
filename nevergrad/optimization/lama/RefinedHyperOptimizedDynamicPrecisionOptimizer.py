import numpy as np


class RefinedHyperOptimizedDynamicPrecisionOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound
        self.ub = 5.0  # Upper bound

    def __call__(self, func):
        # Initialize advanced temperature and cooling parameters
        T = 1.3  # Slightly higher initial temperature for more aggressive early exploration
        T_min = 0.0001  # Lower minimum temperature for fine-grained search near the end
        alpha = 0.90  # Slower cooling rate to maintain exploration capabilities longer

        population_size = 100  # Increased population size for better coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Adaptive mutation strategy and temperate-dependent dynamic acceptance criteria
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutant vector calculation with dynamic mutation factor based on T and evaluations
                dynamic_F = (
                    0.85
                    * np.exp(-0.12 * T)
                    * (0.65 + 0.35 * np.sin(1.5 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                CR = 0.80 + 0.15 * np.cos(
                    1.8 * np.pi * evaluation_count / self.budget
                )  # Dynamic crossover probability
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Modified acceptance criterion, more reactive at lower temperatures with enhanced exploration
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.06 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cooling strategy: include a modulated dynamic non-linear adjustment based on the search progress
            adaptive_cooling = alpha - 0.02 * np.cos(2.0 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
