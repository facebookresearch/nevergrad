import numpy as np


class UltraEnhancedPrecisionEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality
        self.lb = -5.0  # Lower boundary of the search space
        self.ub = 5.0  # Upper boundary of the search space

    def __call__(self, func):
        # Updated thermal dynamics for enhanced exploration and exploitation balance
        T = 1.05  # Initial temperature for broader initial exploration
        T_min = 0.0005  # Minimum temperature for detailed exploitation
        alpha = 0.95  # Cooling rate, optimized for gradual precision enhancement

        # Improved mutation and crossover dynamics
        F = 0.8  # Enhanced mutation factor for aggressive diversification early on
        CR = 0.88  # Crossover probability for maintaining genetic diversity

        population_size = 70  # Optimized population size for budget efficiency
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced evolutionary dynamics with adaptive mutation control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Temperature and evaluation adaptive mutation factor
                dynamic_F = (
                    F * np.exp(-0.09 * T) * (0.65 + 0.35 * np.sin(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criterion that considers delta fitness, temperature, and progress
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.sin(evaluation_count / self.budget)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling strategy
            adaptive_cooling = alpha - 0.015 * np.cos(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
