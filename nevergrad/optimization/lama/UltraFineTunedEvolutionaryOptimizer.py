import numpy as np


class UltraFineTunedEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature parameters for sophisticated annealing
        T_initial = 1.2  # Starting temperature slightly increased for broader initial exploration
        T = T_initial
        T_min = 0.0005  # Lower minimum temperature for extended fine-tuning phases
        alpha = 0.95  # Slower cooling rate to extend exploration phases at each temperature step

        # Optimized mutation and crossover parameters
        F_initial = 0.8  # Higher initial mutation factor to enhance initial global search
        CR = 0.85  # Slightly reduced crossover probability to improve offspring quality

        population_size = 90  # Increased population size for enhanced diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhanced mutation dynamics with adaptive mutation factor
        while evaluation_count < self.budget and T > T_min:
            F = F_initial * np.exp(
                -0.2 * (T_initial - T)
            )  # Adaptive mutation factor decreases with temperature
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion with modified temperature impact
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.03 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling strategy with more periodic modulation
            adaptive_cooling = alpha - 0.015 * np.cos(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
