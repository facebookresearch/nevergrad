import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV25:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling settings
        T = 1.20  # Initial temperature, increased for broader initial search
        T_min = 0.0003  # Minimum temperature, lowered to facilitate deeper search in late stages
        alpha = 0.90  # Cooling rate, optimized to balance search duration and depth

        # Mutation and crossover parameters optimized
        F = 0.78  # Mutation factor, adjusted for optimal balance
        CR = 0.85  # Crossover probability, tweaked to enhance solution diversity

        population_size = 85  # Population size, adjusted for efficient search space coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation with enhanced sigmoid and exponential modulation for mutation strength
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Advanced dynamic mutation factor incorporating temperature and progress
                dynamic_F = (
                    F
                    * (1 - np.exp(-0.1 * T))
                    * (0.6 + 0.4 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria incorporating more sensitive exploitation capability
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Introduction of a sinusoidal term to the cooling schedule for dynamic modulation
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
