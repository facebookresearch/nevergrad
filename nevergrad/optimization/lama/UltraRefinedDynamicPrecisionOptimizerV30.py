import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV30:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.20  # Adjusted starting temperature for extended early exploration
        T_min = 0.0004  # Adjusted lower minimum temperature for deeper late-stage exploration
        alpha = 0.93  # Adjusted cooling rate to extend the search phase more effectively

        # Mutation and crossover parameters are finely tuned
        F = 0.77  # Adjusted Mutation factor for a balance between exploration and exploitation
        CR = 0.88  # Modified Crossover probability to maintain genetic diversity

        population_size = 85  # Adjusted population size to optimize individual evaluations
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation approach with sigmoid-based and exponential adaptations
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adapts combining sigmoid and exponential decay
                dynamic_F = (
                    F
                    * np.exp(-0.07 * T)
                    * (0.75 + 0.25 * np.tanh(3 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # More aggressive acceptance criteria incorporating a temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy using a sinusoidal modulation
            adaptive_cooling = alpha - 0.006 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
