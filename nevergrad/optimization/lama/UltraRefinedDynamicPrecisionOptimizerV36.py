import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV36:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 per problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and advanced cooling parameters
        T = 1.1  # More aggressive starting temperature for broader initial search
        T_min = 0.0001  # Lower minimum temperature to allow finer search towards the end
        alpha = 0.93  # Slower cooling rate to enhance thorough exploration across phases

        # Mutation and crossover parameters are fine-tuned for better adaptability
        F = 0.78  # Mutation factor adjusted for a strong yet controlled explorative push
        CR = 0.85  # Crossover probability adjusted to maintain diversity

        population_size = 85  # Optimized population size to ensure effective coverage and performance
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing a dynamic mutation approach with exponential and sigmoid adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive mutation factor that changes based on the temperature and search progress
                dynamic_F = (
                    F * np.exp(-0.06 * T) * (0.6 + 0.4 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria using a temperature-modulated probabilistic approach
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Cooling strategy with modified sinusoidal modulation for better temperature control
            adaptive_cooling = alpha - 0.007 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
