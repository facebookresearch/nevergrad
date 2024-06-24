import numpy as np


class RefinedHyperRefinedDynamicPrecisionOptimizerV50:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of 5
        self.lb = -5.0  # Lower bound is -5.0
        self.ub = 5.0  # Upper bound is 5.0

    def __call__(self, func):
        # Initiate temperature and cooling parameters with refined values
        T = 1.15  # Slightly increased starting temperature for better early exploration
        T_min = 0.0005  # Lower minimal temperature for deep late-stage search
        alpha = 0.92  # Slow cooling rate to enhance search persistence

        # Mutation and crossover parameters finely-tuned for this problem set
        F = 0.75  # Mutation factor adjusted for a good balance between exploration and exploitation
        CR = 0.87  # Crossover probability adjusted to maintain genetic diversity

        population_size = 80  # Population size optimized for individual evaluations
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation approach with sigmoid adaptation for mutation factor
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F * np.exp(-0.07 * T) * (0.7 + 0.3 * np.tanh(3 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Acceptance criteria with refined temperature dependence
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / T * 1.05
                ):  # Adjusted acceptance probability
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy incorporating sinusoidal modulation
            adaptive_cooling = alpha - 0.008 * np.cos(2.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
