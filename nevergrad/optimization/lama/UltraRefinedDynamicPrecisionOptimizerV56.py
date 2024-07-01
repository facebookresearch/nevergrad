import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV56:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.2  # Enhanced starting temperature for more aggressive exploration
        T_min = 0.0003  # Lower minimum temperature for deeper late-stage exploration
        alpha = 0.90  # Slower cooling rate to extend the effective search phase

        # Mutation and crossover parameters are finely tuned
        F = 0.8  # Dynamic mutation factor for a better balance between exploration and exploitation
        CR = 0.85  # Crossover probability to ensure genetic diversity

        population_size = 85  # A slightly larger population size for better search space coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with enhanced adaptive control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation adapts more responsively near critical phases of the search
                dynamic_F = F / (1 + np.exp(-12 * (evaluation_count / self.budget - 0.5))) + 0.05
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with improved temperature-based adjustments
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adjusted cooling strategy with more refined control based on search progress
            adaptive_cooling = alpha - 0.01 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
