import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV24:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initiate temperature and cooling parameters
        T = 1.18  # Slightly increased starting temperature for more robust initial exploration
        T_min = 0.0004  # Lower minimum temperature to allow more thorough late-stage search
        alpha = 0.91  # Slower cooling rate to prolong the search phase

        # Mutation and crossover parameters are finely tuned
        F = 0.77  # Adjusted Mutation factor for a balance between exploration and exploitation
        CR = 0.89  # Modified Crossover probability to encourage better genetic diversity

        population_size = 83  # Adjusted population size to optimize exploration-exploitation balance
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with enhanced sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor dynamically adapts with an advanced sigmoid function for refined control
                dynamic_F = F * (0.75 + 0.25 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria incorporate a temperature-sensitive function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy with dynamic modulation
            adaptive_cooling = alpha - 0.008 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
