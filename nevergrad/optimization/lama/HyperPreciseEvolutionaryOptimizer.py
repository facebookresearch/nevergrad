import numpy as np


class HyperPreciseEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters with refined values
        T = 1.2  # Starting temperature, slightly higher for more global search initially
        T_min = 0.0008  # Lower threshold temperature for extended fine-tuning
        alpha = 0.92  # Cooling rate, optimized for gradual reduction

        # Mutation and crossover parameters finely tuned
        F_base = 0.6  # Base mutation factor for stability
        CR = 0.85  # Crossover probability to maintain sufficient diversity

        population_size = 80  # Optimal population size considering budget and dimension
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Enhanced mutation dynamics and sophisticated temperature-dependent selection
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=True)]
                # Intricate mutation factor involving time decay and temperature influence
                dynamic_F = (
                    F_base
                    * (1 - np.exp(-0.05 * T))
                    * (0.7 + 0.3 * np.cos(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Advanced acceptance criterion that considers delta fitness and dynamic temperature
                if delta_fitness < 0 or np.random.rand() < np.exp(-delta_fitness / T):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy with a touch of non-linear modulation
            adaptive_cooling = alpha + 0.02 * np.cos(2 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
