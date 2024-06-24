import numpy as np


class SupremeUltraEnhancedEvolutionaryOptimizer:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and enhanced cooling parameters
        T = 1.15  # Initial higher temperature for more aggressive exploration
        T_min = 0.0005  # Lower temperature threshold for fine-tuned exploitation
        alpha = 0.92  # Cooling rate, slightly adjusted for optimal cooling balance

        # Mutation and crossover parameters further refined
        F = 0.75  # Higher mutation factor for aggressive search in early stages
        CR = 0.92  # Increased crossover probability for maintaining high genetic diversity

        population_size = 80  # Increased population size for more diverse initial solutions
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Advanced mutation dynamics with temperature and evaluation adaptive mutation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor with enhanced exploration and exploitation balance
                dynamic_F = (
                    F * np.exp(-0.11 * T) * (0.7 + 0.3 * np.sin(2 * np.pi * evaluation_count / self.budget))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Superior acceptance criterion that considers advanced thermal effects
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Non-linear adaptive cooling with enhanced periodic modulation
            adaptive_cooling = alpha - 0.012 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
