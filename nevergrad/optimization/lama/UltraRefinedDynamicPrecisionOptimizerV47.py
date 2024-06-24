import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV47:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Further refine temperature and cooling parameters
        T = 1.17  # Starting temperature, moderately high for broad early exploration
        T_min = 0.0004  # Lower minimum temperature for more sustained late-stage exploration
        alpha = 0.91  # Slower cooling rate to more extensively utilize the budget

        # Mutation and crossover parameters are finely tuned for enhanced performance
        F = 0.78  # Slightly increased mutation factor to push the boundaries of exploration
        CR = 0.89  # Slightly higher crossover probability to promote diversity

        population_size = 85  # Adjusted population size for better coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = F * (0.75 + 0.25 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhance adaptive cooling strategy with refined sinusoidal modulation
            adaptive_cooling = alpha - 0.0075 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
