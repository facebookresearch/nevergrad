import numpy as np


class HyperOptimizedDynamicPrecisionOptimizerV12:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.20  # Increased starting temperature for broader initial exploration
        T_min = 0.0003  # Reduced minimum temperature for exhaustive end-stage exploration
        alpha = 0.93  # Moderately slow cooling rate to extend exploration duration

        # Mutation and crossover parameters are optimized for exploration and exploitation balance
        F = 0.7  # Decreased Mutation factor to promote more stable search
        CR = 0.85  # Slightly reduced Crossover probability to preserve individual traits longer

        population_size = 85  # Adjusted population size for efficient evaluation coverage
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing a dynamic mutation strategy with a more reactive sigmoid-based modulation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Introduce a more sensitive dynamic mutation factor
                dynamic_F = F * (0.8 + 0.2 * np.tanh(5 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria with adjusted temperature sensitivity
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with sinusoidal amplitude modulation
            adaptive_cooling = alpha - 0.009 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
