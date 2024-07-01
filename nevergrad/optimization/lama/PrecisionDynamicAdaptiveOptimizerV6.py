import numpy as np


class PrecisionDynamicAdaptiveOptimizerV6:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed to 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters
        T = 1.1  # Initial temperature, slightly tuned for a better balance between exploration and exploitation
        T_min = 0.0003  # Reduced minimum temperature for deep late-stage exploration
        alpha = 0.93  # Adjusted cooling rate to prolong effective search duration

        # Mutation and crossover parameters are finely adjusted
        F = 0.78  # Tuned Mutation factor for a subtle balance
        CR = 0.85  # Adjusted Crossover probability to optimize genetic diversity

        # Setting a slightly increased population size to improve sampling
        population_size = 85
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing dynamic mutation with a refined sigmoid-based adaptation
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive mutation factor with refined control
                dynamic_F = (
                    F
                    * np.exp(-0.08 * T)
                    * (0.75 + 0.25 * np.tanh(3.5 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # More sensitive acceptance criteria considering a refined temperature influence
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Enhanced adaptive cooling strategy with a cosine modulation slightly adjusted
            adaptive_cooling = alpha - 0.009 * np.cos(2.8 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
