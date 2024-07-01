import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV17:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize the temperature and cooling schedule with improved adaptive parameters
        T = 1.2  # Increased starting temperature to enhance global search in initial phase
        T_min = 0.0001  # Even lower minimum temperature for very deep late exploration
        alpha = 0.91  # Slightly softer cooling rate to extend effective search time

        # Mutation and crossover parameters are further refined
        F = 0.78  # Mutation factor adjusted for optimal diverse exploration
        CR = 0.88  # Crossover probability finely tuned for better gene mixing

        population_size = 85  # Optimal population size for this problem setting
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing dynamic mutation with a refined sigmoid and adaptive strategy
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Mutation factor dynamically adapts using a sophisticated model
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.75 + 0.25 * np.sin(2 * np.pi * (evaluation_count / self.budget)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with a temperature-sensitive approach
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.055 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Sophisticated adaptive cooling strategy incorporating a periodic modulation
            adaptive_cooling = alpha - 0.009 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
