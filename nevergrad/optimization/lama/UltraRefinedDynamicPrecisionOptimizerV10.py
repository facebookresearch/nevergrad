import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV10:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is set to 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and advanced cooling parameters
        T = 1.2  # Higher initial temperature for a more global exploration initially
        T_min = 0.0003  # Lower minimum temperature for extensive late-stage exploration
        alpha = 0.93  # Gradual cooling to maintain exploration capabilities longer

        # Mutation and crossover parameters for optimal exploration and exploitation
        F = 0.77  # Mutation factor adjusted for a better balance
        CR = 0.90  # Increased crossover probability for robust gene mixing

        population_size = 85  # Optimized population size
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Employing a dynamic mutation strategy with sigmoidal modulation for precision
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.75 + 0.25 * np.tanh(3.5 * (evaluation_count / self.budget - 0.45)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Adaptive acceptance criteria with a temperature-sensitive function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Sophisticated adaptive cooling strategy with a sinusoidal amplitude modulation
            adaptive_cooling = alpha - 0.009 * np.sin(2.8 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
