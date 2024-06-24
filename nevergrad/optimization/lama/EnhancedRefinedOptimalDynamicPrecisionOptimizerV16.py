import numpy as np


class EnhancedRefinedOptimalDynamicPrecisionOptimizerV16:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters with further refinement
        T = 1.2  # Slightly increased starting temperature to enhance early exploration
        T_min = 0.0003  # Reduced minimum temperature to allow deep exploration at late stages
        alpha = 0.93  # Slightly increased cooling rate to maintain exploration capabilities longer

        # Mutation and crossover parameters are finely tuned for optimal performance
        F = 0.77  # Mutation factor adjusted for a better balance of diversity
        CR = 0.85  # Crossover probability fine-tuned to optimize the exploration-exploitation trade-off

        population_size = 85  # Adjusted population size for better convergence properties
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation with sigmoid-based adaptation for responsive control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Dynamic mutation factor using an improved sigmoid for more responsive control
                dynamic_F = (
                    F
                    * np.exp(-0.07 * T)
                    * (0.65 + 0.35 * np.tanh(3.7 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria with adjusted sensitive temperature function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.065 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy with sinusoidal modulation
            adaptive_cooling = alpha - 0.007 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
