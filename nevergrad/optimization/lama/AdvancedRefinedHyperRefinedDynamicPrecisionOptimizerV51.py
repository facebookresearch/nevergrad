import numpy as np


class AdvancedRefinedHyperRefinedDynamicPrecisionOptimizerV51:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality fixed at 5
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Initialize temperature and cooling parameters with refined tuning
        T = 1.2  # Slightly increased starting temperature for enhanced exploration
        T_min = 0.0003  # Lower minimal temperature for deeper late-stage exploration
        alpha = 0.90  # Adjusted slower cooling rate to extend effective search duration

        # Mutation and crossover parameters finely-tuned for this problem set
        F = 0.78  # Slightly increased mutation factor
        CR = 0.85  # Adjusted crossover probability to maintain diversity while fostering convergence

        population_size = 90  # Increased population size to improve diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Dynamic mutation approach with sigmoid adaptation for mutation factor
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Adaptive F with further refined control via sigmoid function
                dynamic_F = (
                    F
                    * np.exp(-0.06 * T)
                    * (0.65 + 0.35 * np.tanh(4 * (evaluation_count / self.budget - 0.55)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Further refined acceptance criteria incorporating a temperature-dependent function
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1.05 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy incorporating a sinusoidal modulation
            adaptive_cooling = alpha - 0.01 * np.sin(1.5 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
