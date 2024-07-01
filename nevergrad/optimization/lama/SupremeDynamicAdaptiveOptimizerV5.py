import numpy as np


class SupremeDynamicAdaptiveOptimizerV5:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Fixed dimensionality of the problem
        self.lb = -5.0  # Lower bound of the search space
        self.ub = 5.0  # Upper bound of the search space

    def __call__(self, func):
        # Enhanced temperature and cooling schedule
        T = 1.2  # Higher initial temperature for more aggressive early exploration
        T_min = 0.0003  # Lower end temperature for extended fine-tuning stage
        alpha = 0.95  # Less aggressive cooling to allow more thorough search

        # Refined mutation and crossover strategy
        F = 0.8  # Slightly increased mutation factor
        CR = 0.9  # Increased crossover probability to foster better information exchange

        population_size = 100  # Increased population size for better sampling of the search space
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Implementing an adaptive mutation strategy with enhanced dynamic components
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # More aggressive dynamic mutation factor with revised formula
                dynamic_F = (
                    F * np.exp(-0.05 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # More refined acceptance criteria with an aggressive threshold for accepting worse solutions
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.05 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Further enhanced adaptive cooling with modified modulation for precision
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
