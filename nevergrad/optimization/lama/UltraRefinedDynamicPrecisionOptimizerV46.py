import numpy as np


class UltraRefinedDynamicPrecisionOptimizerV46:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality fixed at 5
        self.lb = -5.0  # Lower bound of search space
        self.ub = 5.0  # Upper bound of search space

    def __call__(self, func):
        # Initialize temperature and enhance cooling parameters
        T = 1.18  # Slightly greater initial temperature to increase early exploration
        T_min = 0.0004  # Lower minimum temperature to enable deep exploration in late stages
        alpha = 0.92  # Slower cooling rate to maintain the search phase longer

        # Adjust mutation and crossover parameters for optimal search
        F = 0.77  # Mutation factor adjusted for a more aggressive search
        CR = 0.88  # Crossover probability adjusted for enhanced genetic diversity

        population_size = 85  # Incremented population size for better diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Dynamic mutation adapting with a refined sigmoid function
                dynamic_F = F * (0.75 + 0.25 * np.tanh(4 * (evaluation_count / self.budget - 0.5)))
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Enhanced acceptance criteria with temperature function adaptation
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.07 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Advanced adaptive cooling strategy with sinusoidal modulation
            adaptive_cooling = alpha - 0.0075 * np.cos(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
