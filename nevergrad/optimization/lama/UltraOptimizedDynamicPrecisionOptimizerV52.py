import numpy as np


class UltraOptimizedDynamicPrecisionOptimizerV52:
    def __init__(self, budget=10000):
        self.budget = budget
        self.dim = 5  # Dimensionality is fixed at 5 as per the problem description
        self.lb = -5.0  # Lower bound as per the problem description
        self.ub = 5.0  # Upper bound as per the problem description

    def __call__(self, func):
        # Initialize temperature and cooling parameters more aggressively for deeper searches
        T = 1.2  # Higher initial temperature for a broader initial search
        T_min = 0.0001  # Lower minimum temperature to allow for very fine-grained late-stage optimization
        alpha = 0.88  # Slower cooling rate to maintain the search effectiveness over a longer period

        # Mutation and crossover parameters are refined
        F = 0.78  # Slightly higher Mutation factor to promote exploratory behavior
        CR = 0.90  # Increased Crossover probability to ensure effective gene exchange

        population_size = 90  # Increased population size to improve search diversity
        pop = np.random.uniform(self.lb, self.ub, (population_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        f_opt = np.min(fitness)
        x_opt = pop[np.argmin(fitness)]
        evaluation_count = population_size

        # Introduce a dynamic mutation approach with improved adaptive control
        while evaluation_count < self.budget and T > T_min:
            for i in range(population_size):
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                # Mutation factor now includes a decay term that adjusts more smoothly
                dynamic_F = (
                    F * np.exp(-0.04 * T) * (0.8 + 0.2 * np.tanh(4 * (evaluation_count / self.budget - 0.7)))
                )
                mutant = np.clip(a + dynamic_F * (b - c), self.lb, self.ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                trial_fitness = func(trial)
                evaluation_count += 1
                delta_fitness = trial_fitness - fitness[i]

                # Refined acceptance criteria with an improved temperature scaling factor
                if delta_fitness < 0 or np.random.rand() < np.exp(
                    -delta_fitness / (T * (1 + 0.1 * np.abs(delta_fitness)))
                ):
                    pop[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < f_opt:
                        f_opt = trial_fitness
                        x_opt = trial

            # Adaptive cooling strategy with a new sinusoidal modulation for more nuanced adjustments
            adaptive_cooling = alpha - 0.007 * np.sin(3 * np.pi * evaluation_count / self.budget)
            T *= adaptive_cooling

        return f_opt, x_opt
